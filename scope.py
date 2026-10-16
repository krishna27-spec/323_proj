"""Live web scope for the accelerometer on the cDAQ (or a replayed recording).

Starts a small local web server; open the printed address in a browser to see
the X, Y, Z signals scroll, their spectrum and a waterfall, with RMS / peak readouts.
Optional extras: the causal chatter rule of realtime.py (needs --rpm), the trained
models (--models), and a Record button that saves what comes in as an .lvm file in
the same layout as the lab recordings (time, x, y, z), so it can be replayed later.

Live (cDAQ-9174 over USB, NI 9775 in slot 4, ai0..ai3; ai0..ai2 = X, Y, Z):
    .venv/bin/python scope.py
    .venv/bin/python scope.py --rpm 20000 --models
Replay a recording at the true rate, looping:
    .venv/bin/python scope.py --replay Acce_20k_1_r1.lvm --rpm 20000
Then open http://localhost:8050. The Stop / Start button in the page closes and reopens
the DAQ task (add --idle to start stopped).

The website in docs/ (the Chatter Monitor) is served at http://localhost:8050/app/, and
the published copy of it (GitHub Pages) can read this server too: /poll answers any origin.
The rule and the models see the first three channels only (they were trained on X, Y, Z),
and only at 20 kHz.
"""
import argparse
import base64
import json
import queue
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from scipy import signal

import realtime as rt

HERE = Path(__file__).resolve().parent
FIRST_SEND_S = 10.0      # the first poll of a page gets at most this much history
SITE = HERE / 'docs'     # the Chatter Monitor website, served at /app/
MIME = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
        '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png'}


class Scope:
    """Ring buffer of the incoming samples plus the optional rule / model labels."""

    def __init__(self, a):
        self.a, self.fs = a, a.fs
        self.n_buf = int(a.buffer * a.fs)
        self.buf = np.zeros((3 if a.replay else 4, self.n_buf), np.float32)
        self.total = 0                        # samples received since start
        self.lock = threading.Lock()
        self.status, self.error = 'starting', ''
        self.rec_file = self.rec_path = None
        self.rec_n = 0
        self.notice = ''                      # last connection event, shown until the next one
        self.reconnecting = False
        self.labels = []                      # dicts with absolute time t
        self.label_seq = 0                    # id of the next label
        self.q = queue.Queue() if (a.rpm or a.models) else None
        self.det = self.pred = None
        self.reset_analysis = True
        self.running = threading.Event()      # set = acquiring; cleared = DAQ task closed
        if not a.idle:
            self.running.set()

    # ------------------------------------------------------------ acquisition

    def source(self):
        if self.a.replay:
            p = self.a.replay
            self.status = f'loading {p}'
            x = np.fromfile(p, '<f4').reshape(3, -1) if p.endswith('.f32') else np.loadtxt(p, usecols=(1, 2, 3)).T
            hop, t0, i = rt.HOP, time.perf_counter(), 0
            while True:                       # loop the recording at the true rate
                k = i % (x.shape[1] - hop + 1) // hop * hop
                wait = t0 + (i + hop) / self.fs - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                yield x[:, k:k + hop]
                i += hop
        else:
            yield from rt.daq_source(self.a.device, self.a.channels, fs=self.fs, vmax=self.a.vmax)

    def acquire(self):
        while True:
            if not self.running.is_set():
                self.status = 'stopped'
                self.running.wait()
                self.reset_analysis = True    # a gap in the data: the rule calibrates again
            src = self.source()
            try:
                self.status = 'connecting'
                for chunk in src:
                    if not self.running.is_set():
                        break
                    if self.reconnecting:
                        self.notice += f' Reconnected at {datetime.now():%H:%M:%S}.'
                        self.reconnecting = False
                    self.status, self.error = 'running', ''
                    self.push(np.asarray(chunk, np.float32))
            except Exception as e:            # USB pulled, chassis off, driver error ... retry
                self.lost(e)
                time.sleep(2.0)
            finally:
                src.close()                   # leaves the `with nidaqmx.Task()`, so the task is closed

    def lost(self, e):
        """The source failed. Close any recording at the last good sample, so a file never
        joins the data from before and after a gap, and make the rule calibrate again."""
        was_running = self.status == 'running'
        msg = f'{type(e).__name__}: {e}'.strip().splitlines()[0][:300]
        if not self.a.replay and not self.device_present():
            msg = (f'{self.a.device} is not visible: the USB cable is out or the chassis has no power. '
                   'Plug it back in; acquisition restarts by itself.')
        if was_running or msg != self.error:
            print(f'{datetime.now():%H:%M:%S} acquisition: {msg}', file=sys.stderr)
        if was_running:
            note = f'Connection lost at {datetime.now():%H:%M:%S}.'
            if self.rec_file:
                note += f' The recording {self.rec_path.name} was closed at the last good sample ({self.rec_n / self.fs:.1f} s).'
                self.record(False)
            self.notice, self.reconnecting = note, True
        self.status, self.error = 'error', msg
        self.reset_analysis = True

    def device_present(self):
        try:
            from nidaqmx.system import System
            return self.a.device in [d.name for d in System.local().devices]
        except Exception:
            return False

    def set_running(self, on):
        if on:
            self.running.set()
        else:
            self.running.clear()
            self.record(False)

    def push(self, chunk):
        h = chunk.shape[1]
        with self.lock:
            if chunk.shape[0] != self.buf.shape[0]:   # channel count changed: start a fresh buffer
                self.buf = np.zeros((chunk.shape[0], self.n_buf), np.float32)
            start = self.total
            i = start % self.n_buf
            j = min(h, self.n_buf - i)
            self.buf[:, i:i + j] = chunk[:, :j]
            self.buf[:, :h - j] = chunk[:, j:]
            self.total += h
            if self.rec_file:
                t = (self.rec_n + np.arange(h)) / self.fs
                np.savetxt(self.rec_file, np.column_stack([t, chunk.T]), fmt='%.6f', delimiter='\t')
                self.rec_n += h
        if self.q is not None and self.fs == rt.FS:   # rule and models were made for 20 kHz
            self.q.put((start, chunk[:3]))

    def read(self, a, b):
        """Samples [a, b) as a (n_ch, b - a) array; a, b are absolute sample indices."""
        idx = np.arange(a, b) % self.n_buf
        return self.buf[:, idx]

    # ------------------------------------------------------------ analysis

    def analyse(self):
        """Runs the rule (and models) on the stream, in its own thread so a slow
        model never holds up the acquisition."""
        offset = 0
        while True:
            start, chunk = self.q.get()
            if self.reset_analysis:
                self.reset_analysis, offset = False, start
                self.det = rt.OnlineDetector(self.a.rpm) if self.a.rpm else None
                if self.a.models:
                    self.pred = rt.ModelPredictor(self.a.weights)
            r = self.det.update(chunk) if self.det else None
            m = self.pred.update(chunk) if self.pred else None
            if r is None and m is None:
                continue
            t = (start + chunk.shape[1] - rt.WIN / 2) / self.fs if r is None else offset / self.fs + r['t']
            item = dict(id=self.label_seq, t=round(t, 4),
                        rule=r['name'] if r else None,
                        level=None if r is None or np.isnan(r['level_ratio']) else round(r['level_ratio'], 3),
                        drop=None if r is None or np.isnan(r['drop']) else round(r['drop'], 4),
                        models=m, lag=round((self.total - start) / self.fs, 3))
            with self.lock:
                self.labels.append(item)
                self.label_seq += 1
                if len(self.labels) > 4000:
                    del self.labels[:1000]

    def set_fs(self, fs):
        """Closes the DAQ task, starts again at the new rate with an empty buffer."""
        if self.a.replay:
            raise ValueError('the sample rate of a replayed file is fixed')
        fs = int(fs)
        if not 1000 <= fs <= 100_000:
            raise ValueError('sample rate must be 1 to 100 kHz')
        was_running = self.running.is_set()
        self.set_running(False)
        for _ in range(60):                   # the acquire thread closes the task within ~2 s
            if self.status in ('stopped', 'error'):
                break
            time.sleep(0.05)
        with self.lock:
            self.a.fs = self.fs = fs
            self.n_buf = int(self.a.buffer * fs)
            self.buf = np.zeros((self.buf.shape[0], self.n_buf), np.float32)
            self.total = 0
            self.labels = []
        self.reset_analysis = True
        if was_running:
            self.set_running(True)

    # ------------------------------------------------------------ recording

    def record(self, on):
        with self.lock:
            if on and not self.rec_file:
                folder = HERE / self.a.out
                folder.mkdir(exist_ok=True)
                self.rec_path = folder / f"scope_{datetime.now():%Y%m%d_%H%M%S}.lvm"
                self.rec_file, self.rec_n = open(self.rec_path, 'w'), 0
            elif not on and self.rec_file:
                self.rec_file.close()
                self.rec_file = None

    # ------------------------------------------------------------ poll reply

    def poll(self, since, spec_s, label_since):
        with self.lock:
            total = self.total
            a = max(since, total - int(FIRST_SEND_S * self.fs), total - self.n_buf, 0)
            data = self.read(a, total) if total > a else np.zeros((self.buf.shape[0], 0), np.float32)
            n_spec = min(int(spec_s * self.fs), total, self.n_buf)
            seg = self.read(total - n_spec, total) if n_spec >= 256 else None
            labels = [l for l in self.labels if l['id'] >= label_since]
            rec = dict(on=self.rec_file is not None, path=str(self.rec_path.relative_to(HERE)) if self.rec_path else None,
                       seconds=round(self.rec_n / self.fs, 1))
        out = dict(fs=self.fs, start=a, total=total, status=self.status, error=self.error, notice=self.notice,
                   source=self.a.replay or f'{self.a.device}/{self.a.channels}', rpm=self.a.rpm,
                   rule=bool(self.a.rpm), models=self.a.models, acquiring=self.running.is_set(), record=rec, labels=labels,
                   qsize=self.q.qsize() if self.q else 0,
                   data=[base64.b64encode(np.ascontiguousarray(c, '<f4').tobytes()).decode() for c in data])
        if seg is not None:
            f, P = signal.welch(seg, self.fs, nperseg=min(2048, seg.shape[1]), detrend='constant', axis=1)
            out['spec'] = dict(df=float(f[1]), db=np.round(10 * np.log10(P + 1e-20), 1).tolist())
        return out


def make_handler(scope):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, ctype, code=200):
            self.send_response(code)
            self.cors()
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def cors(self):
            # the published site (https://<user>.github.io) reads this local server
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Private-Network', 'true')

        def do_OPTIONS(self):
            self.send_response(204)
            self.cors()
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.end_headers()

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ('/', '/index.html'):
                self.send((HERE / 'scope.html').read_bytes(), 'text/html; charset=utf-8')
            elif u.path == '/app':
                self.send_response(301)
                self.send_header('Location', '/app/')
                self.end_headers()
            elif u.path.startswith('/app/'):
                f = (SITE / (u.path[5:] or 'index.html')).resolve()
                if SITE in f.parents and f.is_file():
                    self.send(f.read_bytes(), MIME.get(f.suffix, 'application/octet-stream'))
                else:
                    self.send(b'not found', 'text/plain', 404)
            elif u.path == '/poll':
                body = scope.poll(int(q.get('since', -1)), float(q.get('spec', 0.5)), int(q.get('lab', 0)))
                self.send(json.dumps(body).encode(), 'application/json')
            else:
                self.send(b'not found', 'text/plain', 404)

        def do_POST(self):
            p = urlparse(self.path).path
            if p == '/record/start':
                scope.record(True)
            elif p == '/record/stop':
                scope.record(False)
            elif p == '/acquire/start':
                scope.set_running(True)
            elif p == '/acquire/stop':
                scope.set_running(False)
            elif p == '/recalibrate':
                scope.reset_analysis = True
            elif p == '/config':
                q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                try:
                    if 'fs' in q:
                        scope.set_fs(float(q['fs']))
                except ValueError as e:
                    return self.send(json.dumps({'error': str(e)}).encode(), 'application/json', 400)
            else:
                return self.send(b'not found', 'text/plain', 404)
            self.send(b'{}', 'application/json')

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cDAQ1Mod4', help='NI-DAQmx module name (nilsdev lists it)')
    ap.add_argument('--channels', default='ai0:3', help='accelerometer channels (the first three = X, Y, Z)')
    ap.add_argument('--fs', type=float, default=rt.FS, help='sample rate, Hz')
    ap.add_argument('--vmax', type=float, default=10.0, help='input range, +/- V')
    ap.add_argument('--replay', help='.lvm or .f32 file to replay (looped, true rate) instead of the DAQ')
    ap.add_argument('--rpm', type=float, help='spindle speed: shows its harmonics and runs the chatter rule')
    ap.add_argument('--models', action='store_true', help='also run the trained models (weights_rt/)')
    ap.add_argument('--weights', default='weights_rt')
    ap.add_argument('--buffer', type=float, default=30.0, help='seconds kept on the server')
    ap.add_argument('--out', default='recordings', help='folder for the Record button')
    ap.add_argument('--idle', action='store_true', help='start with acquisition stopped (press Start in the page)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8050)
    a = ap.parse_args(argv)
    a.fs = int(a.fs)

    scope = Scope(a)
    threading.Thread(target=scope.acquire, daemon=True).start()
    if scope.q is not None:
        threading.Thread(target=scope.analyse, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(scope))
    host = "localhost" if a.host == "127.0.0.1" else a.host
    print(f'Scope on http://{host}:{a.port}   Chatter Monitor on http://{host}:{a.port}/app/   source: '
          f'{a.replay or a.device + "/" + a.channels}   (Ctrl+C to stop)')
    if a.rpm and not a.replay:
        print(f'The rule calibrates for {rt.CALIB_S:.0f} s at the start: spindle on, tool NOT touching the workpiece.')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        scope.record(False)


if __name__ == '__main__':
    main()
