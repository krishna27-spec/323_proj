"""Live chatter detection from the NI 9775 in the cDAQ-9174 (or a replayed recording).

The detector is the entropy rule from labels.py, made causal so it can run
while the machine cuts:

  * the no-cutting reference (level and per-axis entropy) comes from a
    calibration phase at the start, with the spindle on and the tool in the
    air, instead of from the quietest 20 % of the whole record;
  * the median smoothing looks only backwards (SMOOTH_S below), so a label
    lags the signal by about half the smoothing length plus one window;
  * the "chatter stays chatter while the level holds" rule is latched from
    the moment chatter is first seen and released when cutting stops.

Everything is relative to the calibration, so the input can stay in volts.

Live (cDAQ-9174 over USB, NI 9775 in slot 4, accelerometer coupler outputs on BNC ai0..ai2):
    python realtime.py --rpm 20000 --device cDAQ1Mod4
Live now also prints what the three trained models say (weights in weights_rt/, made by
train_realtime.py); add --no-models for the rule alone.
Replay a recording through the same code, to check it against labels.py:
    python realtime.py --rpm 20000 --replay Acce_20k_1_r1.lvm
    python realtime.py --rpm 20000 --replay viewer/data/20k_before.f32
"""
import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

import labels as lb
import causal

FS = lb.FS
WIN, HOP = lb.WIN, lb.HOP
CALIB_S = 2.0        # spindle on, tool in the air, at the start of every session
SMOOTH_S = 0.25      # backward median; labels.py uses 0.5 s centred


class OnlineDetector:
    """Feed HOP new samples (3, HOP) at a time; get one result per call."""

    def __init__(self, rpm, fs=FS, calib_s=CALIB_S, smooth_s=SMOOTH_S, n_ch=3):
        self.rpm, self.fs = rpm, fs
        self.buf = np.zeros((n_ch, WIN))
        self.filled = 0
        self.n_calib = max(int(calib_s * fs / HOP), 5)
        self.k_smooth = max(int(smooth_s * fs / HOP), 1)
        self.hist_lev = deque(maxlen=self.k_smooth)
        self.hist_ent = deque(maxlen=self.k_smooth)
        self.calib_lev, self.calib_ent = [], []
        self.air_level = self.air_ent = None
        self.onset_level = None          # level when chatter was first seen
        self.n = 0                       # windows processed

    @property
    def calibrated(self):
        return self.air_level is not None

    def _features(self):
        ent, lev = [], []
        for v in self.buf:
            tc, f, P = lb.frame_spectra(v, self.fs)          # one frame
            ent.append(lb.notched_entropy(P, f, self.rpm, self.fs)[0])
            lev.append(np.sqrt(P[0, f >= 50].sum()))
        return np.sqrt(np.sum(np.square(lev))), np.array(ent)

    def update(self, chunk):
        chunk = np.asarray(chunk, float)
        h = chunk.shape[1]
        self.buf = np.roll(self.buf, -h, axis=1)
        self.buf[:, -h:] = chunk
        self.filled += h
        if self.filled < WIN:
            return None
        self.n += 1
        t = (self.filled - WIN / 2) / self.fs              # window centre, s
        level, ent = self._features()
        self.hist_lev.append(level)
        self.hist_ent.append(ent)
        level_s = float(np.median(self.hist_lev))
        ent_s = np.median(np.array(self.hist_ent), axis=0)

        if not self.calibrated:
            self.calib_lev.append(level_s)
            self.calib_ent.append(ent_s)
            if len(self.calib_lev) >= self.n_calib:
                self.air_level = float(np.median(self.calib_lev))
                self.air_ent = np.median(np.array(self.calib_ent), axis=0)
            return dict(t=t, label=None, name='calibrating', level_ratio=np.nan, drop=np.nan)

        ratio = level_s / self.air_level
        drop = float(np.mean(1 - ent_s / self.air_ent))
        cutting = ratio > lb.CUT_FACTOR
        if not cutting:
            label, self.onset_level = lb.AIR, None
        elif drop >= lb.DROP_CHATTER:
            label = lb.CHATTER
            if self.onset_level is None:
                self.onset_level = level_s
        elif self.onset_level is not None and level_s >= self.onset_level:
            label = lb.CHATTER                              # broadband chatter
        elif drop < lb.DROP_STABLE:
            label = lb.STABLE
        else:
            label = lb.UNCERTAIN
        return dict(t=t, label=label, name=lb.NAMES[label], level_ratio=ratio, drop=drop)


class ModelPredictor:
    """The three trained models (level only, 1D CNN, CNN + GRU) on the live stream.

    Same input as in training: the causal cleaner of causal.py, divided by the std of the
    first CALIB_S seconds (spindle on, tool in the air), windows of 1024 samples per axis,
    hop 512. Each model is the mean of the probabilities of the 3 seeds in weights_rt/.
    The models answer no cutting / stable / chatter and do not need the spindle speed.
    """
    CLASSES = ['no cutting', 'stable', 'chatter']

    def __init__(self, folder='weights_rt', fs=FS, calib_s=CALIB_S, n_ch=3):
        import torch
        from predict_test import MODELS
        self.torch, self.cleaner = torch, causal.CausalCleaner(n_ch, fs)
        self.nets = {}
        for name, cls in MODELS.items():
            stem = name.replace(' + ', '_').replace(' ', '_')
            files = sorted(Path(folder).glob(f'{stem}_seed*.pt'))
            if not files:
                raise FileNotFoundError(f'no weights for {name} in {folder}/ (run train_realtime.py)')
            nets = []
            for f in files:
                m = cls(); m.load_state_dict(torch.load(f, map_location='cpu')); nets.append(m.eval())
            self.nets[name] = nets
        self.buf = np.zeros((n_ch, WIN)); self.filled = 0
        self.n_calib = int(calib_s * fs); self.calib = []; self.bg = None

    def update(self, chunk):
        """Returns {model name: class name} once a full window exists after calibration, else None."""
        clean = self.cleaner(chunk)
        h = clean.shape[1]
        self.buf = np.roll(self.buf, -h, axis=1); self.buf[:, -h:] = clean
        self.filled += h
        if self.bg is None:
            self.calib.append(clean)
            if sum(c.shape[1] for c in self.calib) >= self.n_calib:
                self.bg = np.concatenate(self.calib, axis=1)[:, :self.n_calib].std(1)[:, None]; self.calib = []
            return None
        if self.filled < WIN:
            return None
        x = self.torch.from_numpy((self.buf / self.bg).astype(np.float32))[None]
        out = {}
        with self.torch.no_grad():
            for name, nets in self.nets.items():
                p = sum(self.torch.softmax(m(x), 1)[0].numpy() for m in nets) / len(nets)
                out[name] = self.CLASSES[int(p.argmax())]
        return out


# ---------------------------------------------------------------- sources

def replay_source(path, fs=FS, realtime=False):
    """Yield (3, HOP) chunks from an .lvm file or a viewer .f32 file."""
    if path.endswith('.f32'):
        x = np.fromfile(path, '<f4').reshape(3, -1)
    else:
        x = np.loadtxt(path)[:, 1:4].T
    t0 = time.perf_counter()
    for i in range(0, x.shape[1] - HOP + 1, HOP):
        if realtime:
            wait = t0 + (i + HOP) / fs - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        yield x[:, i:i + HOP]


def daq_source(device, channels='ai0:2', fs=FS, vmax=10.0):
    """Yield (3, HOP) chunks from an NI C Series module, continuously."""
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    from nidaqmx.errors import DaqReadError
    from nidaqmx.stream_readers import AnalogMultiChannelReader

    # On the cDAQ-9174 + NI 9775, some task starts never deliver a sample (the first read
    # times out with -200284, even after 30 s); a module or chassis reset does not prevent
    # it, but the next start works. So a start whose first block is late is retried.
    for attempt in range(1, 6):
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f'{device}/{channels}', min_val=-vmax, max_val=vmax)
            n_ch = task.number_of_channels
            task.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.CONTINUOUS,
                                            samps_per_chan=fs)      # 1 s driver buffer
            reader = AnalogMultiChannelReader(task.in_stream)
            task.start()
            buf = np.zeros((n_ch, HOP))
            try:
                reader.read_many_sample(buf, HOP, timeout=2.0)
            except DaqReadError:
                print(f'DAQ start {attempt} gave no samples in 2 s, restarting the task', file=sys.stderr)
                continue
            while True:
                yield buf.copy()
                reader.read_many_sample(buf, HOP, timeout=2.0)
    raise RuntimeError(f'{device}: no samples after 5 task starts')


# ---------------------------------------------------------------- main

COLOURS = {'no cutting': '\033[90m', 'stable': '\033[32m', 'chatter': '\033[1;31m',
           'uncertain': '\033[33m', 'calibrating': '\033[36m'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rpm', type=float, required=True, help='spindle speed (used to notch its harmonics)')
    ap.add_argument('--device', default='cDAQ1Mod4', help='NI-DAQmx module name (see NI MAX)')
    ap.add_argument('--channels', default='ai0:2', help='X, Y, Z accelerometer channels')
    ap.add_argument('--replay', help='.lvm or .f32 file to replay instead of the DAQ')
    ap.add_argument('--realtime', action='store_true', help='pace the replay at the true sample rate')
    ap.add_argument('--no-models', action='store_true', help='rule only, do not load the trained models')
    ap.add_argument('--weights', default='weights_rt', help='folder with the trained weights')
    ap.add_argument('--log', help='CSV file for one row per window')
    ap.add_argument('--quiet', action='store_true', help='print only label changes')
    a = ap.parse_args(argv)

    src = replay_source(a.replay, realtime=a.realtime) if a.replay else daq_source(a.device, a.channels)
    det = OnlineDetector(a.rpm)
    pred = None if a.no_models else ModelPredictor(a.weights)
    log = csv.writer(open(a.log, 'w', newline='', buffering=1)) if a.log else None
    if log:
        log.writerow(['t', 'label', 'name', 'level_ratio', 'entropy_drop'] + (list(pred.nets) if pred else []))
    if not a.replay:
        print(f'Calibrating for {CALIB_S:.0f} s: spindle on, tool NOT touching the workpiece.')
    last = None
    try:
        for chunk in src:
            r = det.update(chunk)
            m = pred.update(chunk) if pred else None
            if r is None:
                continue
            if log:
                log.writerow([f"{r['t']:.4f}", r['label'], r['name'],
                              f"{r['level_ratio']:.3f}", f"{r['drop']:.4f}"] + (list(m.values()) if m else []))
            key = (r['name'], tuple(m.values()) if m else None)
            if not a.quiet or key != last:
                tail = '' if not m else '   | ' + '  '.join(f'{k}: {v}' for k, v in m.items())
                print(f"{COLOURS[r['name']]}{r['t']:8.2f} s  rule {r['name']:<11s}"
                      f"  level x{r['level_ratio']:6.2f}  drop {r['drop']:+.3f}{tail}\033[0m")
            last = key
    except KeyboardInterrupt:
        pass
    return det


if __name__ == '__main__':
    sys.exit(main() and 0)
