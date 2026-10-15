"""Windows from the public turning chatter dataset, in the same form as dataset.py.

Khasawneh, Otto, Yesilli (2019), "Turning Dataset for Chatter Diagnosis Using
Machine Learning", Mendeley Data, doi:10.17632/hvm4wh3jzx.1, CC BY 4.0.
Aluminium 6061 turned on a lathe with a boring bar at four stickout lengths,
320-1030 rpm, 160 kHz sampling.

The authors tagged stretches of each cut as stable (s), intermediate chatter (i),
chatter (c) or unknown (u) and published them as filtered 10 kHz signals of one
axis. Here the tags are carried back to the raw recording to get all three axes
of the tool-post accelerometer at our 20 kHz.

    python turning.py            download (7 GB), build the window cache
    python turning.py --clean    delete the download, keep the cache
    python turning.py --purge    delete everything under external/turning
"""
import glob
import json
import os
import shutil
import sys
import urllib.request
import zipfile

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.io import loadmat

import denoise as dn
import labels as lb

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external', 'turning')
CACHE = os.path.join(ROOT, 'turning_windows.npz')
API = 'https://data.mendeley.com/public-api/datasets/hvm4wh3jzx/files?folder_id=root&version=1'
FS_RAW = 160_000
FS_TAG = 10_000
TRIAX = [4, 5, 6]            # columns of d: tool-post accelerometer x, y, z
STICKOUTS = {'2inch': 2.0, '2p5inch': 2.5, '3p5inch': 3.5, '4p5inch': 4.5}
TAGS = {'s': lb.STABLE, 'c': lb.CHATTER, 'i': lb.UNCERTAIN, 'u': lb.UNCERTAIN}
WIN, HOP = lb.WIN, lb.HOP
MIN_MATCH = 0.8              # correlation needed to accept a tag-to-recording match
BG_OUTLIER = 3.0             # first-2-s level this many times the typical one = no idle start


def download():
    """Fetch and unpack the two archives into external/turning."""
    os.makedirs(ROOT, exist_ok=True)
    req = urllib.request.Request(API, headers={'Accept': 'application/vnd.mendeley-public-dataset.1+json'})
    for f in json.load(urllib.request.urlopen(req)):
        name, dest = f['filename'], os.path.join(ROOT, f['filename'])
        if not os.path.exists(dest):
            print('downloading', name, f"{f['size'] / 1e6:.0f} MB")
            # the file host refuses the default urllib user agent
            get = urllib.request.Request(f['content_details']['download_url'], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(get) as r, open(dest, 'wb') as out:
                shutil.copyfileobj(r, out, 1 << 22)
        if name.endswith('.zip'):
            sub = os.path.join(ROOT, 'raw' if 'raw' in name else 'processed')
            if not os.path.isdir(sub):
                zipfile.ZipFile(dest).extractall(sub)


def _tagged_segments():
    """One entry per tagged file: stickout, tag letter, rpm, depth code, 10 kHz time and signal."""
    segs = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'processed', '*', '*.mat'))):
        so = os.path.basename(os.path.dirname(path)).split('_')[0]
        tag, rpm, doc = os.path.basename(path)[:-4].split('_')[:3]
        ts = loadmat(path)['tsDS']
        segs.append(dict(stickout=so, tag=tag, rpm=int(rpm), doc=doc.rstrip('s'), t=ts[:, 0], v=ts[:, 1], path=path))
    return segs


def _match(seg, x_tag):
    """Correlation between a tagged signal and the same stretch of a recording at 10 kHz."""
    i0, n = int(round(seg['t'][0] * FS_TAG)), min(len(seg['v']), 20_000)
    ref = seg['v'][:n] - seg['v'][:n].mean()
    best = 0.0
    for lag in range(-3, 4):
        cand = x_tag[max(i0 + lag, 0):max(i0 + lag, 0) + n]
        if len(cand) == n and cand.std() > 0:
            best = max(best, np.corrcoef(cand, ref)[0, 1])
    return best


def build(verbose=True):
    """Cut every tagged stretch of every recording into windows and cache them.

    Each recording is resampled 160 -> 20 kHz, despiked and high-passed at 50 Hz
    (the two steps of denoise.py that are not specific to our machine), then
    divided by its level in the first 2 s, where the spindle turns and the tool
    is not yet cutting. That is the same scaling dataset.py uses. A recording
    that is already cutting in its first 2 s (level far above the others) is
    scaled by the median background of all recordings instead.
    """
    segs = _tagged_segments()
    parts, cuts, bgs = [], [], []
    for so in STICKOUTS:
        for rec in sorted(glob.glob(os.path.join(ROOT, 'raw', f'timeSeries_{so}Stickout', '*.mat'))):
            name = os.path.basename(rec)
            mine = [s for s in segs if s['stickout'] == so and f"_rpm{s['rpm']}_doc0p{s['doc']}" in name]
            if not mine:
                continue
            d = loadmat(rec)['d'][:, TRIAX]
            x_tag = signal.resample_poly(d[:, 0], 1, FS_RAW // FS_TAG)
            mine = [s for s in mine if _match(s, x_tag) >= MIN_MATCH]
            if not mine:
                continue
            x = signal.resample_poly(d, 1, FS_RAW // dn.FS, axis=0)
            x = np.column_stack([dn.highpass(dn.despike(x[:, c])[0]) for c in range(3)])
            bg = x[:2 * dn.FS].std(0)
            x = x / bg
            cut = len(cuts)
            cuts.append(name)
            bgs.append(bg)
            for s in mine:
                s['used'] = True
                a, b = int(round(s['t'][0] * dn.FS)), int(round(s['t'][-1] * dn.FS))
                if b - a < WIN:
                    continue
                X = sliding_window_view(x[a:b], WIN, axis=0)[::HOP].astype(np.float32)
                n = len(X)
                parts.append(dict(X=X, y=np.full(n, TAGS[s['tag']]), tag=np.full(n, s['tag']),
                                  cut=np.full(n, cut), stickout=np.full(n, STICKOUTS[so]),
                                  rpm=np.full(n, s['rpm']), t=(a + np.arange(n) * HOP + WIN / 2) / dn.FS))
            if verbose:
                print(f"{so:8s} {name:45s} {len(mine)} tagged stretches")
    skipped = [os.path.relpath(s['path'], ROOT) for s in segs if not s.get('used')]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    bgs = np.array(bgs)
    typical = np.median(bgs, 0)
    no_idle = np.flatnonzero((bgs > BG_OUTLIER * typical).any(1))
    for c in no_idle:
        out['X'][out['cut'] == c] *= (bgs[c] / typical).astype(np.float32)[:, None]
    np.savez(CACHE, cuts=np.array(cuts), skipped=np.array(skipped), background=bgs, no_idle=no_idle, **out)
    if verbose:
        print(f"{len(out['y'])} windows from {len(cuts)} recordings, {len(skipped)} tagged files not matched, "
              f"{len(no_idle)} recordings without an idle start: {[cuts[c] for c in no_idle]}")
    return load()


def load():
    """The cached windows: X (N, 3, WIN), y (1 stable, 2 chatter, -1 intermediate or
    unknown), tag (s/c/i/u), cut (index into cuts), stickout (inch), rpm, t."""
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


def segments(d, mask):
    """Continuous signal of every tagged stretch, rebuilt from its overlapping
    windows. Returns a list of (x (3, n), label) for the stretches with windows in mask."""
    idx = np.flatnonzero(mask)
    new = np.r_[True, (np.diff(d['cut'][idx]) != 0) | (d['tag'][idx][1:] != d['tag'][idx][:-1])
                | (np.abs(np.diff(d['t'][idx]) - HOP / dn.FS) > 1e-6)]
    out = []
    for g in np.split(idx, np.flatnonzero(new)[1:]):
        X = d['X'][g]
        out.append((np.concatenate([X[0]] + [w[:, -HOP:] for w in X[1:]], axis=1), int(d['y'][g[0]])))
    return out


def split_cuts(d, test_frac=0.3, seed=0):
    """Assign whole recordings to train/test, so windows of one cut are never on
    both sides. Stratified by whether the recording contains chatter."""
    rng = np.random.default_rng(seed)
    ids = np.unique(d['cut'])
    has_chatter = np.array([(d['y'][d['cut'] == c] == lb.CHATTER).any() for c in ids])
    test_ids = []
    for flag in (False, True):
        pool = rng.permutation(ids[has_chatter == flag])
        test_ids.extend(pool[:int(round(test_frac * len(pool)))])
    return np.isin(d['cut'], test_ids)


if __name__ == '__main__':
    if '--purge' in sys.argv:
        shutil.rmtree(ROOT, ignore_errors=True)
    elif '--clean' in sys.argv:
        for p in glob.glob(os.path.join(ROOT, '*')):
            if p != CACHE:
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    else:
        download()
        build()
