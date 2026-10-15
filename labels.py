"""Weak labels for the micromilling accelerometer runs.

Reproduces the entropy indicator used in the lab's LabVIEW tool (window 1024,
spindle harmonics notched out, normalised spectral entropy) and turns it into
per-window labels:

    0 = no cutting    spindle on, tool not in the workpiece
    1 = stable        cutting, spectrum still flat away from spindle harmonics
    2 = chatter       cutting, energy concentrated at non-harmonic frequencies
   -1 = uncertain     cutting, between the two thresholds (transition)

The spindle speed is used here to notch the harmonics. That is acceptable for
making labels; the model trained on them does not get the spindle speed.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter

FS = 20_000
WIN = 1024           # same window length as the LabVIEW tool (51 ms)
HOP = 512            # 25.6 ms between labels
SMOOTH_S = 0.5       # median smoothing of the indicators
CUT_FACTOR = 2.0     # cutting = level above this many times the no-cutting level
DROP_STABLE = 0.10   # entropy drop below this -> stable
DROP_CHATTER = 0.20  # entropy drop above this -> chatter

AIR, STABLE, CHATTER, UNCERTAIN = 0, 1, 2, -1
NAMES = {AIR: 'no cutting', STABLE: 'stable', CHATTER: 'chatter', UNCERTAIN: 'uncertain'}


def frame_spectra(v, fs=FS, win=WIN, hop=HOP):
    fr = sliding_window_view(v, win)[::hop]
    fr = (fr - fr.mean(1, keepdims=True)) * np.hanning(win)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(win, 1 / fs)
    tc = (np.arange(len(fr)) * hop + win / 2) / fs
    return tc, f, P


def notched_entropy(P, f, rpm, fs=FS, win=WIN, fmin=50.0):
    """Normalised spectral entropy (0..1) after removing spindle harmonics.
    1 = flat spectrum (noise-like), low = energy in a few frequencies."""
    fsp = rpm / 60.0
    k = np.round(f / fsp)
    harmonic = (np.abs(f - k * fsp) <= 1.5 * fs / win) & (k >= 1)
    keep = ~harmonic & (f >= fmin)
    Pn = P[:, keep]
    p = Pn / Pn.sum(1, keepdims=True)
    return -(p * np.log(p + 1e-30)).sum(1) / np.log(keep.sum())


def _smooth(a, fs=FS, hop=HOP):
    return median_filter(a, int(SMOOTH_S * fs / hop) | 1, mode='nearest')


def make_labels(x, rpm, fs=FS):
    """x: (n, 3) raw accelerometer array. Returns a dict of per-window arrays."""
    ent, lev = [], []
    for c in range(x.shape[1]):
        tc, f, P = frame_spectra(x[:, c], fs)
        ent.append(notched_entropy(P, f, rpm, fs))
        lev.append(np.sqrt(P[:, f >= 50].sum(1)))
    ent = np.array(ent)                       # (3, n_windows)
    level = _smooth(np.sqrt((np.array(lev) ** 2).sum(0)))
    ent_s = np.array([_smooth(e) for e in ent])

    # no-cutting reference: the quietest 20 % of the record
    quiet = level <= np.quantile(level, 0.20)
    air_level = np.median(level[quiet])
    air_ent = np.array([np.median(e[quiet]) for e in ent_s])
    cutting = level > CUT_FACTOR * air_level

    # entropy drop relative to the no-cutting value, averaged over the 3 axes
    drop = np.mean(1 - ent_s / air_ent[:, None], axis=0)

    label = np.full(len(tc), AIR)
    label[cutting] = UNCERTAIN
    label[cutting & (drop < DROP_STABLE)] = STABLE
    label[cutting & (drop >= DROP_CHATTER)] = CHATTER

    # Once chatter has started, the vibration can turn broadband: entropy rises
    # again although the cut is at least as violent. Keep those windows as
    # chatter while the level stays at or above the level at chatter onset.
    by_level = np.zeros(len(tc), bool)
    first = np.flatnonzero(label == CHATTER)
    if len(first):
        i0 = first[0]
        by_level[i0:] = cutting[i0:] & (label[i0:] != CHATTER) & (level[i0:] >= level[i0])
        label[by_level] = CHATTER

    return dict(t=tc, level=level, level_ratio=level / air_level, entropy_x=ent_s[0],
                entropy=ent_s, entropy_drop=drop, cutting=cutting, label=label,
                by_level=by_level, air_entropy=air_ent)


def segments(t, label):
    """Collapse per-window labels into (start, end, label) runs."""
    edges = np.flatnonzero(np.diff(label)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(label)]
    half = (t[1] - t[0]) / 2
    return [(t[a] - half, t[b - 1] + half, int(label[a])) for a, b in zip(starts, ends)]
