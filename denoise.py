"""Conservative denoising for the micromilling accelerometer files.

Only removes things identified in the no-cutting segment as background that is
not produced by cutting:
  1. single-sample electrical spikes (Hampel filter)
  2. mean, slow drift and the weak sub-50 Hz machine lines (50 Hz high-pass)
  3. narrow periodic lines: mains harmonics and the 4.466 Hz machine comb

It deliberately leaves alone the broadband floor, the spindle harmonics and
everything above 400 Hz, because cutting and chatter live there.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.ndimage import uniform_filter1d

FS = 20_000
COMB_F0 = 4.466      # Hz, measured from the 20k no-cutting segment
COMB_FMAX = 400.0    # comb lines are not visible above this
MAINS_F0 = 50.0
MAINS_FMAX = 400.0


def load_lvm(path):
    """Return t (s) and x (n, 3) = X feed, Y normal-to-feed, Z axial."""
    raw = np.loadtxt(path)
    return raw[:, 0], raw[:, 1:4]


def despike(v, window=11, n_sigma=6.0, scale_window=201):
    """Replace a sample by the local median when it is more than n_sigma local
    standard deviations away from it. Returns (cleaned, mask).

    The scale is the RMS over a longer window (scale_window samples, 10 ms), not
    the spread of the short median window. During cutting the signal oscillates
    near Nyquist, so a short-window scale flags genuine peaks; the longer window
    follows the signal level and only isolated glitches stand out."""
    half = window // 2
    pad = np.pad(v, half, mode='reflect')
    med = np.median(sliding_window_view(pad, window), axis=1)
    centred = v - uniform_filter1d(v, scale_window, mode='reflect')
    rms = np.sqrt(uniform_filter1d(centred ** 2, scale_window, mode='reflect'))
    # floor the scale at one ADC step so quantised flat stretches are not flagged
    step = np.median(np.diff(np.unique(v)))
    scale = np.maximum(rms, step)
    mask = np.abs(v - med) > n_sigma * scale
    out = v.copy()
    out[mask] = med[mask]
    return out, mask


def highpass(v, fc=50.0, fs=FS, order=4):
    sos = signal.butter(order, fc, 'hp', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, v)


def line_frequencies(comb_f0=COMB_F0, comb_fmax=COMB_FMAX,
                     mains_f0=MAINS_F0, mains_fmax=MAINS_FMAX):
    comb = np.arange(1, int(comb_fmax / comb_f0) + 1) * comb_f0
    mains = np.arange(1, int(mains_fmax / mains_f0) + 1) * mains_f0
    return comb, mains


def remove_lines(v, freqs, half_width, fs=FS):
    """Zero narrow FFT bands around each frequency (zero-phase line removal).
    half_width can be a scalar or one value per frequency (Hz)."""
    n = len(v)
    F = np.fft.rfft(v)
    f = np.fft.rfftfreq(n, 1 / fs)
    hw = np.broadcast_to(half_width, np.shape(freqs))
    kill = np.zeros(len(f), bool)
    for f0, w in zip(freqs, hw):
        kill |= np.abs(f - f0) <= w
    F[kill] = 0
    return np.fft.irfft(F, n), kill.sum() * (f[1] - f[0])


def denoise(x, fs=FS, hp_fc=50.0, comb_hw=0.15, mains_hw=0.3, do_despike=True):
    """Denoise one axis or an (n, k) array. Returns (clean, info)."""
    x = np.asarray(x, float)
    if x.ndim == 2:
        outs, infos = zip(*(denoise(x[:, i], fs, hp_fc, comb_hw, mains_hw, do_despike)
                            for i in range(x.shape[1])))
        return np.column_stack(outs), list(infos)
    info = {}
    v = x
    if do_despike:
        v, mask = despike(v)
        info['spike_mask'] = mask
    v = highpass(v, hp_fc, fs)
    comb, mains = line_frequencies()
    # the comb is a harmonic series, so any error in f0 grows with harmonic number
    comb_w = comb_hw + 0.002 * np.arange(1, len(comb) + 1)
    freqs = np.concatenate([comb, mains])
    hw = np.concatenate([comb_w, np.full(len(mains), mains_hw)])
    v, removed_hz = remove_lines(v, freqs, hw, fs)
    info['removed_bandwidth_hz'] = removed_hz
    return v, info
