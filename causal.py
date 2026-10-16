"""Causal (streaming) version of the cleaning in denoise.py, so the models can run live.

denoise.py uses zero-phase filtering over a whole file, which needs the future.
This keeps only the parts that can run chunk by chunk:
  1. despike (Hampel filter) with 256 samples of context from the previous chunk
  2. 50 Hz high-pass, 4th order Butterworth, filter state carried between chunks
  3. notches at the mains harmonics 50 .. 400 Hz
The 4.466 Hz machine comb is not removed (it needs a long window).

The same class is used to clean the training data (train_realtime.py) and the live
stream (realtime.py), so the models see identical processing in both.
"""
import numpy as np
from scipy import signal

import denoise as dn

CTX = 256


class CausalCleaner:
    def __init__(self, n_ch=3, fs=dn.FS, q=35.0):
        sec = [signal.butter(4, 50.0, 'hp', fs=fs, output='sos')]
        for k in range(1, 9):
            sec.append(signal.tf2sos(*signal.iirnotch(50.0 * k, q, fs=fs)))
        self.sos = np.vstack(sec)
        self.zi = np.zeros((n_ch, self.sos.shape[0], 2))
        self.tail = None

    def __call__(self, chunk):
        """chunk: (n_ch, h) raw samples -> (n_ch, h) cleaned samples."""
        chunk = np.asarray(chunk, float)
        h = chunk.shape[1]
        ext = chunk if self.tail is None else np.concatenate([self.tail, chunk], axis=1)
        self.tail = ext[:, -CTX:]
        out = np.empty_like(chunk)
        for c in range(chunk.shape[0]):
            v, _ = dn.despike(ext[c])
            out[c], self.zi[c] = signal.sosfilt(self.sos, v[-h:], zi=self.zi[c])
        return out


def clean_offline(x, hop=512):
    """x: (n, 3) raw array -> (n', 3) cleaned, chunk by chunk exactly as the live stream."""
    cc = CausalCleaner(x.shape[1])
    n = (len(x) // hop) * hop
    return np.concatenate([cc(x[i:i + hop].T) for i in range(0, n, hop)], axis=1).T
