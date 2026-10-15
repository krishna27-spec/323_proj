"""Write the binary and JSON files that viewer/index.html reads.

Run:  .venv/bin/python export_viewer_data.py
"""
import json
from pathlib import Path
import numpy as np
from scipy import signal
import denoise as dn
import labels as lb

RUNS = {'20k': 20_000, '60k': 60_000, '100k': 100_000}
OUT = Path('viewer/data'); OUT.mkdir(parents=True, exist_ok=True)
meta = {'fs': dn.FS, 'runs': {}}

for name, rpm in RUNS.items():
    t, x = dn.load_lvm(f'Acce_{name}_1_r1.lvm')
    L = lb.make_labels(x, rpm)
    air = L['label'] == lb.AIR
    # 'before' = raw with the no-cutting mean removed, so before and after share a zero
    n_air = np.repeat(air, lb.HOP)[:len(x)]
    before = x - x[:len(n_air)][n_air].mean(0)
    after, _ = dn.denoise(x)
    # axis-contiguous float32: all X, then all Y, then all Z
    before.T.astype('<f4').tofile(OUT / f'{name}_before.f32')
    after.T.astype('<f4').tofile(OUT / f'{name}_after.f32')
    cut = L['t'][L['cutting']]
    json.dump({'t': np.round(L['t'], 4).tolist(),
               'drop': np.round(L['entropy_drop'], 4).tolist(),
               'entropy': np.round(L['entropy'], 4).tolist(),
               'level': np.round(L['level_ratio'], 3).tolist(),
               'label': L['label'].tolist()}, open(OUT / f'{name}_labels.json', 'w'))
    meta['runs'][name] = {'rpm': rpm, 'n': len(x), 'duration': len(x) / dn.FS,
                          'cut': [float(cut.min()), float(cut.max())]}
    # how well the fixed comb frequency fits this run (no-cutting part, X axis)
    seg = slice(0, int(min(cut.min() - 0.5, 20) * dn.FS))
    for tag, arr in (('before', before), ('after', after)):
        f, P = signal.welch(arr[seg, 0] - arr[seg, 0].mean(), dn.FS, nperseg=2 * dn.FS)
        lines = sum(P[np.abs(f - k * dn.COMB_F0) <= 0.5].sum() for k in range(12, 90))
        print(f'{name} {tag:6s} no-cutting X: RMS {arr[seg, 0].std():.5f}, share of power on comb lines {100 * lines / P.sum():.1f}%')

meta['thresholds'] = {'stable': lb.DROP_STABLE, 'chatter': lb.DROP_CHATTER}
json.dump(meta, open(OUT / 'meta.json', 'w'), indent=1)
print('wrote', sorted(p.name for p in OUT.iterdir()))
