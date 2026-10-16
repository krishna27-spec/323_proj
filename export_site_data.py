"""Write the demo recordings that the website in docs/ plays when no DAQ is attached.

For each test run it keeps the cutting part plus a few seconds either side, as int16
(one scale per axis, channel-contiguous), and the per-window labels of the entropy rule
and of the CNN + GRU model, so the page can show the process state as it would live.

Run:  .venv/bin/python export_site_data.py   (needs viewer/data/*_test_* from export_viewer_data.py / predict_test.py)
"""
import json
from pathlib import Path

import numpy as np

SRC = Path('viewer/data')
OUT = Path('docs/data')
RUNS = {'20k_test': 'Test run · 20,000 rpm', '60k_test': 'Test run · 60,000 rpm', '100k_test': 'Test run · 100,000 rpm'}
PAD_BEFORE, PAD_AFTER = 4.0, 3.0      # seconds of no cutting kept around the cut

OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(SRC / 'meta.json'))
fs = meta['fs']
index = []
for run, title in RUNS.items():
    info = meta['runs'][run]
    x = np.fromfile(SRC / f'{run}_before.f32', '<f4').reshape(3, -1)
    a = max(0, int((info['cut'][0] - PAD_BEFORE) * fs))
    b = min(x.shape[1], int((info['cut'][1] + PAD_AFTER) * fs))
    seg = x[:, a:b]
    scale = np.abs(seg).max(1) / 32767
    np.round(seg / scale[:, None]).astype('<i2').tofile(OUT / f'{run}.i16')

    lab = json.load(open(SRC / f'{run}_labels.json'))
    pred = json.load(open(SRC / f'{run}_pred.json'))
    gru = next(s for s in pred['sets'] if s['short'] == 'GRU')['label']
    t = np.array(lab['t'])
    keep = (t >= a / fs) & (t < b / fs)
    json.dump({'fs': fs, 'n': int(seg.shape[1]), 'scale': scale.tolist(), 'rpm': info['rpm'],
               'labels': {'t': np.round(t[keep] - a / fs, 4).tolist(),
                          'rule': np.array(lab['label'])[keep].tolist(),
                          'model': np.array(gru)[keep].tolist()}},
              open(OUT / f'{run}.json', 'w'))
    index.append({'id': run, 'title': title, 'seconds': round(seg.shape[1] / fs, 2)})
    print(f'{run}: {seg.shape[1] / fs:.1f} s, {(OUT / f"{run}.i16").stat().st_size / 1e6:.1f} MB')

json.dump({'runs': index, 'note': 'rule: 0 no cutting, 1 stable, 2 chatter, -1 uncertain; '
                                  'model (CNN + GRU): 0 no cutting, 1 stable, 2 chatter'},
          open(OUT / 'index.json', 'w'), indent=1)
