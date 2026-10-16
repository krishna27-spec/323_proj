"""Retrain the three models on data cleaned with the causal (streaming) cleaner and save
them in weights_rt/, which realtime.py loads. Then compare with the offline-cleaned models
on the test/ files. Run:  .venv/bin/python train_realtime.py
"""
import glob, json
from pathlib import Path
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

import causal, dataset as ds, denoise as dn, labels as lb
import predict_test as pt

OUT = Path('weights_rt'); OUT.mkdir(exist_ok=True)


def windows(x):
    """Raw (n, 3) -> scaled windows (N, 3, WIN) using the live pipeline (first 2 s = background)."""
    clean = causal.clean_offline(x)
    clean = clean / clean[:2 * dn.FS].std(0)
    return sliding_window_view(clean, lb.WIN, axis=0)[::lb.HOP].astype(np.float32)


if __name__ == '__main__':
    Xs, ys = [], []
    for name, rpm in ds.RUNS.items():
        _, x = dn.load_lvm(f'Acce_{name}_1_r1.lvm')
        L = lb.make_labels(x, rpm)['label']
        X = windows(x); n = min(len(X), len(L)); ok = L[:n] >= 0
        Xs.append(X[:n][ok]); ys.append(L[:n][ok])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    print('training windows', len(y), np.bincount(y, minlength=3))
    files = sorted(glob.glob('test/*.lvm'))
    T = {}
    for f in files:
        _, x = dn.load_lvm(f); T[f] = windows(x)
    probs = {f: {} for f in files}
    for mname, cls in pt.MODELS.items():
        for s in pt.SEEDS:
            m = pt.fit(cls, X, y, s)
            torch.save(m.state_dict(), OUT / f"{mname.replace(' + ', '_').replace(' ', '_')}_seed{s}.pt")
            for f in files:
                probs[f][mname] = probs[f].get(mname, 0) + pt.predict(m, T[f]) / len(pt.SEEDS)
        print('done', mname, flush=True)
    summ = {}
    for f in files:
        rpm = int(f.split('Acce_')[1].split('k')[0]) * 1000
        _, x = dn.load_lvm(f); rule = lb.make_labels(x, rpm)['label']
        off = np.load(f'results/test_pred_{rpm // 1000}k.npz')
        summ[f] = {}
        for mname, p in probs[f].items():
            pred = p.argmax(1); n = min(len(pred), len(rule)); k = rule[:n] >= 0
            summ[f][mname] = dict(counts={ds.CLASSES[c]: int((pred == c).sum()) for c in range(3)},
                                  agree_with_offline_models=float((pred[:n] == off[mname].argmax(1)[:n]).mean()),
                                  agree_with_rule=float((pred[:n][k] == rule[:n][k]).mean()))
    json.dump(summ, open('results/realtime_models.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))
