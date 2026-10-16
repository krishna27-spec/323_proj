"""Train the three models on all labelled windows of the original runs and predict the test/ files.

Same windows (1024 samples = 51 ms, hop 512), denoising and background scaling
(first 2 s of each file) as train.py. No test file is used for training.
"""
import glob, json, re, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
import dataset as ds, denoise as dn, labels as lb

SEEDS, EPOCHS, BS = (0, 1, 2), 30, 128
WEIGHTS = Path('weights'); WEIGHTS.mkdir(exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _block(i, o, k, s):
    return nn.Sequential(nn.Conv1d(i, o, k, stride=s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())

class LevelOnly(nn.Module):
    lr = 3e-2
    def __init__(self):
        super().__init__(); self.lin = nn.Linear(3, 3)
    def forward(self, x):
        return self.lin(torch.log(x.pow(2).mean(-1).sqrt() + 1e-6))

class CNN1D(nn.Module):
    lr = 2e-3
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(_block(3, 16, 15, 2), _block(16, 32, 9, 2), _block(32, 64, 9, 2), _block(64, 64, 5, 2))
        self.head = nn.Linear(128, 3)
    def forward(self, x):
        h = self.features(torch.asinh(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))

class CNNGRU(nn.Module):
    lr = 2e-3
    def __init__(self):
        super().__init__()
        self.front = nn.Sequential(_block(3, 16, 15, 2), _block(16, 32, 9, 2), _block(32, 48, 9, 4))
        self.gru = nn.GRU(48, 48, batch_first=True, bidirectional=True)
        self.head = nn.Linear(96, 3)
    def forward(self, x):
        o, _ = self.gru(self.front(torch.asinh(x)).transpose(1, 2))
        return self.head(o.mean(1))

MODELS = {'Level only': LevelOnly, '1D CNN': CNN1D, 'CNN + GRU': CNNGRU}


def fit(cls, X, y, seed):
    torch.manual_seed(seed)
    m = cls().to(dev)
    Xt, yt = torch.from_numpy(X).to(dev), torch.from_numpy(y).long().to(dev)
    c = torch.bincount(yt, minlength=3).float(); pr = c > 0
    w = torch.where(pr, c.sum() / (c.clamp(min=1) * pr.sum()), torch.zeros_like(c))
    steps = int(np.ceil(len(yt) / BS))
    opt = torch.optim.AdamW(m.parameters(), lr=cls.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cls.lr, total_steps=EPOCHS * steps)
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(len(yt), device=dev)
        for s in range(steps):
            i = perm[s * BS:(s + 1) * BS]; xb = Xt[i]
            xb = torch.roll(xb, int(torch.randint(0, xb.shape[-1], (1,))), -1)
            xb = xb * (torch.randint(0, 2, (len(i), 1, 1), device=dev) * 2 - 1)
            xb = xb * torch.exp(0.2 * torch.randn(len(i), 1, 1, device=dev))
            loss = F.cross_entropy(m(xb), yt[i], weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()


def predict(m, X):
    out = []
    with torch.no_grad():
        for s in range(0, len(X), 512):
            out.append(F.softmax(m(torch.from_numpy(X[s:s + 512]).to(dev)), 1).cpu().numpy())
    return np.concatenate(out)


def load_test(path):
    t, x = dn.load_lvm(path)
    rpm = int(re.search(r'Acce_(\d+)k', path).group(1)) * 1000
    clean, _ = dn.denoise(x)
    bg = clean[:2 * dn.FS].std(0)
    X = sliding_window_view(clean / bg, lb.WIN, axis=0)[::lb.HOP].astype(np.float32)
    L = lb.make_labels(x, rpm)
    return dict(X=X, t=L['t'], rule=L['label'], rpm=rpm, n_samples=len(x), bg=bg)


SHORT = {'Level only': 'level', '1D CNN': 'CNN', 'CNN + GRU': 'GRU'}


def export_viewer(files, tests, res):
    """Write the test files in the same format as export_viewer_data.py, as runs '<rpm>k_test'."""
    out = Path('viewer/data'); meta = json.load(open(out / 'meta.json'))
    for f in files:
        d = tests[f]; key = f"{d['rpm'] // 1000}k_test"
        t, x = dn.load_lvm(f)
        L = lb.make_labels(x, d['rpm'])
        air = np.repeat(L['label'] == lb.AIR, lb.HOP)[:len(x)]
        before = x - x[:len(air)][air].mean(0)
        after, _ = dn.denoise(x)
        before.T.astype('<f4').tofile(out / f'{key}_before.f32')
        after.T.astype('<f4').tofile(out / f'{key}_after.f32')
        json.dump({'t': np.round(L['t'], 4).tolist(), 'drop': np.round(L['entropy_drop'], 4).tolist(),
                   'entropy': np.round(L['entropy'], 4).tolist(), 'level': np.round(L['level_ratio'], 3).tolist(),
                   'label': L['label'].tolist()}, open(out / f'{key}_labels.json', 'w'))
        json.dump({'note': 'models trained on the three original runs; pure test file, never used in training; 0 no cutting, 1 stable, 2 chatter',
                   'sets': [{'name': f'{n}, trained on our 3 runs', 'short': SHORT[n], 'label': res[f][n].argmax(1).tolist()}
                            for n in MODELS]}, open(out / f'{key}_pred.json', 'w'))
        cut = L['t'][L['cutting']]
        meta['runs'][key] = {'rpm': d['rpm'], 'n': len(x), 'duration': len(x) / dn.FS,
                             'cut': [float(cut.min()), float(cut.max())] if len(cut) else [0.0, 1.0]}
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)


if __name__ == '__main__':
    files = sorted(glob.glob('test/*.lvm'))
    tests = {f: load_test(f) for f in files}
    tr = ds.build()
    use = tr['usable']
    Xtr, ytr = tr['X'][use], tr['y'][use]
    print('training windows:', len(ytr), np.bincount(ytr, minlength=3))
    res = {f: {} for f in files}
    for name, cls in MODELS.items():
        probs = {f: 0 for f in files}
        for s in SEEDS:
            m = fit(cls, Xtr, ytr, s)
            torch.save(m.state_dict(), WEIGHTS / f"{name.replace(' + ', '_').replace(' ', '_')}_seed{s}.pt")
            for f in files:
                probs[f] = probs[f] + predict(m, tests[f]['X']) / len(SEEDS)
        for f in files:
            res[f][name] = probs[f]
        print('done', name, flush=True)
    summary = {}
    for f in files:
        d = tests[f]; n = len(d['X']); summary[f] = dict(rpm=d['rpm'], samples=d['n_samples'], windows=n, seconds=d['n_samples'] / dn.FS, models={})
        for name, p in res[f].items():
            pred = p.argmax(1)
            summary[f]['models'][name] = {ds.CLASSES[c]: int((pred == c).sum()) for c in range(3)}
        summary[f]['rule_label'] = {lb.NAMES[c]: int((d['rule'] == c).sum()) for c in (0, 1, 2, -1)}
        np.savez_compressed(f"results/test_pred_{d['rpm'] // 1000}k.npz", t=d['t'], rule=d['rule'],
                            **{k: v for k, v in res[f].items()})
    export_viewer(files, tests, res)
    json.dump(summary, open('results/test_predictions.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))
