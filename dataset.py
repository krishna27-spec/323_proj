"""Windows, labels and a leakage-safe train/test split for the three runs.

Each run is cut into time brackets (0.5 s by default). A whole bracket goes to
either train or test, so neighbouring, overlapping windows never end up on both
sides. The first window of every bracket is dropped as a guard, because it
shares samples with the last window of the previous bracket.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import denoise as dn
import labels as lb

RUNS = {'20k': 20_000, '60k': 60_000, '100k': 100_000}
CLASSES = ['no cutting', 'stable', 'chatter']      # label values 0, 1, 2
WIN, HOP = lb.WIN, lb.HOP


def load_run(name, bracket_s=0.5):
    """Windows of the denoised 3-axis signal with their labels and bracket ids."""
    t, x = dn.load_lvm(f'Acce_{name}_1_r1.lvm')
    L = lb.make_labels(x, RUNS[name])
    clean, _ = dn.denoise(x)
    # scale by the background level in the first 2 s (spindle on, no cutting),
    # so the model sees "how many times the background", not sensor units
    clean = clean / clean[:2 * dn.FS].std(0)
    X = sliding_window_view(clean, WIN, axis=0)[::HOP].astype(np.float32)   # (n, 3, WIN)
    assert len(X) == len(L['t'])
    bracket = np.floor(L['t'] / bracket_s).astype(int)
    guard = np.r_[True, np.diff(bracket) != 0]          # first window of each bracket
    return dict(X=X, y=L['label'], t=L['t'], bracket=bracket, guard=guard,
                score=L['entropy_drop'])


def split_brackets(y, bracket, test_frac=0.3, seed=0):
    """Assign whole brackets to train/test, stratified by the bracket's majority label."""
    rng = np.random.default_rng(seed)
    ids = np.unique(bracket)
    major = np.array([np.bincount(y[bracket == b] + 1, minlength=4).argmax() - 1 for b in ids])
    test_ids = []
    for cls in np.unique(major):
        pool = rng.permutation(ids[major == cls])
        k = int(round(test_frac * len(pool)))
        if len(pool) >= 2:
            k = min(max(k, 1), len(pool) - 1)
        test_ids.extend(pool[:k])
    return np.isin(bracket, test_ids), {int(c): int((major == c).sum()) for c in np.unique(major)}


def build(bracket_s=0.5, test_frac=0.3, seed=0):
    """All three runs as one table of windows.

    Returns a dict with X (N, 3, WIN), y, run (index into RUNS), t, bracket,
    is_test, usable (labelled 0/1/2 and not a guard window), and bracket counts.
    """
    parts, counts = [], {}
    for r, name in enumerate(RUNS):
        d = load_run(name, bracket_s)
        is_test, counts[name] = split_brackets(d['y'], d['bracket'], test_frac, seed + r)
        d.update(run=np.full(len(d['y']), r), is_test=is_test)
        parts.append(d)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out['usable'] = (out['y'] >= 0) & ~out['guard']
    out['bracket_counts'] = counts
    return out


def metrics(y_true, y_pred, n_classes=3):
    """Accuracy, per-class precision/recall/F1, macro F1 and the confusion matrix
    (rows = true class, columns = predicted). Classes absent from y_true get NaN
    and are left out of the macro average."""
    cm = np.zeros((n_classes, n_classes), int)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(cm.sum(0) > 0, tp / cm.sum(0), 0.0)
        rec = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(support > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1), np.nan)
        prec = np.where(support > 0, prec, np.nan)
    return dict(accuracy=float(tp.sum() / max(cm.sum(), 1)), precision=prec, recall=rec, f1=f1,
                macro_f1=float(np.nanmean(f1)), support=support, confusion=cm)
