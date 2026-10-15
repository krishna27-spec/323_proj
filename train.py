import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import json, time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import dataset as ds
    import turning as tn

    return F, Path, ds, json, mo, nn, np, plt, time, tn, torch


@app.cell
def intro(mo):
    mo.md("""
    # Chatter classifier: train and test on the three runs

    **Input:** 51 ms windows (1024 samples) of the denoised 3-axis accelerometer signal, scaled by each run's background level.
    **Target:** the weak labels from `labels.py`: no cutting / stable / chatter. Uncertain windows are left out of training and scoring.
    **Split:** each run is cut into 0.5 s brackets; whole brackets go to train (70 %) or test (30 %), stratified by label. The first window of every bracket is dropped so no samples are shared across the split.
    """)
    return


@app.cell
def data(ds, torch):
    data = ds.build(bracket_s=0.5, test_frac=0.3, seed=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUN_NAMES = list(ds.RUNS)
    SEEDS = [0, 1, 2]
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    return RUN_NAMES, SEEDS, data, device, gpu_name


@app.cell
def split_table(RUN_NAMES, data, ds, gpu_name, mo):
    def _split_rows():
        rows = []
        for r, name in enumerate(RUN_NAMES):
            for te in (False, True):
                m = (data["run"] == r) & (data["is_test"] == te) & data["usable"]
                rows.append({"run": name, "split": "test" if te else "train",
                             **{c: int((data["y"][m] == k).sum()) for k, c in enumerate(ds.CLASSES)}})
        return rows
    split_rows = _split_rows()
    mo.vstack([mo.md(f"**Device:** {gpu_name}. **Windows per run, split and class** (brackets per run and label: `{data['bracket_counts']}`, where -1 is uncertain)"),
               mo.ui.table(split_rows, selection=None)])
    return


@app.cell
def models(nn, torch):
    def _block(i, o, k, s):
        return nn.Sequential(nn.Conv1d(i, o, k, stride=s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())


    class LevelOnly(nn.Module):
        """Baseline: logistic regression on the log RMS of each axis. No waveform shape at all."""
        lr = 3e-2

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 3)

        def forward(self, x):
            return self.lin(torch.log(x.pow(2).mean(-1).sqrt() + 1e-6))


    class CNN1D(nn.Module):
        """Four strided 1D convolutions over the raw window, then global pooling."""
        lr = 2e-3

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(_block(3, 16, 15, 2), _block(16, 32, 9, 2), _block(32, 64, 9, 2), _block(64, 64, 5, 2))
            self.head = nn.Linear(128, 3)

        def forward(self, x):
            h = self.features(torch.asinh(x))
            return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


    class CNNGRU(nn.Module):
        """Convolutional front end (16x shorter sequence) followed by a bidirectional GRU."""
        lr = 2e-3

        def __init__(self):
            super().__init__()
            self.front = nn.Sequential(_block(3, 16, 15, 2), _block(16, 32, 9, 2), _block(32, 48, 9, 4))
            self.gru = nn.GRU(48, 48, batch_first=True, bidirectional=True)
            self.head = nn.Linear(96, 3)

        def forward(self, x):
            o, _ = self.gru(self.front(torch.asinh(x)).transpose(1, 2))
            return self.head(o.mean(1))


    MODELS = {"Level only (baseline)": LevelOnly, "1D CNN": CNN1D, "CNN + GRU": CNNGRU}
    n_params = {k: sum(p.numel() for p in m().parameters()) for k, m in MODELS.items()}
    return MODELS, n_params


@app.cell
def train_fn(F, data, device, np, torch):
    def fit_predict(model_cls, train_idx, eval_idx, seed, epochs=30, bs=128):
        """Train on train_idx, return class probabilities for eval_idx."""
        torch.manual_seed(seed)
        model = model_cls().to(device)
        Xtr = torch.from_numpy(data["X"][train_idx]).to(device)
        ytr = torch.from_numpy(data["y"][train_idx]).long().to(device)
        counts = torch.bincount(ytr, minlength=3).float()
        present = counts > 0
        weight = torch.where(present, counts.sum() / (counts.clamp(min=1) * present.sum()), torch.zeros_like(counts))
        steps = int(np.ceil(len(ytr) / bs))
        opt = torch.optim.AdamW(model.parameters(), lr=model_cls.lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=model_cls.lr, total_steps=epochs * steps)
        for _ep in range(epochs):
            model.train()
            perm = torch.randperm(len(ytr), device=device)
            for _s in range(steps):
                idx = perm[_s * bs:(_s + 1) * bs]
                xb = Xtr[idx]
                # augmentation: circular time shift, random sign, +-20 % gain
                xb = torch.roll(xb, int(torch.randint(0, xb.shape[-1], (1,))), -1)
                xb = xb * (torch.randint(0, 2, (len(idx), 1, 1), device=device) * 2 - 1)
                xb = xb * torch.exp(0.2 * torch.randn(len(idx), 1, 1, device=device))
                loss = F.cross_entropy(model(xb), ytr[idx], weight=weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        out = []
        with torch.no_grad():
            for _s in range(0, len(eval_idx), 512):
                xe = torch.from_numpy(data["X"][eval_idx[_s:_s + 512]]).to(device)
                out.append(F.softmax(model(xe), 1).cpu().numpy())
        return np.concatenate(out)


    def summarise(per_seed):
        """Mean and std over seeds of the metrics dicts from ds.metrics."""
        keys = ("accuracy", "macro_f1", "precision", "recall", "f1")
        out = {k: (np.mean([m[k] for m in per_seed], 0), np.std([m[k] for m in per_seed], 0)) for k in keys}
        out["support"] = per_seed[0]["support"]
        out["confusion"] = per_seed[0]["confusion"]
        return out

    return fit_predict, summarise


@app.cell
def exp_split(
    MODELS,
    RUN_NAMES,
    SEEDS,
    data,
    ds,
    fit_predict,
    np,
    summarise,
    time,
):
    # Experiment 1: bracket split. Train on the train brackets of all three runs, test on the test brackets.
    def _run_split():
        tr = np.flatnonzero(data["usable"] & ~data["is_test"])
        ev = np.arange(len(data["y"]))                      # predict every window, score only test ones
        te_mask = data["usable"] & data["is_test"]
        res, preds = {}, {}
        t0 = time.time()
        for name, cls in MODELS.items():
            per = {"all": [], **{r: [] for r in RUN_NAMES}}
            for seed in SEEDS:
                p = fit_predict(cls, tr, ev, seed).argmax(1)
                if seed == SEEDS[0]:
                    preds[name] = p
                per["all"].append(ds.metrics(data["y"][te_mask], p[te_mask]))
                for r, rn in enumerate(RUN_NAMES):
                    m = te_mask & (data["run"] == r)
                    per[rn].append(ds.metrics(data["y"][m], p[m]))
            res[name] = {k: summarise(v) for k, v in per.items()}
        return res, preds, time.time() - t0
    split_results, split_preds, split_seconds = _run_split()
    return split_preds, split_results, split_seconds


@app.cell
def exp_loro(
    MODELS,
    RUN_NAMES,
    SEEDS,
    data,
    ds,
    fit_predict,
    np,
    summarise,
    time,
):
    # Experiment 2: leave one run out. Train on two spindle speeds, test on the third, which the model never saw.
    def _run_loro():
        res = {}
        t0 = time.time()
        for name, cls in MODELS.items():
            res[name] = {}
            for r, rn in enumerate(RUN_NAMES):
                tr = np.flatnonzero(data["usable"] & (data["run"] != r))
                te = np.flatnonzero(data["usable"] & (data["run"] == r))
                res[name][rn] = summarise([ds.metrics(data["y"][te], fit_predict(cls, tr, te, s).argmax(1)) for s in SEEDS])
        return res, time.time() - t0
    loro_results, loro_seconds = _run_loro()
    return loro_results, loro_seconds


@app.cell
def tables(
    MODELS,
    RUN_NAMES,
    SEEDS,
    ds,
    gpu_name,
    loro_results,
    loro_seconds,
    mo,
    n_params,
    np,
    split_results,
    split_seconds,
):
    def _pm(v, i=None):
        m, s = (v[0], v[1]) if i is None else (v[0][i], v[1][i])
        return "n/a" if np.isnan(m) else f"{m:.3f} ± {s:.3f}"


    def result_rows(results, parts):
        rows = []
        for name in MODELS:
            for part in parts:
                r = results[name][part]
                row = {"model": name, "test part": part, "accuracy": _pm(r["accuracy"]), "macro F1": _pm(r["macro_f1"])}
                for k, c in enumerate(ds.CLASSES):
                    row[f"F1 {c}"] = _pm(r["f1"], k)
                    row[f"precision {c}"] = _pm(r["precision"], k)
                    row[f"recall {c}"] = _pm(r["recall"], k)
                    row[f"n {c}"] = int(r["support"][k])
                rows.append(row)
        return rows
    split_rows_out = result_rows(split_results, ["all"] + RUN_NAMES)
    loro_rows_out = result_rows(loro_results, RUN_NAMES)
    mo.vstack([
        mo.md(f"## Experiment 1: bracket split (test brackets of all runs)\nMean ± std over {len(SEEDS)} seeds. Trained in {split_seconds:.0f} s on {gpu_name}. Parameters: `{n_params}`"),
        mo.ui.table(split_rows_out, selection=None, page_size=12),
        mo.md(f"## Experiment 2: leave one run out (test run never seen in training)\nMean ± std over {len(SEEDS)} seeds. Trained in {loro_seconds:.0f} s."),
        mo.ui.table(loro_rows_out, selection=None, page_size=12),
    ])
    return


@app.cell
def save(
    Path,
    SEEDS,
    ds,
    gpu_name,
    json,
    loro_results,
    mo,
    n_params,
    np,
    split_results,
):
    def _plain(r):
        return {k: ([np.asarray(x).tolist() for x in v] if isinstance(v, tuple) else np.asarray(v).tolist()) for k, v in r.items()}
    Path("results").mkdir(exist_ok=True)
    json.dump({"classes": ds.CLASSES, "seeds": SEEDS, "n_params": n_params, "device": gpu_name,
               "note": "tuple entries are [mean, std] over seeds; per-class arrays follow the order in classes",
               "bracket_split": {m: {p: _plain(r) for p, r in d.items()} for m, d in split_results.items()},
               "leave_one_run_out": {m: {p: _plain(r) for p, r in d.items()} for m, d in loro_results.items()}},
              open("results/metrics.json", "w"), indent=1)
    mo.md("Saved `results/metrics.json`.")
    return


@app.cell
def fig_confusion(
    MODELS,
    Path,
    RUN_NAMES,
    ds,
    loro_results,
    np,
    plt,
    split_results,
):
    def _fig_confusion():
        fig, ax = plt.subplots(2, len(MODELS), figsize=(4.6 * len(MODELS), 8.4))
        for j, name in enumerate(MODELS):
            loro_cm = sum(loro_results[name][rn]["confusion"] for rn in RUN_NAMES)
            for i, (cm, title) in enumerate([(split_results[name]["all"]["confusion"], "bracket split"), (loro_cm, "leave one run out")]):
                a = ax[i, j]
                frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
                a.imshow(frac, cmap="Blues", vmin=0, vmax=1)
                for y in range(3):
                    for x in range(3):
                        a.text(x, y, f"{cm[y, x]}\n{100 * frac[y, x]:.0f}%", ha="center", va="center", fontsize=10,
                               color="white" if frac[y, x] > 0.55 else "#0b0b0b")
                a.set_xticks(range(3), ds.CLASSES, fontsize=9)
                a.set_yticks(range(3), ds.CLASSES, fontsize=9, rotation=90, va="center")
                a.set_title(f"{name}\n{title}", fontsize=11, fontweight="bold")
                if j == 0:
                    a.set_ylabel("true label")
                if i == 1:
                    a.set_xlabel("predicted")
        fig.tight_layout()
        Path("figures").mkdir(exist_ok=True)
        fig.savefig("figures/04_confusion.png", dpi=110, facecolor="#fcfcfb")
        return fig
    _fig_confusion()
    return


@app.cell
def fig_timeline(MODELS, RUN_NAMES, data, np, plt, split_preds):
    def _fig_timeline():
        colors = {0: "#c3c2b7", 1: "#2a78d6", 2: "#e34948", -1: "#eda100"}
        names = list(MODELS)
        fig, ax = plt.subplots(len(RUN_NAMES), 1, figsize=(15, 2.6 * len(RUN_NAMES)))
        for r, rn in enumerate(RUN_NAMES):
            m = data["run"] == r
            t, y = data["t"][m], data["y"][m]
            cut = t[y != 0]
            lo, hi = cut.min() - 2, cut.max() + 2
            rows = [("label", y)] + [(n, split_preds[n][m]) for n in names] + [("test brackets", np.where(data["is_test"][m], 9, 8))]
            for k, (lab, v) in enumerate(rows):
                for cls in np.unique(v):
                    c = {8: "#fcfcfb", 9: "#0b0b0b"}.get(int(cls), colors.get(int(cls)))
                    ax[r].scatter(t[v == cls], np.full((v == cls).sum(), -k), s=18, marker="s", color=c, linewidths=0)
            ax[r].set_yticks(-np.arange(len(rows)), [x[0] for x in rows], fontsize=9)
            ax[r].set_xlim(lo, hi)
            ax[r].set_ylim(-len(rows) + 0.4, 0.6)
            ax[r].set_title(f"{rn} rpm", loc="left", fontsize=11, fontweight="bold")
            for s in ("top", "right", "left"):
                ax[r].spines[s].set_visible(False)
        ax[-1].set_xlabel("Time (s)")
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=colors[k], label=l) for k, l in ((0, "no cutting"), (1, "stable"), (-1, "uncertain (not scored)"), (2, "chatter"))] + [Patch(color="#0b0b0b", label="test bracket")],
                   loc="lower center", ncol=5, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig("figures/04_timeline.png", dpi=110, facecolor="#fcfcfb")
        return fig
    _fig_timeline()
    return


@app.cell(hide_code=True)
def transfer_intro(mo):
    mo.md("""
    # Experiment 3: train on public turning data, test on our three runs

    **Training data:** Khasawneh, Otto, Yesilli (2019), *Turning Dataset for Chatter Diagnosis Using Machine Learning*, Mendeley Data, doi:10.17632/hvm4wh3jzx.1 (CC BY 4.0). Aluminium turned with a boring bar at four stickout lengths, 320 to 1030 rpm. `turning.py` carries the authors' tags back to the raw 160 kHz recordings and returns the three axes of the tool-post accelerometer at 20 kHz, in the same 1024-sample windows as ours.
    **Classes:** stable and chatter only. The authors' "intermediate" and "unknown" stretches are left out, and the turning data has no "no cutting" stretches.
    **Test:** every stable and chatter window of our three runs. None of our data is used for training here.
    """)
    return


@app.cell(hide_code=True)
def turning_data(Path, mo, np, tn):
    mo.stop(not Path(tn.CACHE).exists(), mo.md("**Turning windows not found.** Run `python turning.py` to download the dataset (7 GB) and build the window cache."))
    turning = tn.load()
    turning_is_test = tn.split_cuts(turning, test_frac=0.3, seed=0)


    def _turning_rows():
        rows = []
        for so in np.unique(turning["stickout"]):
            for te in (False, True):
                m = (turning["stickout"] == so) & (turning_is_test == te)
                rows.append({"stickout (inch)": float(so), "split": "test" if te else "train", "recordings": len(np.unique(turning["cut"][m])),
                             "stable": int((turning["y"][m] == 1).sum()), "chatter": int((turning["y"][m] == 2).sum()),
                             "intermediate (unused)": int((turning["tag"][m] == "i").sum()), "unknown (unused)": int((turning["tag"][m] == "u").sum())})
        return rows
    turning_rows = _turning_rows()
    mo.vstack([mo.md(f"**Turning windows per stickout and split.** Whole recordings go to train (70 %) or test (30 %). {len(turning['cuts'])} recordings, {len(turning['skipped'])} tagged files could not be matched to a recording."),
               mo.ui.table(turning_rows, selection=None)])
    return turning, turning_is_test


@app.cell(hide_code=True)
def transfer_fn(F, device, np, torch):
    SCALINGS = {"background": "times the no-cutting level (as in experiments 1 and 2)",
                "window": "each window divided by its own RMS, so loudness is removed"}


    def _scale(x, how):
        if how == "background":
            return x
        return x / x.pow(2).mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)


    def fit_transfer(model_cls, Xtr, ytr, evals, seed, how, epochs=20, bs=256):
        """Train on GPU tensors (Xtr, ytr) that hold stable and chatter only. Returns the
        predicted class (1 stable, 2 chatter) for each tensor in evals."""
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        amp = device.type == "cuda"
        model = model_cls().to(device)
        counts = torch.bincount(ytr, minlength=3).float()
        present = counts > 0
        weight = torch.where(present, counts.sum() / (counts.clamp(min=1) * present.sum()), torch.zeros_like(counts))
        steps = int(np.ceil(len(ytr) / bs))
        opt = torch.optim.AdamW(model.parameters(), lr=model_cls.lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=model_cls.lr, total_steps=epochs * steps)
        scaler = torch.amp.GradScaler(enabled=amp)
        for _ep in range(epochs):
            model.train()
            perm = torch.randperm(len(ytr), device=device)
            for _s in range(steps):
                idx = perm[_s * bs:(_s + 1) * bs]
                xb = Xtr[idx]
                # augmentation: circular time shift, random sign, +-20 % gain, and a random
                # axis order, because the axes of the lathe and of our machine do not correspond
                xb = torch.roll(xb, int(torch.randint(0, xb.shape[-1], (1,))), -1)
                xb = xb * (torch.randint(0, 2, (len(idx), 1, 1), device=device) * 2 - 1)
                xb = xb * torch.exp(0.2 * torch.randn(len(idx), 1, 1, device=device))
                xb = _scale(xb[:, torch.randperm(3, device=device)], how)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    loss = F.cross_entropy(model(xb).float(), ytr[idx], weight=weight)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
        model.eval()
        out = []
        with torch.no_grad():
            for Xe in evals:
                p = [model(_scale(Xe[_s:_s + 2048], how))[:, 1:].argmax(1) + 1 for _s in range(0, len(Xe), 2048)]
                out.append(torch.cat(p).cpu().numpy())
        return out

    return SCALINGS, fit_transfer


@app.cell(hide_code=True)
def exp_transfer(
    MODELS,
    RUN_NAMES,
    SCALINGS,
    SEEDS,
    data,
    device,
    ds,
    fit_transfer,
    np,
    summarise,
    time,
    torch,
    turning,
    turning_is_test,
):
    # Experiment 3: train on the turning train recordings, test on the turning test recordings and on all of our cutting windows.
    def _run_transfer():
        def gpu(a):
            return torch.from_numpy(a).to(device)
        labelled = turning["y"] > 0
        tr, te = labelled & ~turning_is_test, labelled & turning_is_test
        ours_cut = data["usable"] & (data["y"] > 0)
        ours_air = data["usable"] & (data["y"] == 0)
        Xtr, ytr = gpu(turning["X"][tr]), gpu(turning["y"][tr]).long()
        evals = [gpu(turning["X"][te]), gpu(data["X"][ours_cut]), gpu(data["X"][ours_air])]
        y_turn, y_ours, run_ours = turning["y"][te], data["y"][ours_cut], data["run"][ours_cut]
        res, air, preds = {}, {}, {}
        t0 = time.time()
        for how in SCALINGS:
            res[how], air[how], preds[how] = {}, {}, {}
            for name, cls in MODELS.items():
                per = {"turning test": [], "ours, all runs": [], **{f"ours, {rn}": [] for rn in RUN_NAMES}}
                air_frac = []
                for seed in SEEDS:
                    p_turn, p_ours, p_air = fit_transfer(cls, Xtr, ytr, evals, seed, how)
                    if seed == SEEDS[0]:
                        preds[how][name] = p_ours
                    per["turning test"].append(ds.metrics(y_turn, p_turn))
                    per["ours, all runs"].append(ds.metrics(y_ours, p_ours))
                    for r, rn in enumerate(RUN_NAMES):
                        per[f"ours, {rn}"].append(ds.metrics(y_ours[run_ours == r], p_ours[run_ours == r]))
                    air_frac.append(float((p_air == 2).mean()))
                res[how][name] = {k: summarise(v) for k, v in per.items()}
                air[how][name] = (float(np.mean(air_frac)), float(np.std(air_frac)))
        return res, air, preds, int(tr.sum()), time.time() - t0
    transfer_results, transfer_air, transfer_preds, transfer_n_train, transfer_seconds = _run_transfer()
    return transfer_air, transfer_n_train, transfer_results, transfer_seconds


@app.cell(hide_code=True)
def transfer_tables(
    MODELS,
    SCALINGS,
    SEEDS,
    gpu_name,
    mo,
    np,
    transfer_air,
    transfer_n_train,
    transfer_results,
    transfer_seconds,
):
    def _pm3(v, i=None):
        m, s = (v[0], v[1]) if i is None else (v[0][i], v[1][i])
        return "n/a" if np.isnan(m) else f"{m:.3f} ± {s:.3f}"


    def _transfer_rows():
        rows = []
        for how in SCALINGS:
            for name in MODELS:
                for part, r in transfer_results[how][name].items():
                    rows.append({"input": how, "model": name, "test part": part, "accuracy": _pm3(r["accuracy"]), "macro F1": _pm3(r["macro_f1"]),
                                 "recall stable": _pm3(r["recall"], 1), "recall chatter": _pm3(r["recall"], 2),
                                 "precision stable": _pm3(r["precision"], 1), "precision chatter": _pm3(r["precision"], 2),
                                 "n stable": int(r["support"][1]), "n chatter": int(r["support"][2])})
        return rows
    transfer_rows = _transfer_rows()
    transfer_air_rows = [{"input": how, "model": name, "no-cutting windows called chatter": f"{transfer_air[how][name][0]:.3f} ± {transfer_air[how][name][1]:.3f}"}
                         for how in SCALINGS for name in MODELS]
    mo.vstack([
        mo.md(f"## Experiment 3: trained on turning, tested on ours\nMean ± std over {len(SEEDS)} seeds. {transfer_n_train} training windows, {len(SCALINGS) * len(MODELS) * len(SEEDS)} trainings in {transfer_seconds:.0f} s on {gpu_name}. Input scalings: `{SCALINGS}`"),
        mo.ui.table(transfer_rows, selection=None, page_size=15),
        mo.md("**Our no-cutting windows.** The turning data has no such class, so the models can only answer stable or chatter:"),
        mo.ui.table(transfer_air_rows, selection=None),
    ])
    return


@app.cell(hide_code=True)
def transfer_save(
    Path,
    SCALINGS,
    SEEDS,
    ds,
    gpu_name,
    json,
    mo,
    np,
    transfer_air,
    transfer_n_train,
    transfer_results,
):
    def _plain3(r):
        return {k: ([np.asarray(x).tolist() for x in v] if isinstance(v, tuple) else np.asarray(v).tolist()) for k, v in r.items()}
    Path("results").mkdir(exist_ok=True)
    json.dump({"classes": ds.CLASSES, "seeds": SEEDS, "device": gpu_name, "scalings": SCALINGS, "n_train_windows": transfer_n_train,
               "train": "turning dataset doi:10.17632/hvm4wh3jzx.1, train recordings, stable and chatter tags only",
               "note": "tuple entries are [mean, std] over seeds; per-class arrays follow the order in classes; class 'no cutting' is absent",
               "results": {how: {m: {p: _plain3(r) for p, r in d.items()} for m, d in by_model.items()} for how, by_model in transfer_results.items()},
               "no_cutting_called_chatter": transfer_air},
              open("results/transfer_metrics.json", "w"), indent=1)
    mo.md("Saved `results/transfer_metrics.json`.")
    return


@app.cell(hide_code=True)
def fig_transfer(MODELS, Path, SCALINGS, ds, np, plt, transfer_results):
    def _fig_transfer():
        names = ds.CLASSES[1:]
        fig, ax = plt.subplots(len(SCALINGS), len(MODELS), figsize=(4.2 * len(MODELS), 4.2 * len(SCALINGS)))
        for i, how in enumerate(SCALINGS):
            for j, name in enumerate(MODELS):
                a = ax[i, j]
                cm = transfer_results[how][name]["ours, all runs"]["confusion"][1:, 1:]
                frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
                a.imshow(frac, cmap="Blues", vmin=0, vmax=1)
                for y in range(2):
                    for x in range(2):
                        a.text(x, y, f"{cm[y, x]}\n{100 * frac[y, x]:.0f}%", ha="center", va="center", fontsize=11,
                               color="white" if frac[y, x] > 0.55 else "#0b0b0b")
                a.set_xticks(range(2), names, fontsize=9)
                a.set_yticks(range(2), names, fontsize=9, rotation=90, va="center")
                a.set_title(f"{name}\ninput: {how}", fontsize=11, fontweight="bold")
                if j == 0:
                    a.set_ylabel("true label (ours)")
                if i == len(SCALINGS) - 1:
                    a.set_xlabel("predicted")
        fig.suptitle("Trained on turning, tested on our three runs (first seed)", fontsize=12)
        fig.tight_layout()
        Path("figures").mkdir(exist_ok=True)
        fig.savefig("figures/05_transfer_confusion.png", dpi=110, facecolor="#fcfcfb")
        return fig
    _fig_transfer()
    return


@app.cell(hide_code=True)
def stretch_intro(mo):
    mo.md("""
    # Experiment 4: the same, with time-stretch augmentation

    In experiment 3 the networks called nearly all of our cutting windows chatter. The chatter tone sits at about 0.9 to 3 kHz in the turning data and at 7.5 to 8.8 kHz in our runs, so a network trained on turning never sees a chatter tone where ours are.

    **Time-stretch:** each training window is cut from the continuous turning signal with a random length of `s` x 1024 samples and resampled to 1024 samples. That moves every frequency in it up by the factor `s` (down for `s` < 1). `s` is drawn between 0.5 and 8, evenly on a log scale. A chatter stretch is never compressed so far that its main tone would leave the 10 kHz band.
    **Control:** the same sampler with `s` = 1, so the only difference between the two rows is the stretch.
    **Test:** unchanged. Our windows are not stretched.
    """)
    return


@app.cell(hide_code=True)
def stretch_fn(F, device, ds, np, tn, torch):
    STRETCH = {"no stretch (control)": (1.0, 1.0), "time-stretch 0.5x to 8x": (0.5, 8.0)}
    TONE_MAX_HZ = 9000      # highest frequency a chatter stretch's main tone may be moved to


    def stretch_bank(segs):
        """All continuous training stretches as one long GPU signal, with the start, length
        and label of each, and the largest compression each one allows."""
        f = np.fft.rfftfreq(ds.WIN, 1 / tn.dn.FS)
        cap = []
        for x, y in segs:
            fr = np.lib.stride_tricks.sliding_window_view(x, ds.WIN, axis=1)[:, ::ds.HOP] * np.hanning(ds.WIN)
            P = (np.abs(np.fft.rfft(fr, axis=-1)) ** 2).sum((0, 1))
            cap.append(TONE_MAX_HZ / f[3 + P[3:].argmax()] if y == 2 else np.inf)
        n = np.array([x.shape[1] for x, _ in segs])
        return dict(sig=torch.from_numpy(np.concatenate([x for x, _ in segs], 1)).to(device), start=np.r_[0, np.cumsum(n)[:-1]],
                    n=n, y=np.array([y for _, y in segs]), cap=np.array(cap))


    def stretched_set(bank, n_windows, s_range, seed, chunk=64):
        """Draw n_windows training windows, half stable and half chatter, in chunks that
        share one stretch factor s. Returns GPU tensors (X, y) and the factors used."""
        rng = np.random.default_rng(seed)
        X, Y, used = [], [], []
        while len(X) * chunk < 2 * n_windows:
            s = float(np.exp(rng.uniform(np.log(s_range[0]), np.log(s_range[1]))))
            L = int(round(ds.WIN * s))
            room = {y: np.where((bank["y"] == y) & (bank["cap"] >= s), bank["n"] - L + 1, 0).clip(min=0) for y in (1, 2)}
            if min(r.sum() for r in room.values()) == 0:
                continue
            for y, r in room.items():
                seg = rng.choice(len(r), chunk // 2, p=r / r.sum())
                starts = torch.as_tensor(bank["start"][seg] + rng.integers(0, r[seg]), device=device)
                x = bank["sig"][:, starts[:, None] + torch.arange(L, device=device)].permute(1, 0, 2)
                if L != ds.WIN:
                    x = F.interpolate(x[:, :, None], size=(1, ds.WIN), mode="bilinear", antialias=True)[:, :, 0]
                # compressing by s low-pass filters the signal, which lowers a broadband level by
                # about sqrt(s); undo that so "times the no-cutting level" keeps its meaning
                X.append(x * max(s, 1.0) ** 0.5)
                Y.append(torch.full((len(seg),), y, device=device))
            used.append(s)
        return torch.cat(X)[:n_windows], torch.cat(Y)[:n_windows], np.array(used)

    return STRETCH, TONE_MAX_HZ, stretch_bank, stretched_set


@app.cell(hide_code=True)
def exp_stretch(
    MODELS,
    RUN_NAMES,
    SCALINGS,
    SEEDS,
    STRETCH,
    data,
    device,
    ds,
    fit_transfer,
    np,
    stretch_bank,
    stretched_set,
    summarise,
    time,
    tn,
    torch,
    transfer_n_train,
    turning,
    turning_is_test,
):
    # Experiment 4: train on stretched turning windows, test on the turning test recordings and on every window of our runs.
    def _run_stretch():
        def gpu(a):
            return torch.from_numpy(a).to(device)
        labelled = turning["y"] > 0
        tr, te = labelled & ~turning_is_test, labelled & turning_is_test
        bank = stretch_bank(tn.segments(turning, tr))
        ours_cut = data["usable"] & (data["y"] > 0)
        ours_air = data["usable"] & (data["y"] == 0)
        evals = [gpu(turning["X"][te]), gpu(data["X"])]
        y_turn, y_ours, run_ours = turning["y"][te], data["y"][ours_cut], data["run"][ours_cut]
        res, air, preds = {}, {}, {}
        t0 = time.time()
        for cond, s_range in STRETCH.items():
            sets = {seed: stretched_set(bank, transfer_n_train, s_range, seed)[:2] for seed in SEEDS}
            res[cond], air[cond], preds[cond] = {}, {}, {}
            for how in SCALINGS:
                res[cond][how], air[cond][how], preds[cond][how] = {}, {}, {}
                for name, cls in MODELS.items():
                    per = {"turning test": [], "ours, all runs": [], **{f"ours, {rn}": [] for rn in RUN_NAMES}}
                    air_frac = []
                    for seed in SEEDS:
                        p_turn, p_all = fit_transfer(cls, *sets[seed], evals, seed, how)
                        p_ours = p_all[ours_cut]
                        if seed == SEEDS[0]:
                            preds[cond][how][name] = p_all
                        per["turning test"].append(ds.metrics(y_turn, p_turn))
                        per["ours, all runs"].append(ds.metrics(y_ours, p_ours))
                        for r, rn in enumerate(RUN_NAMES):
                            per[f"ours, {rn}"].append(ds.metrics(y_ours[run_ours == r], p_ours[run_ours == r]))
                        air_frac.append(float((p_all[ours_air] == 2).mean()))
                    res[cond][how][name] = {k: summarise(v) for k, v in per.items()}
                    air[cond][how][name] = (float(np.mean(air_frac)), float(np.std(air_frac)))
        return res, air, preds, time.time() - t0
    stretch_results, stretch_air, stretch_preds, stretch_seconds = _run_stretch()
    return stretch_air, stretch_preds, stretch_results, stretch_seconds


@app.cell(hide_code=True)
def stretch_tables(
    MODELS,
    SCALINGS,
    SEEDS,
    STRETCH,
    gpu_name,
    mo,
    np,
    stretch_air,
    stretch_results,
    stretch_seconds,
):
    def _pm4(v, i=None):
        m, s = (v[0], v[1]) if i is None else (v[0][i], v[1][i])
        return "n/a" if np.isnan(m) else f"{m:.3f} ± {s:.3f}"


    def _stretch_rows():
        rows = []
        for cond in STRETCH:
            for how in SCALINGS:
                for name in MODELS:
                    for part, r in stretch_results[cond][how][name].items():
                        rows.append({"training": cond, "input": how, "model": name, "test part": part, "accuracy": _pm4(r["accuracy"]),
                                     "macro F1": _pm4(r["macro_f1"]), "recall stable": _pm4(r["recall"], 1), "recall chatter": _pm4(r["recall"], 2),
                                     "precision stable": _pm4(r["precision"], 1), "precision chatter": _pm4(r["precision"], 2),
                                     "n stable": int(r["support"][1]), "n chatter": int(r["support"][2]),
                                     "no-cutting windows called chatter": _pm4(stretch_air[cond][how][name]) if part == "ours, all runs" else ""})
        return rows
    stretch_rows = _stretch_rows()
    mo.vstack([
        mo.md(f"## Experiment 4: trained on time-stretched turning, tested on ours\nMean ± std over {len(SEEDS)} seeds. {2 * len(SCALINGS) * len(MODELS) * len(SEEDS)} trainings in {stretch_seconds:.0f} s on {gpu_name}."),
        mo.ui.table(stretch_rows, selection=None, page_size=15),
    ])
    return


@app.cell(hide_code=True)
def stretch_save(
    SCALINGS,
    SEEDS,
    STRETCH,
    TONE_MAX_HZ,
    ds,
    gpu_name,
    json,
    mo,
    np,
    stretch_air,
    stretch_results,
    transfer_n_train,
):
    def _plain4(r):
        return {k: ([np.asarray(x).tolist() for x in v] if isinstance(v, tuple) else np.asarray(v).tolist()) for k, v in r.items()}
    json.dump({"classes": ds.CLASSES, "seeds": SEEDS, "device": gpu_name, "scalings": SCALINGS, "stretch": STRETCH, "tone_max_hz": TONE_MAX_HZ,
               "n_train_windows": transfer_n_train,
               "note": "tuple entries are [mean, std] over seeds; per-class arrays follow the order in classes; class 'no cutting' is absent",
               "results": {c: {h: {m: {p: _plain4(r) for p, r in d.items()} for m, d in bm.items()} for h, bm in bh.items()} for c, bh in stretch_results.items()},
               "no_cutting_called_chatter": stretch_air},
              open("results/stretch_metrics.json", "w"), indent=1)
    mo.md("Saved `results/stretch_metrics.json`.")
    return


@app.cell(hide_code=True)
def fig_stretch(MODELS, SCALINGS, STRETCH, ds, np, plt, stretch_results):
    def _fig_stretch():
        names, cond = ds.CLASSES[1:], list(STRETCH)[1]
        fig, ax = plt.subplots(len(SCALINGS), len(MODELS), figsize=(4.2 * len(MODELS), 4.2 * len(SCALINGS)))
        for i, how in enumerate(SCALINGS):
            for j, name in enumerate(MODELS):
                a = ax[i, j]
                cm = stretch_results[cond][how][name]["ours, all runs"]["confusion"][1:, 1:]
                frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
                a.imshow(frac, cmap="Blues", vmin=0, vmax=1)
                for y in range(2):
                    for x in range(2):
                        a.text(x, y, f"{cm[y, x]}\n{100 * frac[y, x]:.0f}%", ha="center", va="center", fontsize=11,
                               color="white" if frac[y, x] > 0.55 else "#0b0b0b")
                a.set_xticks(range(2), names, fontsize=9)
                a.set_yticks(range(2), names, fontsize=9, rotation=90, va="center")
                a.set_title(f"{name}\ninput: {how}", fontsize=11, fontweight="bold")
                if j == 0:
                    a.set_ylabel("true label (ours)")
                if i == len(SCALINGS) - 1:
                    a.set_xlabel("predicted")
        fig.suptitle(f"Trained on turning with {cond}, tested on our three runs (first seed)", fontsize=12)
        fig.tight_layout()
        fig.savefig("figures/06_stretch_confusion.png", dpi=110, facecolor="#fcfcfb")
        return fig
    _fig_stretch()
    return


@app.cell(hide_code=True)
def viewer_export(Path, RUN_NAMES, data, json, mo, stretch_preds):
    # Per-window model labels for the signal viewer (viewer/index.html), one file per run.
    VIEWER_SETS = [("1D CNN", "time-stretch 0.5x to 8x", "1D CNN, turning + time-stretch", "CNN stretch"),
                   ("CNN + GRU", "time-stretch 0.5x to 8x", "CNN + GRU, turning + time-stretch", "GRU stretch"),
                   ("1D CNN", "no stretch (control)", "1D CNN, turning, no stretch", "CNN plain"),
                   ("Level only (baseline)", "no stretch (control)", "Level only (loudness baseline), turning", "level")]


    def _export_viewer():
        out = Path("viewer/data")
        for r, rn in enumerate(RUN_NAMES):
            m = data["run"] == r
            json.dump({"note": "trained on the turning dataset only, input scaled by the no-cutting level, first seed; 1 stable, 2 chatter",
                       "sets": [{"name": title, "short": short, "label": stretch_preds[cond]["background"][model][m].tolist()}
                                for model, cond, title, short in VIEWER_SETS]}, open(out / f"{rn}_pred.json", "w"))
        return sorted(p.name for p in out.glob("*_pred.json"))
    viewer_files = _export_viewer()
    mo.md(f"Wrote model labels for the viewer: `{viewer_files}`. Open `viewer/index.html` through a local server to see them.")
    return


if __name__ == "__main__":
    app.run()
