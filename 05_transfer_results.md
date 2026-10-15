# 05 · Train on public turning data, test on our three runs

Code: `turning.py` (download, tag matching, windows) and the last seven cells of the marimo notebook `train.py` (Experiment 3).
Raw numbers: `results/transfer_metrics.json`. Figure: `figures/05_transfer_confusion.png`.

## Result in one paragraph

The networks learn the turning data (macro F1 about 0.91 on turning recordings they did not train on) but do not carry over to our runs. On our data they call almost every cutting window chatter: stable recall is 0.02 to 0.06, and macro F1 (0.44 to 0.48) is the same as a model that always answers "chatter" (0.46). The only model that transfers at all is the 12-parameter loudness baseline with background scaling (macro F1 0.80, stable recall 0.51).

## Training data

Khasawneh, Otto, Yesilli (2019), *Turning Dataset for Chatter Diagnosis Using Machine Learning*, Mendeley Data, doi:10.17632/hvm4wh3jzx.1, CC BY 4.0. Cite it if these results are used.

- Aluminium 6061 turned on a lathe with a boring bar at four stickout lengths (2, 2.5, 3.5, 4.5 inch), 320 to 1030 rpm, several depths of cut. Sampled at 160 kHz.
- The authors tagged stretches of each cut as stable, intermediate chatter, chatter or unknown, from the amplitude and spectrum of one accelerometer axis. They published the tagged stretches as filtered 10 kHz signals of that one axis.
- `turning.py` carries each tag back to the raw recording (matched by correlation, 0.88 or higher on the pairs checked by hand) and takes all three axes of the tool-post accelerometer.

## How it was prepared

- Resampled 160 kHz to 20 kHz, despiked, 50 Hz high-pass. These are the two steps of `denoise.py` that are not specific to our machine; the mains and comb line removal was not applied.
- Divided by each recording's level in its first 2 s (spindle on, tool not cutting), the same scaling as in experiments 1 and 2. Two of the 59 recordings are already cutting in the first 2 s; they are scaled by the median background of the others.
- Cut into the same windows as ours: 1024 samples (51 ms), hop 512.
- 43,787 windows from 59 recordings: 27,585 stable, 7,508 chatter, 3,521 intermediate, 5,173 unknown. Three tagged files (two unknown, one stable) could not be matched to a recording and are left out.
- Only stable and chatter are used. The turning data has no "no cutting" stretches.

## Setup

- **Train:** stable and chatter windows of 42 turning recordings, 24,760 windows.
- **Turning test:** the other 17 turning recordings, 8,520 stable and 1,813 chatter windows. Whole recordings go to one side.
- **Test on ours:** every stable and chatter window of our three runs (207 stable, 1,295 chatter). None of our data is used for training.
- **Models:** the same three as in `04_training_results.md`.
- **Two input scalings:**
  - background: times the no-cutting level, as before.
  - window: each window divided by its own RMS, so loudness is removed and only the waveform shape is left.
- **Training:** 20 epochs, AdamW, class-weighted loss, mixed precision, 3 seeds. Augmentation: time shift, sign flip, ±20 % gain, and a random axis order, because the axes of the lathe and of our machine do not correspond. 18 trainings took 102 s on the RTX 4060 Laptop GPU (8 GB).

## Results

Mean ± std over 3 seeds.

### Background scaling

| Model | Test part | Accuracy | Macro F1 | Recall stable | Recall chatter |
|---|---|---|---|---|---|
| Level only | turning test | 0.803 ± 0.002 | 0.747 ± 0.002 | 0.772 | 0.947 |
| Level only | ours, all runs | 0.921 ± 0.000 | 0.798 ± 0.001 | 0.512 | 0.986 |
| Level only | ours, 20k | 0.895 | 0.669 | 0.246 | 1.000 |
| Level only | ours, 60k | 0.950 | n/a | n/a | 0.950 |
| Level only | ours, 100k | 0.985 | 0.984 | 0.961 | 1.000 |
| 1D CNN | turning test | 0.942 ± 0.004 | 0.906 ± 0.005 | 0.946 | 0.921 |
| 1D CNN | ours, all runs | 0.828 ± 0.014 | 0.469 ± 0.004 | 0.021 | 0.957 |
| 1D CNN | ours, 20k | 0.865 | 0.488 | 0.026 | 1.000 |
| 1D CNN | ours, 60k | 0.995 | n/a | n/a | 0.995 |
| 1D CNN | ours, 100k | 0.353 ± 0.113 | 0.262 ± 0.071 | 0.013 | 0.565 |
| CNN + GRU | turning test | 0.944 ± 0.002 | 0.909 ± 0.003 | 0.949 | 0.921 |
| CNN + GRU | ours, all runs | 0.862 ± 0.004 | 0.484 ± 0.015 | 0.023 | 0.996 |
| CNN + GRU | ours, 20k | 0.866 | 0.496 | 0.033 | 1.000 |
| CNN + GRU | ours, 60k | 0.995 | n/a | n/a | 0.995 |
| CNN + GRU | ours, 100k | 0.602 ± 0.018 | 0.379 ± 0.002 | 0.004 | 0.973 |

### Window scaling (loudness removed)

| Model | Test part | Accuracy | Macro F1 | Recall stable | Recall chatter |
|---|---|---|---|---|---|
| Level only | turning test | 0.725 ± 0.013 | 0.644 ± 0.005 | 0.729 | 0.708 |
| Level only | ours, all runs | 0.831 ± 0.014 | 0.454 ± 0.004 | 0.000 | 0.964 |
| 1D CNN | turning test | 0.948 ± 0.005 | 0.915 ± 0.007 | 0.953 | 0.921 |
| 1D CNN | ours, all runs | 0.758 ± 0.090 | 0.465 ± 0.049 | 0.055 | 0.871 |
| 1D CNN | ours, 100k | 0.300 ± 0.228 | 0.239 ± 0.154 | 0.078 | 0.438 |
| CNN + GRU | turning test | 0.938 ± 0.010 | 0.901 ± 0.014 | 0.940 | 0.927 |
| CNN + GRU | ours, all runs | 0.743 ± 0.119 | 0.437 ± 0.048 | 0.024 | 0.858 |
| CNN + GRU | ours, 100k | 0.247 ± 0.263 | 0.169 ± 0.154 | 0.048 | 0.371 |

With window scaling the level-only model sees only the ratio between the axes. The per-run rows for 20k and 60k are in `results/transfer_metrics.json`.

60k has no stable windows, so its macro F1 is not meaningful and is left out.

### Our no-cutting windows

The models can only answer stable or chatter. Fraction of our no-cutting windows called chatter:

| Model | Background scaling | Window scaling |
|---|---|---|
| Level only | 0.000 | 1.000 |
| 1D CNN | 0.445 ± 0.314 | 0.410 ± 0.299 |
| CNN + GRU | 0.625 ± 0.292 | 0.762 ± 0.145 |

## What the results mean

- **The training worked.** Both networks reach macro F1 about 0.91 on turning recordings they never saw, well above the loudness baseline on the same recordings (0.75). On turning, waveform shape carries information that loudness does not.
- **What they learned does not apply to our machine.** On our runs the networks behave like "always chatter". Accuracy looks acceptable (0.83 to 0.86) only because 86 % of our cutting windows are chatter; macro F1 and stable recall are the honest numbers.
- **Removing loudness does not help.** With window scaling the networks still call our stable windows chatter, and they start missing real chatter as well (recall 0.86 to 0.87, with large spread between seeds).
- **Loudness transfers partly.** "Several times louder than the idle machine" means something on both machines, which is why the baseline reaches 0.80. It still misses half of our stable windows, and three quarters of them in the 20k run.
- **Networks call an idle machine chatter about half the time**, with large spread between seeds, so their output outside cutting is not usable.

**Likely reason (not tested):** the lathe turns at 5 to 17 revolutions per second, so stable turning has almost no periodic content in a 51 ms window, and chatter is the only strong tone. Our spindle turns at 333 to 1667 revolutions per second, so even stable cutting is strongly periodic. A model trained on turning may read any strong tone as chatter.

## What this changes

- Mixing the turning data into training as it is will not fix the stable class on our machine.
- It supports step 3 of `04_training_results.md`: the input has to be made speed-independent (resample each window to a fixed number of samples per spindle revolution, or random time-stretch augmentation) before data from another machine can help. That would be the next thing to try with this dataset.
- The turning set remains a useful outside test: a model that scores well on it without having trained on it would be evidence of a general chatter detector.
- More stable runs of our own are still the most direct fix.

## Limits of this test

- Our labels and the turning tags both come from the vibration signal, by different rules. A disagreement is partly a difference between labelling rules.
- Our test side has 207 stable windows from two runs.
- Only tool-post accelerometer data was used; the boring-bar accelerometers and the microphone in the dataset were not.
- 20 epochs, fixed, with no early stopping.

## Files

- **Deleted after training:** the two downloaded archives and their extracted folders (about 14 GB).
- **Kept:** `external/turning/turning_windows.npz` (540 MB), the prepared windows. The Experiment 3 cells in `train.py` need it to run again.
- **To remove it too:** `python turning.py --purge`. The Experiment 3 cells then stop with a message instead of failing.
- **To rebuild everything:** `python turning.py` (downloads 7 GB, about 7 minutes, then about 1 minute to build).
