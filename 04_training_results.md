# 04 · First training results (three runs: 20k, 60k, 100k rpm)

Code: `dataset.py` (windows, bracket split, metrics) and the marimo notebook `train.py` (models, training on GPU, tables, figures).
Raw numbers: `results/metrics.json`. Figures: `figures/04_confusion.png`, `figures/04_timeline.png`.

## Setup

- **Input:** 51 ms windows (1024 samples, hop 512) of the denoised 3-axis signal, divided by each run's background level (first 2 s).
- **Target:** weak labels from `labels.py`: no cutting / stable / chatter. Uncertain windows are not used for training or scoring.
- **Brackets:** each run is cut into 0.5 s brackets. A whole bracket goes to train (70 %) or test (30 %), stratified by the bracket's majority label. The first window of each bracket is dropped so the two sides share no samples.
- **Windows:** 3008 train, 1273 test.

| Run | Split | No cutting | Stable | Chatter |
|---|---|---|---|---|
| 20k | train / test | 765 / 331 | 96 / 34 | 565 / 242 |
| 60k | train / test | 398 / 168 | 0 / 0 | 254 / 110 |
| 100k | train / test | 795 / 322 | 55 / 22 | 80 / 44 |

- **Models:**
  - Level only (baseline): logistic regression on the log RMS of each axis, 12 parameters. It sees loudness only.
  - 1D CNN: four strided convolutions on the raw window, 45 k parameters.
  - CNN + GRU: convolutional front end and a bidirectional GRU, 48 k parameters.
- **Training:** 30 epochs, AdamW, class-weighted loss, augmentation (time shift, sign flip, ±20 % gain), 3 seeds, RTX 4060 Laptop GPU. All of it took under a minute.

## Experiment 1: bracket split (train and test brackets from the same three runs)

Mean ± std over 3 seeds, on the test brackets.

| Model | Test part | Accuracy | Macro F1 | F1 no cutting | F1 stable | F1 chatter |
|---|---|---|---|---|---|---|
| Level only | all | 0.975 ± 0.001 | 0.919 ± 0.003 | 0.980 | 0.778 | 1.000 |
| Level only | 20k | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Level only | 60k | 1.000 | 1.000 | 1.000 | n/a | 1.000 |
| Level only | 100k | 0.918 ± 0.004 | 0.842 ± 0.005 | 0.948 | 0.579 | 1.000 |
| 1D CNN | all | 1.000 ± 0.000 | 0.999 ± 0.002 | 1.000 | 0.997 | 1.000 |
| 1D CNN | 20k | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1D CNN | 60k | 1.000 | 1.000 | 1.000 | n/a | 1.000 |
| 1D CNN | 100k | 0.999 ± 0.001 | 0.996 ± 0.005 | 1.000 | 0.993 | 0.996 |
| CNN + GRU | all | 0.997 ± 0.003 | 0.998 ± 0.002 | 0.998 | 1.000 | 0.995 |
| CNN + GRU | 20k | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| CNN + GRU | 60k | 1.000 | 1.000 | 1.000 | n/a | 1.000 |
| CNN + GRU | 100k | 0.991 ± 0.010 | 0.985 ± 0.015 | 0.994 | 1.000 | 0.962 |

60k has no stable windows, so its macro F1 is over two classes.

Precision / recall on all test brackets:

| Model | No cutting | Stable | Chatter |
|---|---|---|---|
| Level only | 1.00 / 0.96 | 0.64 / 1.00 | 1.00 / 1.00 |
| 1D CNN | 1.00 / 1.00 | 0.99 / 1.00 | 1.00 / 1.00 |
| CNN + GRU | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 |

## Experiment 2: leave one run out (the test run's spindle speed is never seen in training)

| Model | Test run | Accuracy | Macro F1 | F1 no cutting | F1 stable | F1 chatter |
|---|---|---|---|---|---|---|
| Level only | 20k | 0.973 ± 0.005 | 0.906 ± 0.023 | 0.996 | 0.749 | 0.972 |
| Level only | 60k | 0.911 ± 0.003 | 0.941 ± 0.001 | 0.989 | n/a | 0.892 |
| Level only | 100k | 0.970 ± 0.001 | 0.912 ± 0.004 | 0.987 | 0.795 | 0.954 |
| 1D CNN | 20k | 0.941 ± 0.003 | 0.687 ± 0.029 | 0.993 | 0.130 | 0.939 |
| 1D CNN | 60k | 0.782 ± 0.124 | 0.767 ± 0.188 | 0.999 | n/a | 0.534 |
| 1D CNN | 100k | 0.932 ± 0.010 | 0.742 ± 0.043 | 0.980 | 0.481 | 0.765 |
| CNN + GRU | 20k | 0.944 ± 0.004 | 0.714 ± 0.030 | 0.995 | 0.207 | 0.940 |
| CNN + GRU | 60k | 0.882 ± 0.006 | 0.912 ± 0.006 | 0.999 | n/a | 0.825 |
| CNN + GRU | 100k | 0.964 ± 0.003 | 0.857 ± 0.018 | 0.989 | 0.665 | 0.917 |

Stable-class recall when the run is held out: 1D CNN 0.07 (20k) and 0.58 (100k); CNN + GRU 0.12 and 0.61; level only 0.64 and 1.00.

## What the results mean

- **The bracket split is an easy test.** Train and test brackets are neighbours in the same run, so the same chatter frequency appears on both sides. Scores near 1.0 show the networks can reproduce the labelling rule inside runs they have seen; they do not show that the models detect chatter in general.
- **On an unseen spindle speed the networks do worse than the 12-parameter loudness baseline** (macro F1 0.69–0.91 against 0.91–0.94). They have learned run-specific signatures (which frequency rings in which run), not a general notion of chatter.
- **Stable is the class that fails.** With one run held out, the networks call most stable windows chatter. There are only 207 stable windows in total, from two runs.
- **Loudness alone goes a long way in these three runs** because chatter here is always louder than stable cutting. That will not hold across depths of cut and feeds, so the baseline's good score is also specific to this data.
- **The labels come from the same signal.** A high score means agreement with the entropy rule, not with the machined surface.

## What is needed next

1. **More runs, especially stable ones**, at several spindle speeds, depths and feeds. Leave-one-run-out only becomes meaningful with many runs.
2. **Report leave-one-run-out (or leave-one-speed-out) as the headline number**, with the bracket split as a sanity check.
3. **Make the input speed-independent**, e.g. resample each window to a fixed number of samples per spindle revolution, or train with random time-stretch augmentation, so a chatter tone is not tied to an absolute frequency.
4. **Predict the continuous score** (entropy drop) as a regression target, which also uses the uncertain windows.
5. **Record at 50 kHz or more**; the chatter tones found sit at 5–9 kHz, near the limit of the 20 kHz recordings.
