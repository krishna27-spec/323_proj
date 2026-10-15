# 06 · Time-stretch augmentation: train on turning, test on our three runs

Code: the Experiment 4 cells of the marimo notebook `train.py`, and `turning.segments` in `turning.py`.
Raw numbers: `results/stretch_metrics.json`. Figure: `figures/06_stretch_confusion.png`.
Model labels for the viewer: `viewer/data/*_pred.json`, shown in the new "Model labels" panel of `viewer/index.html`.

## Result in one paragraph

Time-stretch helps the 1D CNN a lot but does not make it better than the loudness baseline. With background scaling, its macro F1 on our runs goes from 0.46 to 0.72 and its stable recall from 0.01 to 0.51, and it stops calling the idle machine chatter (0.85 to 0.02 of no-cutting windows). The loudness baseline is still ahead at 0.80, and the CNN's result varies a lot between seeds. The gain mostly disappears when loudness is removed from the input, so the stretched network is probably leaning on loudness more than on waveform shape.

## What was changed

- **Why:** the chatter tone sits at about 0.9 to 3 kHz in the turning data and at 7.5 to 8.8 kHz in our runs. A network trained on plain turning windows never sees a chatter tone where ours are.
- **Time-stretch:** each training window is cut from the continuous turning signal with a random length of `s` × 1024 samples and resampled to 1024 samples, which moves every frequency in it up by `s`. `s` is drawn between 0.5 and 8, evenly on a log scale.
- **Tone limit:** a chatter stretch is never compressed so far that its main tone would pass 9 kHz and be filtered out.
- **Level correction:** compressing by `s` low-pass filters the signal, which lowers a broadband level by about √`s`. The window is multiplied by √`s` to undo that. This assumes the background is broadband; it was not measured.
- **Control:** the same sampler with `s` = 1 (random crops, half stable and half chatter), so the two rows differ only in the stretch.
- **Everything else as in `05_transfer_results.md`:** same turning train and test recordings, same models, 20 epochs, 3 seeds, 24,760 training windows per training. Our windows are not stretched. 36 trainings took 147 s on the RTX 4060 Laptop GPU.

## Results on our runs

Mean ± std over 3 seeds, all three runs together (207 stable, 1,295 chatter windows).

### Background scaling

| Model | Training | Macro F1 | Recall stable | Recall chatter | No-cutting windows called chatter |
|---|---|---|---|---|---|
| Level only | control | 0.798 ± 0.000 | 0.512 | 0.986 | 0.000 |
| Level only | time-stretch | 0.797 ± 0.001 | 0.510 | 0.986 | 0.000 |
| 1D CNN | control | 0.455 ± 0.006 | 0.010 | 0.944 | 0.846 |
| 1D CNN | time-stretch | 0.718 ± 0.055 | 0.509 ± 0.119 | 0.923 | 0.016 |
| CNN + GRU | control | 0.552 ± 0.124 | 0.127 | 0.991 | 0.658 |
| CNN + GRU | time-stretch | 0.599 ± 0.090 | 0.196 ± 0.146 | 0.975 | 0.345 |

### Window scaling (loudness removed)

| Model | Training | Macro F1 | Recall stable | Recall chatter |
|---|---|---|---|---|
| 1D CNN | control | 0.459 ± 0.035 | 0.026 | 0.909 |
| 1D CNN | time-stretch | 0.507 ± 0.021 | 0.586 ± 0.356 | 0.649 ± 0.155 |
| CNN + GRU | control | 0.422 ± 0.033 | 0.016 | 0.836 |
| CNN + GRU | time-stretch | 0.495 ± 0.020 | 0.348 ± 0.253 | 0.759 ± 0.171 |

The level-only model is left out here: with window scaling it sees only the ratio between axes, and its numbers carry no meaning.

### 1D CNN with time-stretch, per run (background scaling)

| Run | Macro F1 | Recall stable | Recall chatter |
|---|---|---|---|
| 20k | 0.663 ± 0.123 | 0.262 ± 0.171 | 1.000 |
| 60k | n/a | n/a | 0.865 ± 0.030 |
| 100k | 0.694 ± 0.315 | 0.926 ± 0.054 | 0.597 ± 0.428 |

## Cost on the turning data

Macro F1 on the held-out turning recordings (which are not stretched):

| Model | Control | Time-stretch |
|---|---|---|
| 1D CNN | 0.907 ± 0.004 | 0.863 ± 0.029 |
| CNN + GRU | 0.914 ± 0.014 | 0.835 ± 0.058 |

## What the results mean

- **The stretch fixes the worst failure.** Without it the 1D CNN called 99 % of our stable windows and 85 % of the idle machine chatter. With it, it finds half of our stable windows and leaves the idle machine alone.
- **It is still not better than loudness.** The baseline gets 0.80 with 12 parameters and no variation between seeds.
- **The result is not stable across seeds.** On the 100k run the 1D CNN's chatter recall is 0.60 ± 0.43: some seeds find the chatter and some miss most of it.
- **The gain is probably loudness.** With loudness removed (window scaling) the improvement is small and the networks trade chatter recall for stable recall. With background scaling, the stretched 1D CNN's first-seed answers match the loudness baseline on 92 % of our cutting windows, and the stable windows it gets right are quieter (median 3.1× the idle level) than the ones it gets wrong (5.7×). This fits the reading that stretching blurs the frequency cues and pushes the network towards level, but it is an inference from these numbers, not a controlled test.
- **CNN + GRU gains little.** Its control result already varies widely between seeds (0.55 ± 0.12), so the difference to 0.60 is within noise.

## What this changes

- Time-stretch is worth keeping when training on other machines' data: it removes the "everything is chatter" failure at a small cost on the source data.
- It is not enough on its own. The models still do not separate stable from chatter on our machine by waveform shape.
- The next step that would test shape directly is to resample each window to a fixed number of samples per spindle revolution, for both datasets, so the spindle harmonics always land in the same place and only non-harmonic energy (chatter) moves. That needs the spindle speed as an input at preprocessing time.
- More stable runs of our own remain the most direct fix.

## Model labels in the viewer

`viewer/index.html` has a new panel, "Model labels", under the chatter score. It shows one coloured strip per model across each run, with the rule label on top for comparison:

1. 1D CNN, turning + time-stretch
2. CNN + GRU, turning + time-stretch
3. 1D CNN, turning, no stretch
4. Level only (loudness baseline), turning

- All four use background scaling and the first seed only, so a strip can look better or worse than the mean in the tables above.
- The models can only answer stable or chatter. The checkbox "Grey out where the rule says no cutting" (on by default) hides their answers on the idle machine; untick it to see what they say there.
- Hovering over any time panel shows each model's label at the cursor in the readout line.
- To open it: `.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory viewer`, then http://127.0.0.1:8765.
