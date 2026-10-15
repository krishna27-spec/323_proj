# Paper notes: chatter detection in micromilling with sensor fusion + CNN

**Paper:** Gururaja, S., Singh, K.K. (2025). *Explainable framework based on integrated sensor fusion and deep learning to estimate stability in high-speed micromilling of thin-walled titanium alloy.* Engineering Applications of Artificial Intelligence 161, 112326. doi:10.1016/j.engappai.2025.112326
**Affiliation:** Mechanical Engineering, BITS Pilani Hyderabad. Funded by ANRF (CRG/2023/007343).
**File:** `1-s2.0-S0952197625023346-main.pdf` (24 pages)

---

## 1. One-paragraph summary

The paper builds a CNN that classifies a micromilling cut as **stable** or **chatter** from a spectrogram. Vibration (accelerometer, two directions) and audio (microphone) signals are denoised with VMD, fused in the time domain, turned into STFT spectrograms, and fed to a small 2-conv-layer CNN. The CNN is trained on single-sensor spectrograms labelled from SEM images of the machined surface, then tested on fused-signal spectrograms. Of 16 fusion strategies, a plain **weighted average** is ranked best. On the 32 micromilling conditions it gets 31/32 right (96.87 %). The same model, without retraining, is applied to turning (FSS-409) and slot milling (IN-718) and gets 95.31 % and 92.6 %.

## 2. Problem and motivation

- Thin-walled parts (blisks, ribs, turbine blades, stents) lose stiffness with every tool pass, so chatter is close to unavoidable in micromilling.
- Chatter causes dimensional loss, poor surface finish, tool damage. Early onset detection is wanted as feedback for real-time control (e.g. changing spindle speed).
- Classical route (stability lobe diagrams, delay-differential equations, semi-discretization) depends on cutting coefficients and tool dynamics that are material- and condition-specific.
- Existing signal/ML methods (RMS indicators, WPT, EEMD, VMD+entropy, SVM, SOM, ANFIS, SDAE, dual-stream CNNs) either lack accuracy or are too heavy for real time, and generalise badly to unseen data.
- Chatter onset = large rise in vibration amplitude plus a shift in the dominant frequency.

**Gap claimed:** no lightweight, generic, tool/work-independent method that fuses contact and non-contact sensors for real-time onset detection.

## 3. Proposed approach (the six claims)

1. DL approach based on fusing vibration and audio signals.
2. VMD to reconstruct raw signals from only the informative components.
3. Fuse feed-direction vibration (Vx), normal-to-feed vibration (Vy), and audio (A) with several strategies.
4. STFT spectrograms of the fused signal for the DL model, validated against machined-surface analysis.
5. Explainable-DL tools (PDP, ICE) and error ranking to pick the best metric and fusion strategy.
6. Transfer to a different process (turning, slotting) without retraining.

**Why fuse:** accelerometer captures low-to-mid frequency vibration directly but has limited bandwidth and also picks up jig/fixture deflection. Microphone captures high-frequency airborne emission but is sensitive to ambient noise. Fusion needs only basic notch filtering, so low-amplitude chatter content is not filtered away.

## 4. Experimental setup

| Item | Value |
|---|---|
| Machine | High-speed micromachining centre, 3 axes, 0.15 µm encoder resolution, ±2 µm accuracy |
| Spindle | max 80,000 rpm, air-oil cooled, max torque 11.9 N·cm |
| Workpiece | Thin-walled TC4 (Ti-6Al-4V) |
| Tool | Uncoated tungsten carbide micro-end mill, edge radius 2.75 µm, overhang 52 % of length |
| Cutting | Dry, low-immersion radial micromilling, axial depth constant 50 µm |
| Accelerometer | Kistler 8763B1K0AB, tri-axial, ±1000 g (workpiece vibration) |
| Microphone | PCB 130F20, 38.7 mV/Pa, tilt 60°, stand-off 25 mm |
| Displacement | Laser sensor OptoNCDT ILD-2300-2 (normal to cutting direction) |
| Velocity | Laser Doppler Vibrometer, Polytech VGO-200 (along machining direction) |
| Sampling | 50 kHz (Appendix C) |
| Compute | Intel i5-9500 3 GHz, 32 GB RAM, Windows 11, Python 3.12 |

**Design of experiments (32 runs, Table 1):**
- Spindle speed: 20,000 / 40,000 / 60,000 / 80,000 rpm
- Radial depth of cut: 10 / 30 / 50 / 70 µm
- Feed: 3 and 5 µm/flute, which gives feed rates of 120–800 mm/min depending on speed

Feed rate per speed: 20k → 120, 200; 40k → 240, 400; 60k → 360, 600; 80k → 480, 800 mm/min. Conditions are numbered E1–E32, eight per depth of cut.

## 5. Signal processing

### 5.1 VMD denoising
- VMD splits a non-stationary signal into narrow-band intrinsic mode functions (IMFs), each around a centre frequency, without mode mixing. Chosen over wavelets because of better low-frequency resolution.
- Objective: minimise the summed bandwidth of the modes, subject to the modes adding back to the original signal (Eqs. 1–2).
- **Number of modes K:** chosen by kurtosis (Eq. 3). Higher kurtosis = more transient, informative content. K = 8 in Appendix C.
- **Which modes to keep:** Pearson correlation between each IMF and the raw signal (Eq. 4). Threshold = RMS deviation of the correlation coefficients (Eq. 5). Keep IMFs with correlation above the threshold (Eqs. 6–7) and sum them (Eq. 8).
- Low-frequency IMFs = mechanical vibration / tool-work interaction. High-frequency IMFs = acoustic emission / tool state.

### 5.2 Fusion (data level)
Three fusion levels exist: data, feature, decision. The paper uses **data-level** fusion, arguing feature-level can drop low-amplitude chatter content and decision-level costs too much time.

Strategies (Table 2), with Vx, Vy, A as inputs:

| # | Strategy | Formula |
|---|---|---|
| 1 | Mean | F = (Vx + Vy + A) / 3 |
| 2 | Median | F = median(Vx, Vy, A) |
| 3 | RMS | F = sqrt((Vx² + Vy² + A²) / 3) |
| 4 | Weighted average | F = W1·Vx + W2·Vy + W3·A, with W1 + W2 + W3 = 1 |
| 5–7 | Correlation-gated median / RMS / weighted average | Same formulas, applied only when Pearson or Spearman correlation between vibration and audio exceeds T1 = 0.5 or T2 = 0.6 |

4 plain + 3 methods × 2 correlation types × 2 thresholds = **16 strategies**. Naming: `0.6_PC_Median` = threshold 0.6, Pearson, median. `0.5_SC_WAvg` = threshold 0.5, Spearman, weighted average.

Fusion quality check: Pearson correlation between fused signal and each individual signal, 60 % threshold. Below 50 % one sensor dominates the fused signal.

### 5.3 Spectrograms
- STFT, window 512 points, 50 % overlap.
- Images resized to 224 × 224 × 3 (RGB).
- Training: signals segmented (window 512), 1833 segments per signal.
- Testing: no segmentation, one spectrogram per whole signal, to mimic real-time use.

## 6. Labelling

Labels come from **SEM micrographs of the machined surface plus tool condition**, not from the signals.
- Stable: smooth surface, two intact cutting edges. Some small welded chips can still appear.
- Chatter: low-frequency wavy patterns, deep feed marks, flaky waveforms, grooves and tearing, worn or chipped cutting edges.
- Mechanism noted: Ti alloy's low thermal diffusivity raises temperature and contact stress, giving adhesive and diffusion wear, built-up layer, chip welding. Worst at 80,000 rpm and 70 µm depth.
- Result: **15 stable, 17 unstable** out of 32.

## 7. Deep learning model

### 7.1 Architecture (Table 4)

| Layer | Output shape | Params |
|---|---|---|
| Conv2D, 64 filters, 3×3, ReLU, kernel regularisation | 222 × 222 × 64 | 1,792 |
| MaxPool 2×2 | 111 × 111 × 64 | 0 |
| Conv2D, 128 filters, 3×3 | 109 × 109 × 128 | 73,856 |
| MaxPool 2×2 | 54 × 54 × 128 | 0 |
| Flatten | 373,248 | 0 |
| Dense 256, ReLU, kernel regularisation | 256 | 95,551,744 |
| Dropout 50 % | 256 | 0 |
| Dense 2, softmax | 2 | 514 |

About 95.6 M parameters, almost all in the first dense layer.

### 7.2 Training
- Data: 3 signals (Vx, Vy, A) × 32 conditions × 1833 segments × 5 augmentation levels = **879,840 spectrograms**.
- Augmentation: rotation 90°/180°/270°, horizontal and vertical flip.
- Split 70/30 train/validation.
- Adam, categorical cross-entropy, learning rate reduced 5 % per epoch, early stopping, checkpoint on best validation loss.
- Hyperparameter search over pooling size, kernel size, filters, epochs, batch size. 5 runs at 50/100/150/200 epochs, reported as 250/500/750/1000 cumulative.
- Epoch result (Fig. 10): 250 → train 85 %, val 60 % (underfit). **500 → train 96 %, val 89 % (chosen).** 750 → train 89 %, val 65 %. 1000 → train 100 %, val 50 % (overfit).

### 7.3 Pseudo-labelling and fine-tuning (Appendix B)
- Predict on unlabelled samples. Keep those with max softmax ≥ 0.90, discard the rest.
- Loss weight 1.0 for softmax ≥ 0.95, scaled 0.5–0.8 for 0.90–0.95.
- Fine-tune on labelled + pseudo-labelled data, entropy-minimisation loss, early stopping.

## 8. Results

### 8.1 Fusion strategy comparison
- ROC-AUC (Fig. 12): 0.5_PC_Median and 0.5_SC_WAvg = 1.0; weighted average, 0.6_PC_RMS, 0.6_PC_WAvg = 0.97; 0.6_PC_Median = 0.50; 0.5_PC_WAvg = 0.57.
- Confusion matrix (Fig. 13): weighted average = 14 stable + 17 chatter correct, 1 wrong. 0.6_PC_Median = all chatter correct but 15 stable cases called chatter.
- Error ranking (Table 3), top and bottom:

| Strategy | MSE | MAE | R² |
|---|---|---|---|
| **Weighted average** | 0.001248 | 0.027416 | 0.986918 |
| 0.6_SC_WAvg | 0.001975 | 0.037878 | 0.984623 |
| 0.6_SC_Median | 0.002357 | 0.039954 | 0.979624 |
| Mean | 0.00406 | 0.051916 | 0.950235 |
| Median | 0.004699 | 0.05355 | 0.949392 |
| RMS | 0.118066 | 0.166261 | 0.405796 |
| 0.5_SC_RMS (worst) | 0.267815 | 0.218137 | −1.984 |

RMS-based fusion is consistently worst. Spearman-gated strategies come right behind the weighted average.
- PDP/ICE (Fig. 15): partial dependence falls from about 5 to about 1 as accuracy, recall, precision go 0.5 → 1.0. The authors conclude weighted average is optimal and **recall** is the best metric to judge by.

### 8.2 Feature maps (Fig. 9)
- Stable E7 (60,000 rpm): dominant band at 2nd rotational harmonic, 2000 Hz.
- Unstable E5 (40,000 rpm): high-energy band at 9th rotational harmonic, 6000 Hz.
- Band at about 10 kHz = tool entry/exit; band at 0 Hz = flying chips hitting the microphone. The CNN does not treat these as significant.

### 8.3 Ablation: fused vs single sensor (Figs. 17–20)

| Input | Misclassified (of 32) |
|---|---|
| Feed vibration (Vx) | 12 (9 stable, 3 chatter) |
| Normal-to-feed vibration (Vy) | 17 (11 stable, 6 chatter) |
| Audio (A) | 4 chatter |
| **Fused** | **1** |

- Fused: accuracy 0.96, precision 0.97, recall 0.95, F1 0.96.
- Stated gains: accuracy +33 % vs accelerometer, +39 % vs audio; precision +21 % / +39 %; recall +29 % / +9 %; F1 +32 % / +28 %.
- Extra compute for fusion: +22 ms vs Vx, +87.15 ms vs Vy, +26.16 ms vs audio.
- 5-fold CV on micromilling: train 0.92, validation 0.90, std ±1.55 %.
- Accelerometer energy is spread wide over 5000–6500 Hz; audio energy is narrow.

### 8.4 Transfer to other processes (no retraining)

| Process | Setup | Runs | Accuracy | Precision | Recall | F1 | Wrong |
|---|---|---|---|---|---|---|---|
| Turning, FSS-409 | HMT-Praga PTC-200 lathe; depth 0.1–0.4 mm, feed 0.05–0.2 mm/rev, speed 50–200 m/min | 64 | 95.31 % (±0.94 %) | 98 % | 97 % | 97 % | 3 |
| Slot milling, IN-718 | Hardinge Bridgeport GX-600; 4 mm 4-flute end mill; depth 0.05–0.15 mm, feed 0.25–0.75 mm/flute, 2000–6000 rpm, dry | 27 | 92.6 % (±0.82 %) | 91 % | 94 % | 92 % | 2 |

Reasons given for the misses: gradual frequency shift at the stable-to-chatter transition (turning); very low chip load so the signal sits near baseline noise (slotting); tool wear / built-up edge bursts that look like chatter.

### 8.5 Timing (Appendix C)
VMD about 100 ms per sample, STFT about 20 ms per signal, fusion about 15 ms, CNN inference about 20 ms, pseudo-labelling about 10 ms per sample.

## 9. Conclusions as stated by the authors

- 96.87 % accuracy on thin-walled TC4 micromilling.
- Weighted average is the optimal fusion strategy: MAE 0.027416, R² 0.986918.
- 500 epochs is the optimum.
- Fusion improves precision and recall by 21 % and 29 % over single sensors.
- Pretrained model transfers to turning with 61/64 correct.

**Future scope:** add acoustic emission, laser displacement, LDV, cutting force to the fusion; integrate into CNC control as feedback.
**Limitations:** model only accepts RGB spectrograms; needs a class-balanced training set.

## 10. Things to be careful about when citing this paper

These are inconsistencies or unclear points I found in the text. Worth knowing before repeating a number.

- **Depth of cut:** the text says radial depth of 30–180 µm in steps of 50 µm; Table 1 uses 10, 30, 50, 70 µm. Table 1 is the one that matches 32 runs.
- **Number of conv layers:** Table 4 and the text say 2; Appendix C says 3.
- **Early stopping patience:** 3 (Appendix A), 5 (Appendices B and C), 10 (Section 5.2).
- **Epochs:** main text says 500; Appendix A pseudocode says 50, batch 32, lr 0.001.
- **Train/validation split:** 70/30 in the text, test_size 0.2 in Appendix A.
- **Confidence threshold:** 0.90 in the text, with 0.85 also named as the optimum in Appendix B.
- **What was trained on:** the method section says training used single-sensor spectrograms and testing used fused ones; the conclusion says the 879,840 training spectrograms came from fused signals.
- **Fig. 14 "accuracy" above 100 %:** 213 % and 173 % are reported for two strategies and read as overfitting. This is not classification accuracy in the usual sense and the quantity is not defined.
- **Weighted average weights:** W1, W2, W3 are never given, only called "optimal".
- **Regression metrics on a classifier:** MAE, MSE, R² in Table 3 are used to rank a binary classifier; what the target values are is not stated.
- **PDP/ICE:** applied with accuracy/recall/precision as the "feature", which is not the standard use. The conclusion that recall is the best metric rests on this.
- **"33 % improvement":** appears both as fused-vs-accelerometer and as weighted-average-vs-other-fusion-strategies.
- **Small test set:** the headline 96.87 % is 31 of 32 conditions, and the same 32 conditions supplied the training data.
- **Parameter count:** 95.6 M parameters is described as an advantage because the model "can easily memorize the training data"; this is the opposite of what one wants.
- **750-epoch result:** described as a "slight increase" in validation accuracy to 65 %, which is a drop from 89 %.
- **Data availability:** "No data was used for the research described in the article", so the dataset is not public.

## 11. How this connects to our data

Our folder has three accelerometer files. All three are sampled at 20 kHz (dt = 5e-5 s) with a time column and 3 channels:

| File | Duration | Samples |
|---|---|---|
| `Acce_20k_1_r1.lvm` | 60.0 s | 1,200,000 |
| `Acce_60k_1_r1.lvm` | 25.5 s | 510,000 |
| `Acce_100k_1_r1.lvm` | 36.25 s | 725,000 |

- Since the sampling rate is the same in all three, "20k / 60k / 100k" is most likely the **spindle speed in rpm**, not the sampling rate. This is my inference from the file names; confirm it. Note the paper's spindle tops out at 80,000 rpm, so 100k would be outside its test range.
- Three channels fit a tri-axial accelerometer like the paper's. Which channel is feed (Vx) and which is normal-to-feed (Vy) is not known yet.
- We have **vibration only**, no audio file, so the paper's fusion step cannot be reproduced with what is here.
- Our sampling is 20 kHz (10 kHz Nyquist) against the paper's 50 kHz. The paper's 10 kHz entry/exit band would sit right at our limit.

**What we found so far in the 20k file** (see `plot_20k_0-25s.png`):
- 0 to about 20.5 s: flat noise, RMS about 0.004–0.005.
- About 20.5 s: onset. All three channels rise together; channel 3 rises most.
- 30–34 s: steep rise to a peak about 100× baseline. About 39 s: drop. 42–51 s: second smaller rise. About 52 s: back to baseline.
- The flat part before 20.5 s is probably the spindle running with no cutting (air cut), and 20.5 s the tool entering the workpiece. That is my reading, not something the data proves. If so, the onset we found is **cut entry, not chatter onset**. The paper says the same thing about tool entry/exit showing up as its own band.
- The small periodic spikes in the baseline RMS are unexplained.

**Natural next steps if we follow the paper:**
1. Same onset detection on the 60k and 100k files.
2. Trim each file to the cutting part only.
3. FFT / STFT spectrogram of the cutting part (window 512, 50 % overlap as in the paper) and look for energy at spindle-rotation harmonics versus non-harmonic chatter frequencies. Rotation frequency = rpm/60: 333 Hz at 20k, 1000 Hz at 60k, 1667 Hz at 100k.
4. VMD denoising with kurtosis for K and Pearson correlation for mode selection.

## 12. Glossary

- **Chatter:** self-excited vibration between tool and workpiece; unstable cutting.
- **Regenerative chatter:** caused by the tool cutting over the wavy surface left by the previous pass.
- **SLD (stability lobe diagram):** map of chatter-free depth of cut against spindle speed.
- **VMD:** variational mode decomposition. **IMF:** intrinsic mode function, one narrow-band component from VMD.
- **Kurtosis:** how peaked/impulsive a distribution is; high = transient-rich.
- **STFT:** short-time Fourier transform; gives the spectrogram (frequency content against time).
- **PDP / ICE:** partial dependence plot / individual conditional expectation; explainability plots.
- **Pseudo-labelling:** using the model's own confident predictions as labels for unlabelled data.
- **TC4:** Ti-6Al-4V. **FSS-409:** ferritic stainless steel 409. **IN-718:** Inconel 718.
