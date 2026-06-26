# ECG Signal Analysis Algorithm Empowered by AI

Final Year Project — Bachelor of Engineering, Biomedical Engineering
Holy Spirit University of Kaslik (USEK), 2024
Author: Karim Nadjarian Hayek
Supervisor: Dr. Eng. Sandy A. Rihana
Industry partner: Cardiodiagnostics™

## Overview

This project implements a two-stage ECG analysis pipeline:

1. A **novel gradient-based QRS detector** that locates R-peaks through bandpass filtering, second-order differencing, absolute-value accumulation, exponential smoothing, and adaptive thresholding.
2. A **hybrid deep-learning model** that classifies each detected beat into one of four classes drawn from the MIT-BIH annotations: Normal Sinus Rhythm (N), Supraventricular Ectopic (S), Premature Ventricular Contraction (V), and Unclassified (Q).

The work was carried out as an industry-sponsored R&D project aimed at producing a Lebanese-developed algorithm benchmarked against ANSI/AAMI EC57 criteria and the performance levels set by Cardiodiagnostics™ for an eventual FDA 510(k) submission.

## QRS Detector — Gradient-based Algorithm

A new R-peak detection method that departs from amplitude-based thresholding in favour of a gradient-driven pipeline.

**Pipeline.** Raw ECG `r(n)` → first-order Butterworth band-pass filter (5–12 Hz) → second-order central differencing `d(n)` → absolute-value accumulation over a window of `q = 0.126·Fs` samples → squaring → exponential smoothing (α = 0.1) → adaptive thresholding (segment mean over 2700-sample windows with 150-sample overlap) → peak refinement within ±25 samples of each detection.

The squaring stage amplifies QRS energy relative to residual T-wave content, and the windowed-mean threshold adapts locally to amplitude drift.

### Performance

Evaluated record-by-record against four standard databases:

| Database | Sensitivity | Positive Predictivity | EC57 target met |
|----------|-------------|------------------------|------------------|
| MIT-BIH Arrhythmia | 96.77 % | 99.81 % | PPV ✓, Se ✗ (target 98 %) |
| NST (Noise Stress) | 91.24 % | 88.36 % | Both ✓ (targets 90 % / 84 %) |
| ESC ST-T | 97.66 % | 98.97 % | Se ✗, PPV ✗ (target 99 % / 99 %) |
| AHA | 98.97 % | 82.99 % | Se ✓, PPV ✗ (target 97 % / 98 %) |

The PPV on MIT-BIH (99.81 %) matches or exceeds the strongest published comparators including Pan-Tompkins, Elgendi, and BeatLogic; the corresponding sensitivity is the weakest in that comparator set and is the primary axis for future work. Failure modes are concentrated in records 207, 231, and 232 (MIT-BIH), and across noisy AHA recordings where false positives dominate.

## Beat Classification — Hybrid Deep Learning Model

A reduced-complexity adaptation of the HCRNet architecture (Luo et al., 2021), tailored from nine classes down to four and resized to fit the available compute.

### Architecture

Three parallel branches fed by the same normalised beat segment (0.65 s window around each R-peak, Z-score normalised):

- **Block 1 — Spatial:** 2× (Conv1D + MaxPool1D) → GlobalPool1D
- **Block 2 — Spatio-temporal:** 3× (SeparableConv1D + AvgPool1D + Dropout) → GRU
- **Block 3 — Spatio-temporal:** 3× (Conv1D + MaxPool1D + Dropout) → LSTM

Outputs are concatenated (Block1 ⊕ Block2 → R₁; Block1 ⊕ Block3 → R₂; R₁ ⊕ R₂ → R) and passed to a 4-unit Dense layer with SoftMax activation.

Trained with Adam, categorical cross-entropy loss, batch size 16, 10 epochs, ELU activation throughout.

### Data and evaluation

- **Source:** MIT-BIH Arrhythmia database, segmented around reference R-peaks.
- **Class distribution (raw):** 75 027 N, 2 548 S, 7 129 V, 33 Q — extremely imbalanced.
- **Imbalance handling:** SMOTE applied to the training partition only (after the 90/10 train/test split, before 10-fold cross-validation on the training set), to avoid leakage.
- **Cross-validation:** 10-fold; 7 of 10 folds completed on the available hardware.

### Performance

Averaged across the seven completed folds on MIT-BIH:

| Metric | Value |
|--------|-------|
| Accuracy | 93.52 % |
| Sensitivity (recall) | 85.31 % |
| Positive Predictivity (precision) | 76.53 % |
| F1-score | 92.87 % |

Per-fold results vary substantially (precision ranges from 55 % to 98 %), driven largely by how the 33 Q-class instances are split. Generalisation to NST and AHA degraded sharply (precision below 50 % on AHA), indicating that the model overfits MIT-BIH morphology despite the regularisation and reduced depth.

A second variant, a more direct port of the full HCRNet (reported in Section 3.2.6 of the thesis), trained on a stronger workstation over 25 days but produced lower averaged accuracy (68.61 %) than the lighter model and is included for transparency rather than as a recommendation.

## Comparison with State-of-the-Art

| Study | Year | Classes | Technique | Headline result |
|-------|------|---------|-----------|------------------|
| Ramirez et al. | 2019 | 10 | Hybrid NN | Acc 93.8 % |
| Fan et al. | 2019 | 3 | CNN | F1 84 %, Acc 85 % |
| Yao et al. | 2020 | 8 | ATI-CNN | PPV 82.6 %, Se 80.1 %, F1 81.2 % |
| Luo et al. | 2021 | 9 | HCRNet | PPV 99.53 %, Se 99.28 %, Acc 98.7 % |
| Kraft et al. | 2023 | 2 | Deep NN | Sp 92.6 %, Se 96.6 %, Acc 94.6 % |
| **This work** | **2024** | **4** | **Hybrid DL** | **Se 85.31 %, F1 92.87 %, Acc 93.52 %** |

## Datasets

- **MIT-BIH Arrhythmia Database** — 48 half-hour two-channel ambulatory ECG recordings at 360 Hz, used for training and primary evaluation of both stages.
- **European ST-T (ESC) Database** — 90 two-channel records at 250 Hz, used for QRS-detector evaluation.
- **MIT-BIH Noise Stress Test (NST) Database** — 15 records with controlled noise injection at SNRs from 24 dB down to −6 dB; used for QRS robustness testing.
- **AHA Database** — 80 two-channel records at 250 Hz, licensed from ECRI; six corrupted records excluded (1204, 1209, 3210, 8204, 8206, 8208).

All datasets use the first channel.

## Environment

- Python 3 (Anaconda, Jupyter)
- NumPy, SciPy, Matplotlib, statsmodels, OpenCV
- wfdb (PhysioNet I/O)
- TensorFlow, Keras
- scikit-learn, imbalanced-learn, seaborn

## Hardware Used

- **Primary development:** Intel® Core™ i5-1135G7 @ 2.4 GHz, 8 GB RAM, integrated Iris Xe graphics. QRS detector runs at ≈ 34 ms per beat. Deep-learning training averaged ≈ 1000 s per epoch; seven of ten folds completed.
- **Secondary workstation (for the full-HCRNet variant):** dual Intel® Xeon® E5-2637 v3 @ 3.5 GHz, 128 GB RAM. Full ten-fold training took 25 days.

## Reproducibility Notes

- Beats with segment length differing from the expected 0.65 s window are discarded rather than padded.
- Labels with fewer than six instances in the training partition are skipped prior to SMOTE.
- The QRS-detector smoothing factor (α = 0.1), accumulation window (`q = 0.126·Fs`), and threshold-segment length (2700 samples, 150-sample overlap) are the empirically chosen values reported in the thesis; sensitivity to these parameters was not exhaustively swept.

## Known Limitations

1. **QRS sensitivity on MIT-BIH (96.77 %) is below the EC57 target of 98 %.** False negatives concentrate in records with wide or low-amplitude QRS complexes (e.g., 207, 231).
2. **QRS positive predictivity on AHA (82.99 %) is well below target**, driven by false positives on noisy recordings.
3. **Beat-classification evaluation is intra-patient.** AAMI EC57 recommends inter-patient splits (DS1/DS2). Headline numbers should be expected to drop under that protocol.
4. **Class Q has only 33 raw instances**, which produces high per-fold variance and contributes to the spread between best (99 % precision) and worst (55 %) folds.
5. **Generalisation to NST and AHA is poor** for the beat classifier, indicating the model has learned MIT-BIH-specific morphology.
6. **The model is not FDA-cleared** and is not validated for clinical use.

## Future Work

- Inter-patient (DS1/DS2) evaluation under the AAMI EC57 protocol.
- More adaptive thresholding in the QRS stage to reduce false negatives on wide-QRS and low-amplitude records.
- Stratified or grouped cross-validation to stabilise per-fold variance on rare classes.
- Stronger regularisation (mix-up, attention, larger dropout) and architectural search now that compute resources are available through the IRALEB funding programme.
- Cardiologist-in-the-loop validation of detected beats and classifications.

## Acknowledgements

Dr. Eng. Sandy A. Rihana (advisor, USEK); Mr. Ziad Sankari and Eng. Tamara Akl (Cardiodiagnostics™); Mr. Hani Saba (computational setup); Mr. Cris L. Luengo Hendriks (Signal Processing Stack Exchange); IRALEB programme.

## License and Data Access

Code is the author's own work (see thesis plagiarism statement). The MIT-BIH, NST, and ESC databases are publicly available via PhysioNet; the AHA database is licensed from ECRI and is not redistributable.

## Citation

If you reference this work, please cite the thesis:

> Nadjarian Hayek, K. (2024). *ECG Signal Analysis Algorithm Empowered by AI.* Bachelor of Engineering thesis, Holy Spirit University of Kaslik (USEK), Kaslik, Lebanon.
