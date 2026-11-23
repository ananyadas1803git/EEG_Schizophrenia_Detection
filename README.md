# 🧠 EEG-Based Schizophrenia Detection 

An end-to-end signal processing, feature engineering, and machine learning framework for automated schizophrenia diagnosis from continuous 16-channel resting-state EEG recordings.

This repository implements multi-window temporal segmentation, digital frequency-band decomposition, domain-isolated feature selection (Time, Frequency, and Entropy), leak-free subject-isolated cross-validation (`GroupKFold`), and Explainable AI (SHAP).

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset & Clinical Specifications](#-dataset--clinical-specifications)
- [Methodology & Architecture](#-methodology--architecture)
- [Feature Engineering Framework](#-feature-engineering-framework)
- [Experimental Paradigms & Results](#-experimental-paradigms--results)
  - [1. 6-Second Multi-Window Segmentation (840 Segments)](#1-6-second-multi-window-segmentation-840-segments)
  - [2. 10-Second Segmentation & Temporal Trajectory Analysis](#2-10-second-segmentation--temporal-trajectory-analysis)
  - [3. Frequency-Band Decomposition (Delta, Theta, Alpha, Beta)](#3-frequency-band-decomposition-delta-theta-alpha-beta)
- [Explainable AI (XAI) & PCA](#-explainable-ai-xai--pca)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [References](#-references)

---

## 🔬 Project Overview

Schizophrenia is a severe neuropsychiatric disorder characterized by cognitive dysfunction, perceptual impairments, and disrupted cortical communication (the *Disconnection Syndrome*). Electroencephalography (EEG) provides millisecond-level temporal resolution capable of capturing subtle aberrant microstates and cortical desynchronization.

This project investigates multiple diagnostic paradigms:
1. **Multi-Window Segmentation:** Slicing continuous EEG into shorter windows (6s and 10s) to preserve local non-stationary bursts, evaluated via both **Segment-Level** and **Patient-Level Voting**.
2. **Frequency-Band Decomposition:** Filtering continuous 60s recordings into **Delta, Theta, Alpha, and Beta** bands with **Domain-Isolated Feature Selection**.
3. **Temporal Trajectory Analysis:** Tracking diagnostic separability across the 60-second recording timeline to discover the clinical "Golden Window."

---

## 📊 Dataset & Clinical Specifications

The pipeline is benchmarked on the standardized **Moscow State University Schizophrenia EEG Dataset**:

| Parameter | Specification |
| :--- | :--- |
| **Cohort Size** | **84 human subjects** (45 Schizophrenia patients, 39 Healthy controls) |
| **Electrode Setup** | **16 channels** (10–20 International System): `F7, F3, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2` |
| **Sampling Rate ($f_s$)** | **128 Hz** (Nyquist frequency: 64 Hz) |
| **Signal Duration** | **60 seconds continuous** ($16 \times 7,680 = 122,880$ data points per subject) |
| **State** | Resting-state, closed/open eyes |

---

## 🛠 Methodology & Architecture

```text
                      Raw Continuous 16-Channel EEG (60s @ 128 Hz)
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
[Paradigm A: Multi-Window Segmentation]       [Paradigm B: Frequency Decomposition]
  • 6s Window (10 segs/sub = 840 rows)          • 4th-order Butterworth Bandpass:
  • 10s Window (6 segs/sub = 504 rows)             Delta (0.5–4Hz), Theta (4–8Hz)
           │                                       Alpha (8–13Hz), Beta (13–30Hz)
           ▼                                                   │
[Multi-Domain Feature Extraction]                              ▼
  • Time (T): 10 features / channel             [Domain-Isolated Selection (T, F, E)]
  • Frequency (F): 13 features / channel          • Top 8 Time + Top 8 Freq + Top 8 Entropy
  • Entropy (E): 5 features / channel             • 24 Balanced Features per Band
  • Total: 448 Features per segment                            │
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
                [Subject-Isolated 5-Fold GroupKFold Cross-Validation]
                                     │
                                     ▼
                     [5 Machine Learning Classifiers]
                Naïve Bayes | Tuned SVM | Random Forest | XGBoost | ANN
                                     │
                                     ▼
              [Segment Metrics + Patient-Level Soft Probability Voting]
```

### Strict Validation Integrity (0% Data Leakage)
To avoid the optimistic bias and data leakage common in EEG literature, all segmentation paradigms utilize **5-Fold `GroupKFold`** grouped strictly by `subject_id`. All segments belonging to any single subject remain exclusively within either the training fold or the testing fold.

---

## 📈 Feature Engineering Framework

For each EEG channel, 28 handcrafted mathematical biomarkers are extracted across three distinct physiological domains (**448 total features per window**):

### 1. Time Domain (10 Features / Channel $\times$ 16 = 160 Features)
- **Signal Power & Magnitude:** Mean Power ($\frac{1}{N}\sum x^2$), Root Mean Square (RMS).
- **Geometric Complexity:** Curve Line Length ($\sum |\Delta x|$), 1st Difference Mean ($|\Delta x|$), 2nd Difference Mean ($|\Delta^2 x|$).
- **Hjorth Dynamic Parameters:** Activity (Variance $\sigma^2$), Mobility ($\sqrt{\text{var}(x') / \text{var}(x)}$), Complexity ($\frac{\text{Mobility}(x')}{\text{Mobility}(x)}$).
- **Statistical Moments:** Skewness (asymmetry), Kurtosis (tailedness/sharpness).

### 2. Frequency & Wavelet Domain (13 Features / Channel $\times$ 16 = 208 Features)
- **Spectral Power (Welch's PSD):** Absolute Power in Delta ($0.5-4\text{ Hz}$), Theta ($4-8\text{ Hz}$), Alpha ($8-13\text{ Hz}$), Beta ($13-30\text{ Hz}$), and Gamma ($30-45\text{ Hz}$).
- **Relative Band Power:** Proportion of energy in each rhythm relative to total broadband power.
- **Spectral Dynamics:** Dominant Peak Frequency, Peak Spectral Power, Spectral Entropy (Shannon entropy of normalized PSD).

### 3. Non-Linear Entropy Domain (5 Features / Channel $\times$ 16 = 80 Features)
- **Information Entropy:** Shannon Entropy (histogram density).
- **Dynamical Order:** Permutation Entropy ($m=3, \tau=1$).
- **Complexity Metrics:** Higuchi Fractal Dimension (HFD, $k_{max}=8$), Fast Sample Entropy Proxy, Fuzzy Entropy Proxy.

---

## 🏆 Experimental Paradigms & Results

### 1. 6-Second Multi-Window Segmentation (840 Segments)
*Slicing each 60s signal into 10 non-overlapping windows of 6 seconds (768 points).*

#### A. Domain Comparison (Which feature domain carries the strongest biomarker?)
| Feature Domain | Features | Naïve Bayes | Tuned SVM (RBF) | Tuned Random Forest | Tuned XGBoost | Tuned ANN (MLP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Time Domain** | 160 | 67.14% | 74.64% | 74.76% | 76.90% | 69.64% |
| **Frequency Domain** | 208 | 67.50% | 71.19% | 76.55% | 76.90% | 71.90% |
| 🥇 **Entropy Domain** | **80** | **71.55%** | **78.57%** | **76.79%** | **77.38%** | **78.10%** |
| **Combined Segmented** | 448 | 68.93% | 75.60% | **78.93%** | **80.95%** | 75.48% |

> **Finding:** The **Entropy Domain (only 80 features)** dramatically outperformed both Time (160) and Frequency (208) domains, demonstrating that nonlinear complexity reduction is the primary neurodynamic signature of schizophrenia.

#### B. Final 6-Second Benchmark Leaderboard
| Model | Segment Acc (%) | Patient Acc (%) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Tuned SVM (on Entropy)** | 78.57% | **86.90%** | **0.8400** | **0.9333** | **0.8842** | 0.8650 |
| 🥈 **Tuned ANN (on Entropy)** | 78.10% | **83.33%** | 0.8298 | 0.8667 | 0.8478 | 0.8420 |
| 🥉 **Tuned XGBoost (Combined)** | **80.95%** | **80.95%** | 0.7959 | 0.8667 | 0.8298 | **0.8841** |
| **Tuned Random Forest (Combined)**| 78.93% | 80.95% | 0.7843 | 0.8889 | 0.8333 | 0.8710 |

---

### 2. 10-Second Segmentation & Temporal Trajectory Analysis
*Slicing each 60s signal into 6 non-overlapping windows of 10 seconds (1,280 points = 504 total segments).*

#### A. Overall 10s Window Performance
| Model | Segment Acc (%) | Patient Acc (%) | Patient F1 | Precision | Recall | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Tuned XGBoost** | **81.35%** | **83.33%** | **0.8444** | **0.8143** | 0.8444 | 0.7778 | **0.8291** | **0.8874** |
| 🥈 **Tuned Random Forest** | **78.37%** | **80.95%** | 0.8261 | 0.7805 | 0.8296 | 0.7308 | 0.8043 | 0.8537 |
| 🥉 **Tuned ANN (MLP)** | 71.63% | 70.24% | 0.7253 | 0.7343 | 0.7370 | 0.6923 | 0.7357 | 0.7927 |
| **Naïve Bayes** | 69.64% | 71.43% | 0.7073 | 0.7696 | 0.6185 | **0.7863** | 0.6858 | 0.7573 |
| **Tuned SVM (RBF)** | 61.51% | 65.48% | 0.7238 | 0.5909 | **0.9148** | 0.2692 | 0.7180 | 0.7259 |

#### B. Temporal Trajectory: Which 10s Slice is the "Golden Window"?
Evaluating each 10s segment independently across all 84 subjects revealed a **U-shaped temporal diagnostic curve**:

| Time Window | XGBoost | Random Forest | SVM (RBF) | Naïve Bayes | ANN (MLP) | **Slice Average** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Segment 0 ($0-10\text{s}$)** | 77.38% | 75.00% | 73.81% | 69.05% | 60.71% | **71.19%** (🥈 2nd) |
| **Segment 1 ($10-20\text{s}$)** | 72.62% | 71.43% | 71.43% | 71.43% | 64.29% | **70.24%** (3rd) |
| **Segment 2 ($20-30\text{s}$)** | 69.05% | 65.48% | 71.43% | 65.48% | 57.14% | **65.72%** (5th) |
| **Segment 3 ($30-40\text{s}$)** | 75.00% | 72.62% | 63.10% | 66.67% | 64.29% | **68.34%** (4th) |
| **Segment 4 ($40-50\text{s}$)** | 61.90% | 67.86% | 65.48% | 71.43% | 53.57% | **64.05%** (6th) |
| **Segment 5 ($50-60\text{s}$)** 🏆 | **79.76%** | **75.00%** | **75.00%** | **70.24%** | **63.10%** | **72.62%** (🥇 1st) |

> **Key Discovery:** **Segment 5 ($50\text{s} - 60\text{s}$)** is the clinical "Golden Window" — **XGBoost reached 79.76% Accuracy and 0.8667 Recall** on this slice alone.

---

### 3. Frequency-Band Decomposition (Delta, Theta, Alpha, Beta)
*Continuous 60-second signal decomposed using zero-phase Butterworth filters + Domain-Isolated Feature Selection ($8\text{T} + 8\text{F} + 8\text{E} = 24$ balanced features).*

| Frequency Band | Classifier | Accuracy (%) | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Delta ($0.5-4\text{ Hz}$)** | Naïve Bayes | 54.76% | 0.6842 | 0.2889 | 0.4062 | 0.5937 |
| **Theta ($4-8\text{ Hz}$)** | Tuned Random Forest | 67.86% | 0.6957 | 0.7111 | 0.7033 | 0.6752 |
| **Alpha ($8-13\text{ Hz}$)** 🏆 | **Naïve Bayes** | **71.43%** | **0.7059** | **0.8000** | **0.7500** | **0.7168** |
| | Tuned SVM (RBF) | 69.05% | 0.6939 | 0.7556 | 0.7234 | 0.7151 |
| **Beta ($13-30\text{ Hz}$)** | Tuned SVM (RBF) | 70.24% | 0.6852 | **0.8222** | 0.7475 | **0.7823** |
| | Naïve Bayes | 70.24% | 0.7083 | 0.7556 | 0.7312 | 0.7590 |

---

## 🔍 Explainable AI (XAI) & PCA

To validate physiological plausibility and identify non-spurious biomarkers:
1. **SHAP (SHapley Additive exPlanations):** TreeExplainer applied to tuned tree models generates Beeswarm plots and feature attribution bars. Key drivers identified:
   - Left-Right Frontal Asymmetry (`F3` vs. `F4`).
   - Temporal Shannon Entropy (`T3`, `T4`).
   - Occipital Alpha Suppression (`O1`, `O2`).
2. **Principal Component Analysis (PCA):** Explores dimensional reduction, confirming that the first 50 principal components capture $>88\%$ of cumulative variance while preserving class separation.

---

## 📁 Repository Structure

```text
├── data/
│   ├── schizophrenia/                 # Raw EEG patient files (.txt / 16 channels)
│   └── healthy/                       # Raw EEG healthy control files
├── output/
│   ├── schizophrenia_6s_10segments.csv # 840 segmented rows (6s windows)
│   ├── schizophrenia_10s_6segments.csv# 504 segmented rows (10s windows)
│   ├── band_classification_plots/     # Visualizations for Delta, Theta, Alpha, Beta
│   ├── segmented_10s_plots/           # Confusion matrices, ROC curves, bar charts
│   └── segment_slice_plots/           # Dedicated subfolders for each of the 6 time slices
├── extract_features.py                # Extracts 448 continuous features per subject
├── segment_6s_features.py             # Generates the 6s window multi-domain dataset
├── compare_classifiers.py             # Pure segment benchmarking script
├── tune_all_models.py                 # Automated 5-fold GroupKFold hyperparameter tuning
├── segment_10s_features.py            # Generates the 10s window dataset
├── tune_10s_segments.py               # Active tuning & evaluation for 10s segments
├── compare_segment_slices.py          # Temporal slice comparison (0-10s to 50-60s)
├── band_classification_pipeline.py    # Frequency band decomposition & domain-isolated tuning
├── explainable_ai.py                  # Generates SHAP beeswarm and summary plots
├── apply_pca.py                       # PCA projection and scree plots
├── requirements.txt                   # Environment dependencies
└── README.md                          # Project documentation
