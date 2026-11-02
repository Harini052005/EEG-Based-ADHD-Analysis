# ADHD Analysis Using EEG: Testing Accuracy with Fewer Channels and Noise

A reproducible, academic research prototype investigating machine learning classification of pediatric ADHD using electroencephalography (EEG).

---

## 🎯 Research Questions

1. **Channel Reduction (RQ 1)**: Can machine learning models reliably classify ADHD using a predefined, smaller channel set (e.g., 4–5 frontal/central electrodes) compared to the full 19-channel standard 10–20 montage?
2. **Noise Robustness (RQ 2)**: How does signal degradation (controlled simulated noise) affect classifier accuracy across full versus reduced channel montages?

> ⚠️ **Important Boundary**: This repository contains a **research prototype, not a clinical diagnostic tool**. Band powers and the Theta/Beta Ratio (TBR) are exploratory features, not standalone markers. Dataset labels are treated as research labels.

---

## 🔬 Scientific Methodology & Guardrails

- **Participant-Wise Splitting (No Data Leakage)**: Segments from the same child are never split across training and test sets. Evaluation uses group-aware cross-validation (`StratifiedGroupKFold` across 121 children).
- **Pipeline Isolation**: Scaling and feature transformations are fitted strictly on training data inside scikit-learn pipelines.
- **Controlled Noise Simulation**: Noise tests are conducted with fixed random seeds on test partitions only, keeping raw signals untouched.

---

## 📁 Project Directory Structure

```text
EEG-Based ADHD Analysis/
├── backend/
│   ├── app.py                      # Flask REST API & static UI server
│   ├── data_service.py             # Data loading and preprocessing provider
│   ├── inference.py                # Real-time ML inference service
│   ├── train_export_models.py      # Production model serialization script
│   └── models/                     # Saved model pipelines (*.joblib)
├── frontend/
│   ├── index.html                  # Responsive dashboard UI
│   ├── css/
│   │   └── styles.css              # Custom scientific design system
│   └── js/
│       ├── api.js                  # Asynchronous client API wrapper
│       ├── charts.js               # Interactive Chart.js graphs
│       ├── components.js           # Dynamic DOM component templates
│       └── app.js                  # Main controller & navigation router
├── src/
│   ├── inspect_data.py             # Phase 1: Dataset audit & inspection
│   ├── preprocess.py               # Phase 2: Bandpass/notch filtering & epoching
│   ├── extract_features.py         # Phase 3: Welch PSD (Theta, Alpha, Beta, TBR)
│   ├── split_data.py               # Phase 4: Participant-wise StratifiedGroupKFold
│   ├── train_models.py             # Phase 5: Random Forest & SVM baselines
│   ├── channel_experiments.py      # Phase 6: Full 19ch vs reduced montages (RQ 1)
│   └── noise_simulation.py         # Phase 7: Controlled noise testing (RQ 2)
├── documents/                      # Milestone completion reports (Phases 1-7)
├── data/
│   ├── raw/                        # Raw EEG dataset (adhdata.csv)
│   └── processed/                  # Preprocessed epochs & feature tables
├── results/
│   ├── figures/                    # Generated publication-quality figures
│   └── metrics/                    # JSON & CSV cross-validation evaluations
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart: Launching the Web UI

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python backend/app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

The interactive dashboard allows you to:
- 📊 Explore the **Dataset Audit & Pipeline Architecture**.
- 🧠 View **Live Scalp EEG Waveforms** and **Welch Power Spectral Density (PSD)** for any electrode.
- 📉 Inspect **Channel Reduction Results** comparing Full 19-Channel vs Frontal-Central 5-Channel, 4-Channel, and Midline 2-Channel setups.
- 🌪️ Interact with the **Noise Robustness Simulator** testing SNR levels from $+20\text{ dB}$ down to $-5\text{ dB}$.
- 🔬 Run **Live Subject Inference** with soft voting across any participant in the cohort.

---

## 🧪 Scientific Pipeline Execution (Phases 1 to 7)

To reproduce the experiments from scratch:

```bash
# Phase 1: Audit raw dataset
python src/inspect_data.py

# Phase 2: Zero-phase filtering & artifact rejection
python src/preprocess.py

# Phase 3: Welch PSD feature extraction
python src/extract_features.py

# Phase 4: 5-Fold StratifiedGroupKFold anti-leakage splitting
python src/split_data.py

# Phase 5: Train baseline RF & SVM classifiers
python src/train_models.py

# Phase 6: Channel reduction experiments (Research Question 1)
python src/channel_experiments.py

# Phase 7: Controlled noise robustness simulation (Research Question 2)
python src/noise_simulation.py
```
