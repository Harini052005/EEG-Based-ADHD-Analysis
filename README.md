# ADHD Analysis Using EEG: Testing Accuracy with Fewer Channels and Noise

A reproducible, academic research prototype investigating machine learning classification of pediatric ADHD using electroencephalography (EEG).

---

## 🎯 Research Questions

1. **Channel Reduction**: Can machine learning models reliably classify ADHD using a predefined, smaller channel set (e.g., 4–5 frontal/central electrodes) compared to the full 19-channel standard 10–20 montage?
2. **Noise Robustness**: How does signal degradation (controlled simulated noise) affect classifier accuracy across full versus reduced channel montages?

> ⚠️ **Important Boundary**: This repository contains a **research prototype, not a clinical diagnostic tool**. Band powers and the Theta/Beta Ratio (TBR) are exploratory features, not standalone markers. Dataset labels are treated as research labels.

---

## 🔬 Scientific Methodology & Guardrails

- **Participant-Wise Splitting (No Data Leakage)**: Segments from the same child are never split across training and test sets. Evaluation uses group-aware cross-validation (`StratifiedGroupKFold`).
- **Pipeline Isolation**: Scaling and feature transformations are fitted strictly on training data inside scikit-learn pipelines.
- **Controlled Noise Simulation**: Noise tests are conducted with fixed random seeds on test partitions only, keeping raw signals untouched.

---

## 📁 Project Structure

```text
EEG-Based ADHD Analysis/
├── data/
│   ├── raw/                 # Raw EEG files (git-ignored)
│   └── README.md            # Dataset download and placement instructions
├── documents/
│   └── dataset_details.txt  # Detailed dataset specifications and project guide
├── src/
│   └── inspect_data.py      # Dataset inspection and audit script
├── results/
│   ├── figures/             # Output plots and graphs
│   └── metrics/             # Saved metric evaluations
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the **Nasrabadi Pediatric ADHD EEG Dataset** (DOI: `10.21227/rzfh-zn36`) from [Kaggle](https://www.kaggle.com/datasets/eeg-dataset-for-adhd) or [IEEE DataPort](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children) and place the files into `data/raw/` (see `data/README.md` for details).

### 3. Inspect Data
Run the dataset audit script:
```bash
python src/inspect_data.py
```
