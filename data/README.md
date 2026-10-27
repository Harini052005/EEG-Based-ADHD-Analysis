# EEG Dataset: Download Instructions & Placement

This project uses the **Nasrabadi Pediatric ADHD EEG Dataset** (Ali Motie Nasrabadi et al., Shahed University).

---

## Dataset Summary
- **Subjects**: 121 children (61 diagnosed with ADHD, 60 age-matched healthy controls).
- **Ages**: 7–12 years.
- **Recording Channels**: 19 channels (10–20 standard montage: `Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2`).
- **Reference**: Ear electrodes A1 and A2.
- **Sampling Frequency**: 128 Hz.
- **Protocol**: Task-based visual attention task (counting cartoon characters).

---

## Where to Download

### Option A: Kaggle (Easiest & Fastest — Free download)
1. Go to Kaggle and search for **"EEG Dataset for ADHD"** or **"EEG ADHD Nasrabadi MAT"**:
   - Link 1: [https://www.kaggle.com/datasets/eeg-dataset-for-adhd](https://www.kaggle.com/datasets/eeg-dataset-for-adhd)
   - Link 2: [https://www.kaggle.com/datasets/eeg-adhd-nasrabadi-mat](https://www.kaggle.com/datasets/eeg-adhd-nasrabadi-mat)
2. Click **Download** (approx. 100–300 MB).

### Option B: IEEE DataPort (Official Open-Access Archive)
1. Visit the IEEE DataPort repository:
   - Search: **"EEG Dataset for ADHD"** (DOI: `10.21227/rzfh-zn36`)
   - URL: [https://ieee-dataport.org/open-access/eeg-data-adhd-control-children](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)
2. Download the `.zip` archive containing the MATLAB `.mat` or ASCII/text recordings.

---

## Where to Place the Files

Extract the downloaded archive into the `data/raw/` folder in this project:

```text
EEG-Based ADHD Analysis/
└── data/
    ├── README.md
    └── raw/
        ├── ADHD_part1/     (or .mat files: v1p.mat, v2p.mat, etc.)
        └── Control_part1/  (or .mat files: v1c.mat, v2c.mat, etc.)
```
*(Exact folder or file names may vary slightly depending on whether you download the MAT or CSV version. Once placed in `data/raw/`, our inspection script will automatically detect and verify the format).*
