# Dataset Audit Report: Nasrabadi Pediatric ADHD EEG

- **Audit Date**: 2026-10-27
- **File Path**: `E:\PG-Academic\MINI - PROJECT\EEG-Based ADHD Analysis\data\raw\adhdata.csv`
- **File Size**: 254.90 MB
- **Total Rows (Samples)**: 2,166,383
- **Total Columns**: 21

## 1. Participants & Class Balance
- **Total Unique Participants (Children)**: 121
- **ADHD Participants**: 61
- **Control Participants**: 60
- **Sampling Frequency ($f_s$)**: 128 Hz
- **Recording Duration per Child**: Min: 62.4s (7,983 samples) | Max: 337.9s (43,252 samples) | Mean: 139.9s (17,904 samples)

## 2. Channel Montage (10-20 Standard)
- **Expected Channels (19)**: `['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']`
- **Channels Found in File (19)**: `['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']`
- **Missing Channels**: None
- **Extra Columns**: None (Clean metadata: `Class`, `ID`)

## 3. Data Integrity & Missing Values
- **Total Missing Values (NaN/Null)**: 0
- **Signal Amplitude Range**: Min = -13237.00, Max = 13813.00

## 4. Participant List & Sample Distribution
| ID | Class | Samples | Duration (s) |
| :--- | :--- | :--- | :--- |
| v10p | ADHD | 14,304 | 111.8 |
| v12p | ADHD | 17,604 | 137.5 |
| v14p | ADHD | 17,562 | 137.2 |
| v15p | ADHD | 43,252 | 337.9 |
| v173 | ADHD | 24,241 | 189.4 |
| v18p | ADHD | 25,003 | 195.3 |
| v19p | ADHD | 23,063 | 180.2 |
| v1p | ADHD | 12,258 | 95.8 |
| v20p | ADHD | 35,328 | 276.0 |
| v21p | ADHD | 16,574 | 129.5 |
| ... | ... | ... | *(Showing 10 of 121 participants)* |

## 5. Audit Conclusions & Guardrails
1. **Participant-Wise Splitting is Fully Supported**: Because each row contains the `ID` column, we can group all rows by child and apply `StratifiedGroupKFold`. Segment-level random splitting is strictly prohibited to prevent data leakage.
2. **Channel Mapping Verified**: All 19 standard electrodes are present. Reduced montage experiments can map standard frontal/central sites (`Fz, Cz, F3, F4`).
3. **Zero Missing Values**: The dataset is fully intact without NaNs.