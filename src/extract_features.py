"""
Phase 3: Feature Extraction Script
Dataset: Nasrabadi Pediatric ADHD EEG Dataset (128 Hz, 19 Channels)
Input:
  - data/processed/preprocessed_epochs.npz (5,136 clean 2.0s epochs, 121 participants)
Extracted Features (Welch PSD per channel):
  - Absolute Band Powers:
      * Delta: 0.5 - 4.0 Hz
      * Theta: 4.0 - 8.0 Hz
      * Alpha: 8.0 - 13.0 Hz
      * Beta:  13.0 - 30.0 Hz
      * Total: 0.5 - 30.0 Hz
  - Relative Band Powers (Normalized to Total 0.5 - 30.0 Hz Power):
      * Relative Delta, Theta, Alpha, Beta
  - Clinical Ratio Feature:
      * Theta/Beta Ratio (TBR = Theta / Beta)
Artifacts Generated:
  - data/processed/features_extracted.csv (Epoch-level, 5,136 rows, 194 columns)
  - data/processed/features_participant_level.csv (Participant-level, 121 rows)
  - results/figures/psd_and_features_comparison.png
  - documents/phase_3_feature_extraction.txt
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "documents"

INPUT_NPZ_PATH = PROCESSED_DIR / "preprocessed_epochs.npz"
EPOCH_FEATURES_CSV_PATH = PROCESSED_DIR / "features_extracted.csv"
PARTICIPANT_FEATURES_CSV_PATH = PROCESSED_DIR / "features_participant_level.csv"
FIGURE_PATH = FIGURES_DIR / "psd_and_features_comparison.png"
DOC_PATH = DOCS_DIR / "phase_3_feature_extraction.txt"

# Frequency bands (Hz)
BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'total': (0.5, 30.0)
}

FS = 128
NPERSEG = 256  # 2.0-second window -> 0.5 Hz frequency resolution


def compute_epoch_psd(X, fs=FS, nperseg=NPERSEG):
    """
    Compute Welch Power Spectral Density across all epochs and channels.
    Parameters:
        X: numpy array of shape (n_epochs, n_channels, n_samples)
        fs: sampling frequency (128 Hz)
        nperseg: length of Welch window (256 samples)
    Returns:
        freqs: array of frequency bins (129,)
        psd: array of power spectral densities (n_epochs, n_channels, 129)
    """
    freqs, psd = welch(X, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd


def extract_spectral_features(freqs, psd, channels):
    """
    Extract absolute power, relative power, and TBR for each channel.
    Parameters:
        freqs: 1D array of frequency bins
        psd: 3D array of shape (n_epochs, n_channels, n_freqs)
        channels: list of channel names (19,)
    Returns:
        feature_dict: dictionary mapping feature name to 1D array of shape (n_epochs,)
    """
    n_epochs, n_channels, _ = psd.shape
    features = {}

    # Precompute frequency masks
    masks = {}
    for band_name, (f_low, f_high) in BANDS.items():
        masks[band_name] = (freqs >= f_low) & (freqs <= f_high)

    # Compute absolute band powers for all channels simultaneously
    abs_powers = {}
    for band_name, mask in masks.items():
        # Trapezoidal integration across frequency axis
        band_pwr = trapezoid(psd[:, :, mask], freqs[mask], axis=-1)  # (n_epochs, n_channels)
        abs_powers[band_name] = band_pwr

    total_pwr = abs_powers['total']  # (n_epochs, n_channels)
    # Numerical safeguard for division
    safe_total = np.maximum(total_pwr, 1e-12)
    safe_beta = np.maximum(abs_powers['beta'], 1e-12)

    # Compute relative powers and ratios per channel
    for ch_idx, ch_name in enumerate(channels):
        # 1. Absolute powers
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'total']:
            features[f"{ch_name}_{band_name}_abs"] = abs_powers[band_name][:, ch_idx]

        # 2. Relative powers (normalized to 0.5 - 30 Hz total power)
        for band_name in ['delta', 'theta', 'alpha', 'beta']:
            features[f"{ch_name}_{band_name}_rel"] = (
                abs_powers[band_name][:, ch_idx] / safe_total[:, ch_idx]
            )

        # 3. Theta/Beta Ratio (TBR)
        features[f"{ch_name}_tbr"] = (
            abs_powers['theta'][:, ch_idx] / safe_beta[:, ch_idx]
        )

    return features


def build_feature_dataframes(features_dict, y, groups, epoch_indices):
    """
    Build epoch-level and participant-level feature DataFrames.
    """
    # Metadata columns
    meta_df = pd.DataFrame({
        'epoch_id': np.arange(len(y)),
        'participant_id': groups,
        'class_label': y,
        'label': (y == 'ADHD').astype(int)  # ADHD: 1, Control: 0
    })

    # Feature columns
    feat_df = pd.DataFrame(features_dict)

    # Combine
    full_epoch_df = pd.concat([meta_df, feat_df], axis=1)

    # Participant-level aggregation (mean of features for each child)
    numeric_cols = list(features_dict.keys())
    participant_df = full_epoch_df.groupby(['participant_id', 'class_label', 'label'])[numeric_cols].mean().reset_index()

    # Reorder participant columns so metadata is upfront
    meta_p_cols = ['participant_id', 'class_label', 'label']
    participant_df = participant_df[meta_p_cols + numeric_cols]

    return full_epoch_df, participant_df


def plot_psd_and_features(freqs, psd, y, channels, participant_df, figure_path):
    """
    Generate publication-quality diagnostic figure for Phase 3:
    - Grand Average Welch PSD for ADHD vs Control (Fz and Cz) with shaded bands.
    - Topographic/Midline boxplots for Relative Theta and Theta/Beta Ratio (TBR).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    adhd_mask = (y == 'ADHD')
    ctrl_mask = (y == 'Control')

    fz_idx = list(channels).index('Fz')
    cz_idx = list(channels).index('Cz')

    # Subplot 1: Grand Average PSD at Frontal Midline (Fz)
    ax1 = axes[0, 0]
    psd_adhd_fz = np.mean(psd[adhd_mask, fz_idx, :], axis=0)
    psd_ctrl_fz = np.mean(psd[ctrl_mask, fz_idx, :], axis=0)

    ax1.plot(freqs, psd_adhd_fz, color='#d9534f', linewidth=2.0, label='ADHD (Mean PSD)')
    ax1.plot(freqs, psd_ctrl_fz, color='#2e6da4', linewidth=2.0, label='Control (Mean PSD)')

    # Shaded bands
    ax1.axvspan(0.5, 4.0, color='gray', alpha=0.15, label='Delta (0.5-4 Hz)')
    ax1.axvspan(4.0, 8.0, color='orange', alpha=0.15, label='Theta (4-8 Hz)')
    ax1.axvspan(8.0, 13.0, color='green', alpha=0.15, label='Alpha (8-13 Hz)')
    ax1.axvspan(13.0, 30.0, color='purple', alpha=0.15, label='Beta (13-30 Hz)')

    ax1.set_xlim(0.5, 30.0)
    ax1.set_ylim(0, max(psd_adhd_fz.max(), psd_ctrl_fz.max()) * 1.15)
    ax1.set_title("Grand Average PSD at Fz (Frontal Midline)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Frequency (Hz)", fontsize=10)
    ax1.set_ylabel(r"Power Spectral Density ($\mu V^2 / Hz$)", fontsize=10)
    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Grand Average PSD at Central Midline (Cz)
    ax2 = axes[0, 1]
    psd_adhd_cz = np.mean(psd[adhd_mask, cz_idx, :], axis=0)
    psd_ctrl_cz = np.mean(psd[ctrl_mask, cz_idx, :], axis=0)

    ax2.plot(freqs, psd_adhd_cz, color='#d9534f', linewidth=2.0, label='ADHD (Mean PSD)')
    ax2.plot(freqs, psd_ctrl_cz, color='#2e6da4', linewidth=2.0, label='Control (Mean PSD)')

    ax2.axvspan(0.5, 4.0, color='gray', alpha=0.15)
    ax2.axvspan(4.0, 8.0, color='orange', alpha=0.15)
    ax2.axvspan(8.0, 13.0, color='green', alpha=0.15)
    ax2.axvspan(13.0, 30.0, color='purple', alpha=0.15)

    ax2.set_xlim(0.5, 30.0)
    ax2.set_ylim(0, max(psd_adhd_cz.max(), psd_ctrl_cz.max()) * 1.15)
    ax2.set_title("Grand Average PSD at Cz (Central Vertex)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Frequency (Hz)", fontsize=10)
    ax2.set_ylabel(r"Power Spectral Density ($\mu V^2 / Hz$)", fontsize=10)
    ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Relative Theta Power Comparison (Participant Level)
    ax3 = axes[1, 0]
    p_adhd = participant_df[participant_df['class_label'] == 'ADHD']
    p_ctrl = participant_df[participant_df['class_label'] == 'Control']

    box_data_rel_theta = [
        p_adhd['Fz_theta_rel'], p_ctrl['Fz_theta_rel'],
        p_adhd['Cz_theta_rel'], p_ctrl['Cz_theta_rel']
    ]
    bp1 = ax3.boxplot(
        box_data_rel_theta,
        patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 1.5}
    )
    ax3.set_xticks([1, 2, 3, 4])
    ax3.set_xticklabels(['Fz (ADHD)', 'Fz (Control)', 'Cz (ADHD)', 'Cz (Control)'])
    colors = ['#ff9999', '#99ccff', '#ff9999', '#99ccff']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)

    ax3.set_title("Participant-Level Relative Theta Power (Fz vs Cz)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Relative Power (Theta / Total)", fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Subplot 4: Theta/Beta Ratio (TBR) Comparison (Participant Level)
    ax4 = axes[1, 1]
    box_data_tbr = [
        p_adhd['Fz_tbr'], p_ctrl['Fz_tbr'],
        p_adhd['Cz_tbr'], p_ctrl['Cz_tbr']
    ]
    bp2 = ax4.boxplot(
        box_data_tbr,
        patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 1.5}
    )
    ax4.set_xticks([1, 2, 3, 4])
    ax4.set_xticklabels(['Fz (ADHD)', 'Fz (Control)', 'Cz (ADHD)', 'Cz (Control)'])
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)

    ax4.set_title("Participant-Level Theta/Beta Ratio (TBR)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("TBR (Theta / Beta)", fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Phase 3: Welch PSD & Spectral Feature Extraction Summary", fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Verification figure saved to: {figure_path}")


def write_phase_3_document(doc_path, epoch_df, participant_df, channels):
    """
    Generate mandatory Phase 3 milestone completion document.
    """
    adhd_p = participant_df[participant_df['class_label'] == 'ADHD']
    ctrl_p = participant_df[participant_df['class_label'] == 'Control']

    n_features = len([c for c in epoch_df.columns if c not in ['epoch_id', 'participant_id', 'class_label', 'label']])

    doc_text = [
        "=" * 80,
        "PHASE 3 EXECUTION REPORT: SPECTRAL FEATURE EXTRACTION (WELCH PSD & TBR)",
        "=" * 80,
        "Phase Name: Phase 3 - Spectral Feature Extraction",
        "Date Completed: November 2026",
        f"Input File: {INPUT_NPZ_PATH.name} (5,136 clean 2.0s epochs, 19 channels, 121 participants)",
        "Output Files:",
        f"  - data/processed/{EPOCH_FEATURES_CSV_PATH.name} ({len(epoch_df):,} rows, {len(epoch_df.columns)} columns)",
        f"  - data/processed/{PARTICIPANT_FEATURES_CSV_PATH.name} ({len(participant_df)} rows, {len(participant_df.columns)} columns)",
        f"  - results/figures/{FIGURE_PATH.name}",
        "",
        "-" * 80,
        "1. SPECTRAL EXTRACTION SPECIFICATIONS & METHODOLOGY",
        "-" * 80,
        f"* Method: Welch's Power Spectral Density (scipy.signal.welch)",
        f"* Sampling Frequency (fs): {FS} Hz",
        f"* Window Size (nperseg): {NPERSEG} samples (2.0 seconds)",
        f"* Window Function: Hann window",
        f"* Frequency Resolution (df): {FS / NPERSEG:.2f} Hz",
        f"* Numerical Integration: Composite Trapezoidal Rule (scipy.integrate.trapezoid)",
        f"* Frequency Band Definitions:",
        f"    - Delta:  0.5 - 4.0 Hz",
        f"    - Theta:  4.0 - 8.0 Hz",
        f"    - Alpha:  8.0 - 13.0 Hz",
        f"    - Beta:  13.0 - 30.0 Hz",
        f"    - Total:  0.5 - 30.0 Hz (consistent with preprocessed filter passband)",
        f"* Features Extracted Per Channel ({len(channels)} channels):",
        f"    1. Absolute Powers: [band]_abs for delta, theta, alpha, beta, total (5 features)",
        f"    2. Relative Powers: [band]_rel = [band]_abs / total_abs for delta, theta, alpha, beta (4 features)",
        f"    3. Clinical Ratio:  tbr = theta_abs / beta_abs (1 feature)",
        f"    Total Per Channel: 10 features x 19 channels = {len(channels) * 10} features per epoch.",
        "",
        "-" * 80,
        "2. DATA INTEGRITY & SANITY CHECKS",
        "-" * 80,
        f"* Total Epochs Processed: {len(epoch_df):,}",
        f"    - ADHD Epochs:    {len(epoch_df[epoch_df['class_label'] == 'ADHD']):,} (57.05%)",
        f"    - Control Epochs: {len(epoch_df[epoch_df['class_label'] == 'Control']):,} (42.95%)",
        f"* Total Participants: {len(participant_df)}",
        f"    - ADHD Children:    {len(adhd_p)} (50.41%)",
        f"    - Control Children: {len(ctrl_p)} (49.59%)",
        f"* Missing / NaN Values across all features: {epoch_df.isna().sum().sum()}",
        f"* Infinite Values across all features:      {np.isinf(epoch_df.select_dtypes(include=np.number)).sum().sum()}",
        f"* Zero Division Protection: Applied 1e-12 floor to denominators.",
        "",
        "-" * 80,
        "3. DESCRIPTIVE STATISTICS FOR KEY FRONTAL & VERTEX SITES",
        "-" * 80,
        "Participant-Level Distributions (Mean +/- Std):",
        f"  - Fz Relative Theta:",
        f"      ADHD:    {adhd_p['Fz_theta_rel'].mean():.4f} +/- {adhd_p['Fz_theta_rel'].std():.4f}",
        f"      Control: {ctrl_p['Fz_theta_rel'].mean():.4f} +/- {ctrl_p['Fz_theta_rel'].std():.4f}",
        f"  - Cz Relative Theta:",
        f"      ADHD:    {adhd_p['Cz_theta_rel'].mean():.4f} +/- {adhd_p['Cz_theta_rel'].std():.4f}",
        f"      Control: {ctrl_p['Cz_theta_rel'].mean():.4f} +/- {ctrl_p['Cz_theta_rel'].std():.4f}",
        f"  - Fz Theta/Beta Ratio (TBR):",
        f"      ADHD:    {adhd_p['Fz_tbr'].mean():.4f} +/- {adhd_p['Fz_tbr'].std():.4f}",
        f"      Control: {ctrl_p['Fz_tbr'].mean():.4f} +/- {ctrl_p['Fz_tbr'].std():.4f}",
        f"  - Cz Theta/Beta Ratio (TBR):",
        f"      ADHD:    {adhd_p['Cz_tbr'].mean():.4f} +/- {adhd_p['Cz_tbr'].std():.4f}",
        f"      Control: {ctrl_p['Cz_tbr'].mean():.4f} +/- {ctrl_p['Cz_tbr'].std():.4f}",
        "",
        "Scientific Observation on TBR:",
        "- High inter-individual variance is evident across both groups.",
        "- As warned in SKILL_updated.md, TBR exhibits notable overlap between groups",
        "  and must NOT be claimed as a standalone diagnostic biomarker, but rather",
        "  evaluated within multivariate machine-learning models.",
        "",
        "-" * 80,
        "4. ANTI-LEAKAGE VERIFICATION FOR DOWNSTREAM PIPELINES",
        "-" * 80,
        "- The output dataset strictly maintains the 'participant_id' and 'label' columns.",
        "- No cross-participant normalisation or feature scaling was performed here.",
        "- All scaling (StandardScaler / RobustScaler) and model fitting must be restricted",
        "  strictly within training folds in Phase 4 & Phase 5.",
        "",
        "-" * 80,
        "5. STATUS & NEXT MILESTONE",
        "-" * 80,
        "STATUS: PHASE 3 COMPLETED AND VERIFIED.",
        "Next Phase: Phase 4 - Participant-Wise Group Splitting (StratifiedGroupKFold)",
        "=" * 80
    ]

    doc_path.write_text("\n".join(doc_text), encoding="utf-8")
    print(f"[OK] Completion document written to: {doc_path}")


def run_feature_extraction():
    print("=" * 80)
    print("PHASE 3: SPECTRAL FEATURE EXTRACTION (WELCH PSD & TBR)")
    print("=" * 80)

    if not INPUT_NPZ_PATH.exists():
        print(f"[!] Error: Preprocessed epochs file not found at: {INPUT_NPZ_PATH}")
        sys.exit(1)

    print(f"[*] Loading preprocessed epochs: {INPUT_NPZ_PATH.name}...")
    npz_data = np.load(INPUT_NPZ_PATH)
    X = npz_data['X']                    # (5136, 19, 256)
    y = npz_data['y']                    # (5136,)
    groups = npz_data['groups']          # (5136,)
    epoch_indices = npz_data['epoch_indices']  # (5136,)
    channels = npz_data['channels']      # (19,)
    fs = int(npz_data['fs'])

    print(f"[OK] Loaded array shapes:")
    print(f"     - X: {X.shape} (N_epochs, N_channels, N_samples)")
    print(f"     - y: {y.shape} (ADHD: {(y == 'ADHD').sum()}, Control: {(y == 'Control').sum()})")
    print(f"     - groups: {groups.shape} ({len(np.unique(groups))} unique children)")
    print(f"     - channels: {list(channels)}")

    print(f"\n[*] Computing Welch PSD across all {len(X)} epochs and {len(channels)} channels...")
    freqs, psd = compute_epoch_psd(X, fs=fs, nperseg=NPERSEG)
    print(f"[OK] Welch PSD computed. Frequency bins: {len(freqs)} (range {freqs[0]:.1f} - {freqs[-1]:.1f} Hz, df={freqs[1]-freqs[0]:.2f} Hz)")

    print(f"\n[*] Extracting spectral band powers and clinical ratios...")
    features_dict = extract_spectral_features(freqs, psd, channels)
    n_features = len(features_dict)
    print(f"[OK] Extracted {n_features} features per epoch ({n_features // len(channels)} features x {len(channels)} channels).")

    print(f"\n[*] Constructing DataFrames...")
    epoch_df, participant_df = build_feature_dataframes(features_dict, y, groups, epoch_indices)

    # Sanity checks
    print(f"\n[*] Running Data Quality & Sanity Assertions...")
    assert len(epoch_df) == len(X), "Epoch count mismatch!"
    assert len(participant_df) == len(np.unique(groups)), "Participant count mismatch!"
    assert not epoch_df.isna().any().any(), "Found NaNs in epoch features!"
    assert not np.isinf(epoch_df.select_dtypes(include=np.number)).any().any(), "Found Infs in epoch features!"
    print(f"[OK] All sanity checks passed: Zero NaNs, Zero Infs, 100% integrity.")

    print(f"\n[*] Saving extracted feature datasets...")
    epoch_df.to_csv(EPOCH_FEATURES_CSV_PATH, index=False)
    print(f"[OK] Epoch-level features saved: {EPOCH_FEATURES_CSV_PATH} ({epoch_df.shape[0]} rows, {epoch_df.shape[1]} cols)")
    participant_df.to_csv(PARTICIPANT_FEATURES_CSV_PATH, index=False)
    print(f"[OK] Participant-level features saved: {PARTICIPANT_FEATURES_CSV_PATH} ({participant_df.shape[0]} rows, {participant_df.shape[1]} cols)")

    print(f"\n[*] Generating PSD & feature distribution plots...")
    plot_psd_and_features(freqs, psd, y, channels, participant_df, FIGURE_PATH)

    print(f"\n[*] Writing Phase 3 milestone completion document...")
    write_phase_3_document(DOC_PATH, epoch_df, participant_df, channels)

    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_feature_extraction()
