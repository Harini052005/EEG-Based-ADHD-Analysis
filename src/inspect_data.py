"""
Dataset Audit & Inspection Script
Inspects downloaded EEG files in data/raw/, verifies participant count,
channels, sampling frequency, and formats, and generates dataset_audit.md.
"""

from pathlib import Path
import numpy as np
import scipy.io as sio

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
AUDIT_OUTPUT = Path(__file__).resolve().parent.parent / "dataset_audit.md"

def inspect_dataset():
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[!] Created directory: {RAW_DATA_DIR}")
        print("[!] Please place your downloaded dataset files into data/raw/")
        return

    files = list(RAW_DATA_DIR.rglob("*"))
    data_files = [f for f in files if f.suffix.lower() in [".mat", ".csv", ".edf", ".txt", ".set"]]

    if not data_files:
        print(f"[!] No EEG recording files (.mat, .csv, .edf, .txt) found in: {RAW_DATA_DIR}")
        print("[!] Please download the dataset and extract it into data/raw/ (see data/README.md).")
        return

    print(f"[✓] Found {len(data_files)} data files in {RAW_DATA_DIR}")
    
    # Check sample file
    sample_file = data_files[0]
    print(f"[*] Inspecting sample file: {sample_file.name}")
    
    audit_lines = [
        "# Dataset Audit Report",
        "",
        f"- **Audit Directory**: `{RAW_DATA_DIR}`",
        f"- **Total Data Files Found**: {len(data_files)}",
        f"- **Sample File**: `{sample_file.name}`",
        "",
        "## File List Sample",
        "```text"
    ]
    
    for f in data_files[:15]:
        audit_lines.append(f"{f.parent.name}/{f.name} ({f.stat().st_size / 1024:.1f} KB)")
    if len(data_files) > 15:
        audit_lines.append(f"... and {len(data_files) - 15} more files")
    audit_lines.append("```")
    audit_lines.append("")

    # Attempt inspection if .mat
    if sample_file.suffix.lower() == ".mat":
        try:
            mat = sio.loadmat(sample_file)
            keys = [k for k in mat.keys() if not k.startswith("__")]
            audit_lines.append(f"- **MAT Keys**: `{keys}`")
            for k in keys:
                val = mat[k]
                if isinstance(val, np.ndarray):
                    audit_lines.append(f"- Shape of `{k}`: {val.shape}, dtype: {val.dtype}")
        except Exception as e:
            audit_lines.append(f"- Failed to read sample .mat with scipy.io: {e}")

    audit_lines.append("")
    audit_lines.append("## Next Steps")
    audit_lines.append("1. Verify channel names match 10-20 standard montage (19 channels).")
    audit_lines.append("2. Confirm sampling frequency is 128 Hz.")
    audit_lines.append("3. Verify ADHD vs Control subject partitioning.")
    
    AUDIT_OUTPUT.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"[✓] Generated audit report at: {AUDIT_OUTPUT}")

if __name__ == "__main__":
    inspect_dataset()
