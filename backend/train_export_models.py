"""
Backend Utility: Train and export final production pipeline models.
Models saved:
  - backend/models/rf_full_19ch.joblib
  - backend/models/svm_full_19ch.joblib
  - backend/models/rf_frontal_5ch.joblib
  - backend/models/svm_frontal_5ch.joblib
  - backend/models/rf_midline_2ch.joblib
  - backend/models/svm_midline_2ch.joblib
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BACKEND_DIR = PROJECT_ROOT / "backend"
MODELS_DIR = BACKEND_DIR / "models"

INPUT_CSV = PROCESSED_DIR / "features_with_splits.csv"
RANDOM_STATE = 42

MONTAGES = {
    "full_19ch": ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                  'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'],
    "frontal_5ch": ['Fz', 'Cz', 'Pz', 'F3', 'F4'],
    "midline_2ch": ['Fz', 'Cz']
}


def export_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Loading feature dataset from {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV)
    y = df['label'].values

    all_feat_cols = [c for c in df.columns if c not in ['epoch_id', 'participant_id', 'class_label', 'label', 'fold']]

    for montage_name, ch_list in MONTAGES.items():
        print(f"\n[*] Training and exporting models for montage: {montage_name} ({len(ch_list)} channels)...")
        sub_cols = [c for c in all_feat_cols if any(c.startswith(f"{ch}_") for ch in ch_list)]
        X = df[sub_cols].values

        # 1. Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
        )
        rf.fit(X, y)
        rf_path = MODELS_DIR / f"rf_{montage_name}.joblib"
        joblib.dump({"model": rf, "feature_names": sub_cols, "channels": ch_list}, rf_path)
        print(f"    [OK] Saved Random Forest model to: {rf_path.name}")

        # 2. SVM Pipeline
        svm = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=1.0, kernel='rbf', probability=True, random_state=RANDOM_STATE, class_weight='balanced'))
        ])
        svm.fit(X, y)
        svm_path = MODELS_DIR / f"svm_{montage_name}.joblib"
        joblib.dump({"model": svm, "feature_names": sub_cols, "channels": ch_list}, svm_path)
        print(f"    [OK] Saved SVM model to: {svm_path.name}")

    print("\n[OK] All models exported successfully!")


if __name__ == "__main__":
    export_models()

