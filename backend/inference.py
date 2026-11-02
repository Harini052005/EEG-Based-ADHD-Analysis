"""
Backend Inference Service: Loads trained models and runs live predictions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from backend.data_service import data_service
MODELS_DIR = PROJECT_ROOT / "backend" / "models"


class InferenceService:
    def __init__(self):
        self._models = {}
        self._load_all_models()

    def _load_all_models(self):
        for model_prefix in ["rf", "svm"]:
            for montage in ["full_19ch", "frontal_5ch", "midline_2ch"]:
                key = f"{model_prefix}_{montage}"
                model_file = MODELS_DIR / f"{key}.joblib"
                if model_file.exists():
                    self._models[key] = joblib.load(model_file)

    def predict_participant(self, participant_id, model_type="svm", montage="full_19ch"):
        """
        Run subject-level prediction across all clean epochs of a participant.
        """
        data_service.load_data()
        df = data_service._df
        if df is None:
            raise ValueError("Feature dataset not loaded.")

        model_key = f"{model_type.lower()}_{montage.lower()}"
        if model_key not in self._models:
            raise ValueError(f"Model key '{model_key}' not available.")

        bundle = self._models[model_key]
        model = bundle["model"]
        feature_names = bundle["feature_names"]

        p_mask = (df['participant_id'] == participant_id)
        if not p_mask.any():
            raise ValueError(f"Participant '{participant_id}' not found.")

        p_df = df[p_mask]
        X = p_df[feature_names].values
        true_label = int(p_df['label'].iloc[0])
        true_class = p_df['class_label'].iloc[0]

        # Predict per epoch
        epoch_probs = model.predict_proba(X)[:, 1]
        epoch_preds = (epoch_probs >= 0.5).astype(int)

        # Soft voting aggregation
        mean_prob = float(np.mean(epoch_probs))
        subject_pred = 1 if mean_prob >= 0.5 else 0
        subject_class = "ADHD" if subject_pred == 1 else "Control"
        is_correct = bool(subject_pred == true_label)

        # Confidence (distance from decision boundary 0.5)
        confidence = abs(mean_prob - 0.5) * 2 * 100

        # Epoch prediction breakdown
        adhd_epoch_votes = int(epoch_preds.sum())
        ctrl_epoch_votes = int(len(epoch_preds) - adhd_epoch_votes)

        return {
            "participant_id": participant_id,
            "true_diagnosis": true_class,
            "true_label": true_label,
            "predicted_class": subject_class,
            "predicted_label": subject_pred,
            "adhd_probability": round(mean_prob, 4),
            "confidence_percent": round(confidence, 1),
            "is_correct": is_correct,
            "model_used": model_type.upper(),
            "montage_used": montage,
            "channels_used": bundle["channels"],
            "total_epochs": len(p_df),
            "epoch_votes": {
                "adhd_votes": adhd_epoch_votes,
                "control_votes": ctrl_epoch_votes,
                "adhd_vote_ratio": round(adhd_epoch_votes / len(epoch_preds), 3)
            },
            "disclaimer": "Research Prototype: Band powers & ML predictions are exploratory research markers, NOT a clinical diagnostic tool."
        }

    def predict_epoch(self, epoch_id, model_type="svm", montage="full_19ch"):
        """Run single-epoch prediction."""
        data_service.load_data()
        df = data_service._df
        if epoch_id < 0 or epoch_id >= len(df):
            raise ValueError(f"Epoch ID {epoch_id} out of bounds.")

        model_key = f"{model_type.lower()}_{montage.lower()}"
        bundle = self._models[model_key]
        model = bundle["model"]
        feature_names = bundle["feature_names"]

        row = df.iloc[epoch_id]
        X = row[feature_names].values.reshape(1, -1)
        true_label = int(row['label'])
        true_class = row['class_label']
        p_id = row['participant_id']

        prob = float(model.predict_proba(X)[0, 1])
        pred = 1 if prob >= 0.5 else 0
        pred_class = "ADHD" if pred == 1 else "Control"

        return {
            "epoch_id": int(epoch_id),
            "participant_id": p_id,
            "true_diagnosis": true_class,
            "predicted_class": pred_class,
            "adhd_probability": round(prob, 4),
            "confidence_percent": round(abs(prob - 0.5) * 2 * 100, 1),
            "is_correct": bool(pred == true_label),
            "model_used": model_type.upper(),
            "montage_used": montage
        }


# Singleton instance
inference_service = InferenceService()
