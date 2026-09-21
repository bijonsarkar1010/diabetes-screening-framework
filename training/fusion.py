"""
Fusion layer — combines the EHR-model and imaging-model outputs into the
single structured record described in the paper (Section VI-D), and reports
the fused evaluation numbers to paste into Section VI-E.

This is a simple late-fusion baseline (weighted average of the two models'
predicted probabilities). It assumes ehr_model.py and imaging_model.py have
already been run and that you have a way to pair each patient's EHR row
with their retinal image (a shared patient/encounter ID). Adjust the
pairing logic to match how your two datasets are actually linked — Pima
and APTOS are NOT natively linked, so for a first pass you can pair them
synthetically (e.g., by matching test-set indices) purely to validate the
fusion mechanics; a real deployment would use a single linked cohort.
"""

import json
import uuid
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


def fuse_scores(ehr_proba, imaging_proba, w_ehr=0.5, w_imaging=0.5):
    """Weighted-average late fusion. Replace with a learned fusion layer
    once you have a real linked dataset."""
    return w_ehr * ehr_proba + w_imaging * imaging_proba


def risk_category(score_0_100):
    if score_0_100 < 33:
        return "low", "routine follow-up"
    elif score_0_100 < 66:
        return "moderate", "specialist referral"
    else:
        return "high", "urgent referral"


def build_output_record(patient_id, ehr_proba, imaging_proba, fused_proba, model_version="v0.1-baseline"):
    score = round(float(fused_proba) * 100, 1)
    category, action = risk_category(score)
    total = ehr_proba + imaging_proba
    return {
        "encounter_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "risk_score": score,
        "risk_category": category,
        "modality_contributions": {
            "retinal_imaging": round(imaging_proba / total, 3) if total > 0 else 0.0,
            "ehr_risk_factors": round(ehr_proba / total, 3) if total > 0 else 0.0,
            "wearable_trends": 0.0,  # not yet implemented — see paper Section VI-A
        },
        "explanation_summary": (
            f"Fused risk score driven {'primarily by retinal findings' if imaging_proba > ehr_proba else 'primarily by EHR risk factors'}; "
            f"see attached SHAP/Grad-CAM outputs for detail."
        ),
        "recommended_action": action,
        "model_version": model_version,
        "audit_log_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    # --- Example wiring: replace with your real, aligned test-set arrays ---
    # ehr_proba = np.load("ehr_test_proba.npy")
    # imaging_proba = np.load("imaging_test_proba.npy")
    # y_true = np.load("fused_test_labels.npy")
    #
    # For a first dry run without real linked data, synthesize a toy example:
    ehr_proba = np.array([0.72, 0.15, 0.44])
    imaging_proba = np.array([0.65, 0.20, 0.80])
    y_true = np.array([1, 0, 1])

    fused_proba = fuse_scores(ehr_proba, imaging_proba)
    fused_pred = (fused_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, fused_pred),
        "roc_auc": roc_auc_score(y_true, fused_proba) if len(set(y_true)) > 1 else None,
        "sensitivity_recall": recall_score(y_true, fused_pred),
        "specificity": specificity_score(y_true, fused_pred),
    }
    print("=== Fused model metrics (paste into paper Section VI-E) ===")
    print(json.dumps(metrics, indent=2))

    print("\n=== Example structured output record (paper Section VI-D schema) ===")
    record = build_output_record(
        patient_id="P0001",
        ehr_proba=float(ehr_proba[0]),
        imaging_proba=float(imaging_proba[0]),
        fused_proba=float(fused_proba[0]),
    )
    print(json.dumps(record, indent=2))

    with open("fusion_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
