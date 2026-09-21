"""
orchestrator.py — the single system described in the paper's Fig. 1.

Takes whatever patient data is actually available (EHR fields, a retinal
image, wearable data — any subset), detects which modalities are present,
routes each to its own model, and fuses only the results that exist into
one risk score and explanation.

IMPORTANT — DEMO MODE:
If a trained model file isn't found (ehr_model.json from ehr_model.py, or
imaging_model.pt from imaging_model.py), this falls back to a clearly
labeled placeholder heuristic so the system is runnable end-to-end before
you've trained anything. Every output says explicitly whether a real
trained model or a placeholder produced that modality's score. Never
present demo-mode output as a validated clinical result — swap in your
real trained models before using this for anything beyond a UI walkthrough.
"""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np

EHR_MODEL_PATH = "ehr_model.json"
IMAGING_MODEL_PATH = "imaging_model.pt"


@dataclass
class PatientInput:
    ehr_data: Optional[Dict[str, float]] = None      # e.g. {"Glucose": 148, "BMI": 33.6, ...}
    image_path: Optional[str] = None                  # path to a retinal image
    wearable_data: Optional[Dict[str, Any]] = None     # e.g. {"avg_heart_rate": 78, "steps": 4200}


@dataclass
class ModalityResult:
    modality: str
    probability: float           # 0.0-1.0 risk probability from this modality alone
    explanation: str
    source: str                  # "trained_model" or "demo_placeholder"


def detect_modalities(patient: PatientInput):
    """Return the list of modalities actually present for this patient."""
    present = []
    if patient.ehr_data:
        present.append("ehr")
    if patient.image_path and os.path.exists(patient.image_path):
        present.append("imaging")
    if patient.wearable_data:
        present.append("wearable")
    return present


def run_ehr_model(ehr_data: Dict[str, float]) -> ModalityResult:
    if os.path.exists(EHR_MODEL_PATH):
        import xgboost as xgb
        import pandas as pd
        model = xgb.XGBClassifier()
        model.load_model(EHR_MODEL_PATH)
        X = pd.DataFrame([ehr_data])
        proba = float(model.predict_proba(X)[0, 1])
        # Real SHAP explanation would be generated here against the loaded model.
        explanation = "Risk driven by EHR risk factors (see SHAP output for detail)."
        return ModalityResult("ehr", proba, explanation, "trained_model")

    # --- DEMO PLACEHOLDER: simple, transparent heuristic, NOT a validated model ---
    glucose = ehr_data.get("Glucose", 100)
    bmi = ehr_data.get("BMI", 25)
    age = ehr_data.get("Age", 30)
    score = (
        0.5 * min(max((glucose - 90) / 120, 0), 1)
        + 0.3 * min(max((bmi - 18) / 22, 0), 1)
        + 0.2 * min(max((age - 20) / 50, 0), 1)
    )
    return ModalityResult(
        "ehr", float(np.clip(score, 0, 1)),
        "[DEMO PLACEHOLDER] Heuristic score from glucose, BMI, and age only — not a trained model.",
        "demo_placeholder",
    )


def run_imaging_model(image_path: str) -> ModalityResult:
    if os.path.exists(IMAGING_MODEL_PATH):
        import torch
        from torchvision import transforms, models
        from PIL import Image
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(IMAGING_MODEL_PATH, map_location="cpu"))
        model.eval()
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        x = tf(image).unsqueeze(0)
        with torch.no_grad():
            proba = float(torch.softmax(model(x), dim=1)[0, 1])
        explanation = "Risk driven by retinal findings (see Grad-CAM output for detail)."
        return ModalityResult("imaging", proba, explanation, "trained_model")

    # --- DEMO PLACEHOLDER: deterministic pseudo-score from image bytes, NOT a real model ---
    with open(image_path, "rb") as f:
        content_hash = sum(f.read()[:4096]) % 1000
    score = content_hash / 1000
    return ModalityResult(
        "imaging", float(score),
        "[DEMO PLACEHOLDER] Deterministic pseudo-score from image bytes — not a trained model.",
        "demo_placeholder",
    )


def run_wearable_model(wearable_data: Dict[str, Any]) -> ModalityResult:
    # Per the paper (Section VI-A), the wearable modality is proposed but not
    # yet validated. We still surface a transparent placeholder rather than
    # silently ignoring the data the patient provided.
    hr = wearable_data.get("avg_heart_rate", 75)
    steps = wearable_data.get("steps", 5000)
    score = 0.6 * min(max((hr - 60) / 60, 0), 1) + 0.4 * (1 - min(steps / 10000, 1))
    return ModalityResult(
        "wearable", float(np.clip(score, 0, 1)),
        "[DEMO PLACEHOLDER — not yet a validated modality per the paper] Heuristic from heart rate and activity.",
        "demo_placeholder",
    )


MODEL_RUNNERS = {
    "ehr": lambda p: run_ehr_model(p.ehr_data),
    "imaging": lambda p: run_imaging_model(p.image_path),
    "wearable": lambda p: run_wearable_model(p.wearable_data),
}

# Relative weight given to each modality when fusing — renormalized over
# whichever modalities are actually present for a given patient.
BASE_WEIGHTS = {"ehr": 0.4, "imaging": 0.45, "wearable": 0.15}


def fuse(results: Dict[str, ModalityResult]) -> float:
    present = list(results.keys())
    total_weight = sum(BASE_WEIGHTS[m] for m in present)
    fused = sum(BASE_WEIGHTS[m] / total_weight * results[m].probability for m in present)
    return fused


def risk_category(score_0_100: float):
    if score_0_100 < 33:
        return "low", "routine follow-up"
    elif score_0_100 < 66:
        return "moderate", "specialist referral"
    else:
        return "high", "urgent referral"


def run_screening(patient: PatientInput, patient_id: str = "unknown") -> Dict[str, Any]:
    """The single entry point: detect modalities, route, run, fuse, return output."""
    present = detect_modalities(patient)
    if not present:
        raise ValueError("No usable patient data provided — need at least one of EHR, image, or wearable data.")

    results: Dict[str, ModalityResult] = {}
    for modality in present:
        results[modality] = MODEL_RUNNERS[modality](patient)

    fused_proba = fuse(results)
    score = round(fused_proba * 100, 1)
    category, action = risk_category(score)

    modality_contributions = {
        "retinal_imaging": round(results["imaging"].probability, 3) if "imaging" in results else None,
        "ehr_risk_factors": round(results["ehr"].probability, 3) if "ehr" in results else None,
        "wearable_trends": round(results["wearable"].probability, 3) if "wearable" in results else None,
    }
    any_demo = any(r.source == "demo_placeholder" for r in results.values())

    explanation_parts = [r.explanation for r in results.values()]
    return {
        "encounter_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "modalities_used": present,
        "risk_score": score,
        "risk_category": category,
        "modality_contributions": modality_contributions,
        "explanation_summary": " | ".join(explanation_parts),
        "recommended_action": action,
        "demo_mode": any_demo,
        "model_version": "v0.1-orchestrator",
        "audit_log_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    # Quick self-test with EHR-only data (no image, no trained models present)
    patient = PatientInput(ehr_data={"Glucose": 148, "BMI": 33.6, "Age": 45})
    import json
    print(json.dumps(run_screening(patient, patient_id="TEST001"), indent=2))
