"""
EHR-modality baseline for the multimodal diabetes screening pipeline.

Trains a gradient-boosted tree model (XGBoost) on the Pima Indians Diabetes
Database and produces SHAP explanations, corresponding to Framework Layer B
(Model & Explainability) and the EHR branch of Fig. 1 in the paper.

Expected input: data/pima.csv with the standard Pima columns
(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age, Outcome). If you use `ucimlrepo` instead,
swap the loading block below.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    confusion_matrix, classification_report,
)
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/pima.csv"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "Outcome"]
    X = df[feature_cols]
    y = df["Outcome"]
    return X, y, feature_cols


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


def main():
    X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "sensitivity_recall": recall_score(y_test, y_pred),
        "specificity": specificity_score(y_test, y_pred),
    }
    print("=== EHR model (XGBoost on Pima) ===")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred))

    # SHAP explainability — Framework Layer B
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("ehr_shap_summary.png", dpi=150)
    print("Saved SHAP summary plot to ehr_shap_summary.png")

    model.save_model("ehr_model.json")
    with open("ehr_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
