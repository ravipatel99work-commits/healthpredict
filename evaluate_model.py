"""
evaluate_models.py
---------------------------------
Generates Normalized Confusion Matrix, ROC Curve,
and Precision–Recall Curve for all trained models.
Saves plots into /evaluation_results folder.
"""

import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# -------------------------------
# 🔧 Define paths
# -------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"                 # your datasets folder
EVAL_DIR = ROOT / "evaluation_results"   # new folder for output plots
EVAL_DIR.mkdir(exist_ok=True)

# -------------------------------
# 🔗 Model mapping (update if names differ)
# -------------------------------
MODEL_MAP = {
    "diabetes":  {"model": "diabetes_model.pkl", "imputer": "diabetes_imputer.pkl", "features": "diabetes_features.json", "title": "Diabetes"},
    "flu":       {"model": "flu_model.pkl", "imputer": "flu_imputer.pkl", "features": "flu_features.json", "title": "Influenza (Flu)"},
    "pneumonia": {"model": "pneumonia_model.pkl", "imputer": "pneumonia_imputer.pkl", "features": "pneumonia_features.json", "title": "Pneumonia"},
    "heart":     {"model": "heart_disease_model.pkl", "imputer": "heart_disease_imputer.pkl", "features": "heart_disease_features.json", "title": "Heart Disease"},
    "kidney":    {"model": "ckd_model.pkl", "imputer": "ckd_imputer.pkl", "features": "ckd_features.json", "title": "Chronic Kidney Disease"},
}

# -------------------------------
# 🎨 Helper functions
# -------------------------------
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, normalize="true")  # normalized matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title(f"Normalized Confusion Matrix – {title}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_score, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {title}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_pr_curve(y_true, y_score, title, filename):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, color="green", lw=2, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve – {title}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# -------------------------------
# 🚀 Evaluation function
# -------------------------------
def evaluate(disease):
    print(f"\n🔍 Evaluating {disease.title()} model...")

    # Load model, imputer, and features
    files = MODEL_MAP[disease]
    model = joblib.load(MODEL_DIR / files["model"])
    imputer = joblib.load(MODEL_DIR / files["imputer"])
    with open(MODEL_DIR / files["features"]) as f:
        features = json.load(f)

    # Load dataset (must include 'target_disease' or 'label')
    data_path = DATA_DIR / f"{disease}_dataset.csv"
    if not data_path.exists():
        print(f"⚠️ Missing dataset for {disease}")
        return

    df = pd.read_csv(data_path)
    target_col = "target_disease" if "target_disease" in df.columns else "label"
    X = df[features]
    y = df[target_col].astype(int)

    # Impute missing data
    X_imp = imputer.transform(X)

    # Predictions and probabilities
    y_pred = model.predict(X_imp)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_imp)[:, 1]
    else:
        y_score = y_pred

    # Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"✅ Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    # Save plots
    plot_confusion_matrix(y, y_pred, files["title"], EVAL_DIR / f"cm_{disease}.png")
    plot_roc_curve(y, y_score, files["title"], EVAL_DIR / f"roc_{disease}.png")
    plot_pr_curve(y, y_score, files["title"], EVAL_DIR / f"pr_{disease}.png")

    print(f"🖼️ Saved plots for {disease}: cm_{disease}.png, roc_{disease}.png, pr_{disease}.png")

# -------------------------------
# 🔁 Run for all trained models
# -------------------------------
if __name__ == "__main__":
    for d in MODEL_MAP.keys():
        evaluate(d)
    print(f"\n🎉 All evaluation plots saved in: {EVAL_DIR.resolve()}")
