"""Evaluate the saved model on the held-out test set and save plots.

Run from the project root:   python -m src.evaluate
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.config import FIGURES_DIR, MODEL_PATH, TEST_DATA_PATH
from src.predict import load_metrics


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    X_test, y_test = joblib.load(TEST_DATA_PATH)
    threshold = load_metrics().get("threshold", 0.5)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    print(f"Decision threshold: {threshold}\n")
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stay", "Churn"], cmap="Oranges")
    plt.title(f"Confusion matrix (threshold {threshold})")
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

    RocCurveDisplay.from_predictions(y_test, proba)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4); plt.title("ROC curve")
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight"); plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, proba)
    plt.title("Precision-Recall curve")
    plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=150, bbox_inches="tight"); plt.close()

    # Threshold sweep
    ts = np.arange(0.1, 0.81, 0.05)
    p = [precision_score(y_test, (proba >= t).astype(int), zero_division=0) for t in ts]
    r = [recall_score(y_test, (proba >= t).astype(int)) for t in ts]
    f = [f1_score(y_test, (proba >= t).astype(int)) for t in ts]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ts, p, label="Precision"); plt.plot(ts, r, label="Recall"); plt.plot(ts, f, label="F1")
    plt.axvline(threshold, color="gray", ls="--", label=f"Chosen ({threshold})")
    plt.xlabel("Threshold"); plt.legend(); plt.title("Precision / recall trade-off")
    plt.savefig(FIGURES_DIR / "threshold_sweep.png", dpi=150, bbox_inches="tight"); plt.close()
    print("\nSaved plots to", FIGURES_DIR)


if __name__ == "__main__":
    main()
