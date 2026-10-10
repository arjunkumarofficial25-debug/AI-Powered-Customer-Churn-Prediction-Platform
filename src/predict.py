"""Loading the model and making predictions (used by the Streamlit app)."""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import METRICS_PATH, MODEL_PATH
from src.preprocess import prepare_features


def load_metrics() -> dict:
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return {}


def load_or_train():
    """Load the saved model. If it is missing or was built with a different
    scikit-learn version, retrain automatically (about 1-2 minutes)."""
    metrics = load_metrics()
    same_version = metrics.get("sklearn_version") == sklearn.__version__
    if MODEL_PATH.exists() and same_version:
        try:
            return joblib.load(MODEL_PATH), metrics
        except Exception:
            pass
    from src.train import train_and_save
    out = train_and_save(verbose=False)
    return out["model"], out["metrics"]


def risk_level(p: float, threshold: float) -> str:
    """High = above the decision threshold, Medium = within 60% of it, else Low."""
    if p >= threshold:
        return "High"
    if p >= 0.6 * threshold:
        return "Medium"
    return "Low"


def predict_df(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add churn_probability, prediction and risk_level columns to df."""
    proba = model.predict_proba(prepare_features(df))[:, 1]
    out = df.copy()
    out["churn_probability"] = np.round(proba, 4)
    out["prediction"] = np.where(proba >= threshold, "Churn", "Stay")
    out["risk_level"] = [risk_level(p, threshold) for p in proba]
    return out


def recommendations(row: dict) -> list:
    """Simple rule-based retention ideas for one customer."""
    tips = []
    if row.get("Contract") == "Month-to-month":
        tips.append("Offer a discount to move to a 1- or 2-year contract.")
    if row.get("tenure", 99) < 12:
        tips.append("New customer: trigger an onboarding call and early-life engagement offers.")
    if row.get("PaymentMethod") == "Electronic check":
        tips.append("Encourage automatic payment (card / bank transfer) with a small credit.")
    if row.get("InternetService") == "Fiber optic" and row.get("TechSupport") == "No":
        tips.append("Offer a free tech-support trial for fiber customers.")
    if row.get("OnlineSecurity") == "No" and row.get("InternetService") != "No":
        tips.append("Bundle online security / backup at a promotional price.")
    if row.get("MonthlyCharges", 0) > 80:
        tips.append("High bill: review the plan and propose a cheaper bundle.")
    return tips or ["No specific risk drivers found: keep the customer engaged as usual."]
