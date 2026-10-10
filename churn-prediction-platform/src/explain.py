"""Model explainability: SHAP (preferred) with a permutation-importance fallback.

Run from the project root:   python -m src.explain
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from src.config import CATEGORICAL, FIGURES_DIR, MODEL_PATH, RANDOM_STATE, TEST_DATA_PATH


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def pretty_name(name: str) -> str:
    """'cat__Contract_Month-to-month' -> 'Contract = Month-to-month'; 'num__tenure' -> 'tenure'."""
    name = name.split("__", 1)[-1]
    for col in CATEGORICAL:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name


def feature_names(pipe):
    return [pretty_name(n) for n in pipe.named_steps["prep"].get_feature_names_out()]


def _to_2d(values):
    """Normalise SHAP output to (n_samples, n_features) for the churn class."""
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def build_explainer(pipe, X_background):
    import shap
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["model"]
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, prep.transform(X_background))
    return shap.TreeExplainer(clf)


def shap_values(pipe, explainer, X):
    """Return (shap_values 2-D, transformed X) for the churn class."""
    Xt = pipe.named_steps["prep"].transform(X)
    return _to_2d(explainer.shap_values(Xt)), Xt


def explain_row(pipe, explainer, X_row: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top feature contributions (log-odds scale) for ONE customer."""
    vals, _ = shap_values(pipe, explainer, X_row)
    df = pd.DataFrame({"feature": feature_names(pipe), "contribution": vals[0]})
    df["abs"] = df["contribution"].abs()
    return df.sort_values("abs", ascending=False).head(top_n).drop(columns="abs").reset_index(drop=True)


def permutation_table(pipe, X, y, n_repeats=5) -> pd.DataFrame:
    """Model-agnostic importance per ORIGINAL column (drop in ROC-AUC when shuffled)."""
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pipe = joblib.load(MODEL_PATH)
    X_test, y_test = joblib.load(TEST_DATA_PATH)

    # Permutation importance (always available)
    perm = permutation_table(pipe, X_test, y_test)
    top = perm.head(12).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance"], xerr=top["std"], color="#E4572E")
    plt.xlabel("Drop in ROC-AUC when feature is shuffled")
    plt.title("Permutation importance (test set)")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "permutation_importance.png", dpi=150); plt.close()
    print("Saved permutation_importance.png")

    if not shap_available():
        print("SHAP is not installed -> skipped SHAP plots (pip install shap)")
        return

    import shap
    sample = X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)
    explainer = build_explainer(pipe, X_test)
    vals, Xt = shap_values(pipe, explainer, sample)
    names = feature_names(pipe)

    shap.summary_plot(vals, Xt, feature_names=names, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

    shap.summary_plot(vals, Xt, feature_names=names, plot_type="bar", show=False, max_display=15)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()

    base = explainer.expected_value
    base = float(np.ravel(base)[-1]) if np.ndim(base) else float(base)
    exp = shap.Explanation(values=vals[0], base_values=base, data=Xt[0], feature_names=names)
    shap.plots.waterfall(exp, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "shap_waterfall.png", dpi=150, bbox_inches="tight"); plt.close()
    print("Saved SHAP summary, bar and waterfall plots to", FIGURES_DIR)


if __name__ == "__main__":
    main()
