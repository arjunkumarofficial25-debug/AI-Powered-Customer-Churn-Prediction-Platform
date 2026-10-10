"""Train, compare, tune and save the churn model.

Run from the project root:   python -m src.train
"""
import json
import time

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline

from src.config import (METRICS_PATH, MODEL_PATH, RANDOM_STATE, TEST_DATA_PATH)
from src.preprocess import build_preprocessor, load_and_clean, split_data

try:  # XGBoost is optional
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def get_candidates(y_train):
    """Candidate models + hyper-parameter search spaces."""
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [200, 400],
             "model__max_depth": [6, 8, 12, None],
             "model__min_samples_leaf": [2, 5, 10]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 200, 300],
             "model__learning_rate": [0.03, 0.05, 0.1],
             "model__max_depth": [2, 3, 4],
             "model__subsample": [0.8, 1.0]},
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                          random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [100, 200, 300],
             "model__learning_rate": [0.03, 0.05, 0.1],
             "model__max_depth": [2, 3, 4]},
        )
    return models


def best_threshold(y_true, proba, lo=0.10, hi=0.70):
    """Threshold that maximises F1 (computed on out-of-fold predictions)."""
    grid = np.arange(lo, hi + 1e-9, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
    }


def train_and_save(verbose=True, n_iter=12, save=True):
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()

    df = load_and_clean()
    X_train, X_test, y_train, y_test = split_data(df)
    log(f"Data: {len(df)} rows | train {len(X_train)} | test {len(X_test)} | "
        f"churn rate {y_train.mean():.1%}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = get_candidates(y_train)

    # 1) Compare models with 5-fold stratified CV (ROC-AUC)
    cv_auc = {}
    for name, (est, _) in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_auc[name] = round(float(scores.mean()), 4)
        log(f"  CV ROC-AUC {name:<20} {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_auc, key=cv_auc.get)
    log(f"Best model: {best_name}")

    # 2) Tune the winner
    est, space = candidates[best_name]
    pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
    search = RandomizedSearchCV(pipe, space, n_iter=min(n_iter, int(np.prod([len(v) for v in space.values()]))),
                                cv=cv, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    log(f"Tuned CV ROC-AUC: {search.best_score_:.4f} | params: {search.best_params_}")

    # 3) Choose the decision threshold on out-of-fold training predictions (no test leakage)
    oof = cross_val_predict(clone(best), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof)
    log(f"Chosen decision threshold: {threshold}")

    # 4) Final evaluation on the untouched test set
    proba = best.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at(y_test, proba, threshold)
    default_metrics = metrics_at(y_test, proba, 0.5)
    log(f"Test metrics @ {threshold}: {test_metrics}")

    result = {
        "best_model": best_name,
        "best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
        "threshold": threshold,
        "cv_roc_auc": cv_auc,
        "tuned_cv_roc_auc": round(float(search.best_score_), 4),
        "test_metrics": test_metrics,
        "test_metrics_at_0.5": default_metrics,
        "sklearn_version": sklearn.__version__,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    if save:
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best, MODEL_PATH, compress=3)
            joblib.dump((X_test, y_test), TEST_DATA_PATH, compress=3)
            METRICS_PATH.write_text(json.dumps(result, indent=2))
            log(f"Saved model -> {MODEL_PATH}")
        except OSError as e:  # e.g. read-only file system on some hosts
            log(f"Could not save artifacts: {e}")

    log(f"Done in {time.time() - t0:.0f}s")
    return {"model": best, "metrics": result}


if __name__ == "__main__":
    train_and_save()
