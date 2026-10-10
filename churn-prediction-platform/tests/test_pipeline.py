import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.analytics import create_database, load_queries, run_query
from src.config import FEATURES
from src.preprocess import build_preprocessor, load_and_clean, prepare_features, split_data, validate_input
from src.predict import predict_df, risk_level


@pytest.fixture(scope="module")
def df():
    return load_and_clean()


def test_cleaning(df):
    assert df.shape[0] == 7043
    assert df["TotalCharges"].isnull().sum() == 0
    assert set(df["Churn"].unique()) == {0, 1}
    assert "customerID" not in df.columns


def test_split_is_stratified(df):
    _, _, y_train, y_test = split_data(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_preprocessor_output(df):
    X = df[FEATURES]
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == len(X) and Xt.shape[1] > len(FEATURES)


def test_validate_input(df):
    assert validate_input(df.drop(columns="Churn")) == []
    assert "Missing" in validate_input(df.drop(columns=["tenure"]))[0]


def test_model_predictions(df):
    from src.predict import load_or_train
    model, metrics = load_or_train()
    out = predict_df(model, df.head(50), metrics["threshold"])
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["risk_level"]) <= {"Low", "Medium", "High"}
    assert metrics["test_metrics"]["roc_auc"] > 0.8


def test_risk_level():
    assert risk_level(0.9, 0.4) == "High"
    assert risk_level(0.3, 0.4) == "Medium"
    assert risk_level(0.05, 0.4) == "Low"


def test_sql_layer(df):
    conn = create_database(":memory:", load_and_clean(keep_id=True))
    for sql in load_queries().values():
        assert len(run_query(conn, sql)) > 0
    with pytest.raises(ValueError):
        run_query(conn, "DELETE FROM customers")
