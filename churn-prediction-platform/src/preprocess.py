"""Data loading, cleaning and the scikit-learn preprocessing transformer."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (CATEGORICAL, DATA_PATH, FEATURES, ID_COL, NUMERIC,
                        RANDOM_STATE, TARGET, TEST_SIZE)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data quirks. Safe to call on training data AND on uploaded CSVs."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # TotalCharges contains blank strings (new customers with tenure = 0)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if {"tenure", "MonthlyCharges"} <= set(df.columns):
            df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def load_and_clean(path=DATA_PATH, keep_id: bool = False) -> pd.DataFrame:
    """Load the Telco CSV, clean it and encode the target (Yes -> 1, No -> 0)."""
    df = clean(pd.read_csv(path))
    if not keep_id:
        df = df.drop(columns=[ID_COL], errors="ignore")
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categorical columns."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split -> X_train, X_test, y_train, y_test."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def validate_input(df: pd.DataFrame) -> list:
    """Return a list of human-readable problems with an uploaded dataframe (empty = OK)."""
    problems = []
    df = clean(df)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        problems.append(f"Missing required columns: {missing}")
        return problems
    nulls = df[FEATURES].isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        problems.append(f"Empty values found in: {nulls.to_dict()}")
    return problems


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and select exactly the columns the model was trained on."""
    return clean(df)[FEATURES]
