"""Central configuration: paths, column groups and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_PATH = ROOT / "data" / "churn.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
TEST_DATA_PATH = ROOT / "models" / "test_data.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL

TEST_SIZE = 0.2
RANDOM_STATE = 42
