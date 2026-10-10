"""Churn analytics helpers and the SQL layer (SQLite)."""
import re
import sqlite3

import pandas as pd

from src.config import DB_PATH, SQL_PATH
from src.preprocess import load_and_clean


def kpis(df: pd.DataFrame) -> dict:
    churned = df[df["Churn"] == 1]
    return {
        "customers": len(df),
        "churn_rate": df["Churn"].mean(),
        "avg_monthly": df["MonthlyCharges"].mean(),
        "revenue_at_risk": churned["MonthlyCharges"].sum(),
    }


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.groupby(col)["Churn"].agg(["mean", "count"]).reset_index()
    out.columns = [col, "churn_rate", "customers"]
    return out.sort_values("churn_rate", ascending=False)


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_group"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 100],
                                labels=["0-12", "13-24", "25-48", "49+"]).astype(str)
    return df


# ---------------------------------------------------------------- SQL layer
def create_database(path=DB_PATH, df: pd.DataFrame = None) -> sqlite3.Connection:
    """Create (or overwrite) the SQLite database file. path=':memory:' for in-memory."""
    df = load_and_clean(keep_id=True) if df is None else df
    conn = sqlite3.connect(str(path), check_same_thread=False)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    return conn


def load_queries(path=SQL_PATH) -> dict:
    """Parse sql/queries.sql into {title: query}."""
    text = path.read_text()
    queries = {}
    for block in re.split(r"^-- name:", text, flags=re.M)[1:]:
        title, _, body = block.partition("\n")
        queries[title.strip()] = body.strip().rstrip(";").strip()
    return queries


def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Run a read-only query. Only SELECT / WITH statements are allowed."""
    stripped = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", stripped, flags=re.I) or ";" in stripped:
        raise ValueError("Only a single SELECT query is allowed.")
    return pd.read_sql_query(stripped, conn)
