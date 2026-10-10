"""Create data/churn.db (SQLite) from the raw CSV.   Run: python -m src.load_db"""
from src.analytics import create_database
from src.config import DB_PATH

if __name__ == "__main__":
    conn = create_database(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    print(f"Created {DB_PATH} with {n} rows in table 'customers'")
