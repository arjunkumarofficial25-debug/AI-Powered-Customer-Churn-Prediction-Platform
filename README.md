# 📉 AI-Powered Customer Churn Prediction Platform

An end-to-end machine learning project that predicts which telecom customers are likely to leave, explains **why**, and lets business users act on it through an interactive Streamlit app.

**Live demo:** _add your Streamlit Cloud URL here_

## Problem
Acquiring a new customer costs far more than keeping an existing one. Using the IBM Telco Customer Churn dataset (7,043 customers, 21 columns; ~26.5% churn), this project predicts churn risk per customer, highlights the drivers, and suggests retention actions.

## Tech stack
Python · Pandas · NumPy · Scikit-learn · SHAP · Streamlit · Plotly · SQL (SQLite) · Git/GitHub

## Features
| Page | What it does |
|---|---|
| **Single Prediction** | Form for one customer → churn probability, risk level, SHAP explanation, retention suggestions |
| **Analytics Dashboard** | KPIs and churn rate by contract, tenure, payment method, internet service |
| **Batch Prediction** | Upload a CSV → scored table, download results; validates columns and empty values |
| **Model & Explainability** | Test metrics, model comparison, global SHAP summary |
| **SQL Explorer** | 8 ready-made business queries plus a read-only custom SQL box on SQLite |

## Approach
1. **EDA** (`notebooks/01_eda.ipynb`): cleaned `TotalCharges` blanks, checked class balance, found churn drivers.
2. **Preprocessing** (`src/preprocess.py`): `ColumnTransformer` (StandardScaler + OneHotEncoder) inside a `Pipeline`, so the app reuses the exact training transforms and there is no leakage.
3. **Modeling** (`src/train.py`): Logistic Regression, Random Forest, Gradient Boosting (+ XGBoost if installed), compared with 5-fold stratified CV on ROC-AUC, then tuned with `RandomizedSearchCV`.
4. **Threshold selection**: chosen on out-of-fold training predictions (not the test set) to maximise F1. Missing a churner usually costs more than a false alarm.
5. **Evaluation** (`src/evaluate.py`): precision, recall, F1, ROC-AUC, confusion matrix, PR curve.
6. **Explainability** (`src/explain.py`): SHAP globally and per customer.

## Results (held-out test set, 1,409 customers)
| Model | 5-fold CV ROC-AUC |
|---|---|
| Logistic Regression | 0.846 |
| Random Forest | 0.822 |
| **Gradient Boosting (tuned: 0.851)** | **0.848** |

Final model: Gradient Boosting at threshold **0.32**

| Metric | Threshold 0.32 (chosen) | Threshold 0.50 (default) |
|---|---|---|
| ROC-AUC | 0.846 | 0.846 |
| Recall | **0.741** | 0.516 |
| Precision | 0.555 | 0.675 |
| F1-score | **0.635** | 0.585 |
| Accuracy | 0.774 | 0.806 |

Lowering the threshold catches ~74% of churners (vs ~52% at 0.5) at the cost of more false alarms, a sensible trade for retention campaigns.

## Key insights
- Month-to-month customers churn at **42.7%**, versus 11.3% (1-year) and 2.8% (2-year).
- Customers in their first 12 months churn at **47.4%**, versus 17.1% afterwards.
- Electronic-check payers (45.3%) and fiber-optic customers (41.9%) churn most.
- Customers without tech support churn at 41.6% versus 15.2% with it.
- Churned customers pay more on average ($74.44 vs $61.27 per month).

_Add the SHAP plots (`reports/figures/shap_summary.png`) and app screenshots here._

## Project structure
```
churn-prediction-platform/
├── app/app.py              # Streamlit app (5 pages)
├── data/raw/               # Telco CSV
├── models/                 # trained model, metrics.json, test split
├── notebooks/01_eda.ipynb
├── reports/figures/        # evaluation + explainability plots
├── sql/queries.sql         # business queries (SQLite)
├── src/                    # config, preprocess, train, evaluate, explain, predict, analytics, load_db
├── tests/test_pipeline.py
├── requirements.txt        # runtime deps (Streamlit Cloud)
└── requirements-dev.txt    # + jupyter, pytest, xgboost
```

## Run locally
```bash
git clone https://github.com/<your-username>/churn-prediction-platform.git
cd churn-prediction-platform

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

python -m pip install --upgrade pip
pip install -r requirements-dev.txt

python -m src.train            # (optional) retrain, ~2 min
python -m src.evaluate         # metrics + plots
python -m src.explain          # SHAP plots
streamlit run app/app.py
```
Run all commands from the project root. Run tests with `python -m pytest`.

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub (the model and CSV are small enough to commit).
2. Go to share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `app/app.py`.
3. Deploy, then paste the URL at the top of this README.

If the cloud scikit-learn version differs from the one used for training, the app detects it and retrains automatically on startup (about 1-2 minutes).

## Future improvements
Retention ROI calculator (offer cost vs. customer lifetime value) · FastAPI endpoint · model monitoring and drift detection · probability calibration · Dockerfile.

_Dataset: IBM Telco Customer Churn sample data._
