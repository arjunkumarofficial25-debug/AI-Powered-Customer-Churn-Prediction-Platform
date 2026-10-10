"""Customer Churn Prediction Platform - Streamlit app.

Run from the project root:   streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (add_tenure_group, churn_rate_by, create_database,
                           kpis, load_queries, run_query)
from src.config import CATEGORICAL, DATA_PATH, FEATURES, FIGURES_DIR
from src.explain import (build_explainer, explain_row, feature_names,
                         permutation_table, shap_available, shap_values)
from src.predict import load_or_train, predict_df, recommendations, risk_level
from src.preprocess import load_and_clean, prepare_features, validate_input

st.set_page_config(page_title="Churn Prediction Platform", page_icon="📉", layout="wide")

RISK_COLORS = {"High": "#D64545", "Medium": "#F0A202", "Low": "#2E9E6B"}


# ----------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading model (retrains automatically if needed)...")
def get_model():
    return load_or_train()


@st.cache_data(show_spinner=False)
def get_data():
    return load_and_clean(keep_id=True)


@st.cache_resource(show_spinner=False)
def get_sql_connection():
    return create_database(":memory:", get_data())


@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def get_explainer(_model):
    data = get_data()
    return build_explainer(_model, data[FEATURES].sample(300, random_state=42))


@st.cache_data(show_spinner="Computing global explanations...")
def get_global_shap():
    model, _ = get_model()
    data = get_data()
    sample = data[FEATURES].sample(400, random_state=42)
    explainer = get_explainer(model)
    vals, Xt = shap_values(model, explainer, sample)
    return vals, Xt, feature_names(model)


model, metrics = get_model()
df = get_data()
THRESHOLD = metrics.get("threshold", 0.5)

# ----------------------------------------------------------------- sidebar
st.sidebar.title("📉 Churn Platform")
page = st.sidebar.radio("Navigate", ["Single Prediction", "Analytics Dashboard",
                                     "Batch Prediction", "Model & Explainability", "SQL Explorer"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{metrics.get('best_model', 'n/a')}**")
st.sidebar.caption(f"Decision threshold: **{THRESHOLD}**")
tm = metrics.get("test_metrics", {})
if tm:
    st.sidebar.caption(f"Test ROC-AUC: **{tm.get('roc_auc')}** | Recall: **{tm.get('recall')}**")


# ----------------------------------------------------------------- page 1
def page_single():
    st.title("Single Customer Prediction")
    st.write("Enter a customer's details to estimate their churn probability.")

    def opts(col):
        return sorted(df[col].dropna().unique().tolist())

    with st.form("customer_form"):
        st.subheader("Account")
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tenure (months)", 0, 72, 12)
        monthly = c2.number_input("Monthly charges ($)", 15.0, 130.0, 70.0, step=0.5)
        contract = c3.selectbox("Contract", opts("Contract"))
        payment = c1.selectbox("Payment method", opts("PaymentMethod"))
        paperless = c2.selectbox("Paperless billing", opts("PaperlessBilling"))
        senior = c3.selectbox("Senior citizen", ["No", "Yes"])

        st.subheader("Services")
        c1, c2, c3 = st.columns(3)
        internet = c1.selectbox("Internet service", opts("InternetService"))
        phone = c2.selectbox("Phone service", opts("PhoneService"))
        lines = c3.selectbox("Multiple lines", opts("MultipleLines"))
        security = c1.selectbox("Online security", opts("OnlineSecurity"))
        backup = c2.selectbox("Online backup", opts("OnlineBackup"))
        protection = c3.selectbox("Device protection", opts("DeviceProtection"))
        support = c1.selectbox("Tech support", opts("TechSupport"))
        tv = c2.selectbox("Streaming TV", opts("StreamingTV"))
        movies = c3.selectbox("Streaming movies", opts("StreamingMovies"))

        st.subheader("Demographics")
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", opts("gender"))
        partner = c2.selectbox("Partner", opts("Partner"))
        dependents = c3.selectbox("Dependents", opts("Dependents"))
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if not submitted:
        return

    row = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
    }
    X_new = prepare_features(pd.DataFrame([row]))
    p = float(model.predict_proba(X_new)[0, 1])
    level = risk_level(p, THRESHOLD)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{p:.1%}")
    c2.metric("Prediction", "Likely to churn" if p >= THRESHOLD else "Likely to stay")
    c3.markdown(f"**Risk level**<h2 style='color:{RISK_COLORS[level]};margin:0'>{level}</h2>",
                unsafe_allow_html=True)
    st.progress(min(max(p, 0.0), 1.0))

    left, right = st.columns(2)
    with left:
        st.subheader("Suggested retention actions")
        for tip in recommendations(row):
            st.write(f"- {tip}")
    with right:
        st.subheader("Why this prediction? (SHAP)")
        if shap_available():
            try:
                contrib = explain_row(model, get_explainer(model), X_new, top_n=10)
                contrib["direction"] = contrib["contribution"].apply(
                    lambda v: "Raises churn risk" if v > 0 else "Lowers churn risk")
                fig = px.bar(contrib.iloc[::-1], x="contribution", y="feature", orientation="h",
                             color="direction",
                             color_discrete_map={"Raises churn risk": "#D64545",
                                                 "Lowers churn risk": "#2E9E6B"})
                fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_title="SHAP value (log-odds)", yaxis_title=None,
                                  legend_title=None)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Could not compute SHAP explanation: {e}")
        else:
            st.info("Install `shap` (pip install shap) to see per-customer explanations.")


# ----------------------------------------------------------------- page 2
def page_analytics():
    st.title("Churn Analytics Dashboard")
    k = kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{k['customers']:,}")
    c2.metric("Churn rate", f"{k['churn_rate']:.1%}")
    c3.metric("Avg monthly charge", f"${k['avg_monthly']:.2f}")
    c4.metric("Monthly revenue lost", f"${k['revenue_at_risk']:,.0f}")

    d = add_tenure_group(df)
    charts = [("Contract", "Churn rate by contract"), ("tenure_group", "Churn rate by tenure (months)"),
              ("PaymentMethod", "Churn rate by payment method"),
              ("InternetService", "Churn rate by internet service")]
    cols = st.columns(2)
    for i, (col, title) in enumerate(charts):
        data = churn_rate_by(d, col)
        if col == "tenure_group":
            data = data.sort_values(col)
        fig = px.bar(data, x=col, y="churn_rate", text=data["churn_rate"].map("{:.0%}".format),
                     hover_data=["customers"], title=title, color_discrete_sequence=["#E4572E"])
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn rate", xaxis_title=None)
        cols[i % 2].plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly charges distribution")
    d["Status"] = d["Churn"].map({1: "Churned", 0: "Retained"})
    fig = px.histogram(d, x="MonthlyCharges", color="Status", barmode="overlay", opacity=0.65, nbins=40,
                       color_discrete_map={"Churned": "#D64545", "Retained": "#2E9E6B"})
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------- page 3
def page_batch():
    st.title("Batch Prediction")
    st.write("Upload a CSV with the same columns as the Telco dataset (the `Churn` column is optional).")
    sample = df.drop(columns=["Churn"]).head(25)
    st.download_button("Download a sample CSV", sample.to_csv(index=False), "sample_customers.csv", "text/csv")

    file = st.file_uploader("Upload CSV", type="csv")
    if not file:
        return
    try:
        data = pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    problems = validate_input(data)
    if problems:
        for p in problems:
            st.error(p)
        return

    result = predict_df(model, data, THRESHOLD)
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", f"{len(result):,}")
    c2.metric("Predicted to churn", f"{(result['prediction'] == 'Churn').sum():,}")
    c3.metric("High risk", f"{(result['risk_level'] == 'High').sum():,}")

    result = result.sort_values("churn_probability", ascending=False)
    st.dataframe(result, use_container_width=True, height=420)
    st.download_button("Download predictions", result.to_csv(index=False), "churn_predictions.csv", "text/csv")

    fig = px.histogram(result, x="churn_probability", nbins=30, title="Distribution of churn probabilities",
                       color_discrete_sequence=["#E4572E"])
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------- page 4
def page_model():
    st.title("Model Performance & Explainability")
    if not metrics:
        st.warning("No metrics found. Run `python -m src.train`.")
        return

    st.subheader("Test-set performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", tm.get("roc_auc"))
    c2.metric("Precision", tm.get("precision"))
    c3.metric("Recall", tm.get("recall"))
    c4.metric("F1-score", tm.get("f1"))
    c5.metric("Accuracy", tm.get("accuracy"))
    st.caption(f"Threshold {THRESHOLD} was chosen on cross-validated training predictions to balance "
               "precision and recall (missing a churner usually costs more than a false alarm).")

    st.subheader("Model comparison (5-fold CV ROC-AUC)")
    cmp_df = pd.DataFrame({"Model": list(metrics["cv_roc_auc"]), "CV ROC-AUC": list(metrics["cv_roc_auc"].values())})
    st.plotly_chart(px.bar(cmp_df, x="Model", y="CV ROC-AUC", text="CV ROC-AUC",
                           color_discrete_sequence=["#E4572E"], range_y=[0.7, 0.9]), use_container_width=True)

    st.subheader("What drives churn?")
    if shap_available():
        try:
            import matplotlib.pyplot as plt
            import shap
            vals, Xt, names = get_global_shap()
            fig = plt.figure()
            shap.summary_plot(vals, Xt, feature_names=names, show=False, max_display=15)
            st.pyplot(fig, bbox_inches="tight")
            st.caption("Each dot is a customer. Red = high feature value, blue = low. "
                       "Dots to the right push the prediction toward churn.")
            return
        except Exception as e:
            st.info(f"SHAP plot unavailable ({e}). Showing permutation importance instead.")
    import joblib
    from src.config import TEST_DATA_PATH
    try:
        X_test, y_test = joblib.load(TEST_DATA_PATH)
        perm = permutation_table(model, X_test, y_test, n_repeats=3).head(12).iloc[::-1]
        st.plotly_chart(px.bar(perm, x="importance", y="feature", orientation="h",
                               color_discrete_sequence=["#E4572E"],
                               labels={"importance": "Drop in ROC-AUC when shuffled"}), use_container_width=True)
    except Exception:
        img = FIGURES_DIR / "permutation_importance.png"
        if img.exists():
            st.image(str(img))


# ----------------------------------------------------------------- page 5
def page_sql():
    st.title("SQL Explorer")
    st.write("The customer data is loaded into a SQLite table called `customers`. "
             "Pick a ready-made query or write your own (read-only).")
    conn = get_sql_connection()
    queries = load_queries()
    choice = st.selectbox("Saved queries", ["(custom)"] + list(queries))
    default = queries.get(choice, "SELECT Contract, COUNT(*) AS n, ROUND(AVG(Churn)*100,1) AS churn_pct\n"
                                  "FROM customers\nGROUP BY Contract")
    sql = st.text_area("SQL", default, height=200, key=f"sql_{choice}")
    if st.button("Run query", type="primary"):
        try:
            result = run_query(conn, sql)
            st.dataframe(result, use_container_width=True)
            st.download_button("Download result", result.to_csv(index=False), "query_result.csv", "text/csv")
        except Exception as e:
            st.error(str(e))
    with st.expander("Table columns"):
        st.write(", ".join(df.columns))


{"Single Prediction": page_single, "Analytics Dashboard": page_analytics,
 "Batch Prediction": page_batch, "Model & Explainability": page_model,
 "SQL Explorer": page_sql}[page]()
