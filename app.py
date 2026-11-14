from pathlib import Path
from datetime import datetime
import os
import pandas as pd
import streamlit as st

from model import load_model, predict_transaction
from database import init_db, save_transaction, get_transactions
from train_model import main as train_model, create_demo_data

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "fraud_model.joblib"
DB_PATH = BASE / "database" / "fraud_detection.db"

st.set_page_config(page_title="Bank Fraud Detection", page_icon="🛡️", layout="wide")

# A small amount of styling to make the existing Streamlit app easier to navigate.
st.markdown("""
<style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
    .hero {padding: 1.25rem 1.4rem; border-radius: 14px; background: linear-gradient(110deg,#10243a,#174b67); color: white; margin-bottom: 1rem;}
    .hero h1 {color: white; margin: 0; font-size: 2rem;}
    .hero p {color: #dceaf4; margin: .35rem 0 0 0;}
    div[data-testid="stMetric"] {background: #f4f7fb; padding: 14px; border-radius: 12px; border: 1px solid #e4eaf1;}
    .small-note {color: #667085; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

init_db(DB_PATH)

# Streamlit Community Cloud starts from a clean deployment. Train a demo model
# automatically if the model artifact is not present in the repository/runtime.
if not MODEL_PATH.exists():
    with st.spinner("Preparing the fraud-detection model for the first run..."):
        train_model()

try:
    MODEL_BUNDLE = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"The machine-learning model could not be loaded: {exc}")
    st.stop()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def login():
    st.markdown('<div class="hero"><h1>🛡️ Bank Payment Fraud Detection</h1><p>Administrator access to transaction monitoring and fraud analysis.</p></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Administrator Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            # Demo credentials retained from the original project. Configure
            # ADMIN_USERNAME and ADMIN_PASSWORD as Streamlit secrets for deployment.
            expected_user = st.secrets.get("ADMIN_USERNAME", os.getenv("ADMIN_USERNAME", "admin"))
            expected_password = st.secrets.get("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", "admin123"))
            if username == expected_user and password == expected_password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.caption("Demo login: admin / admin123. Change these credentials before using real or sensitive data.")


if not st.session_state.logged_in:
    login()
    st.stop()

with st.sidebar:
    st.title("🛡️ Fraud Monitor")
    st.caption("Bank Payment Fraud Detection")
    page = st.radio("Navigation", ["Dashboard", "Transaction Detection", "Transaction History", "Reports", "Model Performance"])
    st.divider()
    st.caption("Model: Random Forest")
    if st.button("Log out", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()


def get_summary(df):
    total = len(df)
    fraud = int((df["Prediction"] == "POTENTIALLY FRAUDULENT").sum()) if total else 0
    legit = total - fraud
    rate = (fraud / total * 100) if total else 0
    return total, legit, fraud, rate


def add_demo_transactions(count=30):
    demo = create_demo_data(n=count, seed=int(datetime.now().timestamp()) % 100000)
    for _, row in demo.iterrows():
        tx_time = pd.to_datetime(row["Transaction_Time"])
        record = pd.DataFrame([{
            "Transaction_Amount": float(row["Transaction_Amount"]),
            "Transaction_Type": row["Transaction_Type"],
            "Transaction_Time": tx_time,
            "Location": row["Location"],
            "Payment_Channel": row["Payment_Channel"],
            "Transaction_Frequency": int(row["Transaction_Frequency"]),
            "Previous_Pattern": float(row["Previous_Pattern"]),
        }])
        label, probability = predict_transaction(MODEL_BUNDLE, record)
        save_transaction(DB_PATH, f"DEMO-{int(datetime.now().timestamp())}-{_}",
                         float(row["Transaction_Amount"]), row["Transaction_Type"],
                         tx_time.isoformat(), row["Location"], row["Payment_Channel"],
                         int(row["Transaction_Frequency"]), float(row["Previous_Pattern"]),
                         label, probability)


if page == "Dashboard":
    st.markdown('<div class="hero"><h1>Fraud Analysis Dashboard</h1><p>Overview of transactions analysed by the machine-learning model.</p></div>', unsafe_allow_html=True)
    df = get_transactions(DB_PATH)
    total, legit, fraud, fraud_rate = get_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", f"{total:,}")
    c2.metric("Legitimate", f"{legit:,}")
    c3.metric("Potential fraud", f"{fraud:,}")
    c4.metric("Fraud alert rate", f"{fraud_rate:.1f}%")

    if df.empty:
        st.info("No transactions have been analysed yet. Use Transaction Detection to analyse a transaction, or load clearly labelled sample data for a project demonstration.")
        if st.button("Generate 30 sample transactions", type="primary"):
            add_demo_transactions(30)
            st.success("30 synthetic demonstration transactions have been added. They are not real bank records.")
            st.rerun()
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Transaction classification")
            counts = df["Prediction"].value_counts().rename_axis("Classification").reset_index(name="Transactions")
            st.bar_chart(counts.set_index("Classification"))
        with right:
            st.subheader("Recent fraud probabilities")
            recent = df.head(12).copy().sort_values("Transaction_ID")
            st.bar_chart(recent.set_index("Transaction_ID")["Fraud_Probability"])
        st.subheader("Latest transactions")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        if st.button("Add 30 more sample transactions"):
            add_demo_transactions(30)
            st.success("Added 30 synthetic demonstration transactions.")
            st.rerun()
    st.caption("Note: synthetic sample records are for demonstration only. Model predictions are not proof that a real transaction is fraudulent.")

elif page == "Transaction Detection":
    st.markdown('<div class="hero"><h1>Transaction Detection</h1><p>Enter transaction information and review the model assessment.</p></div>', unsafe_allow_html=True)
    with st.form("transaction_form"):
        c1, c2 = st.columns(2)
        with c1:
            account_id = st.text_input("Account ID", "ACC-1001")
            amount = st.number_input("Transaction amount", min_value=0.0, value=5000.0, step=500.0)
            tx_type = st.selectbox("Transaction type", ["Transfer", "POS", "ATM", "Online", "Mobile"])
            location = st.selectbox("Transaction location", ["Lagos", "Ibadan", "Abuja", "Port Harcourt", "Kano", "Other"])
        with c2:
            channel = st.selectbox("Payment channel", ["Mobile", "POS", "ATM", "Internet Banking"])
            frequency = st.number_input("Number of recent transactions", min_value=0, value=3, step=1)
            previous_pattern = st.number_input("Usual transaction amount", min_value=0.0, value=4500.0, step=500.0)
        submitted = st.form_submit_button("Analyse transaction", type="primary", use_container_width=True)

    if submitted:
        if not account_id.strip():
            st.error("Please enter an account ID.")
        else:
            now = datetime.now()
            row = pd.DataFrame([{
                "Transaction_Amount": amount, "Transaction_Type": tx_type,
                "Transaction_Time": now, "Location": location,
                "Payment_Channel": channel, "Transaction_Frequency": frequency,
                "Previous_Pattern": previous_pattern,
            }])
            label, probability = predict_transaction(MODEL_BUNDLE, row)
            tx_id = save_transaction(DB_PATH, account_id.strip(), amount, tx_type,
                                     now.isoformat(), location, channel, frequency,
                                     previous_pattern, label, probability)
            st.divider()
            st.subheader(f"Analysis result — Transaction #{tx_id}")
            r1, r2 = st.columns(2)
            r1.metric("Model classification", label)
            r2.metric("Estimated fraud probability", f"{probability:.1%}")
            if label == "POTENTIALLY FRAUDULENT":
                st.error("⚠️ Fraud risk alert: send this transaction for review by an authorised analyst.")
            else:
                st.success("The model classified this transaction as legitimate.")
            st.caption("This is a machine-learning estimate based on the supplied fields and demonstration training data, not a bank-authorisation decision.")

elif page == "Transaction History":
    st.markdown('<div class="hero"><h1>Transaction History</h1><p>Search and review previously analysed transactions.</p></div>', unsafe_allow_html=True)
    df = get_transactions(DB_PATH)
    if df.empty:
        st.info("No transaction records yet. Analyse a transaction or add sample transactions from the Dashboard.")
    else:
        search = st.text_input("Search by account ID")
        classification = st.selectbox("Filter by classification", ["All", "LEGITIMATE", "POTENTIALLY FRAUDULENT"])
        if search:
            df = df[df["Account_ID"].astype(str).str.contains(search, case=False, na=False)]
        if classification != "All":
            df = df[df["Prediction"] == classification]
        st.write(f"Showing {len(df)} transaction(s).")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download transaction history (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="transaction_history.csv", mime="text/csv")

elif page == "Reports":
    st.markdown('<div class="hero"><h1>Fraud Detection Reports</h1><p>Summary and export of analysed transaction records.</p></div>', unsafe_allow_html=True)
    df = get_transactions(DB_PATH)
    if df.empty:
        st.info("No data available for reporting yet. Analyse transactions or generate sample data from the Dashboard.")
    else:
        total, legit, fraud, fraud_rate = get_summary(df)
        a, b, c = st.columns(3)
        a.metric("Transactions analysed", total)
        b.metric("Potential fraud cases", fraud)
        c.metric("Fraud rate", f"{fraud_rate:.1f}%")
        st.subheader("Classification summary")
        summary = df["Prediction"].value_counts().rename_axis("Prediction").reset_index(name="Count")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.subheader("Full report")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV report", df.to_csv(index=False).encode("utf-8"),
                           file_name="fraud_detection_report.csv", mime="text/csv", type="primary")
        st.caption("Reports contain model classifications. Confirm suspicious transactions through an authorised review process.")


elif page == "Model Performance":
    st.markdown('<div class="hero"><h1>Model Performance</h1><p>Evaluation results from the model training test set.</p></div>', unsafe_allow_html=True)
    metrics = MODEL_BUNDLE.get("metrics", {})
    if not metrics:
        st.warning("No evaluation metrics are stored with this model. Retrain the model to generate them.")
    else:
        st.info(MODEL_BUNDLE.get("evaluation_note", "Metrics describe the held-out test data used during training."))
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        m2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        m3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        m4.metric("F1-score", f"{metrics.get('f1', 0):.1%}")
        m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        st.subheader("Confusion matrix")
        cm = metrics.get("confusion_matrix")
        if cm and len(cm) == 2 and all(len(row) == 2 for row in cm):
            cm_df = pd.DataFrame(cm, index=["Actual legitimate", "Actual fraud"],
                                 columns=["Predicted legitimate", "Predicted fraud"])
            st.dataframe(cm_df, use_container_width=True)
            st.caption("Rows are actual classes; columns are model predictions. False positives are legitimate transactions flagged as fraud. False negatives are fraudulent transactions classified as legitimate.")
        else:
            st.info("Confusion matrix is not available in this saved model. Retrain the model to generate it.")
        st.write(f"Test samples: {metrics.get('test_samples', 'Not recorded')}")
        st.write(f"Fraud samples in test set: {metrics.get('fraud_test_samples', 'Not recorded')}")
        st.write(f"Legitimate samples in test set: {metrics.get('legitimate_test_samples', 'Not recorded')}")
        st.warning("Important: high scores on synthetic/demo data do not prove that the system will detect fraud accurately in real banking data. A properly labelled, representative real-world dataset and independent validation are required.")
