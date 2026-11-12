import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
from model import load_model, predict_transaction
from database import init_db, save_transaction, get_transactions

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "fraud_model.joblib"
DB_PATH = BASE / "database" / "fraud_detection.db"

st.set_page_config(page_title="Bank Fraud Detection", layout="wide")
init_db(DB_PATH)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("Bank Payment Fraud Detection System")
    st.subheader("Administrator Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

if not st.session_state.logged_in:
    login()
    st.stop()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Dashboard", "Transaction Detection", "Transaction History", "Reports"]
)

if page == "Dashboard":
    st.title("Fraud Analysis Dashboard")
    df = get_transactions(DB_PATH)
    if df.empty:
        st.info("No transactions have been processed yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", len(df))
        c2.metric("Legitimate", int((df["Prediction"] == "LEGITIMATE").sum()))
        c3.metric("Potentially Fraudulent", int((df["Prediction"] == "POTENTIALLY FRAUDULENT").sum()))
        c4.metric("Alerts", int((df["Prediction"] == "POTENTIALLY FRAUDULENT").sum()))
        st.dataframe(df, use_container_width=True)

elif page == "Transaction Detection":
    st.title("Transaction Detection")
    if not MODEL_PATH.exists():
        st.warning("Model not found. Run `python train_model.py` first.")
        st.stop()

    model_bundle = load_model(MODEL_PATH)
    with st.form("transaction_form"):
        account_id = st.text_input("Account ID", "ACC-1001")
        amount = st.number_input("Transaction Amount", min_value=0.0, value=5000.0)
        tx_type = st.selectbox("Transaction Type", ["Transfer", "POS", "ATM", "Online", "Mobile"])
        location = st.text_input("Location", "Lagos")
        channel = st.selectbox("Payment Channel", ["Mobile", "POS", "ATM", "Internet Banking"])
        frequency = st.number_input("Transaction Frequency", min_value=0, value=3, step=1)
        previous_pattern = st.number_input(
            "Previous Transaction Pattern (average amount)",
            min_value=0.0, value=4500.0
        )
        submitted = st.form_submit_button("Check Transaction")

    if submitted:
        now = datetime.now()
        row = {
            "Transaction_Amount": amount,
            "Transaction_Type": tx_type,
            "Transaction_Time": now,
            "Location": location,
            "Payment_Channel": channel,
            "Transaction_Frequency": frequency,
            "Previous_Pattern": previous_pattern,
        }
        label, probability = predict_transaction(model_bundle, pd.DataFrame([row]))
        tx_id = save_transaction(
            DB_PATH, account_id, amount, tx_type, now.isoformat(),
            location, channel, frequency, previous_pattern,
            label, probability
        )
        st.success(f"Transaction #{tx_id}: {label}")
        st.metric("Fraud Probability", f"{probability:.2%}")
        if label == "POTENTIALLY FRAUDULENT":
            st.error("Fraud alert generated. The transaction should be reviewed by an authorized analyst.")
        else:
            st.info("Transaction classified as legitimate by the trained model.")

elif page == "Transaction History":
    st.title("Transaction History")
    df = get_transactions(DB_PATH)
    if df.empty:
        st.info("No transaction records available.")
    else:
        search = st.text_input("Search by Account ID")
        if search:
            df = df[df["Account_ID"].astype(str).str.contains(search, case=False, na=False)]
        st.dataframe(df, use_container_width=True)

elif page == "Reports":
    st.title("Fraud Detection Reports")
    df = get_transactions(DB_PATH)
    if df.empty:
        st.info("No data available for reporting.")
    else:
        st.write("Summary")
        st.dataframe(df["Prediction"].value_counts().rename_axis("Prediction").reset_index(name="Count"))
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Report",
            data=csv,
            file_name="fraud_detection_report.csv",
            mime="text/csv"
        )
