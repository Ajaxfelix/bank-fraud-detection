from pathlib import Path
import joblib
import pandas as pd

FEATURES = [
    "Transaction_Amount",
    "Transaction_Type",
    "Transaction_Time",
    "Location",
    "Payment_Channel",
    "Transaction_Frequency",
    "Previous_Pattern",
]

def load_model(path):
    return joblib.load(path)

def prepare_input(df):
    df = df.copy()
    df["Transaction_Time"] = pd.to_datetime(df["Transaction_Time"])
    df["Transaction_Hour"] = df["Transaction_Time"].dt.hour
    df["Transaction_DayOfWeek"] = df["Transaction_Time"].dt.dayofweek
    df["Amount_Deviation"] = (
        df["Transaction_Amount"] - df["Previous_Pattern"]
    ).abs()
    return df.drop(columns=["Transaction_Time"])

def predict_transaction(bundle, df):
    processed = prepare_input(df)
    probability = float(bundle["pipeline"].predict_proba(processed)[:, 1][0])
    threshold = float(bundle.get("threshold", 0.50))
    prediction = (
        "POTENTIALLY FRAUDULENT"
        if probability >= threshold
        else "LEGITIMATE"
    )
    return prediction, probability
