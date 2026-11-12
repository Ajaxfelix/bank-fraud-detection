from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "transactions.csv"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"

def create_demo_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    amount = np.round(rng.lognormal(8.0, 1.0, n), 2)
    frequency = rng.poisson(4, n)
    previous = np.round(amount * rng.uniform(0.5, 1.5, n), 2)
    tx_type = rng.choice(["Transfer", "POS", "ATM", "Online", "Mobile"], n)
    channel = rng.choice(["Mobile", "POS", "ATM", "Internet Banking"], n)
    location = rng.choice(["Lagos", "Ibadan", "Abuja", "Port Harcourt", "Kano"], n)
    hour = rng.integers(0, 24, n)
    score = (
        (amount > np.quantile(amount, .92)).astype(int)
        + (frequency > 8).astype(int)
        + ((hour < 5) | (hour > 23)).astype(int)
        + (np.abs(amount - previous) > np.quantile(np.abs(amount - previous), .85)).astype(int)
        + (channel == "Internet Banking").astype(int)
    )
    fraud = (score >= 3).astype(int)
    times = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 180*24*3600, n), unit="s"
    )
    return pd.DataFrame({
        "Transaction_Amount": amount,
        "Transaction_Type": tx_type,
        "Transaction_Time": times,
        "Location": location,
        "Payment_Channel": channel,
        "Transaction_Frequency": frequency,
        "Previous_Pattern": previous,
        "Fraud_Label": fraud
    })

def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = create_demo_data()
        df.to_csv(DATA_PATH, index=False)
    return df

def main():
    df = load_data()
    target_candidates = ["Fraud_Label", "Fraud", "is_fraud", "Class", "label"]
    target = next((c for c in target_candidates if c in df.columns), None)
    if target is None:
        raise ValueError(
            "No fraud target column found. Use one of: " + ", ".join(target_candidates)
        )

    # Standardize target to 0/1 where possible.
    y = df[target]
    if y.dtype == object:
        mapping = {
            "fraud": 1, "fraudulent": 1, "true": 1, "1": 1,
            "legitimate": 0, "legit": 0, "false": 0, "0": 0
        }
        y = y.astype(str).str.strip().str.lower().map(mapping)
    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    required = [
        "Transaction_Amount", "Transaction_Type", "Transaction_Time",
        "Location", "Payment_Channel", "Transaction_Frequency",
        "Previous_Pattern"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[required].copy()
    X["Transaction_Time"] = pd.to_datetime(X["Transaction_Time"], errors="coerce")
    X["Transaction_Hour"] = X["Transaction_Time"].dt.hour.fillna(0)
    X["Transaction_DayOfWeek"] = X["Transaction_Time"].dt.dayofweek.fillna(0)
    X["Amount_Deviation"] = (
        X["Transaction_Amount"] - X["Previous_Pattern"]
    ).abs()
    X = X.drop(columns=["Transaction_Time"])

    numeric = [
        "Transaction_Amount", "Transaction_Frequency",
        "Previous_Pattern", "Transaction_Hour",
        "Transaction_DayOfWeek", "Amount_Deviation"
    ]
    categorical = ["Transaction_Type", "Location", "Payment_Channel"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    print("\nMODEL PERFORMANCE")
    print("-----------------")
    print("Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 4))
    print("F1-score :", round(f1_score(y_test, pred, zero_division=0), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, prob), 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification Report:")
    print(classification_report(y_test, pred, zero_division=0))

    bundle = {
        "pipeline": pipeline,
        "threshold": 0.50,
        "features": required,
        "metrics": {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob)
        }
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
