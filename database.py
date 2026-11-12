import sqlite3
from pathlib import Path
import pandas as pd

def init_db(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS Transactions (
            Transaction_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Account_ID TEXT,
            Transaction_Amount REAL NOT NULL,
            Transaction_Type TEXT,
            Transaction_Time TEXT,
            Location TEXT,
            Payment_Channel TEXT,
            Transaction_Frequency INTEGER,
            Previous_Pattern REAL,
            Prediction TEXT,
            Fraud_Probability REAL,
            Created_At TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

def save_transaction(db_path, account_id, amount, tx_type, tx_time,
                     location, channel, frequency, previous_pattern,
                     prediction, probability):
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("""
        INSERT INTO Transactions
        (Account_ID, Transaction_Amount, Transaction_Type, Transaction_Time,
         Location, Payment_Channel, Transaction_Frequency, Previous_Pattern,
         Prediction, Fraud_Probability)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            account_id, amount, tx_type, tx_time, location, channel,
            frequency, previous_pattern, prediction, probability
        ))
        conn.commit()
        return cur.lastrowid

def get_transactions(db_path):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM Transactions ORDER BY Transaction_ID DESC", conn
        )
