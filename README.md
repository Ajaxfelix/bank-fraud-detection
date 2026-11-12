# Machine Learning-Based Fraud Detection System for Bank Payment Transactions

A Python/Streamlit prototype for classifying bank payment transactions as LEGITIMATE or POTENTIALLY FRAUDULENT.

## Requirements
- Python 3.10+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- streamlit
- joblib
- matplotlib
- seaborn

Install:
```bash
pip install -r requirements.txt
```

Run:
```bash
streamlit run app.py
```

## Included modules
- Dataset loading and preprocessing
- Feature engineering
- Random Forest fraud model
- Model evaluation
- Transaction prediction
- Fraud alerts
- Transaction history
- Dashboard
- CSV report generation
- SQLite database
- Admin login

## Demo login
Username: `admin`
Password: `admin123`

Change the demo password before any real deployment.

## Dataset
Place a CSV dataset at `data/transactions.csv`. The application accepts common target names such as:
`Fraud_Label`, `Fraud`, `is_fraud`, `Class`, or `label`.

If no dataset is supplied, the training script can generate a synthetic demonstration dataset.
