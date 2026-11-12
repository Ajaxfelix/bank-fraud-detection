import tempfile
from pathlib import Path
from database import init_db, save_transaction, get_transactions

def test_database_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.db"
        init_db(db)
        tx_id = save_transaction(
            db, "ACC-1", 1000, "POS", "2026-09-15T09:00:00",
            "Ibadan", "POS", 2, 900, "LEGITIMATE", 0.12
        )
        assert tx_id == 1
        df = get_transactions(db)
        assert len(df) == 1
        assert df.iloc[0]["Prediction"] == "LEGITIMATE"
