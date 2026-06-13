import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "inventory.db"
MODEL_DIR = BASE_DIR / "models"


def load_invoice_data():

    conn = sqlite3.connect(DB_PATH)

    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            SUM(p.Quantity) AS total_item_quantity,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(
                julianday(p.ReceivingDate)
                - julianday(p.PODate)
            ) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )

    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,

        (
            julianday(vi.InvoiceDate)
            - julianday(vi.PODate)
        ) AS days_po_to_invoice,

        (
            julianday(vi.PayDate)
            - julianday(vi.InvoiceDate)
        ) AS days_to_pay,

        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay

    FROM vendor_invoice vi

    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    try:
        df = pd.read_sql_query(
            query,
            conn
        )
    finally:
        conn.close()

    return df


def create_invoice_risk_label(row):

    if abs(
        row["invoice_dollars"]
        - row["total_item_dollars"]
    ) > 5:
        return 1

    if row["avg_receiving_delay"] > 10:
        return 1

    return 0


def apply_labels(df):

    df["flag_invoice"] = (
        (
            abs(
                df["invoice_dollars"]
                - df["total_item_dollars"]
            ) > 5
        )
        |
        (
            df["avg_receiving_delay"] > 10
        )
    ).astype(int)

    return df


def clean_data(df):

    numeric_columns = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "days_po_to_invoice",
        "days_to_pay",
        "total_brands",
        "total_item_quantity",
        "total_item_dollars",
        "avg_receiving_delay"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.fillna(0)

    return df


def split_data(
    df,
    features,
    target
):

    X = df[features]

    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def save_scaler_placeholder():
    """
    Kept only for compatibility.
    RandomForest does not require scaling.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {"scaler": None},
        MODEL_DIR / "scaler.pkl"
    )