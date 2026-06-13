import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path):
    """
    Load vendor invoice data from SQLite database.
    """

    conn = sqlite3.connect(db_path)

    try:
        query = """
        SELECT *
        FROM vendor_invoice
        """

        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()

    return df


def prepare_features(df: pd.DataFrame):
    """
    Prepare freight prediction features and target.
    """

    required_columns = ["Dollars", "Freight"]

    missing_cols = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    # Convert to numeric
    df["Dollars"] = pd.to_numeric(
        df["Dollars"],
        errors="coerce"
    )

    df["Freight"] = pd.to_numeric(
        df["Freight"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=["Dollars", "Freight"]
    )

    X = df[["Dollars"]]

    y = df["Freight"]

    return X, y


def split_data(
    X,
    y,
    test_size=0.2,
    random_state=42
):
    """
    Split dataset into train and test sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )