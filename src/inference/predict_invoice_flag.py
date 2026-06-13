import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "predict_flag_invoice.pkl"
)


def load_model_package():
    return joblib.load(MODEL_PATH)


MODEL_PACKAGE = load_model_package()

MODEL = MODEL_PACKAGE["model"]

FEATURES = MODEL_PACKAGE["features"]


def predict_invoice_flag(input_data):

    input_df = pd.DataFrame(input_data)

    missing = [
        col
        for col in FEATURES
        if col not in input_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    for col in FEATURES:

        input_df[col] = pd.to_numeric(
            input_df[col],
            errors="coerce"
        )

    if input_df[FEATURES].isna().any().any():

        raise ValueError(
            "Invalid numeric values detected."
        )

    X = input_df[FEATURES]

    prediction = MODEL.predict(X)

    probability = (
        MODEL.predict_proba(X)[:, 1]
        * 100
    )

    result_df = input_df.copy()

    result_df["Predicted_Flag"] = prediction

    result_df["Risk_Probability"] = (
        probability.round(2)
    )

    result_df["Risk_Status"] = (
        result_df["Predicted_Flag"]
        .map({
            0: "Safe",
            1: "Manual Approval Required"
        })
    )

    return result_df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [100],
        "invoice_dollars": [5000],
        "Freight": [50],
        "total_item_quantity": [100],
        "total_item_dollars": [5000]
    }

    result = predict_invoice_flag(
        sample_data
    )

    print(result)