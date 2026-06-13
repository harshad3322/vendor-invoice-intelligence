import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "predict_freight_model.pkl"
)


def load_model():

    model_package = joblib.load(MODEL_PATH)

    return model_package["model"]


MODEL = load_model()


def predict_freight_cost(input_data):
    """
    Predict freight cost from invoice dollars.
    """

    input_df = pd.DataFrame(input_data)

    required_columns = ["Dollars"]

    missing = [
        col
        for col in required_columns
        if col not in input_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    input_df["Dollars"] = pd.to_numeric(
        input_df["Dollars"],
        errors="coerce"
    )

    if input_df["Dollars"].isna().any():
        raise ValueError(
            "Invalid value found in Dollars."
        )

    X = input_df[["Dollars"]]

    predictions = MODEL.predict(X)

    result_df = input_df.copy()

    result_df["Predicted_Freight"] = (
        predictions.round(2)
    )

    return result_df


if __name__ == "__main__":

    sample_data = {
        "Dollars": [
            18500,
            9000,
            3000,
            200
        ]
    }

    result = predict_freight_cost(
        sample_data
    )

    print(result)