import joblib
from pathlib import Path
import pandas as pd

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data,
)

from modeling_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "inventory.db"
MODEL_DIR = BASE_DIR / "models"


def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load data
    df = load_vendor_invoice_data(DB_PATH)

    # Prepare data
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Train models
    lr_model = train_linear_regression(
        X_train,
        y_train
    )

    dt_model = train_decision_tree(
        X_train,
        y_train
    )

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    # Evaluate models
    results = []

    results.append(
        evaluate_model(
            lr_model,
            X_test,
            y_test,
            "Linear Regression"
        )
    )

    results.append(
        evaluate_model(
            dt_model,
            X_test,
            y_test,
            "Decision Tree Regression"
        )
    )

    results.append(
        evaluate_model(
            rf_model,
            X_test,
            y_test,
            "Random Forest Regression"
        )
    )

    results_df = pd.DataFrame(results)

    print("\nModel Comparison")
    print(results_df)

    # Select best model based on lowest MAE
    best_model_info = min(
        results,
        key=lambda x: x["mae"]
    )

    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model,
    }[best_model_name]

    model_package = {
        "model": best_model,
        "model_name": best_model_name,
        "metrics": best_model_info
    }

    model_path = (
        MODEL_DIR
        / "predict_freight_model.pkl"
    )

    joblib.dump(
        model_package,
        model_path
    )

    print(
        f"\nBest model saved: {best_model_name}"
    )

    print(
        f"Model path: {model_path}"
    )


if __name__ == "__main__":
    main()