import joblib
from pathlib import Path
import pandas as pd

from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    clean_data,
    split_data
)

from modeling_evaluation import (
    train_random_forest,
    evaluate_classifier
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load data
    df = load_invoice_data()

    # Clean data
    df = clean_data(df)

    # Create labels
    df = apply_labels(df)

    # Prepare train/test data
    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    # Train model
    grid_search = train_random_forest(
        X_train,
        y_train
    )

    best_model = (
        grid_search.best_estimator_
    )

    # Evaluate model
    metrics = evaluate_classifier(
        best_model,
        X_test,
        y_test,
        "Random Forest Classifier"
    )

    # Save model package
    model_package = {
        "model": best_model,
        "features": FEATURES,
        "metrics": metrics,
        "best_params": grid_search.best_params_
    }

    model_path = (
        MODEL_DIR
        / "predict_flag_invoice.pkl"
    )

    joblib.dump(
        model_package,
        model_path
    )

    print(
        f"\nModel saved successfully:"
    )
    print(model_path)

    print("\nFeature List:")
    print(FEATURES)

    print("\nBest Parameters:")
    print(grid_search.best_params_)


if __name__ == "__main__":
    main()