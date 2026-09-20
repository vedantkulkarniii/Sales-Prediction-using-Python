"""Command-line interface for training and prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .predict import predict_sales
from .config import TrainingConfig
from .logging_utils import configure_logging
from .train_model import train_model


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse options."""
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def minimum_two_int(value: str) -> int:
    """Parse an integer suitable for cross-validation folds."""
    parsed_value = int(value)
    if parsed_value < 2:
        raise argparse.ArgumentTypeError("value must be at least 2")
    return parsed_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Sales prediction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save a model")
    train_parser.add_argument("data_path")
    train_parser.add_argument("target_column")
    train_parser.add_argument("--model-path", default="models/sales_model.joblib")
    train_parser.add_argument("--n-estimators", type=positive_int, default=300)
    train_parser.add_argument("--cv-folds", type=minimum_two_int, default=5)

    predict_parser = subparsers.add_parser("predict", help="Predict sales from a CSV")
    predict_parser.add_argument("data_path")
    predict_parser.add_argument("--model-path", default="models/sales_model.joblib")
    predict_parser.add_argument("--output-path", default="outputs/predictions/predictions.csv")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logs")

    arguments = parser.parse_args()
    configure_logging(arguments.verbose)
    if arguments.command == "train":
        config = TrainingConfig(n_estimators=arguments.n_estimators, cv_folds=arguments.cv_folds)
        print(train_model(arguments.data_path, arguments.target_column, arguments.model_path, config=config))
        return

    predictions = predict_sales(pd.read_csv(arguments.data_path), arguments.model_path)
    Path(arguments.output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(arguments.output_path, index=False)
    print(f"Saved predictions to {arguments.output_path}")


if __name__ == "__main__":
    main()
