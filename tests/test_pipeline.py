import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.cli import minimum_two_int, positive_int
from src.evaluation import evaluate_regression
from src.predict import predict_sales
from src.predict import load_model_artifact
from src.train_model import train_model


class PipelineTests(unittest.TestCase):
    def test_cli_parameter_parsers_reject_invalid_values(self):
        self.assertEqual(positive_int("3"), 3)
        self.assertEqual(minimum_two_int("2"), 2)

        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            minimum_two_int("1")

    def test_evaluation_returns_named_regression_metrics(self):
        metrics = evaluate_regression(pd.Series([10, 20]), pd.Series([12, 18]))

        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertAlmostEqual(metrics["rmse"], 2.0)
        self.assertAlmostEqual(metrics["mape"], 0.15)
        self.assertAlmostEqual(metrics["r2"], 0.84)

    def test_training_and_prediction_share_the_same_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "sales_model.joblib"
            metrics = train_model(
                "data/raw/sample_sales.csv",
                "sales",
                str(model_path),
            )
            inputs = pd.read_csv("data/raw/sample_sales.csv").drop(columns=["sales"]).head(2)

            predictions = predict_sales(inputs, str(model_path))

            self.assertTrue(model_path.exists())
            self.assertIn("mae", metrics)
            self.assertIn("r2", metrics)
            self.assertIn("cv_mae", metrics)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(list(predictions.columns), ["predicted_sales"])

            artifact = load_model_artifact(str(model_path))
            self.assertEqual(artifact["schema_version"], 2)
            self.assertIn("trained_at", artifact)


if __name__ == "__main__":
    unittest.main()
