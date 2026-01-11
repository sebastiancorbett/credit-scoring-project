"""
Unit tests for evaluation module.

Tests model evaluation metrics and visualization functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import ModelEvaluator


class TestModelEvaluator:
    """Test suite for ModelEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create ModelEvaluator instance for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ModelEvaluator(output_dir=Path(tmpdir))

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        n_samples = 200

        y_true = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        y_pred = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        y_pred_proba = np.random.rand(n_samples)

        return y_true, y_pred, y_pred_proba

    def test_initialization(self, evaluator):
        """Test ModelEvaluator initialization."""
        assert evaluator is not None
        assert evaluator.output_dir.exists()
        assert isinstance(evaluator.results, dict)

    def test_calculate_metrics(self, evaluator, sample_predictions):
        """Test metrics calculation."""
        y_true, y_pred, y_pred_proba = sample_predictions

        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)

        assert "roc_auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "accuracy" in metrics

        # Check metric ranges for score metrics (0-1)
        score_metrics = [
            "roc_auc",
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "recall_negative",
            "positive_prediction_ratio",
            "balance_score",
        ]
        for key, value in metrics.items():
            if key in score_metrics:
                assert (
                    0 <= value <= 1
                ), f"Metric {key} value {value} is out of range [0, 1]"

    def test_evaluate_model(self, evaluator, sample_predictions):
        """Test single model evaluation."""
        y_true, y_pred, y_pred_proba = sample_predictions

        results = evaluator.evaluate_model("TestModel", y_true, y_pred, y_pred_proba)

        assert "model_name" in results
        assert results["model_name"] == "TestModel"
        assert "metrics" in results
        assert "classification_report" in results
        assert "confusion_matrix" in results

    def test_evaluate_all_models(self, evaluator):
        """Test evaluation of multiple models."""
        np.random.seed(42)
        n_samples = 200

        y_true = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])

        models_predictions = {
            "Model1": (
                np.random.choice([0, 1], n_samples),
                np.random.rand(n_samples),
            ),
            "Model2": (
                np.random.choice([0, 1], n_samples),
                np.random.rand(n_samples),
            ),
        }

        all_results = evaluator.evaluate_all_models(models_predictions, y_true)

        assert len(all_results) == 2
        assert "Model1" in all_results
        assert "Model2" in all_results

    def test_plot_roc_curves(self, evaluator):
        """Test ROC curve plotting."""
        np.random.seed(42)
        n_samples = 200

        y_true = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])

        models_predictions = {
            "Model1": (
                np.random.choice([0, 1], n_samples),
                np.random.rand(n_samples),
            ),
        }

        # Should not raise errors
        evaluator.plot_roc_curves(models_predictions, y_true, save=True)

        # Check if file was created
        roc_file = evaluator.output_dir / "roc_curves.png"
        assert roc_file.exists()

    def test_plot_confusion_matrices(self, evaluator):
        """Test confusion matrix plotting."""
        np.random.seed(42)
        n_samples = 200

        y_true = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])

        models_predictions = {
            "Model1": (
                np.random.choice([0, 1], n_samples),
                np.random.rand(n_samples),
            ),
        }

        # Should not raise errors
        evaluator.plot_confusion_matrices(models_predictions, y_true, save=True)

        # Check if file was created
        cm_file = evaluator.output_dir / "confusion_matrices.png"
        assert cm_file.exists()

    def test_generate_comparison_report(self, evaluator):
        """Test comparison report generation."""
        np.random.seed(42)
        n_samples = 200

        y_true = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])

        models_predictions = {
            "Model1": (
                np.random.choice([0, 1], n_samples),
                np.random.rand(n_samples),
            ),
            "Model2": (
                np.random.choice([0, 1], n_samples),
                np.random.rand(n_samples),
            ),
        }

        all_results = evaluator.evaluate_all_models(models_predictions, y_true)
        comparison_df = evaluator.generate_comparison_report(all_results, save=True)

        assert len(comparison_df) == 2
        assert "Model" in comparison_df.columns
        assert "roc_auc" in comparison_df.columns

        # Check if report was saved
        report_file = evaluator.output_dir / "model_comparison_report.csv"
        assert report_file.exists()


class TestModelEvaluatorEdgeCases:
    """Test edge cases and error handling."""

    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(output_dir=Path(tmpdir))

            y_true = np.array([0, 0, 1, 1, 0, 1])
            y_pred = np.array([0, 0, 1, 1, 0, 1])
            y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.95])

            metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)

            assert metrics["accuracy"] == 1.0
            assert metrics["precision"] == 1.0
            assert metrics["recall"] == 1.0

    def test_all_same_class(self):
        """Test evaluation when all predictions are same class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(output_dir=Path(tmpdir))

            y_true = np.array([0, 1, 0, 1, 1])
            y_pred = np.array([0, 0, 0, 0, 0])  # All zeros
            y_pred_proba = np.array([0.1, 0.2, 0.3, 0.15, 0.25])

            # Should handle gracefully with zero_division parameter
            metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)

            assert "roc_auc" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
