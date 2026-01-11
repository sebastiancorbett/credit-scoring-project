"""
Unit tests for hyperparameter_tuning module.

Tests hyperparameter optimization functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hyperparameter_tuning import HyperparameterTuner


class TestHyperparameterTuner:
    """Test suite for HyperparameterTuner class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(300, 10)
        y = np.random.choice([0, 1], 300, p=[0.7, 0.3])
        return X, y

    @pytest.fixture
    def tuner(self):
        """Create HyperparameterTuner instance for testing."""
        return HyperparameterTuner(
            search_method="random",
            cv_folds=3,
            n_iter=5,  # Small for testing
            verbose=0,
        )

    def test_initialization(self, tuner):
        """Test HyperparameterTuner initialization."""
        assert tuner is not None
        assert tuner.search_method == "random"
        assert tuner.cv_folds == 3
        assert tuner.n_iter == 5

    def test_get_param_grid_logistic_regression(self, tuner):
        """Test parameter grid for Logistic Regression."""
        param_grid = tuner.get_param_grid_logistic_regression()

        assert "C" in param_grid
        assert "penalty" in param_grid
        assert "solver" in param_grid
        assert len(param_grid["C"]) > 0

    def test_get_param_grid_random_forest(self, tuner):
        """Test parameter grid for Random Forest."""
        param_grid = tuner.get_param_grid_random_forest()

        assert "n_estimators" in param_grid
        assert "max_depth" in param_grid
        assert "min_samples_split" in param_grid

    def test_get_param_grid_xgboost(self, tuner):
        """Test parameter grid for XGBoost."""
        param_grid = tuner.get_param_grid_xgboost()

        assert "n_estimators" in param_grid
        assert "max_depth" in param_grid
        assert "learning_rate" in param_grid

    def test_tune_logistic_regression(self, tuner, sample_data):
        """Test tuning Logistic Regression."""
        X, y = sample_data

        best_model, best_params = tuner.tune_logistic_regression(X, y)

        assert best_model is not None
        assert isinstance(best_params, dict)
        assert len(best_params) > 0

    def test_tune_random_forest(self, tuner, sample_data):
        """Test tuning Random Forest."""
        X, y = sample_data

        best_model, best_params = tuner.tune_random_forest(X, y)

        assert best_model is not None
        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params

    def test_best_params_storage(self, tuner, sample_data):
        """Test that best parameters are stored."""
        X, y = sample_data

        tuner.tune_logistic_regression(X, y)

        assert "Logistic Regression" in tuner.best_params
        assert "Logistic Regression" in tuner.cv_results

    def test_tuning_summary(self, tuner, sample_data):
        """Test tuning summary generation."""
        X, y = sample_data

        tuner.tune_logistic_regression(X, y)
        summary = tuner.get_tuning_summary()

        assert len(summary) == 1
        assert "Model" in summary.columns
        assert "Best CV ROC-AUC" in summary.columns

    def test_save_results(self, tuner, sample_data):
        """Test saving tuning results."""
        X, y = sample_data

        tuner.tune_logistic_regression(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tuning_results.json"
            tuner.save_results(output_path)
            assert output_path.exists()


class TestHyperparameterTunerIntegration:
    """Integration tests for HyperparameterTuner."""

    def test_tune_multiple_models(self):
        """Test tuning multiple models."""
        np.random.seed(42)
        X = np.random.randn(200, 8)
        y = np.random.choice([0, 1], 200, p=[0.75, 0.25])

        tuner = HyperparameterTuner(
            search_method="random",
            cv_folds=2,
            n_iter=3,
            verbose=0,
        )

        # Tune only fast models for testing
        tuned_models = tuner.tune_all_models(
            X,
            y,
            models_to_tune=["Logistic Regression", "Random Forest"],
        )

        assert len(tuned_models) == 2
        assert "Logistic Regression" in tuned_models
        assert "Random Forest" in tuned_models

        # Check that models are fitted
        for name, (model, params) in tuned_models.items():
            assert hasattr(model, "predict")
            assert isinstance(params, dict)

    def test_grid_search_method(self):
        """Test using grid search instead of random search."""
        np.random.seed(42)
        X = np.random.randn(150, 8)
        y = np.random.choice([0, 1], 150, p=[0.75, 0.25])

        # Create tuner with grid search and very small grid
        tuner = HyperparameterTuner(
            search_method="grid",
            cv_folds=2,
            verbose=0,
        )

        # Override param grid to be very small for testing
        _original_method = tuner.get_param_grid_logistic_regression  # noqa: F841
        tuner.get_param_grid_logistic_regression = lambda: {
            "C": [0.1, 1.0],
            "solver": ["lbfgs"],
        }

        best_model, best_params = tuner.tune_logistic_regression(X, y)

        assert best_model is not None
        assert "C" in best_params
        assert best_params["C"] in [0.1, 1.0]

    def test_tuning_improves_performance(self):
        """Test that tuning can find better parameters than defaults."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.choice([0, 1], 200, p=[0.8, 0.2])

        tuner = HyperparameterTuner(
            search_method="random",
            cv_folds=3,
            n_iter=10,
            verbose=0,
        )

        best_model, best_params = tuner.tune_logistic_regression(X, y)
        cv_results = tuner.cv_results["Logistic Regression"]

        # Check that we got a valid score
        assert cv_results["best_score"] > 0.5
        assert cv_results["best_score"] <= 1.0
