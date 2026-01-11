"""
Unit tests for models module.

Tests model training, prediction, and persistence functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    SVMModel,
    NeuralNetworkModel,
    ModelTrainer,
)


class TestModels:
    """Test suite for individual model classes."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.choice([0, 1], 200, p=[0.7, 0.3])
        return X, y

    def test_logistic_regression_model(self, sample_data):
        """Test Logistic Regression model."""
        X, y = sample_data
        model = LogisticRegressionModel()

        # Test fitting
        model.fit(X, y)
        assert model.is_fitted

        # Test prediction
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert set(y_pred).issubset({0, 1})

        # Test probability prediction
        y_proba = model.predict_proba(X)
        assert y_proba.shape == (len(X), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)

    def test_random_forest_model(self, sample_data):
        """Test Random Forest model."""
        X, y = sample_data
        model = RandomForestModel()

        model.fit(X, y)
        assert model.is_fitted

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)

    def test_xgboost_model(self, sample_data):
        """Test XGBoost model."""
        X, y = sample_data
        model = XGBoostModel()

        model.fit(X, y)
        assert model.is_fitted

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == X.shape[1]

    def test_svm_model(self, sample_data):
        """Test SVM model."""
        X, y = sample_data
        # Use smaller sample for SVM (faster)
        X_small, y_small = X[:50], y[:50]

        model = SVMModel()
        model.fit(X_small, y_small)
        assert model.is_fitted

        y_pred = model.predict(X_small)
        assert len(y_pred) == len(y_small)

    def test_neural_network_model(self, sample_data):
        """Test Neural Network model."""
        X, y = sample_data
        params = {
            "hidden_layer_sizes": (50,),
            "max_iter": 100,
            "random_state": 42,
        }
        model = NeuralNetworkModel(params)

        model.fit(X, y)
        assert model.is_fitted

        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

    def test_model_save_load(self, sample_data):
        """Test model saving and loading."""
        X, y = sample_data
        model = LogisticRegressionModel()
        model.fit(X, y)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.pkl"
            model.save(filepath)
            assert filepath.exists()

            # Load model
            new_model = LogisticRegressionModel()
            new_model.load(filepath)
            assert new_model.is_fitted

            # Compare predictions
            y_pred_original = model.predict(X)
            y_pred_loaded = new_model.predict(X)
            assert np.array_equal(y_pred_original, y_pred_loaded)

    def test_model_before_fitting(self, sample_data):
        """Test error when predicting before fitting."""
        X, y = sample_data
        model = LogisticRegressionModel()

        with pytest.raises(ValueError):
            model.predict(X)


class TestModelTrainer:
    """Test suite for ModelTrainer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(300, 10)
        y = np.random.choice([0, 1], 300, p=[0.8, 0.2])
        return X, y

    def test_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer()
        assert trainer is not None
        assert len(trainer.models) > 0
        assert trainer.handle_imbalance

    def test_train_all_models(self, sample_data):
        """Test training all models."""
        X, y = sample_data
        trainer = ModelTrainer(handle_imbalance=False)  # Skip SMOTE for speed

        trained_models = trainer.train_all(X, y, perform_cv=False)

        assert len(trained_models) > 0
        for model in trained_models.values():
            assert model.is_fitted

    def test_get_predictions(self, sample_data):
        """Test getting predictions from all models."""
        X, y = sample_data
        trainer = ModelTrainer(handle_imbalance=False)
        trainer.train_all(X, y, perform_cv=False)

        predictions = trainer.get_predictions(X, return_proba=True)

        assert len(predictions) > 0
        for preds in predictions.values():
            assert len(preds) == len(X)
            assert np.all((preds >= 0) & (preds <= 1))

    def test_save_all_models(self, sample_data):
        """Test saving all trained models."""
        X, y = sample_data
        trainer = ModelTrainer(handle_imbalance=False)
        trainer.train_all(X, y, perform_cv=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            trainer.save_all_models(output_dir)

            # Check that files were created
            saved_files = list(output_dir.glob("*.pkl"))
            assert len(saved_files) > 0


class TestModelTrainerIntegration:
    """Integration tests for ModelTrainer."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline with small dataset."""
        np.random.seed(42)
        X = np.random.randn(150, 8)
        y = np.random.choice([0, 1], 150, p=[0.85, 0.15])

        # Create trainer with subset of models for speed
        from src.models import LogisticRegressionModel, RandomForestModel

        models = {
            "Logistic Regression": LogisticRegressionModel(),
            "Random Forest": RandomForestModel(),
        }

        trainer = ModelTrainer(models=models, handle_imbalance=False)
        trained_models = trainer.train_all(X, y, perform_cv=False)

        assert len(trained_models) == 2

        # Test predictions
        predictions = trainer.get_predictions(X)
        assert len(predictions) == 2

        for name, preds in predictions.items():
            assert len(preds) == len(X)
