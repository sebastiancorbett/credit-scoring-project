"""
Integration tests for the complete pipeline.

Tests end-to-end workflow from data loading to model evaluation.
"""

import pytest  # noqa: F401
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer, LogisticRegressionModel
from src.evaluation import ModelEvaluator


class TestEndToEndPipeline:
    """Integration tests for complete pipeline."""

    def test_full_pipeline_small_dataset(self):
        """Test complete pipeline with small synthetic dataset."""
        # Step 1: Load data
        loader = DataLoader()
        df = loader._create_sample_data(n_samples=300)

        # Step 2: Clean data
        df_clean = loader.clean_data(df)
        assert len(df_clean) > 0

        # Step 3: Engineer features
        engineer = FeatureEngineer()
        df_engineered = engineer.engineer_all_features(df_clean)
        assert df_engineered.shape[1] > df_clean.shape[1]

        # Step 4: Encode and prepare data (three-way split)
        df_encoded = loader.encode_categorical(df_engineered)
        X, y = loader.prepare_features_target(df_encoded)
        splits = loader.split_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        X_train_scaled, X_val_scaled, X_test_scaled = loader.scale_features(
            X_train, X_val, X_test
        )

        # Step 5: Train model (single model for speed)
        models = {"Logistic Regression": LogisticRegressionModel()}
        trainer = ModelTrainer(models=models, handle_imbalance=False)
        trained_models = trainer.train_all(
            X_train_scaled, y_train.values, perform_cv=False
        )

        assert len(trained_models) == 1

        # Step 6: Evaluate
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(output_dir=Path(tmpdir))

            models_predictions = {}
            for name, model in trained_models.items():
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                models_predictions[name] = (y_pred, y_pred_proba)

            all_results = evaluator.evaluate_all_models(
                models_predictions, y_test.values
            )

            assert len(all_results) == 1
            assert "metrics" in all_results["Logistic Regression"]

            # Generate visualizations
            evaluator.create_all_visualizations(
                models_predictions, y_test.values, all_results
            )

            # Check that output files were created
            assert (Path(tmpdir) / "roc_curves.png").exists()
            assert (Path(tmpdir) / "confusion_matrices.png").exists()

    def test_pipeline_with_feature_engineering(self):
        """Test that feature engineering improves feature count."""
        loader = DataLoader()
        df = loader._create_sample_data(n_samples=200)

        # Without feature engineering
        df_clean = loader.clean_data(df)
        original_feature_count = df_clean.shape[1]

        # With feature engineering
        engineer = FeatureEngineer()
        df_engineered = engineer.engineer_all_features(df_clean)
        engineered_feature_count = df_engineered.shape[1]

        assert engineered_feature_count > original_feature_count
        assert len(engineer.get_feature_names()) > 0

    def test_model_persistence(self):
        """Test that models can be saved and loaded."""
        loader = DataLoader()
        df = loader._create_sample_data(n_samples=200)
        df_clean = loader.clean_data(df)

        X, y = loader.prepare_features_target(df_clean)
        splits = loader.split_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        X_train_scaled, _, _ = loader.scale_features(X_train, X_val, X_test)

        # Train model
        model = LogisticRegressionModel()
        model.fit(X_train_scaled, y_train.values)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            model_path = Path(tmpdir) / "test_model.pkl"
            model.save(model_path)
            assert model_path.exists()

            # Load model
            new_model = LogisticRegressionModel()
            new_model.load(model_path)
            assert new_model.is_fitted

            # Compare predictions
            y_pred1 = model.predict(X_train_scaled)
            y_pred2 = new_model.predict(X_train_scaled)
            assert np.array_equal(y_pred1, y_pred2)

    def test_cross_validation_execution(self):
        """Test that cross-validation executes successfully."""
        loader = DataLoader()
        df = loader._create_sample_data(n_samples=200)
        df_clean = loader.clean_data(df)

        X, y = loader.prepare_features_target(df_clean)
        splits = loader.split_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        X_train_scaled, _, _ = loader.scale_features(X_train, X_val, X_test)

        model = LogisticRegressionModel()
        model.fit(X_train_scaled, y_train.values)

        # Perform cross-validation
        cv_scores = model.cross_validate(X_train_scaled, y_train.values, cv=3)

        assert "roc_auc_mean" in cv_scores
        assert "roc_auc_std" in cv_scores
        assert 0 <= cv_scores["roc_auc_mean"] <= 1


class TestDataIntegrity:
    """Test data integrity throughout pipeline."""

    def test_no_data_leakage(self):
        """Ensure no data leakage between train/val/test sets."""
        loader = DataLoader()
        df = loader._create_sample_data(n_samples=500)
        df_clean = loader.clean_data(df)

        X, y = loader.prepare_features_target(df_clean)
        splits = loader.split_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = splits

        # Check no overlap in indices
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)

        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0

    def test_class_balance_preservation(self):
        """Test that stratification preserves class balance."""
        loader = DataLoader()
        df = loader._create_sample_data(n_samples=1000)
        df_clean = loader.clean_data(df)

        X, y = loader.prepare_features_target(df_clean)
        splits = loader.split_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = splits

        # Calculate class proportions
        train_prop = y_train.mean()
        val_prop = y_val.mean()
        test_prop = y_test.mean()
        overall_prop = y.mean()

        # Proportions should be similar (within 5%)
        assert abs(train_prop - overall_prop) < 0.05
        assert abs(val_prop - overall_prop) < 0.05
        assert abs(test_prop - overall_prop) < 0.05
