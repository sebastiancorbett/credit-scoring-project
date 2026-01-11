"""
Machine learning models module for credit default prediction.

This module implements multiple classification models including Logistic Regression,
Random Forest, XGBoost, SVM, and Neural Networks with proper class imbalance handling
and threshold optimization.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

try:
    from imblearn.over_sampling import SMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = None
    IMBLEARN_AVAILABLE = False

import xgboost as xgb
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, save_pickle, load_pickle
from config import (
    MODEL_PARAMS,
    RANDOM_STATE,
    CV_FOLDS,
    IMBALANCE_STRATEGY,
    SMOTE_SAMPLING_STRATEGY,
    MIN_RECALL_POSITIVE,
    MIN_RECALL_NEGATIVE,
    TARGET_RECALL_POSITIVE_MIN,
    TARGET_RECALL_POSITIVE_MAX,
    TARGET_RECALL_NEGATIVE_MIN,
    TARGET_RECALL_NEGATIVE_MAX,
    THRESHOLD_WEIGHT_F1,
    THRESHOLD_WEIGHT_RECALL_POS,
    THRESHOLD_WEIGHT_RECALL_NEG,
    BALANCE_PENALTY_WEIGHT,
    compute_scale_pos_weight,
)

logger = setup_logging(__name__)


class CreditRiskModel:
    """
    Base class for credit risk prediction models.

    Provides common interface and functionality for all model types.
    """

    def __init__(self, model_type: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize CreditRiskModel.

        Args:
            model_type: Type of model (e.g., 'logistic_regression', 'random_forest')
            params: Model hyperparameters (uses defaults if None)
        """
        self.model_type = model_type
        self.params = params or MODEL_PARAMS.get(model_type, {}).copy()
        self.model = None
        self.is_fitted = False
        logger.info(f"Initialized {model_type} model")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CreditRiskModel":
        """
        Train the model.

        Args:
            X: Training features
            y: Training target

        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features for prediction

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features for prediction

        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = CV_FOLDS
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Target
            cv: Number of folds

        Returns:
            Dictionary with cross-validation scores
        """
        logger.info(f"Performing {cv}-fold cross-validation for {self.model_type}")

        cv_strategy = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=RANDOM_STATE
        )

        scoring = ["roc_auc", "precision", "recall", "f1"]
        scores = {}

        for metric in scoring:
            cv_scores = cross_val_score(
                self.model, X, y, cv=cv_strategy, scoring=metric, n_jobs=-1
            )
            scores[f"{metric}_mean"] = cv_scores.mean()
            scores[f"{metric}_std"] = cv_scores.std()

        logger.info(f"Cross-validation complete. ROC-AUC: {scores['roc_auc_mean']:.4f}")
        return scores

    def save(self, filepath: Path) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save file
        """
        save_pickle({"model": self.model, "params": self.params}, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Path) -> None:
        """
        Load model from file.

        Args:
            filepath: Path to model file
        """
        data = load_pickle(filepath)
        self.model = data["model"]
        self.params = data["params"]
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


class LogisticRegressionModel(CreditRiskModel):
    """Logistic Regression model for credit risk prediction."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize Logistic Regression model."""
        super().__init__("logistic_regression", params)
        self.model = LogisticRegression(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression model...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Logistic Regression training complete")
        return self


class RandomForestModel(CreditRiskModel):
    """Random Forest model for credit risk prediction."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize Random Forest model."""
        super().__init__("random_forest", params)
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Random Forest training complete")
        return self

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        return self.model.feature_importances_


class XGBoostModel(CreditRiskModel):
    """XGBoost model for credit risk prediction."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost model."""
        super().__init__("xgboost", params)
        # Remove scale_pos_weight if None (will be computed dynamically)
        if self.params.get("scale_pos_weight") is None:
            self.params.pop("scale_pos_weight", None)
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        """
        Train XGBoost model.

        Dynamically computes scale_pos_weight based on class distribution.
        """
        logger.info("Training XGBoost model...")

        # Compute scale_pos_weight dynamically if not set
        if "scale_pos_weight" not in self.model.get_params():
            scale_weight = compute_scale_pos_weight(y)
            self.model.set_params(scale_pos_weight=scale_weight)
            logger.info(f"  Computed scale_pos_weight: {scale_weight:.2f}")

        self.model.fit(X, y, verbose=False)
        self.is_fitted = True
        logger.info("XGBoost training complete")
        return self

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        return self.model.feature_importances_


class SVMModel(CreditRiskModel):
    """Support Vector Machine model for credit risk prediction."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize SVM model."""
        super().__init__("svm", params)
        self.model = SVC(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        """Train SVM model."""
        logger.info("Training SVM model...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("SVM training complete")
        return self


class NeuralNetworkModel(CreditRiskModel):
    """Neural Network (MLP) model for credit risk prediction."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize Neural Network model."""
        super().__init__("neural_network", params)
        self.model = MLPClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetworkModel":
        """Train Neural Network model."""
        logger.info("Training Neural Network model...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(
            f"Neural Network training complete. Iterations: {self.model.n_iter_}"
        )
        return self


class ModelTrainer:
    """
    Class for training and managing multiple models with imbalance handling.

    Implements proper threshold optimization on validation data to prevent
    data leakage to the test set.
    """

    def __init__(
        self,
        models: Optional[Dict[str, CreditRiskModel]] = None,
        handle_imbalance: bool = True,
        optimize_thresholds: bool = True,
        min_recall_positive: float = MIN_RECALL_POSITIVE,
        min_recall_negative: float = MIN_RECALL_NEGATIVE,
    ):
        """
        Initialize ModelTrainer.

        Args:
            models: Dictionary of models to train
            handle_imbalance: Whether to apply SMOTE for class imbalance
            optimize_thresholds: Whether to optimize decision thresholds
            min_recall_positive: Minimum recall for positive class (from config)
            min_recall_negative: Minimum recall for negative class (from config)
        """
        self.models = models or self._create_default_models()
        self.handle_imbalance = handle_imbalance
        self.optimize_thresholds = optimize_thresholds
        self.min_recall_positive = min_recall_positive
        self.min_recall_negative = min_recall_negative
        self.trained_models = {}
        self.cv_results = {}
        self.optimal_thresholds = {}

        logger.info(f"ModelTrainer initialized with {len(self.models)} models")
        if optimize_thresholds:
            logger.info(
                f"Threshold optimization enabled: "
                f"recall_pos>={min_recall_positive:.0%}, "
                f"recall_neg>={min_recall_negative:.0%}"
            )

    def _create_default_models(self) -> Dict[str, CreditRiskModel]:
        """Create default set of models."""
        return {
            "Logistic Regression": LogisticRegressionModel(),
            "Random Forest": RandomForestModel(),
            "XGBoost": XGBoostModel(),
            "Neural Network": NeuralNetworkModel(),
            "SVM": SVMModel(),
        }

    def _apply_smote(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to handle class imbalance.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Tuple of (resampled X, resampled y)
        """
        if not IMBLEARN_AVAILABLE or SMOTE is None:
            logger.warning(
                "SMOTE not available, skipping resampling. Using class_weight instead."
            )
            return X_train, y_train

        logger.info("Applying SMOTE for class imbalance...")
        smote = SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            random_state=RANDOM_STATE,
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        logger.info(
            f"Original shape: {X_train.shape}, Resampled shape: {X_resampled.shape}"
        )
        logger.info(f"Original class distribution: {np.bincount(y_train.astype(int))}")
        logger.info(
            f"Resampled class distribution: {np.bincount(y_resampled.astype(int))}"
        )

        return X_resampled, y_resampled

    def _optimize_threshold(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str
    ) -> float:
        """
        Find optimal threshold for balanced predictions.

        Uses configurable weights from config.py for the composite score.
        Thresholds are found on VALIDATION data (not test) to prevent leakage.

        The scoring formula includes:
        1. Base score: weighted sum of F1 and recalls
        2. Balance penalty: penalizes large differences between recalls
        3. Target range bonus: rewards recalls within target range

        Args:
            y_true: True labels (from validation set)
            y_pred_proba: Predicted probabilities
            model_name: Name of the model

        Returns:
            Optimal threshold
        """
        best_threshold = 0.5
        best_score = -np.inf
        best_recalls = (0, 0)
        thresholds = np.linspace(0.01, 0.99, 200)

        valid_thresholds_found = 0

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape != (2, 2):
                continue

            tn, fp, fn, tp = cm.ravel()

            # Calculate recalls for both classes
            recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Check minimum constraints (hard constraints)
            if (
                recall_pos < self.min_recall_positive
                or recall_neg < self.min_recall_negative
            ):
                continue

            # Check maximum constraints (hard constraints for balanced predictions)
            # This prevents models from being too aggressive in one direction
            if (
                recall_pos > TARGET_RECALL_POSITIVE_MAX
                or recall_neg > TARGET_RECALL_NEGATIVE_MAX
            ):
                continue

            # Check if within ideal target range
            in_target_range = (
                TARGET_RECALL_POSITIVE_MIN <= recall_pos <= TARGET_RECALL_POSITIVE_MAX
                and TARGET_RECALL_NEGATIVE_MIN
                <= recall_neg
                <= TARGET_RECALL_NEGATIVE_MAX
            )

            valid_thresholds_found += 1

            # Calculate F1
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Base score with configurable weights
            base_score = (
                THRESHOLD_WEIGHT_F1 * f1
                + THRESHOLD_WEIGHT_RECALL_POS * recall_pos
                + THRESHOLD_WEIGHT_RECALL_NEG * recall_neg
            )

            # Balance penalty: penalize large differences between recalls
            # This encourages more balanced predictions
            recall_diff = abs(recall_pos - recall_neg)
            balance_penalty = BALANCE_PENALTY_WEIGHT * recall_diff

            # Target range bonus: prefer recalls within target range
            target_bonus = 0.05 if in_target_range else 0

            # Final score
            score = base_score - balance_penalty + target_bonus

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_recalls = (recall_pos, recall_neg)

        if valid_thresholds_found == 0:
            logger.warning(
                f"  {model_name}: No threshold satisfies both recall constraints. "
                f"Using default 0.5"
            )
        else:
            logger.info(
                f"  {model_name}: Optimal threshold={best_threshold:.4f} "
                f"(recall_pos={best_recalls[0]:.2%}, recall_neg={best_recalls[1]:.2%}, "
                f"from {valid_thresholds_found} valid candidates)"
            )

        return best_threshold

    def optimize_thresholds_on_validation(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> Dict[str, float]:
        """
        Optimize decision thresholds on VALIDATION data (not test).

        This is the proper methodology to prevent data leakage:
        - Thresholds are tuned on validation set
        - Test set remains untouched for final evaluation

        Args:
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary of optimal thresholds
        """
        if not self.optimize_thresholds:
            return {name: 0.5 for name in self.trained_models.keys()}

        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZING DECISION THRESHOLDS ON VALIDATION SET")
        logger.info("(This prevents data leakage to test set)")
        logger.info("=" * 80)

        for name, model in self.trained_models.items():
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            optimal_threshold = self._optimize_threshold(y_val, y_pred_proba, name)
            self.optimal_thresholds[name] = optimal_threshold

        logger.info("=" * 80)
        return self.optimal_thresholds

    def train_all(
        self, X_train: np.ndarray, y_train: np.ndarray, perform_cv: bool = True
    ) -> Dict[str, CreditRiskModel]:
        """
        Train all models.

        Args:
            X_train: Training features
            y_train: Training target
            perform_cv: Whether to perform cross-validation

        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 80)
        logger.info("Starting model training pipeline")
        logger.info("=" * 80)

        # Apply SMOTE if enabled
        if self.handle_imbalance and IMBALANCE_STRATEGY in ["SMOTE", "both"]:
            X_train_resampled, y_train_resampled = self._apply_smote(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Train each model
        for name, model in self.models.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training {name}")
            logger.info(f"{'=' * 60}")

            try:
                # Train model
                model.fit(X_train_resampled, y_train_resampled)
                self.trained_models[name] = model

                # Perform cross-validation if requested
                if perform_cv:
                    cv_scores = model.cross_validate(
                        X_train_resampled, y_train_resampled
                    )
                    self.cv_results[name] = cv_scores

                    logger.info(f"\nCross-validation results for {name}:")
                    logger.info(
                        f"  ROC-AUC: {cv_scores['roc_auc_mean']:.4f} "
                        f"(+/- {cv_scores['roc_auc_std']:.4f})"
                    )
                    logger.info(
                        f"  Precision: {cv_scores['precision_mean']:.4f} "
                        f"(+/- {cv_scores['precision_std']:.4f})"
                    )
                    logger.info(
                        f"  Recall: {cv_scores['recall_mean']:.4f} "
                        f"(+/- {cv_scores['recall_std']:.4f})"
                    )
                    logger.info(
                        f"  F1: {cv_scores['f1_mean']:.4f} "
                        f"(+/- {cv_scores['f1_std']:.4f})"
                    )

            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue

        logger.info("\n" + "=" * 80)
        logger.info(
            f"Model training complete. Trained {len(self.trained_models)} models"
        )
        logger.info("=" * 80)

        return self.trained_models

    def get_predictions(
        self, X: np.ndarray, return_proba: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from all trained models.

        Args:
            X: Features for prediction
            return_proba: Whether to return probabilities or class labels

        Returns:
            Dictionary of predictions for each model
        """
        predictions = {}

        for name, model in self.trained_models.items():
            try:
                if return_proba:
                    predictions[name] = model.predict_proba(X)[:, 1]
                else:
                    predictions[name] = model.predict(X)
            except Exception as e:
                logger.error(f"Error getting predictions from {name}: {str(e)}")
                continue

        return predictions

    def save_all_models(self, output_dir: Path) -> None:
        """
        Save all trained models.

        Args:
            output_dir: Directory to save models
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.trained_models.items():
            filename = f"{name.lower().replace(' ', '_')}_model.pkl"
            filepath = output_dir / filename
            model.save(filepath)

        logger.info(f"Saved {len(self.trained_models)} models to {output_dir}")


def main():
    """Main function for testing ModelTrainer."""
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer

    # Load and prepare data
    loader = DataLoader()
    df = loader.load_data()
    df_clean = loader.clean_data(df)

    # Engineer features
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_all_features(df_clean)

    # Prepare data
    df_encoded = loader.encode_categorical(df_engineered)
    X, y = loader.prepare_features_target(df_encoded)

    # Use three-way split
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = loader.scale_features(
        X_train, X_val, X_test
    )

    # Train models
    trainer = ModelTrainer()
    trained_models = trainer.train_all(X_train_scaled, y_train.values)

    # Optimize thresholds on VALIDATION set (not test)
    trainer.optimize_thresholds_on_validation(X_val_scaled, y_val.values)

    logger.info(f"\nTrained {len(trained_models)} models successfully")
    logger.info(f"Optimal thresholds: {trainer.optimal_thresholds}")


if __name__ == "__main__":
    main()
