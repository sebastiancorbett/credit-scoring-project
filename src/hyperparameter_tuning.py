"""
Hyperparameter tuning module for credit risk models.

This module implements automated hyperparameter optimization using
GridSearchCV and RandomizedSearchCV for all model types.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, save_json, print_section_header
from config import RANDOM_STATE, CV_FOLDS

logger = setup_logging(__name__)


class HyperparameterTuner:
    """
    Class for automated hyperparameter tuning of ML models.

    Supports both Grid Search and Randomized Search with cross-validation.
    Uses stratified sampling to maintain class distribution during tuning.
    """

    def __init__(
        self,
        search_method: str = "random",
        cv_folds: int = CV_FOLDS,
        n_iter: int = 50,
        n_jobs: int = -1,
        verbose: int = 1,
    ):
        """
        Initialize HyperparameterTuner.

        Args:
            search_method: 'grid' for GridSearch or 'random' for RandomizedSearch
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for RandomizedSearch
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        self.search_method = search_method
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_params = {}
        self.cv_results = {}

        logger.info(f"HyperparameterTuner initialized with {search_method} search")

    def get_param_grid_logistic_regression(self) -> Dict[str, list]:
        """Get parameter grid for Logistic Regression."""
        return {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear", "saga"],
            "max_iter": [500, 1000, 2000],
            "class_weight": ["balanced", None],
        }

    def get_param_grid_random_forest(self) -> Dict[str, list]:
        """Get parameter grid for Random Forest."""
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
            "class_weight": ["balanced", "balanced_subsample", None],
            "bootstrap": [True, False],
        }

    def get_param_grid_xgboost(self) -> Dict[str, list]:
        """Get parameter grid for XGBoost."""
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2],
            "scale_pos_weight": [1, 3, 5, 10],
        }

    def get_param_grid_svm(self) -> Dict[str, list]:
        """Get parameter grid for SVM."""
        return {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "class_weight": ["balanced", None],
            "degree": [2, 3, 4],  # Only for poly kernel
        }

    def get_param_grid_neural_network(self) -> Dict[str, list]:
        """Get parameter grid for Neural Network."""
        return {
            "hidden_layer_sizes": [
                (50,),
                (100,),
                (50, 50),
                (100, 50),
                (100, 100),
                (100, 50, 25),
                (150, 100, 50),
            ],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
            "batch_size": [64, 128, 256],
            "max_iter": [500, 1000],
        }

    def _stratified_subsample(
        self, X: np.ndarray, y: np.ndarray, max_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a stratified random subsample of the data.

        This ensures class distribution is preserved in the subsample,
        which is important for imbalanced classification.

        Args:
            X: Features
            y: Target
            max_samples: Maximum number of samples

        Returns:
            Tuple of (X_sample, y_sample)
        """
        if len(X) <= max_samples:
            return X, y

        # Use stratified sampling to preserve class distribution
        np.random.seed(RANDOM_STATE)

        # Get indices for each class
        classes = np.unique(y)
        sampled_indices = []

        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            # Calculate proportional sample size
            n_cls_samples = int(max_samples * len(cls_indices) / len(y))
            # Ensure at least some samples from each class
            n_cls_samples = max(n_cls_samples, min(10, len(cls_indices)))

            if len(cls_indices) > n_cls_samples:
                sampled = np.random.choice(cls_indices, n_cls_samples, replace=False)
            else:
                sampled = cls_indices

            sampled_indices.extend(sampled)

        sampled_indices = np.array(sampled_indices)
        np.random.shuffle(sampled_indices)

        logger.info(
            f"  Stratified subsample: {len(X)} -> {len(sampled_indices)} samples"
        )
        logger.info(f"  Original distribution: {np.bincount(y.astype(int))}")
        logger.info(
            f"  Subsample distribution: {np.bincount(y[sampled_indices].astype(int))}"
        )

        return X[sampled_indices], y[sampled_indices]

    def tune_logistic_regression(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[LogisticRegression, Dict[str, Any]]:
        """Tune Logistic Regression hyperparameters."""
        logger.info("Tuning Logistic Regression hyperparameters...")

        base_model = LogisticRegression(random_state=RANDOM_STATE, n_jobs=self.n_jobs)
        param_grid = self.get_param_grid_logistic_regression()

        best_model, best_params = self._perform_search(
            base_model, param_grid, X, y, "Logistic Regression"
        )

        return best_model, best_params

    def tune_random_forest(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """Tune Random Forest hyperparameters."""
        logger.info("Tuning Random Forest hyperparameters...")

        base_model = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=self.n_jobs
        )
        param_grid = self.get_param_grid_random_forest()

        best_model, best_params = self._perform_search(
            base_model, param_grid, X, y, "Random Forest"
        )

        return best_model, best_params

    def tune_xgboost(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """Tune XGBoost hyperparameters."""
        logger.info("Tuning XGBoost hyperparameters...")

        base_model = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=self.n_jobs,
            tree_method="hist",
        )
        param_grid = self.get_param_grid_xgboost()

        best_model, best_params = self._perform_search(
            base_model, param_grid, X, y, "XGBoost"
        )

        return best_model, best_params

    def tune_svm(self, X: np.ndarray, y: np.ndarray) -> Tuple[SVC, Dict[str, Any]]:
        """
        Tune SVM hyperparameters.

        Uses stratified subsampling for computational efficiency while
        preserving class distribution.
        """
        logger.info("Tuning SVM hyperparameters...")

        # Use stratified subsample for SVM (computationally expensive)
        max_samples = min(2000, len(X))
        X_sample, y_sample = self._stratified_subsample(X, y, max_samples)

        base_model = SVC(random_state=RANDOM_STATE, probability=True)
        param_grid = self.get_param_grid_svm()

        best_model, best_params = self._perform_search(
            base_model, param_grid, X_sample, y_sample, "SVM"
        )

        # Retrain on full dataset with best params
        logger.info("Retraining SVM on full dataset with best parameters...")
        best_model.fit(X, y)

        return best_model, best_params

    def tune_neural_network(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[MLPClassifier, Dict[str, Any]]:
        """Tune Neural Network hyperparameters."""
        logger.info("Tuning Neural Network hyperparameters...")

        base_model = MLPClassifier(
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
        )
        param_grid = self.get_param_grid_neural_network()

        best_model, best_params = self._perform_search(
            base_model, param_grid, X, y, "Neural Network"
        )

        return best_model, best_params

    def _perform_search(
        self,
        base_model: Any,
        param_grid: Dict[str, list],
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter search.

        Args:
            base_model: Base model instance
            param_grid: Parameter grid to search
            X: Training features
            y: Training target
            model_name: Name of the model

        Returns:
            Tuple of (best model, best parameters)
        """
        cv_strategy = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        if self.search_method == "grid":
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_strategy,
                scoring="roc_auc",
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True,
            )
        else:  # random
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                cv=cv_strategy,
                scoring="roc_auc",
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=RANDOM_STATE,
                return_train_score=True,
            )

        logger.info(f"Starting {self.search_method} search for {model_name}...")
        search.fit(X, y)

        # Store results
        self.best_params[model_name] = search.best_params_
        self.cv_results[model_name] = {
            "best_score": search.best_score_,
            "best_params": search.best_params_,
            "cv_results": pd.DataFrame(search.cv_results_),
        }

        logger.info(f"\n{'=' * 70}")
        logger.info(f"Best parameters for {model_name}:")
        for param, value in search.best_params_.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Best CV ROC-AUC score: {search.best_score_:.4f}")
        logger.info(f"{'=' * 70}\n")

        return search.best_estimator_, search.best_params_

    def tune_all_models(
        self, X: np.ndarray, y: np.ndarray, models_to_tune: Optional[list] = None
    ) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """
        Tune all models.

        Args:
            X: Training features
            y: Training target
            models_to_tune: List of model names to tune (None = all)

        Returns:
            Dictionary mapping model names to (best_model, best_params)
        """
        print_section_header("HYPERPARAMETER TUNING")

        available_models = {
            "Logistic Regression": self.tune_logistic_regression,
            "Random Forest": self.tune_random_forest,
            "XGBoost": self.tune_xgboost,
            "SVM": self.tune_svm,
            "Neural Network": self.tune_neural_network,
        }

        if models_to_tune is None:
            models_to_tune = list(available_models.keys())

        tuned_models = {}

        for model_name in models_to_tune:
            if model_name not in available_models:
                logger.warning(f"Model {model_name} not found, skipping...")
                continue

            try:
                logger.info(f"\n{'*' * 80}")
                logger.info(f"Tuning {model_name}")
                logger.info(f"{'*' * 80}\n")

                tune_func = available_models[model_name]
                best_model, best_params = tune_func(X, y)
                tuned_models[model_name] = (best_model, best_params)

            except Exception as e:
                logger.error(f"Error tuning {model_name}: {str(e)}")
                continue

        logger.info("\n" + "=" * 80)
        logger.info(f"Hyperparameter tuning complete for {len(tuned_models)} models")
        logger.info("=" * 80)

        return tuned_models

    def save_results(self, output_path: Path) -> None:
        """
        Save tuning results to file.

        Args:
            output_path: Path to save results
        """
        results_dict = {
            model_name: {
                "best_params": results["best_params"],
                "best_score": float(results["best_score"]),
            }
            for model_name, results in self.cv_results.items()
        }

        save_json(results_dict, output_path)
        logger.info(f"Tuning results saved to {output_path}")

    def get_tuning_summary(self) -> pd.DataFrame:
        """
        Get summary of tuning results.

        Returns:
            DataFrame with tuning summary
        """
        summary_data = []

        for model_name, results in self.cv_results.items():
            summary_data.append(
                {
                    "Model": model_name,
                    "Best CV ROC-AUC": results["best_score"],
                    "Best Parameters": str(results["best_params"]),
                }
            )

        return pd.DataFrame(summary_data).sort_values(
            "Best CV ROC-AUC", ascending=False
        )


def main():
    """Main function for testing HyperparameterTuner."""
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
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = loader.scale_features(
        X_train, X_val, X_test
    )

    # Tune models (using small subset for testing)
    tuner = HyperparameterTuner(
        search_method="random",
        n_iter=10,  # Small for testing
        cv_folds=3,
    )

    # Tune only fast models for testing
    _tuned_models = tuner.tune_all_models(  # noqa: F841
        X_train_scaled,
        y_train.values,
        models_to_tune=["Logistic Regression", "Random Forest"],
    )

    # Print summary
    summary = tuner.get_tuning_summary()
    print("\n" + "=" * 80)
    print("TUNING SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
