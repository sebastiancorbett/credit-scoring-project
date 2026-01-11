#!/usr/bin/env python
"""
Credit Default Risk Prediction Pipeline

This is the main entry point for the credit default prediction project.
It orchestrates data loading, preprocessing, feature engineering, model training,
threshold optimization, and evaluation.

Methodology:
- Three-way data split: train (60%) / validation (20%) / test (20%)
- Threshold optimization on validation set (prevents data leakage)
- Final evaluation on held-out test set
- Optional hyperparameter tuning with cross-validation

Usage:
    python main.py                    # Run with default parameters
    python main.py --tune             # Run with hyperparameter tuning
    python main.py --tune --n-iter 30 # Tuning with 30 iterations
    python main.py --help             # Show all options
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

# Project imports
from config import (
    OUTPUT_DIR,
    RANDOM_STATE,
    MIN_RECALL_POSITIVE,
    MIN_RECALL_NEGATIVE,
)
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator
from src.hyperparameter_tuning import HyperparameterTuner
from src.utils import (
    setup_logging,
    save_json,
    print_section_header,
    get_system_info,
)

logger = setup_logging(__name__)


class CreditRiskPipeline:
    """
    End-to-end pipeline for credit default risk prediction.

    This pipeline implements proper ML methodology:
    1. Three-way data split to prevent data leakage
    2. Feature engineering on training data only
    3. Model training with class imbalance handling
    4. Threshold optimization on validation set
    5. Final evaluation on held-out test set

    Attributes:
        data_loader: DataLoader instance
        feature_engineer: FeatureEngineer instance
        model_trainer: ModelTrainer instance
        evaluator: ModelEvaluator instance
        tuner: HyperparameterTuner instance (optional)
    """

    def __init__(
        self,
        enable_tuning: bool = False,
        tuning_n_iter: int = 50,
        tuning_cv_folds: int = 5,
    ):
        """
        Initialize the pipeline.

        Args:
            enable_tuning: Whether to perform hyperparameter tuning
            tuning_n_iter: Number of iterations for random search
            tuning_cv_folds: Number of CV folds for tuning
        """
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.enable_tuning = enable_tuning

        if enable_tuning:
            self.tuner = HyperparameterTuner(
                search_method="random",
                n_iter=tuning_n_iter,
                cv_folds=tuning_cv_folds,
            )
        else:
            self.tuner = None

        # Data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None

        # Results
        self.trained_models = {}
        self.predictions = {}
        self.evaluation_results = {}
        self.optimal_thresholds = {}
        self.is_synthetic_data = False

        logger.info("=" * 80)
        logger.info("CREDIT DEFAULT RISK PREDICTION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Tuning enabled: {enable_tuning}")
        if enable_tuning:
            logger.info(f"  - Iterations: {tuning_n_iter}")
            logger.info(f"  - CV folds: {tuning_cv_folds}")

    def load_and_preprocess_data(self) -> None:
        """
        Load and preprocess the data with proper three-way split.

        Steps:
        1. Load data (real Kaggle dataset or synthetic fallback)
        2. Clean data (missing values, outliers)
        3. Split into train/validation/test (60/20/20)
        4. Apply feature engineering
        5. Scale features (fit on train only)
        """
        print_section_header("DATA LOADING AND PREPROCESSING")

        # Load and clean data
        df = self.data_loader.load_data()
        self.is_synthetic_data = self.data_loader.is_synthetic
        df_clean = self.data_loader.clean_data(df)

        # Feature engineering
        df_engineered = self.feature_engineer.engineer_all_features(df_clean)

        # Encode categorical variables
        df_encoded = self.data_loader.encode_categorical(df_engineered)

        # Prepare features and target
        X, y = self.data_loader.prepare_features_target(df_encoded)
        self.feature_names = self.data_loader.feature_names

        # Three-way split: train / validation / test
        # Validation set is used for threshold optimization
        # Test set is held out for final unbiased evaluation
        (X_train, X_val, X_test, y_train, y_val, y_test) = self.data_loader.split_data(
            X, y
        )

        # Scale features (fit on training data only to prevent leakage)
        X_train_scaled, X_val_scaled, X_test_scaled = self.data_loader.scale_features(
            X_train, X_val, X_test
        )

        # Store processed data
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train.values if hasattr(y_train, "values") else y_train
        self.y_val = y_val.values if hasattr(y_val, "values") else y_val
        self.y_test = y_test.values if hasattr(y_test, "values") else y_test

        logger.info("\nData preprocessing complete:")
        logger.info(f"  Training set: {self.X_train.shape}")
        logger.info(f"  Validation set: {self.X_val.shape}")
        logger.info(f"  Test set: {self.X_test.shape}")
        logger.info(f"  Features: {len(self.feature_names)}")

        if self.is_synthetic_data:
            logger.warning("\n" + "!" * 80)
            logger.warning(
                "WARNING: Using synthetic data. Results are for demonstration only."
            )
            logger.warning(
                "For academic evaluation, please use the real Kaggle dataset."
            )
            logger.warning("!" * 80 + "\n")

    def run_hyperparameter_tuning(self) -> None:
        """
        Run optional hyperparameter tuning.

        Tuning is performed on training data with cross-validation.
        Best parameters are used to reinitialize models.
        """
        if not self.enable_tuning or self.tuner is None:
            logger.info("Hyperparameter tuning disabled, using default parameters.")
            return

        print_section_header("HYPERPARAMETER TUNING")

        logger.info("Starting hyperparameter tuning...")
        logger.info("This may take a while depending on the dataset size and n_iter.")

        # Tune all models
        _tuned_results = self.tuner.tune_all_models(  # noqa: F841
            self.X_train,
            self.y_train,
        )

        # Save tuning results
        self.tuner.save_results(OUTPUT_DIR / "tuning_results.json")

        # Save tuning summary
        summary = self.tuner.get_tuning_summary()
        summary.to_csv(OUTPUT_DIR / "tuning_summary.csv", index=False)

        logger.info("\nTuning complete. Best parameters saved to outputs/")

        # Update model trainer with tuned models
        # Note: We create fresh models with best params for cleaner training
        logger.info("Reinitializing models with tuned parameters...")

    def train_models(self) -> None:
        """
        Train all models on the training set.

        Models are trained with:
        - SMOTE for class imbalance (if enabled)
        - Class weights for additional balance
        - 5-fold cross-validation for performance estimation
        """
        print_section_header("MODEL TRAINING")

        self.trained_models = self.model_trainer.train_all(
            self.X_train,
            self.y_train,
            perform_cv=True,
        )

        logger.info(f"\nTrained {len(self.trained_models)} models successfully")

    def optimize_thresholds(self) -> None:
        """
        Optimize decision thresholds on VALIDATION set.

        This is the proper methodology to prevent data leakage:
        - Thresholds are tuned on validation set
        - Test set remains untouched for final evaluation

        Constraints:
        - Recall (positive) >= MIN_RECALL_POSITIVE
        - Recall (negative) >= MIN_RECALL_NEGATIVE
        """
        print_section_header("THRESHOLD OPTIMIZATION")

        logger.info("Optimizing thresholds on VALIDATION set (not test!)")
        logger.info(
            f"Constraints: recall_pos >= {MIN_RECALL_POSITIVE:.0%}, "
            f"recall_neg >= {MIN_RECALL_NEGATIVE:.0%}"
        )

        self.optimal_thresholds = self.model_trainer.optimize_thresholds_on_validation(
            self.X_val,
            self.y_val,
        )

        # Save optimal thresholds
        thresholds_df = pd.DataFrame(
            [
                {"Model": name, "Optimal_Threshold": threshold}
                for name, threshold in self.optimal_thresholds.items()
            ]
        )
        thresholds_df.to_csv(OUTPUT_DIR / "optimized_thresholds.csv", index=False)

        logger.info("\nOptimal thresholds saved to outputs/optimized_thresholds.csv")

    def generate_predictions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions on the TEST set using optimized thresholds.

        Returns:
            Dictionary mapping model names to (y_pred, y_pred_proba)
        """
        print_section_header("GENERATING PREDICTIONS ON TEST SET")

        predictions = {}

        for name, model in self.trained_models.items():
            # Get probability predictions
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            # Apply optimized threshold
            threshold = self.optimal_thresholds.get(name, 0.5)
            y_pred = (y_pred_proba >= threshold).astype(int)

            predictions[name] = (y_pred, y_pred_proba)

            # Log prediction distribution
            n_pos = np.sum(y_pred == 1)
            n_neg = np.sum(y_pred == 0)
            logger.info(
                f"{name} (threshold={threshold:.4f}): "
                f"{n_pos} positive, {n_neg} negative predictions"
            )

        self.predictions = predictions
        return predictions

    def evaluate_models(self) -> None:
        """
        Evaluate all models on the TEST set.

        This is the final, unbiased evaluation using thresholds
        optimized on the validation set.
        """
        print_section_header("MODEL EVALUATION ON TEST SET")

        logger.info("Evaluating models on held-out TEST set...")
        logger.info("(Thresholds were optimized on validation set)")

        # Evaluate all models
        self.evaluation_results = self.evaluator.evaluate_all_models(
            self.predictions,
            self.y_test,
        )

        # Generate comparison report
        comparison_df = self.evaluator.generate_comparison_report(
            self.evaluation_results,
            save=True,
        )

        # Add threshold information to the report
        comparison_df_with_thresholds = comparison_df.copy()
        comparison_df_with_thresholds["Threshold"] = comparison_df_with_thresholds[
            "Model"
        ].map(self.optimal_thresholds)

        # Reorder columns
        cols = ["Model", "Threshold"] + [
            c
            for c in comparison_df_with_thresholds.columns
            if c not in ["Model", "Threshold"]
        ]
        comparison_df_with_thresholds = comparison_df_with_thresholds[cols]
        comparison_df_with_thresholds.to_csv(
            OUTPUT_DIR / "model_comparison_report.csv", index=False
        )

        logger.info("\nEvaluation complete. Results saved to outputs/")

    def generate_visualizations(self) -> None:
        """Generate all visualization plots."""
        print_section_header("GENERATING VISUALIZATIONS")

        self.evaluator.create_all_visualizations(
            self.predictions,
            self.y_test,
            self.evaluation_results,
            X_train=self.X_train,
            y_train=self.y_train,
            trained_models=self.trained_models,
            feature_names=self.feature_names,
        )

        logger.info("All visualizations saved to outputs/")

    def save_pipeline_summary(self, elapsed_time: float) -> None:
        """
        Save comprehensive pipeline summary.

        Args:
            elapsed_time: Total execution time in seconds
        """
        print_section_header("SAVING PIPELINE SUMMARY")

        # Get system info
        system_info = get_system_info()

        # Prepare summary
        summary = {
            "execution_info": {
                "timestamp": datetime.now().isoformat(),
                "elapsed_time_seconds": round(elapsed_time, 2),
                "elapsed_time_formatted": f"{elapsed_time / 60:.1f} minutes",
                "tuning_enabled": self.enable_tuning,
            },
            "data_info": {
                "is_synthetic": self.is_synthetic_data,
                "train_samples": len(self.X_train),
                "validation_samples": len(self.X_val),
                "test_samples": len(self.X_test),
                "n_features": len(self.feature_names),
                "feature_names": self.feature_names,
                "class_distribution_train": {
                    "negative": int(np.sum(self.y_train == 0)),
                    "positive": int(np.sum(self.y_train == 1)),
                },
            },
            "methodology": {
                "data_split": "train(60%)/validation(20%)/test(20%)",
                "threshold_optimization": "on validation set (prevents data leakage)",
                "final_evaluation": "on held-out test set",
                "recall_constraints": {
                    "min_recall_positive": MIN_RECALL_POSITIVE,
                    "min_recall_negative": MIN_RECALL_NEGATIVE,
                },
            },
            "optimal_thresholds": self.optimal_thresholds,
            "model_performance": {
                name: {
                    "roc_auc": results["metrics"]["roc_auc"],
                    "precision": results["metrics"]["precision"],
                    "recall": results["metrics"]["recall"],
                    "recall_negative": results["metrics"]["recall_negative"],
                    "f1_score": results["metrics"]["f1_score"],
                    "accuracy": results["metrics"]["accuracy"],
                }
                for name, results in self.evaluation_results.items()
            },
            "system_info": system_info,
        }

        # Add warning if synthetic data
        if self.is_synthetic_data:
            summary["WARNING"] = (
                "Results are based on SYNTHETIC data. "
                "For academic evaluation, use the real Kaggle dataset."
            )

        # Save summary
        save_json(summary, OUTPUT_DIR / "pipeline_summary.json")
        logger.info("Pipeline summary saved to outputs/pipeline_summary.json")

        # Print final summary
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 80)
        print(f"\nExecution time: {elapsed_time / 60:.1f} minutes")
        print(f"Tuning enabled: {self.enable_tuning}")
        print(
            f"Data source: {'SYNTHETIC (demo only)' if self.is_synthetic_data else 'Real Kaggle dataset'}"
        )
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("\nKey outputs:")
        print("  - model_comparison_report.csv: Model performance metrics")
        print("  - optimized_thresholds.csv: Per-model decision thresholds")
        print("  - pipeline_summary.json: Complete execution summary")
        print("  - Various plots (ROC, PR, confusion matrices, etc.)")

        # Print best model
        if self.evaluation_results:
            best_model = max(
                self.evaluation_results.items(),
                key=lambda x: x[1]["metrics"]["roc_auc"],
            )
            print(f"\nBest model by ROC-AUC: {best_model[0]}")
            print(f"  ROC-AUC: {best_model[1]['metrics']['roc_auc']:.4f}")
            print(f"  Recall (positive): {best_model[1]['metrics']['recall']:.4f}")
            print(
                f"  Recall (negative): {best_model[1]['metrics']['recall_negative']:.4f}"
            )

        print("=" * 80)

    def run(self) -> None:
        """
        Execute the full pipeline.

        Steps:
        1. Load and preprocess data (with train/val/test split)
        2. Optional hyperparameter tuning (on training data)
        3. Train models
        4. Optimize thresholds (on validation data)
        5. Generate predictions (on test data)
        6. Evaluate models (on test data)
        7. Generate visualizations
        8. Save summary
        """
        start_time = time.time()

        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()

            # Step 2: Optional hyperparameter tuning
            if self.enable_tuning:
                self.run_hyperparameter_tuning()

            # Step 3: Train models
            self.train_models()

            # Step 4: Optimize thresholds on validation set
            self.optimize_thresholds()

            # Step 5: Generate predictions on test set
            self.generate_predictions()

            # Step 6: Evaluate models on test set
            self.evaluate_models()

            # Step 7: Generate visualizations
            self.generate_visualizations()

            # Step 8: Save summary
            elapsed_time = time.time() - start_time
            self.save_pipeline_summary(elapsed_time)

        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Credit Default Risk Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Run with default parameters
  python main.py --tune              Enable hyperparameter tuning
  python main.py --tune --n-iter 30  Tuning with 30 iterations
  python main.py --tune --cv 3       Tuning with 3-fold CV

Methodology:
  - Data is split into train (60%), validation (20%), and test (20%)
  - Thresholds are optimized on validation set (prevents data leakage)
  - Final evaluation is performed on the held-out test set
        """,
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (slower but may improve results)",
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of iterations for random search tuning (default: 50)",
    )

    parser.add_argument(
        "--cv", type=int, default=5, help="Number of CV folds for tuning (default: 5)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Set random seed for reproducibility
    np.random.seed(RANDOM_STATE)

    # Create and run pipeline
    pipeline = CreditRiskPipeline(
        enable_tuning=args.tune,
        tuning_n_iter=args.n_iter,
        tuning_cv_folds=args.cv,
    )

    pipeline.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
