"""
Model evaluation and interpretability module.

This module provides comprehensive evaluation metrics, visualization,
and SHAP-based interpretability for credit risk models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, save_json, print_section_header
from config import (
    OUTPUT_DIR,
    SHAP_SAMPLE_SIZE,
    SHAP_TOP_FEATURES,
    FIGURE_SIZE,
    DPI,
    RANDOM_STATE,
    DEFAULT_COST_FALSE_NEGATIVE,
    DEFAULT_COST_FALSE_POSITIVE,
    DEFAULT_BENEFIT_TRUE_POSITIVE,
    DEFAULT_BENEFIT_TRUE_NEGATIVE,
)

logger = setup_logging(__name__)


class ModelEvaluator:
    """
    Class for evaluating and interpreting machine learning models.

    Provides comprehensive metrics, visualizations, and SHAP analysis.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize ModelEvaluator.

        Args:
            output_dir: Directory for saving outputs (uses default if None)
        """
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        logger.info(f"ModelEvaluator initialized. Output: {self.output_dir}")

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of metric scores
        """
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Recall for both classes (important for bidirectional constraint)
        recall_positive = recall_score(y_true, y_pred, zero_division=0)
        # Specificity = TN / (TN + FP) = recall for negative class
        recall_negative = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Prediction distribution stats
        n_total = len(y_pred)
        n_positive_pred = np.sum(y_pred == 1)
        n_negative_pred = np.sum(y_pred == 0)
        positive_ratio = n_positive_pred / n_total if n_total > 0 else 0

        metrics = {
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_positive,
            "recall_negative": recall_negative,  # Specificity
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "positive_predictions": int(n_positive_pred),
            "negative_predictions": int(n_negative_pred),
            "positive_prediction_ratio": positive_ratio,
        }

        return metrics

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary with all evaluation results
        """
        logger.info(f"Evaluating {model_name}...")

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)

        # Generate classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        results = {
            "model_name": model_name,
            "metrics": metrics,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
        }

        self.results[model_name] = results

        logger.info(f"{model_name} evaluation complete:")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall (Positive): {metrics['recall']:.4f}")
        logger.info(
            f"  Recall (Negative/Specificity): {metrics['recall_negative']:.4f}"
        )
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(
            f"  Predictions: {metrics['positive_predictions']} pos / {metrics['negative_predictions']} neg"
        )

        return results

    def evaluate_all_models(
        self,
        models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models.

        Args:
            models_predictions: Dict mapping model names to (y_pred, y_pred_proba)
            y_true: True labels

        Returns:
            Dictionary of evaluation results for all models
        """
        print_section_header("MODEL EVALUATION")

        all_results = {}

        for model_name, (y_pred, y_pred_proba) in models_predictions.items():
            results = self.evaluate_model(model_name, y_true, y_pred, y_pred_proba)
            all_results[model_name] = results

        return all_results

    def plot_roc_curves(
        self,
        models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
        save: bool = True,
    ) -> None:
        """
        Plot ROC curves for all models.

        Args:
            models_predictions: Dict mapping model names to (y_pred, y_pred_proba)
            y_true: True labels
            save: Whether to save the plot
        """
        plt.figure(figsize=FIGURE_SIZE)

        for model_name, (_, y_pred_proba) in models_predictions.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            plt.plot(
                fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2
            )

        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves - Model Comparison", fontsize=14, pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "roc_curves.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"ROC curves saved to {filepath}")

        plt.close()

    def plot_precision_recall_curves(
        self,
        models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
        save: bool = True,
    ) -> None:
        """
        Plot Precision-Recall curves for all models.

        Args:
            models_predictions: Dict mapping model names to (y_pred, y_pred_proba)
            y_true: True labels
            save: Whether to save the plot
        """
        plt.figure(figsize=FIGURE_SIZE)

        for model_name, (_, y_pred_proba) in models_predictions.items():
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            plt.plot(recall, precision, label=model_name, linewidth=2)

        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curves - Model Comparison", fontsize=14, pad=20)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "precision_recall_curves.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Precision-Recall curves saved to {filepath}")

        plt.close()

    def plot_confusion_matrices(
        self,
        models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
        save: bool = True,
    ) -> None:
        """
        Plot confusion matrices for all models.

        Args:
            models_predictions: Dict mapping model names to (y_pred, y_pred_proba)
            y_true: True labels
            save: Whether to save the plot
        """
        n_models = len(models_predictions)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, (y_pred, _)) in enumerate(models_predictions.items()):
            cm = confusion_matrix(y_true, y_pred)

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                cbar=True,
                square=True,
            )
            axes[idx].set_title(f"{model_name}", fontsize=12, pad=10)
            axes[idx].set_ylabel("True Label", fontsize=10)
            axes[idx].set_xlabel("Predicted Label", fontsize=10)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "confusion_matrices.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Confusion matrices saved to {filepath}")

        plt.close()

    def plot_individual_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save: bool = True,
    ) -> None:
        """
        Plot detailed confusion matrix for a single model.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save: Whether to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))

        # Calculate percentages
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create annotations with both counts and percentages
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=True,
            square=True,
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            linewidths=2,
            linecolor="white",
        )

        plt.title(
            f"Confusion Matrix - {model_name}", fontsize=16, pad=20, fontweight="bold"
        )
        plt.ylabel("True Label", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")

        # Add text annotations for interpretation
        plt.text(
            0.5,
            -0.15,
            f"True Negatives: {cm[0, 0]}  |  False Positives: {cm[0, 1]}\n"
            f"False Negatives: {cm[1, 0]}  |  True Positives: {cm[1, 1]}",
            horizontalalignment="center",
            transform=plt.gca().transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()

        if save:
            filepath = (
                self.output_dir
                / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            )
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Individual confusion matrix saved to {filepath}")

        plt.close()

    def plot_learning_curves(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        cv: int = 5,
        save: bool = True,
    ) -> None:
        """
        Plot learning curves showing training and validation scores.

        Args:
            model: Trained model
            X: Feature data
            y: Target labels
            model_name: Name of the model
            cv: Number of cross-validation folds
            save: Whether to save the plot
        """
        logger.info(f"Generating learning curves for {model_name}...")

        # Use subset of data for faster computation
        max_samples = min(2000, len(X))
        train_sizes = np.linspace(0.1, 1.0, 10)

        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model,
                X[:max_samples],
                y[:max_samples],
                train_sizes=train_sizes,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )

            # Calculate mean and std
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)

            # Create plot
            plt.figure(figsize=(12, 7))

            # Plot learning curves
            plt.plot(
                train_sizes_abs,
                train_mean,
                "o-",
                color="royalblue",
                label="Training Score",
                linewidth=2,
                markersize=8,
            )
            plt.fill_between(
                train_sizes_abs,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.2,
                color="royalblue",
            )

            plt.plot(
                train_sizes_abs,
                val_mean,
                "o-",
                color="orangered",
                label="Validation Score",
                linewidth=2,
                markersize=8,
            )
            plt.fill_between(
                train_sizes_abs,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.2,
                color="orangered",
            )

            plt.xlabel("Number of Training Samples", fontsize=13, fontweight="bold")
            plt.ylabel("ROC-AUC Score", fontsize=13, fontweight="bold")
            plt.title(
                f"Learning Curves - {model_name}",
                fontsize=16,
                pad=20,
                fontweight="bold",
            )
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(True, alpha=0.3, linestyle="--")

            # Add interpretation text
            gap = train_mean[-1] - val_mean[-1]
            if gap > 0.1:
                interpretation = "⚠️ High variance (overfitting)"
                color = "red"
            elif gap > 0.05:
                interpretation = "⚡ Moderate variance"
                color = "orange"
            else:
                interpretation = "✅ Good generalization"
                color = "green"

            plt.text(
                0.02,
                0.98,
                interpretation,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
            )

            plt.tight_layout()

            if save:
                filepath = (
                    self.output_dir
                    / f"learning_curve_{model_name.lower().replace(' ', '_')}.png"
                )
                plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
                logger.info(f"Learning curve saved to {filepath}")

            plt.close()

        except Exception as e:
            logger.error(f"Error generating learning curve for {model_name}: {str(e)}")
            plt.close()

    def plot_prediction_distributions(
        self,
        models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
        save: bool = True,
    ) -> None:
        """
        Plot histograms of prediction probabilities for each class.

        Args:
            models_predictions: Dict mapping model names to (y_pred, y_pred_proba)
            y_true: True labels
            save: Whether to save the plot
        """
        n_models = len(models_predictions)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, (_, y_pred_proba)) in enumerate(
            models_predictions.items()
        ):
            ax = axes[idx]

            # Separate predictions by true class
            proba_no_default = y_pred_proba[y_true == 0]
            proba_default = y_pred_proba[y_true == 1]

            # Plot histograms
            ax.hist(
                proba_no_default,
                bins=50,
                alpha=0.6,
                color="green",
                label="No Default (True)",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.hist(
                proba_default,
                bins=50,
                alpha=0.6,
                color="red",
                label="Default (True)",
                edgecolor="black",
                linewidth=0.5,
            )

            ax.axvline(
                x=0.5,
                color="black",
                linestyle="--",
                linewidth=2,
                label="Decision Threshold",
            )

            ax.set_xlabel(
                "Predicted Probability of Default", fontsize=11, fontweight="bold"
            )
            ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
            ax.set_title(f"{model_name}", fontsize=13, fontweight="bold", pad=10)
            ax.legend(loc="upper center", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Prediction Probability Distributions",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "prediction_distributions.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Prediction distributions saved to {filepath}")

        plt.close()

    def plot_feature_distributions(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str],
        top_n: int = 12,
        save: bool = True,
    ) -> None:
        """
        Plot distributions of top features by class.

        Args:
            X: Feature data
            y: Target labels
            feature_names: List of feature names
            top_n: Number of top features to plot
            save: Whether to save the plot
        """
        logger.info(f"Plotting feature distributions for top {top_n} features...")

        # Select top features to plot
        features_to_plot = feature_names[:top_n]
        n_features = len(features_to_plot)

        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(features_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Get feature data
            if isinstance(X, np.ndarray):
                feature_idx = feature_names.index(feature)
                feature_data = X[:, feature_idx]
            else:
                feature_data = X[feature].values

            # Separate by class
            no_default_data = feature_data[y == 0]
            default_data = feature_data[y == 1]

            # Plot histograms
            ax.hist(
                no_default_data,
                bins=30,
                alpha=0.6,
                color="green",
                label="No Default",
                edgecolor="black",
                linewidth=0.5,
                density=True,
            )
            ax.hist(
                default_data,
                bins=30,
                alpha=0.6,
                color="red",
                label="Default",
                edgecolor="black",
                linewidth=0.5,
                density=True,
            )

            ax.set_xlabel(feature, fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(feature, fontsize=10, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Feature Distributions by Class", fontsize=16, fontweight="bold", y=1.00
        )
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "feature_distributions.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Feature distributions saved to {filepath}")

        plt.close()

    def plot_metrics_comparison(
        self, all_results: Dict[str, Dict[str, Any]], save: bool = True
    ) -> None:
        """
        Plot bar chart comparing metrics across models.

        Args:
            all_results: Evaluation results for all models
            save: Whether to save the plot
        """
        metrics_df = pd.DataFrame(
            {model: results["metrics"] for model, results in all_results.items()}
        ).T

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        metrics_to_plot = ["roc_auc", "precision", "recall", "f1_score", "accuracy"]

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            metrics_df[metric].sort_values(ascending=False).plot(
                kind="bar", ax=ax, color="steelblue", edgecolor="black"
            )
            ax.set_title(metric.upper().replace("_", " "), fontsize=12, pad=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_xlabel("Model", fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=45)

        # Hide the last subplot
        axes[-1].axis("off")

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "metrics_comparison.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Metrics comparison saved to {filepath}")

        plt.close()

    def analyze_feature_importance_shap(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        model_name: str,
        sample_size: Optional[int] = None,
        save: bool = True,
    ) -> None:
        """
        Analyze feature importance using SHAP values.

        Args:
            model: Trained model
            X: Feature data
            feature_names: List of feature names
            model_name: Name of the model
            sample_size: Number of samples for SHAP analysis
            save: Whether to save plots
        """
        if not SHAP_AVAILABLE or shap is None:
            logger.warning(
                f"SHAP not available, skipping SHAP analysis for {model_name}"
            )
            return

        try:
            logger.info(f"Computing SHAP values for {model_name}...")

            sample_size = sample_size or min(SHAP_SAMPLE_SIZE, X.shape[0])
            X_sample = X[:sample_size]

            # Create SHAP explainer
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                # For binary classification, use the positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            except Exception:
                # Fallback to KernelExplainer for models without tree structure
                logger.info(f"Using KernelExplainer for {model_name}")
                background = shap.sample(X, min(100, X.shape[0]))
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:, 1], background
                )
                shap_values = explainer.shap_values(X_sample)

            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
                max_display=SHAP_TOP_FEATURES,
            )
            plt.title(f"SHAP Feature Importance - {model_name}", fontsize=14, pad=20)
            plt.tight_layout()

            if save:
                filepath = (
                    self.output_dir
                    / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
                )
                plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
                logger.info(f"SHAP summary plot saved to {filepath}")

            plt.close()

            # Bar plot of mean absolute SHAP values
            shap_importance = np.abs(shap_values).mean(axis=0)
            importance_df = (
                pd.DataFrame({"feature": feature_names, "importance": shap_importance})
                .sort_values("importance", ascending=False)
                .head(SHAP_TOP_FEATURES)
            )

            plt.figure(figsize=(10, 6))
            plt.barh(
                importance_df["feature"], importance_df["importance"], color="steelblue"
            )
            plt.xlabel("Mean |SHAP Value|", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.title(
                f"Top {SHAP_TOP_FEATURES} Features by SHAP - {model_name}",
                fontsize=14,
                pad=20,
            )
            plt.gca().invert_yaxis()
            plt.tight_layout()

            if save:
                filepath = (
                    self.output_dir
                    / f"shap_bar_{model_name.lower().replace(' ', '_')}.png"
                )
                plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
                logger.info(f"SHAP bar plot saved to {filepath}")

            plt.close()

            logger.info(f"SHAP analysis complete for {model_name}")

        except Exception as e:
            logger.error(f"Error in SHAP analysis for {model_name}: {str(e)}")

    def generate_comparison_report(
        self, all_results: Dict[str, Dict[str, Any]], save: bool = True
    ) -> pd.DataFrame:
        """
        Generate comprehensive comparison report.

        Args:
            all_results: Evaluation results for all models
            save: Whether to save the report

        Returns:
            DataFrame with comparison metrics
        """
        print_section_header("MODEL COMPARISON REPORT")

        comparison_data = []

        for model_name, results in all_results.items():
            metrics = results["metrics"]
            row = {"Model": model_name, **metrics}
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("roc_auc", ascending=False)

        # Print formatted table
        print(comparison_df.to_string(index=False, float_format="%.4f"))

        # Identify best model
        best_model = comparison_df.iloc[0]["Model"]
        best_auc = comparison_df.iloc[0]["roc_auc"]

        print(f"\n{'=' * 80}")
        print(f"BEST PERFORMING MODEL: {best_model}")
        print(f"ROC-AUC Score: {best_auc:.4f}")
        print(f"{'=' * 80}\n")

        if save:
            filepath = self.output_dir / "model_comparison_report.csv"
            comparison_df.to_csv(filepath, index=False)
            logger.info(f"Comparison report saved to {filepath}")

            # Save detailed results as JSON
            json_filepath = self.output_dir / "detailed_results.json"
            save_json(all_results, json_filepath)
            logger.info(f"Detailed results saved to {json_filepath}")

        return comparison_df

    def create_all_visualizations(
        self,
        models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
        all_results: Dict[str, Dict[str, Any]],
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        trained_models: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Create all visualization plots.

        Args:
            models_predictions: Dict mapping model names to predictions
            y_true: True labels
            all_results: Evaluation results for all models
            X_train: Training features (optional, for learning curves)
            y_train: Training labels (optional, for learning curves)
            trained_models: Dictionary of trained models (optional)
            feature_names: List of feature names (optional)
        """
        print_section_header("GENERATING VISUALIZATIONS")

        # Basic evaluation plots
        self.plot_roc_curves(models_predictions, y_true)
        self.plot_precision_recall_curves(models_predictions, y_true)
        self.plot_confusion_matrices(models_predictions, y_true)
        self.plot_metrics_comparison(all_results)

        # Individual confusion matrices for each model
        logger.info("Generating individual confusion matrices...")
        for model_name, (y_pred, _) in models_predictions.items():
            self.plot_individual_confusion_matrix(y_true, y_pred, model_name)

        # Prediction distributions
        logger.info("Generating prediction distribution plots...")
        self.plot_prediction_distributions(models_predictions, y_true)

        # Learning curves (if training data provided)
        if X_train is not None and y_train is not None and trained_models is not None:
            logger.info("Generating learning curves...")
            for model_name, model in trained_models.items():
                try:
                    self.plot_learning_curves(
                        model.model, X_train, y_train, model_name, cv=3
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not generate learning curve for {model_name}: {str(e)}"
                    )

        # Feature distributions (if feature names provided)
        if feature_names is not None and X_train is not None and y_train is not None:
            logger.info("Generating feature distribution plots...")
            # Convert to DataFrame for easier handling
            if isinstance(X_train, np.ndarray):
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
            else:
                X_train_df = X_train
            self.plot_feature_distributions(X_train_df, y_train, feature_names)

        # NEW ADVANCED VISUALIZATIONS
        logger.info("Generating advanced visualizations...")

        # Calibration curves
        self.plot_calibration_curves(models_predictions, y_true)

        # Threshold analysis
        self.plot_threshold_analysis(models_predictions, y_true)

        # Cost-benefit analysis
        self.plot_cost_benefit_analysis(models_predictions, y_true)

        # Radar chart comparison
        self.plot_model_comparison_radar(all_results)

        # Feature correlation heatmap (if feature names provided)
        if feature_names is not None and X_train is not None:
            if isinstance(X_train, np.ndarray):
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
            else:
                X_train_df = X_train
            self.plot_feature_correlation_heatmap(X_train_df, feature_names)

        # SHAP analysis for tree-based models
        if (
            trained_models is not None
            and feature_names is not None
            and X_train is not None
        ):
            logger.info("Generating SHAP analysis for tree-based models...")
            shap_models = ["XGBoost", "Random Forest"]
            for model_name in shap_models:
                if model_name in trained_models:
                    try:
                        self.analyze_feature_importance_shap(
                            trained_models[model_name].model,
                            X_train,
                            feature_names,
                            model_name,
                        )
                    except Exception as e:
                        logger.warning(
                            f"SHAP analysis failed for {model_name}: {str(e)}"
                        )

        logger.info("All visualizations generated successfully")

    def plot_calibration_curves(
        self,
        models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
        n_bins: int = 10,
        save: bool = True,
    ) -> None:
        """
        Plot probability calibration curves for all models.

        Calibration curves show how well the predicted probabilities match
        the actual frequency of the positive class.

        Args:
            models_predictions: Dict mapping model names to (y_pred, y_pred_proba)
            y_true: True labels
            n_bins: Number of bins for calibration
            save: Whether to save the plot
        """
        logger.info("Generating calibration curves...")

        n_models = len(models_predictions)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, (_, y_pred_proba)) in enumerate(
            models_predictions.items()
        ):
            ax = axes[idx]

            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
            )

            # Plot calibration curve
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                marker="o",
                linewidth=2,
                markersize=8,
                label=f"{model_name}",
                color="royalblue",
            )

            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration")

            ax.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
            ax.set_ylabel("Fraction of Positives", fontsize=11, fontweight="bold")
            ax.set_title(
                f"Calibration Curve - {model_name}",
                fontsize=13,
                fontweight="bold",
                pad=10,
            )
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            # Calculate calibration error
            calibration_error = np.mean(
                np.abs(fraction_of_positives - mean_predicted_value)
            )
            ax.text(
                0.05,
                0.95,
                f"Calibration Error: {calibration_error:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Probability Calibration Analysis", fontsize=16, fontweight="bold", y=1.00
        )
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "calibration_curves.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Calibration curves saved to {filepath}")

        plt.close()

    def plot_feature_correlation_heatmap(
        self,
        X: np.ndarray,
        feature_names: List[str],
        top_n: int = 20,
        save: bool = True,
    ) -> None:
        """
        Plot correlation heatmap for top features.

        Args:
            X: Feature data
            feature_names: List of feature names
            top_n: Number of top features to include
            save: Whether to save the plot
        """
        logger.info(f"Generating correlation heatmap for top {top_n} features...")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X

        # Select top features
        features_to_plot = feature_names[: min(top_n, len(feature_names))]
        df_subset = df[features_to_plot]

        # Calculate correlation matrix
        corr_matrix = df_subset.corr()

        # Create heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8},
        )

        plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Features", fontsize=12, fontweight="bold")
        plt.ylabel("Features", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "feature_correlation_heatmap.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Correlation heatmap saved to {filepath}")

        plt.close()

    def plot_threshold_analysis(
        self,
        models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
        save: bool = True,
    ) -> None:
        """
        Plot threshold optimization analysis showing precision, recall, and F1 vs threshold.

        Args:
            models_predictions: Dict mapping model names to (y_pred, y_pred_proba)
            y_true: True labels
            save: Whether to save the plot
        """
        logger.info("Generating threshold analysis plots...")

        n_models = len(models_predictions)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, (_, y_pred_proba)) in enumerate(
            models_predictions.items()
        ):
            ax = axes[idx]

            # Calculate metrics at different thresholds
            thresholds = np.linspace(0, 1, 100)
            precisions = []
            recalls = []
            f1_scores = []

            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)

                prec = precision_score(y_true, y_pred_thresh, zero_division=0)
                rec = recall_score(y_true, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_true, y_pred_thresh, zero_division=0)

                precisions.append(prec)
                recalls.append(rec)
                f1_scores.append(f1)

            # Plot metrics
            ax.plot(
                thresholds, precisions, label="Precision", linewidth=2, color="blue"
            )
            ax.plot(thresholds, recalls, label="Recall", linewidth=2, color="green")
            ax.plot(thresholds, f1_scores, label="F1-Score", linewidth=2, color="red")

            # Find optimal threshold (max F1)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]

            ax.axvline(
                x=optimal_threshold,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Optimal Threshold: {optimal_threshold:.2f}",
            )
            ax.axvline(
                x=0.5,
                color="gray",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                label="Default (0.5)",
            )

            ax.set_xlabel("Decision Threshold", fontsize=11, fontweight="bold")
            ax.set_ylabel("Score", fontsize=11, fontweight="bold")
            ax.set_title(
                f"Threshold Optimization - {model_name}",
                fontsize=13,
                fontweight="bold",
                pad=10,
            )
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Decision Threshold Analysis", fontsize=16, fontweight="bold", y=1.00
        )
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "threshold_analysis.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Threshold analysis saved to {filepath}")

        plt.close()

    def plot_cost_benefit_analysis(
        self,
        models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
        cost_fp: float = None,
        cost_fn: float = None,
        benefit_tp: float = None,
        benefit_tn: float = None,
        save: bool = True,
    ) -> None:
        """
        Plot cost-benefit analysis for different decision thresholds.

        Cost parameters are configured in config.py with documented justifications.
        Default ratio FN/FP = 5:1 reflects that missed defaults are typically
        more costly than rejecting good customers.

        Args:
            models_predictions: Dict mapping model names to (y_pred, y_pred_proba)
            y_true: True labels
            cost_fp: Cost of false positive (default from config)
            cost_fn: Cost of false negative (default from config)
            benefit_tp: Benefit of true positive (default from config)
            benefit_tn: Benefit of true negative (default from config)
            save: Whether to save the plot
        """
        # Use defaults from config if not provided
        cost_fp = cost_fp if cost_fp is not None else DEFAULT_COST_FALSE_POSITIVE
        cost_fn = cost_fn if cost_fn is not None else DEFAULT_COST_FALSE_NEGATIVE
        benefit_tp = (
            benefit_tp if benefit_tp is not None else DEFAULT_BENEFIT_TRUE_POSITIVE
        )
        benefit_tn = (
            benefit_tn if benefit_tn is not None else DEFAULT_BENEFIT_TRUE_NEGATIVE
        )
        logger.info("Generating cost-benefit analysis...")

        n_models = len(models_predictions)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, (_, y_pred_proba)) in enumerate(
            models_predictions.items()
        ):
            ax = axes[idx]

            # Calculate net benefit at different thresholds
            thresholds = np.linspace(0, 1, 100)
            net_benefits = []

            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)

                cm = confusion_matrix(y_true, y_pred_thresh)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

                # Calculate total benefit
                total_benefit = (
                    tp * benefit_tp + tn * benefit_tn - fp * cost_fp - fn * cost_fn
                )

                net_benefits.append(total_benefit)

            # Plot net benefit
            ax.plot(thresholds, net_benefits, linewidth=2.5, color="darkgreen")

            # Find optimal threshold
            optimal_idx = np.argmax(net_benefits)
            optimal_threshold = thresholds[optimal_idx]
            max_benefit = net_benefits[optimal_idx]

            ax.axvline(
                x=optimal_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Optimal: {optimal_threshold:.2f}",
            )
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            ax.set_xlabel("Decision Threshold", fontsize=11, fontweight="bold")
            ax.set_ylabel("Net Benefit ($)", fontsize=11, fontweight="bold")
            ax.set_title(
                f"Cost-Benefit Analysis - {model_name}",
                fontsize=13,
                fontweight="bold",
                pad=10,
            )
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add annotation for max benefit
            ax.annotate(
                f"Max Benefit: ${max_benefit:.0f}",
                xy=(optimal_threshold, max_benefit),
                xytext=(10, 20),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Cost-Benefit Analysis\n(FP=${cost_fp}, FN=${cost_fn}, TP=${benefit_tp}, TN=${benefit_tn})",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "cost_benefit_analysis.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Cost-benefit analysis saved to {filepath}")

        plt.close()

    def plot_model_comparison_radar(
        self,
        all_results: Dict[str, Dict[str, Any]],
        save: bool = True,
    ) -> None:
        """
        Create radar chart comparing models across multiple metrics.

        Args:
            all_results: Evaluation results for all models
            save: Whether to save the plot
        """
        logger.info("Generating model comparison radar chart...")

        # Prepare data
        metrics_to_plot = ["roc_auc", "precision", "recall", "f1_score", "accuracy"]
        models = list(all_results.keys())

        # Create figure with polar projection
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)

        # Number of metrics
        num_metrics = len(metrics_to_plot)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        for idx, (model_name, color) in enumerate(zip(models, colors)):
            values = [
                all_results[model_name]["metrics"][metric] for metric in metrics_to_plot
            ]
            values += values[:1]  # Complete the circle

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=model_name,
                color=color,
                markersize=8,
            )
            ax.fill(angles, values, alpha=0.15, color=color)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [m.replace("_", " ").title() for m in metrics_to_plot],
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.title(
            "Model Performance Comparison - Radar Chart",
            fontsize=16,
            fontweight="bold",
            pad=30,
        )
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "model_comparison_radar.png"
            plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
            logger.info(f"Radar chart saved to {filepath}")

        plt.close()


def main():
    """Main function for testing ModelEvaluator."""
    # Create sample predictions for testing
    np.random.seed(42)
    n_samples = 1000

    y_true = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

    models_predictions = {
        "Model A": (
            np.random.choice([0, 1], size=n_samples),
            np.random.rand(n_samples),
        ),
        "Model B": (
            np.random.choice([0, 1], size=n_samples),
            np.random.rand(n_samples),
        ),
    }

    evaluator = ModelEvaluator()
    all_results = evaluator.evaluate_all_models(models_predictions, y_true)

    # Test new visualization functions
    evaluator.plot_prediction_distributions(models_predictions, y_true)

    # Test individual confusion matrices
    for name, (y_pred, _) in models_predictions.items():
        evaluator.plot_individual_confusion_matrix(y_true, y_pred, name)

    evaluator.generate_comparison_report(all_results)


if __name__ == "__main__":
    main()
