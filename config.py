"""
Configuration settings for the Credit Default Risk Prediction project.

This module contains all configuration parameters, paths, and hyperparameters
used throughout the project. All parameters are documented with justifications
based on literature and best practices.
"""

from pathlib import Path
from typing import Dict, Any

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================
# Using Kaggle's "Give Me Some Credit" dataset
# Reference: https://www.kaggle.com/c/GiveMeSomeCredit
DATASET_NAME = "give_me_some_credit"
DATASET_URL = "https://www.kaggle.com/c/GiveMeSomeCredit/data"
DATA_FILE = DATA_DIR / "cs-training.csv"

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
# Fixed seed for reproducibility (standard practice in ML research)
RANDOM_STATE = 42

# =============================================================================
# DATA SPLIT CONFIGURATION
# =============================================================================
# Three-way split: train (60%) / validation (20%) / test (20%)
# Justification:
# - Validation set is required for threshold optimization without data leakage
# - 60/20/20 is a standard split ratio (Hastie et al., ESL 2009)
# - Test set remains untouched until final evaluation
TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2

# Cross-validation settings
# 5-fold CV is standard for datasets of this size (Kohavi, 1995)
CV_FOLDS = 5

# =============================================================================
# CLASS IMBALANCE HANDLING
# =============================================================================
# Strategy options: "SMOTE", "class_weight", "both"
# "both" combines SMOTE oversampling with class-weighted loss
IMBALANCE_STRATEGY = "both"

# SMOTE sampling strategy
# Justification: We don't want 1:1 ratio (can cause overfitting on synthetic samples)
# A ratio of 0.5 means minority class will be 50% of majority class size
# This is a moderate oversampling that balances recall improvement vs overfitting risk
# Reference: Chawla et al. (2002) - original SMOTE paper recommends conservative ratios
SMOTE_SAMPLING_STRATEGY = 0.5

# =============================================================================
# RECALL CONSTRAINTS
# =============================================================================
# Minimum recall requirements for threshold optimization
# Justification for 30% minimum:
# - In credit risk, missing defaults (FN) is costly, but rejecting good customers (FP) also hurts
# - 30% ensures neither class is completely ignored
# - This creates a meaningful trade-off between precision and recall
# - Higher minimum (e.g., 50%) would force too many false positives
MIN_RECALL_POSITIVE = 0.30  # Minimum recall for default class (catching defaulters)
MIN_RECALL_NEGATIVE = (
    0.30  # Minimum recall for non-default class (not rejecting good customers)
)

# Target recall range for balanced predictions
# We aim for 35-75% recall on positive class for balanced results
# This prevents extreme predictions in either direction
TARGET_RECALL_POSITIVE_MIN = 0.35
TARGET_RECALL_POSITIVE_MAX = 0.75  # Allow up to 75% to give optimizer room
TARGET_RECALL_NEGATIVE_MIN = 0.35
TARGET_RECALL_NEGATIVE_MAX = (
    0.90  # Higher allowed for negative class due to class imbalance
)

# Balance penalty weight - penalizes large differences between recalls
# Higher values enforce more balanced predictions
BALANCE_PENALTY_WEIGHT = 0.15

# =============================================================================
# THRESHOLD OPTIMIZATION WEIGHTS
# =============================================================================
# Composite score for threshold selection:
# score = w_f1 * F1 + w_recall_pos * Recall_pos + w_recall_neg * Recall_neg
#
# Justification for equal weighting (0.33 each):
# - F1 balances precision and recall naturally
# - Equal weights for both recalls ensure bidirectional constraint satisfaction
# - No business-specific preference given in requirements, so symmetric weighting is appropriate
# - Sum = 1.0 for interpretability
THRESHOLD_WEIGHT_F1 = 0.34
THRESHOLD_WEIGHT_RECALL_POS = 0.33
THRESHOLD_WEIGHT_RECALL_NEG = 0.33

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Default hyperparameters based on literature and common practice
# These serve as starting points; hyperparameter tuning can optimize further

MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        # max_iter: Increased for convergence on complex datasets
        "max_iter": 1000,
        # class_weight: "balanced" automatically adjusts weights inversely proportional
        # to class frequencies, equivalent to setting w_i = n_samples / (n_classes * n_samples_i)
        # Reference: King & Zeng (2001) - Logistic Regression in Rare Events Data
        "class_weight": "balanced",
        # solver: LBFGS is efficient for medium-sized datasets with L2 regularization
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "random_forest": {
        # n_estimators: 200 trees provide stable predictions without excessive computation
        # Reference: Breiman (2001) - more trees reduce variance but have diminishing returns
        "n_estimators": 200,
        # max_depth: Limited to 15 to prevent overfitting on imbalanced data
        # Deeper trees can memorize minority class
        "max_depth": 15,
        # min_samples_split/leaf: Conservative values to prevent overfitting
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        # class_weight: "balanced" for imbalanced classification
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 200,
        # max_depth: 6 is XGBoost default, prevents overfitting
        # Reference: Chen & Guestrin (2016) - XGBoost paper
        "max_depth": 6,
        # learning_rate: 0.1 is standard, balances speed and accuracy
        "learning_rate": 0.1,
        # subsample/colsample: 0.8 adds regularization through randomness
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # scale_pos_weight: Will be computed dynamically based on class ratio
        # Formula: sum(negative) / sum(positive)
        # For ~7% positive rate: scale_pos_weight ≈ 13
        # We use a moderate value to balance recall without hurting precision too much
        "scale_pos_weight": None,  # Computed dynamically in training
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    },
    "svm": {
        # RBF kernel: Good default for non-linear classification
        # Linear kernel is faster but may underfit
        "kernel": "rbf",
        # C: Regularization parameter, moderate value
        "C": 1.0,
        # gamma: 'scale' uses 1 / (n_features * X.var())
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,  # Required for predict_proba
        "random_state": RANDOM_STATE,
        "max_iter": 5000,
    },
    "neural_network": {
        # Architecture: Two hidden layers with decreasing size
        # 100->50 is a common pattern for tabular data
        "hidden_layer_sizes": (100, 50),
        # ReLU: Standard activation, avoids vanishing gradient
        "activation": "relu",
        # Adam: Adaptive learning rate optimizer, robust default
        "solver": "adam",
        # alpha: L2 regularization, prevents overfitting
        "alpha": 0.001,
        "batch_size": 256,
        # adaptive learning rate: Reduces LR when loss plateaus
        "learning_rate": "adaptive",
        "max_iter": 500,
        # Early stopping: Prevents overfitting
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": RANDOM_STATE,
    },
}

# =============================================================================
# FEATURE ENGINEERING SETTINGS
# =============================================================================
FEATURE_CONFIG = {
    "debt_to_income_ratio": True,
    "loan_to_asset_ratio": True,
    "monthly_payment_ratio": True,
    "age_group_encoding": True,
    "credit_utilization_categories": True,
}

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================
EVALUATION_METRICS = [
    "roc_auc",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
]

# =============================================================================
# COST-BENEFIT ANALYSIS PARAMETERS
# =============================================================================
# These costs are illustrative defaults for visualization purposes.
# In production, they should be calibrated to actual portfolio data.
#
# Justification for default values:
# - FN cost (500): Missing a defaulter leads to loan loss (typically 20-40% of loan value)
# - FP cost (100): Rejecting a good customer loses potential interest income
# - Ratio FN/FP = 5:1 reflects that defaults are more costly than lost opportunities
# - Reference: Hand & Henley (1997) - Statistical Classification Methods in Consumer Credit
#
# These can be overridden via command-line arguments or environment variables
DEFAULT_COST_FALSE_NEGATIVE = 500  # Cost of missing a defaulter
DEFAULT_COST_FALSE_POSITIVE = 100  # Cost of rejecting a good customer
DEFAULT_BENEFIT_TRUE_POSITIVE = 50  # Benefit of correctly identifying defaulter
DEFAULT_BENEFIT_TRUE_NEGATIVE = 10  # Benefit of approving good customer

# =============================================================================
# SHAP ANALYSIS SETTINGS
# =============================================================================
SHAP_SAMPLE_SIZE = 500  # Samples for SHAP (computational constraint)
SHAP_TOP_FEATURES = 10  # Top features to display

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
FIGURE_SIZE = (12, 8)
DPI = 300
STYLE = "seaborn-v0_8-darkgrid"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def compute_scale_pos_weight(y):
    """
    Compute optimal scale_pos_weight for XGBoost based on class distribution.

    Formula: sum(negative instances) / sum(positive instances)

    Args:
        y: Target array

    Returns:
        Computed scale_pos_weight value
    """
    import numpy as np

    n_positive = np.sum(y == 1)
    n_negative = np.sum(y == 0)

    if n_positive == 0:
        return 1.0

    # Compute raw ratio
    raw_ratio = n_negative / n_positive

    # Cap at reasonable value to avoid extreme predictions
    # A cap of 10 is used to prevent overcompensation
    return min(raw_ratio, 10.0)
