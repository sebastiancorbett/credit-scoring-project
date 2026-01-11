"""
Utility functions for the Credit Default Risk Prediction project.

This module provides helper functions for logging, file operations,
and common data transformations.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import LOG_LEVEL, LOG_FORMAT


def setup_logging(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        name: Logger name
        level: Logging level (default from config)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def save_pickle(obj: Any, filepath: Path) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Path) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_json(data: Dict, filepath: Path, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
        indent: JSON indentation level
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: Path) -> Dict:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        filepath: Path to save file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def calculate_class_distribution(y: np.ndarray) -> Dict[str, float]:
    """
    Calculate class distribution statistics.

    Args:
        y: Target labels

    Returns:
        Dictionary with class distribution metrics
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    distribution = {
        "total_samples": total,
        "class_counts": dict(zip(unique.tolist(), counts.tolist())),
        "class_percentages": dict(
            zip(unique.tolist(), (counts / total * 100).tolist())
        ),
        "imbalance_ratio": float(counts.max() / counts.min()),
    }
    return distribution


def print_class_distribution(y: np.ndarray, title: str = "Class Distribution") -> None:
    """
    Print formatted class distribution.

    Args:
        y: Target labels
        title: Title for the distribution report
    """
    dist = calculate_class_distribution(y)
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")
    print(f"Total Samples: {dist['total_samples']}")
    print("\nClass Breakdown:")
    for cls, count in dist["class_counts"].items():
        pct = dist["class_percentages"][cls]
        print(f"  Class {cls}: {count:,} ({pct:.2f}%)")
    print(f"\nImbalance Ratio: {dist['imbalance_ratio']:.2f}:1")
    print(f"{'=' * 50}\n")


def create_correlation_plot(
    df: pd.DataFrame, output_path: Optional[Path] = None, figsize: tuple = (14, 10)
) -> None:
    """
    Create and save correlation heatmap.

    Args:
        df: DataFrame with features
        output_path: Path to save plot (optional)
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    correlation_matrix = df.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=16, pad=20)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def format_metric(value: float, metric_name: str) -> str:
    """
    Format metric value for display.

    Args:
        value: Metric value
        metric_name: Name of the metric

    Returns:
        Formatted string
    """
    if metric_name.lower() in [
        "auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]:
        return f"{value:.4f}"
    else:
        return f"{value:.2f}"


def print_section_header(title: str, width: int = 80) -> None:
    """
    Print formatted section header.

    Args:
        title: Section title
        width: Total width of header
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}\n")


def validate_dataframe(
    df: pd.DataFrame, required_columns: Optional[List[str]] = None
) -> bool:
    """
    Validate DataFrame structure and content.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, raises ValueError otherwise
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    return True


def get_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get DataFrame memory usage statistics.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with memory usage statistics
    """
    mem_usage = df.memory_usage(deep=True)
    total_mb = mem_usage.sum() / 1024**2

    return {
        "total_mb": f"{total_mb:.2f} MB",
        "per_column_kb": {col: f"{mem_usage[col] / 1024:.2f} KB" for col in df.columns},
    }


def get_system_info() -> Dict[str, str]:
    """
    Get system information for reproducibility documentation.

    Returns:
        Dictionary with system information
    """
    import platform
    import sys

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "platform_release": platform.release(),
        "processor": platform.processor() or "Unknown",
    }
