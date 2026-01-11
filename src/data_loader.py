"""
Data loading and preprocessing module.

This module handles loading credit risk data, cleaning, encoding categorical
variables, and preparing data for model training with proper train/validation/test splits.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, print_class_distribution
from config import RANDOM_STATE, TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, DATA_FILE

logger = setup_logging(__name__)


class DataLoader:
    """
    Class for loading and preprocessing credit risk data.

    This class handles data loading, cleaning, encoding, and splitting
    with proper validation and error handling.

    The split strategy follows best practices:
    - Training set (60%): Used for model fitting
    - Validation set (20%): Used for threshold optimization and hyperparameter tuning
    - Test set (20%): Held out for final unbiased evaluation
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to data file (uses config default if None)
        """
        self.data_path = data_path or DATA_FILE
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_name = None
        self._is_synthetic = False
        logger.info(f"DataLoader initialized with path: {self.data_path}")

    @property
    def is_synthetic(self) -> bool:
        """Return whether synthetic data was used."""
        return self._is_synthetic

    def load_data(self) -> pd.DataFrame:
        """
        Load data from file with error handling.

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        try:
            if not self.data_path.exists():
                logger.warning(f"Data file not found: {self.data_path}")
                logger.warning("=" * 60)
                logger.warning("IMPORTANT: Using synthetic data for demonstration.")
                logger.warning("Results should NOT be used for academic evaluation.")
                logger.warning("Please download the Kaggle dataset for real results.")
                logger.warning("=" * 60)
                self._is_synthetic = True
                return self._create_sample_data()

            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            self._is_synthetic = False
            logger.info(f"Real dataset loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.warning("Creating synthetic sample data as fallback")
            self._is_synthetic = True
            return self._create_sample_data()

    def _create_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Create synthetic credit risk data for demonstration.

        Note: This data is for demonstration purposes only.
        Academic evaluation should use the real Kaggle dataset.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Synthetic DataFrame
        """
        np.random.seed(RANDOM_STATE)

        # Target: ~7% default rate (typical for credit data)
        default_rate = 0.07

        # Generate synthetic features mimicking real credit data
        data = {
            "SeriousDlqin2yrs": np.random.choice(
                [0, 1], size=n_samples, p=[1 - default_rate, default_rate]
            ),
            "RevolvingUtilizationOfUnsecuredLines": np.clip(
                np.random.exponential(0.3, n_samples), 0, 5
            ),
            "age": np.random.randint(21, 90, n_samples),
            "NumberOfTime30-59DaysPastDueNotWorse": np.random.poisson(0.3, n_samples),
            "DebtRatio": np.clip(np.random.exponential(0.5, n_samples), 0, 10),
            "MonthlyIncome": np.random.lognormal(8.5, 0.8, n_samples),
            "NumberOfOpenCreditLinesAndLoans": np.random.randint(0, 30, n_samples),
            "NumberOfTimes90DaysLate": np.random.poisson(0.2, n_samples),
            "NumberRealEstateLoansOrLines": np.random.randint(0, 10, n_samples),
            "NumberOfTime60-89DaysPastDueNotWorse": np.random.poisson(0.2, n_samples),
            "NumberOfDependents": np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
        }

        # Add realistic correlation with target
        mask = data["SeriousDlqin2yrs"] == 1
        # Defaulters tend to have more delinquencies
        data["NumberOfTime30-59DaysPastDueNotWorse"][mask] += np.random.randint(
            0, 3, mask.sum()
        )
        # Defaulters tend to have higher debt ratios
        data["DebtRatio"][mask] *= 1.5
        # Defaulters tend to have higher utilization
        data["RevolvingUtilizationOfUnsecuredLines"][mask] *= 1.3

        df = pd.DataFrame(data)
        logger.info(f"Created synthetic dataset with shape: {df.shape}")
        logger.info(f"Default rate: {df['SeriousDlqin2yrs'].mean():.2%}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.

        Outlier handling strategy:
        - Uses IQR method with 3x multiplier (conservative)
        - Outliers are capped at boundaries (not replaced with median)
        - This preserves more information while limiting extreme values

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()

        # Log initial missing values
        missing_before = df_clean.isnull().sum()
        if missing_before.sum() > 0:
            logger.info(
                f"Missing values before cleaning:\n{missing_before[missing_before > 0]}"
            )

        # Handle missing values: Fill numerical columns with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                logger.debug(
                    f"Filled {col} missing values with median: {median_val:.2f}"
                )

        # Handle outliers using IQR method with capping (not replacement)
        # Capping preserves rank ordering while limiting extreme values
        outlier_cols = [
            "RevolvingUtilizationOfUnsecuredLines",
            "DebtRatio",
            "MonthlyIncome",
        ]

        for col in outlier_cols:
            if col in df_clean.columns:
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr

                # Cap outliers at boundaries (winsorization)
                n_lower = (df_clean[col] < lower_bound).sum()
                n_upper = (df_clean[col] > upper_bound).sum()

                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

                if n_lower + n_upper > 0:
                    logger.debug(
                        f"Capped {n_lower + n_upper} outliers in {col} "
                        f"(lower: {n_lower}, upper: {n_upper})"
                    )

        # Remove duplicates
        n_duplicates = df_clean.duplicated().sum()
        if n_duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            logger.info(f"Removed {n_duplicates} duplicate rows")

        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        return df_clean

    def encode_categorical(
        self, df: pd.DataFrame, categorical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical columns to encode

        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()

        if categorical_cols is None:
            categorical_cols = df_encoded.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        if not categorical_cols:
            logger.info("No categorical columns to encode")
            return df_encoded

        logger.info(f"Encoding categorical columns: {categorical_cols}")

        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                logger.debug(f"Encoded {col} with {len(le.classes_)} unique values")

        return df_encoded

    def prepare_features_target(
        self, df: pd.DataFrame, target_col: str = "SeriousDlqin2yrs"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.

        Args:
            df: Input DataFrame
            target_col: Name of target column

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.feature_names = X.columns.tolist()
        self.target_name = target_col

        logger.info(
            f"Prepared features (shape: {X.shape}) and target (shape: {y.shape})"
        )
        print_class_distribution(y.values, "Target Variable Distribution")

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_size: float = TRAIN_SIZE,
        validation_size: float = VALIDATION_SIZE,
        test_size: float = TEST_SIZE,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split data into training, validation, and test sets with stratification.

        This implements a proper three-way split to prevent data leakage:
        - Training: Used for model fitting
        - Validation: Used for threshold optimization (prevents leakage to test)
        - Test: Held out for final unbiased evaluation

        Args:
            X: Features DataFrame
            y: Target Series
            train_size: Proportion of training set (default: 0.6)
            validation_size: Proportion of validation set (default: 0.2)
            test_size: Proportion of test set (default: 0.2)

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Validate split proportions
        total = train_size + validation_size + test_size
        if not np.isclose(total, 1.0):
            logger.warning(f"Split proportions sum to {total}, normalizing...")
            train_size /= total
            validation_size /= total
            test_size /= total

        logger.info(
            f"Splitting data: train={train_size:.0%}, "
            f"val={validation_size:.0%}, test={test_size:.0%}"
        )

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )

        # Second split: separate validation from training
        # Adjust validation size relative to remaining data
        val_size_adjusted = validation_size / (train_size + validation_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=RANDOM_STATE,
            stratify=y_temp,
        )

        logger.info(
            f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
        )
        print_class_distribution(y_train.values, "Training Set Distribution")
        print_class_distribution(y_val.values, "Validation Set Distribution")
        print_class_distribution(y_test.values, "Test Set Distribution")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def split_data_two_way(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Legacy two-way split for backward compatibility.

        Note: For proper methodology, use split_data() which provides
        train/validation/test split.

        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of test set

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.warning(
            "Using two-way split. Consider using split_data() for "
            "train/validation/test split to prevent data leakage."
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print_class_distribution(y_train.values, "Training Set Distribution")
        print_class_distribution(y_test.values, "Test Set Distribution")

        return X_train, X_test, y_train, y_test

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
    ) -> Tuple:
        """
        Scale features using StandardScaler fitted on training data only.

        This prevents data leakage by fitting the scaler only on training data
        and applying the same transformation to validation and test sets.

        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)

        Returns:
            Tuple of scaled arrays (X_train_scaled, [X_val_scaled], [X_test_scaled])
        """
        logger.info("Scaling features (fit on training data only)...")

        X_train_scaled = self.scaler.fit_transform(X_train)

        result = [X_train_scaled]

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            result.append(X_val_scaled)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            result.append(X_test_scaled)

        logger.info("Feature scaling complete")

        return tuple(result) if len(result) > 1 else X_train_scaled

    def get_full_pipeline(
        self, target_col: str = "SeriousDlqin2yrs", scale: bool = True
    ) -> Tuple:
        """
        Execute full data loading and preprocessing pipeline.

        Args:
            target_col: Name of target column
            scale: Whether to scale features

        Returns:
            Tuple of processed data and splits
        """
        logger.info("=" * 80)
        logger.info("Starting full data preprocessing pipeline")
        logger.info("=" * 80)

        # Load data
        df = self.load_data()

        # Clean data
        df_clean = self.clean_data(df)

        # Encode categorical variables
        df_encoded = self.encode_categorical(df_clean)

        # Prepare features and target
        X, y = self.prepare_features_target(df_encoded, target_col)

        # Split data (three-way)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Scale features if requested
        if scale:
            X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
                X_train, X_val, X_test
            )
            result = (
                X_train_scaled,
                X_val_scaled,
                X_test_scaled,
                y_train.values,
                y_val.values,
                y_test.values,
                X_train,
                X_val,
                X_test,
            )
        else:
            result = (
                X_train.values,
                X_val.values,
                X_test.values,
                y_train.values,
                y_val.values,
                y_test.values,
            )

        logger.info("=" * 80)
        logger.info("Data preprocessing pipeline complete")
        logger.info("=" * 80)

        return result


def main():
    """Main function for testing DataLoader."""
    loader = DataLoader()
    result = loader.get_full_pipeline()

    if len(result) == 9:
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            X_train_df,
            X_val_df,
            X_test_df,
        ) = result
        logger.info("\nFinal shapes:")
        logger.info(f"X_train: {X_train.shape}")
        logger.info(f"X_val: {X_val.shape}")
        logger.info(f"X_test: {X_test.shape}")
        logger.info(f"y_train: {y_train.shape}")
        logger.info(f"y_val: {y_val.shape}")
        logger.info(f"y_test: {y_test.shape}")


if __name__ == "__main__":
    main()
