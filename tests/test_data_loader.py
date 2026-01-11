"""
Unit tests for data_loader module.

Tests data loading, cleaning, encoding, and splitting functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""

    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance for testing."""
        return DataLoader()

    @pytest.fixture
    def sample_data(self):
        """Create sample credit risk data."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "SeriousDlqin2yrs": np.random.choice([0, 1], 100, p=[0.9, 0.1]),
                "age": np.random.randint(21, 80, 100),
                "DebtRatio": np.random.rand(100),
                "MonthlyIncome": np.random.randint(1000, 10000, 100),
                "NumberOfOpenCreditLinesAndLoans": np.random.randint(0, 20, 100),
            }
        )

    def test_initialization(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader is not None
        assert data_loader.scaler is not None
        assert isinstance(data_loader.label_encoders, dict)

    def test_load_data(self, data_loader):
        """Test data loading (creates synthetic data)."""
        df = data_loader.load_data()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "SeriousDlqin2yrs" in df.columns

    def test_clean_data(self, data_loader, sample_data):
        """Test data cleaning."""
        sample_data.loc[0:5, "MonthlyIncome"] = np.nan
        df_clean = data_loader.clean_data(sample_data)
        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) > 0

    def test_encode_categorical(self, data_loader):
        """Test categorical encoding."""
        df = pd.DataFrame({"cat_col": ["A", "B", "A", "C"], "num_col": [1, 2, 3, 4]})
        df_encoded = data_loader.encode_categorical(df, ["cat_col"])
        assert "cat_col" in df_encoded.columns
        assert df_encoded["cat_col"].dtype in [np.int32, np.int64]

    def test_prepare_features_target(self, data_loader, sample_data):
        """Test feature and target separation."""
        X, y = data_loader.prepare_features_target(sample_data)
        assert X.shape[0] == sample_data.shape[0]
        assert X.shape[1] == sample_data.shape[1] - 1
        assert len(y) == len(sample_data)
        assert "SeriousDlqin2yrs" not in X.columns

    def test_split_data(self, data_loader, sample_data):
        """Test data splitting (three-way: train/val/test)."""
        X, y = data_loader.prepare_features_target(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)
        assert len(X_train) >= len(X_test)

    def test_scale_features(self, data_loader, sample_data):
        """Test feature scaling (three-way split)."""
        X, y = data_loader.prepare_features_target(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
        X_train_scaled, X_val_scaled, X_test_scaled = data_loader.scale_features(
            X_train, X_val, X_test
        )
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        assert X_test_scaled.shape == X_test.shape
        assert isinstance(X_train_scaled, np.ndarray)
        assert np.abs(X_train_scaled.mean()) < 0.5
        assert np.abs(X_train_scaled.std() - 1.0) < 0.5

    def test_full_pipeline(self, data_loader):
        """Test complete preprocessing pipeline (three-way split)."""
        result = data_loader.get_full_pipeline()
        assert len(result) == 9
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
        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) == X_train.shape[0]
        assert len(y_val) == X_val.shape[0]
        assert len(y_test) == X_test.shape[0]


class TestDataLoaderEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_target_column(self):
        """Test error when target column is missing."""
        loader = DataLoader()
        df = pd.DataFrame({"feature1": [1, 2, 3]})
        with pytest.raises(ValueError):
            loader.prepare_features_target(df, target_col="NonExistentTarget")

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        loader = DataLoader()
        df = pd.DataFrame()
        df_clean = loader.clean_data(df)
        assert len(df_clean) == 0
