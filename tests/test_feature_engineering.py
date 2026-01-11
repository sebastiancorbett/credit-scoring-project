"""
Unit tests for feature_engineering module.

Tests feature creation and transformation functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance for testing."""
        return FeatureEngineer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for feature engineering."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "age": np.random.randint(21, 80, 100),
                "MonthlyIncome": np.random.randint(2000, 10000, 100),
                "DebtRatio": np.random.rand(100),
                "NumberOfOpenCreditLinesAndLoans": np.random.randint(0, 20, 100),
                "NumberRealEstateLoansOrLines": np.random.randint(0, 5, 100),
                "RevolvingUtilizationOfUnsecuredLines": np.random.rand(100),
                "NumberOfTime30-59DaysPastDueNotWorse": np.random.randint(0, 5, 100),
                "NumberOfTime60-89DaysPastDueNotWorse": np.random.randint(0, 3, 100),
                "NumberOfTimes90DaysLate": np.random.randint(0, 2, 100),
                "NumberOfDependents": np.random.randint(0, 5, 100),
            }
        )

    def test_initialization(self, feature_engineer):
        """Test FeatureEngineer initialization."""
        assert feature_engineer is not None
        assert feature_engineer.config is not None
        assert isinstance(feature_engineer.engineered_features, list)

    def test_create_debt_to_income_ratio(self, feature_engineer, sample_data):
        """Test debt-to-income ratio feature creation."""
        df_new = feature_engineer.create_debt_to_income_ratio(sample_data)

        assert "DebtToIncomeRatio" in df_new.columns
        assert "DebtToIncomeCategory" in df_new.columns
        assert len(df_new) == len(sample_data)

    def test_create_loan_to_asset_ratio(self, feature_engineer, sample_data):
        """Test loan-to-asset ratio feature creation."""
        df_new = feature_engineer.create_loan_to_asset_ratio(sample_data)

        assert "LoanToAssetProxy" in df_new.columns
        assert "RealEstateLoanRatio" in df_new.columns
        assert len(df_new) == len(sample_data)

    def test_create_monthly_payment_ratio(self, feature_engineer, sample_data):
        """Test monthly payment ratio feature creation."""
        df_new = feature_engineer.create_monthly_payment_ratio(sample_data)

        assert "MonthlyDebtPayment" in df_new.columns
        assert "DiscretionaryIncome" in df_new.columns
        assert len(df_new) == len(sample_data)

    def test_create_age_group_features(self, feature_engineer, sample_data):
        """Test age group feature creation."""
        df_new = feature_engineer.create_age_group_features(sample_data)

        assert "AgeGroup" in df_new.columns
        assert "AgeSquared" in df_new.columns
        assert "IsSenior" in df_new.columns
        assert df_new["IsSenior"].isin([0, 1]).all()

    def test_create_credit_utilization_features(self, feature_engineer, sample_data):
        """Test credit utilization feature creation."""
        df_new = feature_engineer.create_credit_utilization_features(sample_data)

        assert "CreditUtilizationCategory" in df_new.columns
        assert "IsOverUtilized" in df_new.columns
        assert df_new["IsOverUtilized"].isin([0, 1]).all()

    def test_create_delinquency_features(self, feature_engineer, sample_data):
        """Test delinquency feature creation."""
        df_new = feature_engineer.create_delinquency_features(sample_data)

        assert "TotalDelinquencies" in df_new.columns
        assert "HasDelinquency" in df_new.columns
        assert "DelinquencySeverity" in df_new.columns

    def test_create_credit_activity_features(self, feature_engineer, sample_data):
        """Test credit activity feature creation."""
        df_new = feature_engineer.create_credit_activity_features(sample_data)

        assert "CreditActivityLevel" in df_new.columns
        assert "NoCreditActivity" in df_new.columns

    def test_engineer_all_features(self, feature_engineer, sample_data):
        """Test complete feature engineering pipeline."""
        df_engineered = feature_engineer.engineer_all_features(sample_data)

        # Should have more columns than original
        assert df_engineered.shape[1] > sample_data.shape[1]
        assert df_engineered.shape[0] == sample_data.shape[0]

        # Check that features were recorded
        assert len(feature_engineer.engineered_features) > 0

    def test_get_feature_names(self, feature_engineer, sample_data):
        """Test getting engineered feature names."""
        _ = feature_engineer.engineer_all_features(sample_data)  # noqa: F841
        feature_names = feature_engineer.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0


class TestFeatureEngineerEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_columns(self):
        """Test handling when expected columns are missing."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({"column1": [1, 2, 3]})

        # Should not raise errors, just skip features
        df_new = engineer.create_debt_to_income_ratio(df)
        assert len(df_new) == len(df)

    def test_zero_values(self):
        """Test handling of zero values in ratios."""
        engineer = FeatureEngineer()
        df = pd.DataFrame(
            {
                "MonthlyIncome": [0, 100, 200],
                "DebtRatio": [0, 0.5, 1.0],
                "NumberOfDependents": [0, 1, 2],
            }
        )

        # Should handle zero values without errors
        df_new = engineer.create_monthly_payment_ratio(df)
        assert len(df_new) == len(df)
        assert not df_new.isnull().any().any()
