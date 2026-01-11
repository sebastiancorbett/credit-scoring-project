"""
Feature engineering module for credit risk prediction.

This module creates derived features such as debt-to-income ratio,
loan-to-asset ratio, and other financial indicators to improve model performance.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging
from config import FEATURE_CONFIG

logger = setup_logging(__name__)


class FeatureEngineer:
    """
    Class for engineering features from credit risk data.

    This class creates derived financial indicators and categorical encodings
    to enhance model predictive power.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize FeatureEngineer.

        Args:
            config: Feature engineering configuration (uses defaults if None)
        """
        self.config = config or FEATURE_CONFIG
        self.engineered_features = []
        logger.info("FeatureEngineer initialized")

    def create_debt_to_income_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create debt-to-income ratio feature.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new feature
        """
        if not self.config.get("debt_to_income_ratio", False):
            return df

        df_new = df.copy()

        if "DebtRatio" in df_new.columns and "MonthlyIncome" in df_new.columns:
            # DebtRatio is already debt/income, but we create additional variants
            df_new["DebtToIncomeRatio"] = df_new["DebtRatio"]

            # Create categorical bins
            df_new["DebtToIncomeCategory"] = pd.cut(
                df_new["DebtToIncomeRatio"],
                bins=[0, 0.3, 0.5, 1.0, np.inf],
                labels=["Low", "Medium", "High", "VeryHigh"],
            )
            df_new["DebtToIncomeCategory"] = df_new["DebtToIncomeCategory"].cat.codes

            self.engineered_features.append("DebtToIncomeRatio")
            self.engineered_features.append("DebtToIncomeCategory")
            logger.info("Created debt-to-income ratio features")

        return df_new

    def create_loan_to_asset_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create loan-to-asset ratio feature.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new feature
        """
        if not self.config.get("loan_to_asset_ratio", False):
            return df

        df_new = df.copy()

        if (
            "NumberOfOpenCreditLinesAndLoans" in df_new.columns
            and "MonthlyIncome" in df_new.columns
        ):
            # Approximate loan-to-asset using credit lines and income
            df_new["LoanToAssetProxy"] = df_new["NumberOfOpenCreditLinesAndLoans"] / (
                df_new["MonthlyIncome"] / 1000 + 1
            )

            # Cap extreme values
            df_new["LoanToAssetProxy"] = df_new["LoanToAssetProxy"].clip(0, 10)

            self.engineered_features.append("LoanToAssetProxy")
            logger.info("Created loan-to-asset proxy feature")

        if "NumberRealEstateLoansOrLines" in df_new.columns:
            df_new["RealEstateLoanRatio"] = df_new["NumberRealEstateLoansOrLines"] / (
                df_new["NumberOfOpenCreditLinesAndLoans"] + 1
            )
            self.engineered_features.append("RealEstateLoanRatio")
            logger.info("Created real estate loan ratio feature")

        return df_new

    def create_monthly_payment_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create monthly payment ratio features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        if not self.config.get("monthly_payment_ratio", False):
            return df

        df_new = df.copy()

        if "MonthlyIncome" in df_new.columns and "DebtRatio" in df_new.columns:
            # Estimate monthly debt payment
            df_new["MonthlyDebtPayment"] = df_new["MonthlyIncome"] * df_new["DebtRatio"]

            # Discretionary income (income minus debt)
            df_new["DiscretionaryIncome"] = (
                df_new["MonthlyIncome"] - df_new["MonthlyDebtPayment"]
            )
            df_new["DiscretionaryIncome"] = df_new["DiscretionaryIncome"].clip(0, None)

            # Income per dependent
            if "NumberOfDependents" in df_new.columns:
                df_new["IncomePerDependent"] = df_new["MonthlyIncome"] / (
                    df_new["NumberOfDependents"] + 1
                )
                self.engineered_features.append("IncomePerDependent")

            self.engineered_features.extend(
                ["MonthlyDebtPayment", "DiscretionaryIncome"]
            )
            logger.info("Created monthly payment ratio features")

        return df_new

    def create_age_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age group categorical features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        if not self.config.get("age_group_encoding", False):
            return df

        df_new = df.copy()

        if "age" in df_new.columns:
            # Age groups
            df_new["AgeGroup"] = pd.cut(
                df_new["age"],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
            )
            df_new["AgeGroup"] = df_new["AgeGroup"].cat.codes

            # Age squared for non-linear relationships
            df_new["AgeSquared"] = df_new["age"] ** 2

            # Senior citizen flag
            df_new["IsSenior"] = (df_new["age"] >= 65).astype(int)

            self.engineered_features.extend(["AgeGroup", "AgeSquared", "IsSenior"])
            logger.info("Created age group features")

        return df_new

    def create_credit_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create credit utilization categorical features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        if not self.config.get("credit_utilization_categories", False):
            return df

        df_new = df.copy()

        if "RevolvingUtilizationOfUnsecuredLines" in df_new.columns:
            # Credit utilization categories
            df_new["CreditUtilizationCategory"] = pd.cut(
                df_new["RevolvingUtilizationOfUnsecuredLines"],
                bins=[0, 0.3, 0.5, 0.7, 1.0, np.inf],
                labels=["Excellent", "Good", "Fair", "Poor", "Critical"],
            )
            df_new["CreditUtilizationCategory"] = df_new[
                "CreditUtilizationCategory"
            ].cat.codes

            # Flag for over-utilization
            df_new["IsOverUtilized"] = (
                df_new["RevolvingUtilizationOfUnsecuredLines"] > 1.0
            ).astype(int)

            self.engineered_features.extend(
                ["CreditUtilizationCategory", "IsOverUtilized"]
            )
            logger.info("Created credit utilization features")

        return df_new

    def create_delinquency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated delinquency features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        df_new = df.copy()

        delinquency_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ]

        available_cols = [col for col in delinquency_cols if col in df_new.columns]

        if available_cols:
            # Total delinquencies
            df_new["TotalDelinquencies"] = df_new[available_cols].sum(axis=1)

            # Has any delinquency flag
            df_new["HasDelinquency"] = (df_new["TotalDelinquencies"] > 0).astype(int)

            # Severity score (weighted by lateness)
            if len(available_cols) == 3:
                df_new["DelinquencySeverity"] = (
                    df_new[available_cols[0]] * 1
                    + df_new[available_cols[1]] * 2
                    + df_new[available_cols[2]] * 3
                )
                self.engineered_features.append("DelinquencySeverity")

            self.engineered_features.extend(["TotalDelinquencies", "HasDelinquency"])
            logger.info("Created delinquency aggregation features")

        return df_new

    def create_credit_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to credit activity.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        df_new = df.copy()

        if "NumberOfOpenCreditLinesAndLoans" in df_new.columns:
            # Credit activity level
            df_new["CreditActivityLevel"] = pd.cut(
                df_new["NumberOfOpenCreditLinesAndLoans"],
                bins=[0, 3, 7, 12, np.inf],
                labels=["Low", "Medium", "High", "VeryHigh"],
            )
            df_new["CreditActivityLevel"] = df_new["CreditActivityLevel"].cat.codes

            # Flag for no credit activity
            df_new["NoCreditActivity"] = (
                df_new["NumberOfOpenCreditLinesAndLoans"] == 0
            ).astype(int)

            self.engineered_features.extend(["CreditActivityLevel", "NoCreditActivity"])
            logger.info("Created credit activity features")

        return df_new

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        df_new = df.copy()

        # Age and income interaction
        if "age" in df_new.columns and "MonthlyIncome" in df_new.columns:
            df_new["AgeIncomeInteraction"] = df_new["age"] * np.log1p(
                df_new["MonthlyIncome"]
            )
            self.engineered_features.append("AgeIncomeInteraction")

        # Debt and delinquency interaction
        if "DebtRatio" in df_new.columns and "TotalDelinquencies" in df_new.columns:
            df_new["DebtDelinquencyInteraction"] = (
                df_new["DebtRatio"] * df_new["TotalDelinquencies"]
            )
            self.engineered_features.append("DebtDelinquencyInteraction")

        # Credit lines and utilization interaction
        if (
            "NumberOfOpenCreditLinesAndLoans" in df_new.columns
            and "RevolvingUtilizationOfUnsecuredLines" in df_new.columns
        ):
            df_new["CreditLinesUtilizationInteraction"] = (
                df_new["NumberOfOpenCreditLinesAndLoans"]
                * df_new["RevolvingUtilizationOfUnsecuredLines"]
            )
            self.engineered_features.append("CreditLinesUtilizationInteraction")

        if self.engineered_features:
            logger.info("Created interaction features")

        return df_new

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all engineered features
        """
        logger.info("=" * 80)
        logger.info("Starting feature engineering pipeline")
        logger.info("=" * 80)

        df_engineered = df.copy()

        # Apply all feature engineering methods
        df_engineered = self.create_debt_to_income_ratio(df_engineered)
        df_engineered = self.create_loan_to_asset_ratio(df_engineered)
        df_engineered = self.create_monthly_payment_ratio(df_engineered)
        df_engineered = self.create_age_group_features(df_engineered)
        df_engineered = self.create_credit_utilization_features(df_engineered)
        df_engineered = self.create_delinquency_features(df_engineered)
        df_engineered = self.create_credit_activity_features(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)

        n_new_features = len(self.engineered_features)
        logger.info(f"Created {n_new_features} new engineered features")
        logger.info(f"Engineered features: {self.engineered_features}")
        logger.info(f"Final feature count: {df_engineered.shape[1]}")

        logger.info("=" * 80)
        logger.info("Feature engineering pipeline complete")
        logger.info("=" * 80)

        return df_engineered

    def get_feature_names(self) -> List[str]:
        """
        Get list of engineered feature names.

        Returns:
            List of feature names
        """
        return self.engineered_features.copy()


def main():
    """Main function for testing FeatureEngineer."""
    from src.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    df = loader.load_data()
    df_clean = loader.clean_data(df)

    # Engineer features
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_all_features(df_clean)

    logger.info(f"\nOriginal shape: {df.shape}")
    logger.info(f"Engineered shape: {df_engineered.shape}")
    logger.info(f"\nNew features created: {len(engineer.get_feature_names())}")


if __name__ == "__main__":
    main()
