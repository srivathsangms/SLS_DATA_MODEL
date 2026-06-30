"""
Statistical analysis module.

Correlation matrices, PCA, feature ranking, and summary statistics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Perform statistical analysis on the master dataset."""

    def compute_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptive statistics for all numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        summary = numeric_df.describe().T
        summary["variance"] = numeric_df.var()
        summary["skewness"] = numeric_df.skew()
        summary["kurtosis"] = numeric_df.kurtosis()
        summary["missing"] = numeric_df.isnull().sum()
        summary["missing_pct"] = (numeric_df.isnull().sum() / len(df) * 100).round(2)
        return summary

    def compute_pearson_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Pearson correlation matrix."""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method="pearson")

    def compute_spearman_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Spearman rank correlation matrix."""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method="spearman")

    def compute_pca(
        self, df: pd.DataFrame, n_components: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Perform PCA on numeric features.

        Returns
        -------
        scores : pd.DataFrame
            PCA scores (transformed data)
        loadings : pd.DataFrame
            PCA loadings (component weights)
        explained_variance : np.ndarray
            Explained variance ratio per component
        """
        numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        numeric_df = numeric_df.fillna(numeric_df.mean())

        # Remove zero-variance columns
        non_zero_var = numeric_df.columns[numeric_df.std() > 1e-10]
        numeric_df = numeric_df[non_zero_var]

        if numeric_df.shape[1] < 2:
            logger.warning("Not enough features for PCA")
            return pd.DataFrame(), pd.DataFrame(), np.array([])

        n_components = min(n_components, numeric_df.shape[1], numeric_df.shape[0])

        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(scaled)

        score_cols = [f"PC{i + 1}" for i in range(n_components)]
        scores_df = pd.DataFrame(scores, columns=score_cols, index=numeric_df.index)

        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=score_cols,
            index=numeric_df.columns,
        )

        return scores_df, loadings_df, pca.explained_variance_ratio_

    def compute_feature_ranking(
        self, df: pd.DataFrame, target_col: str = "Stage"
    ) -> pd.DataFrame:
        """
        Rank features by importance using Random Forest on stage classification.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain the target column and numeric features.
        target_col : str
            Column to use as classification target.

        Returns
        -------
        pd.DataFrame
            Feature importance ranking.
        """
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return pd.DataFrame()

        numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        numeric_df = numeric_df.fillna(numeric_df.mean())

        # Remove zero-variance
        non_zero_var = numeric_df.columns[numeric_df.std() > 1e-10]
        numeric_df = numeric_df[non_zero_var]

        if numeric_df.shape[1] < 1:
            return pd.DataFrame()

        # Encode target
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].fillna("unknown"))

        X = numeric_df.values

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
        )
        rf.fit(X, y)

        importance_df = pd.DataFrame({
            "Feature": numeric_df.columns,
            "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        importance_df["Rank"] = range(1, len(importance_df) + 1)
        importance_df["Cumulative_Importance"] = importance_df["Importance"].cumsum()

        return importance_df
