"""
Final ML-ready dataset builder.

Assembles raw, normalized, and encoded datasets ready for
Random Forest, XGBoost, LightGBM, CatBoost, PINN, and SHAP analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

STAGE_ORDINAL = {
    "equilibrium": 0,
    "bed": 1,
    "laser": 2,
    "hold": 3,
    "cooling": 4,
}


class DatasetBuilder:
    """Build final ML-ready datasets from master feature data."""

    def __init__(self, normalization: str = "minmax", stage_encoding: str = "ordinal"):
        self.normalization = normalization
        self.stage_encoding = stage_encoding

    def build_raw_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the raw dataset with consistent column ordering."""
        meta_cols = ["Composition", "Temperature", "Stage", "Frame", "NumAtoms", "SimTime", "Timestep"]
        existing_meta = [c for c in meta_cols if c in df.columns]
        feature_cols = sorted([c for c in df.columns if c not in meta_cols])
        return df[existing_meta + feature_cols].copy()

    def build_normalized_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a normalized dataset (features only, metadata preserved)."""
        meta_cols = ["Composition", "Temperature", "Stage", "Frame", "NumAtoms", "SimTime", "Timestep"]
        existing_meta = [c for c in meta_cols if c in df.columns]
        feature_cols = [c for c in df.columns if c not in meta_cols]

        numeric_features = df[feature_cols].select_dtypes(include=[np.number])
        non_numeric = df[feature_cols].select_dtypes(exclude=[np.number])

        if self.normalization == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        # Handle NaN and zero-variance columns
        cols_to_scale = numeric_features.columns[numeric_features.std() > 1e-10]
        scaled_data = numeric_features.copy()
        if len(cols_to_scale) > 0:
            scaled_data[cols_to_scale] = scaler.fit_transform(
                numeric_features[cols_to_scale].fillna(0)
            )

        result = pd.concat([df[existing_meta], scaled_data, non_numeric], axis=1)
        return result

    def build_ml_ready_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a fully ML-ready dataset:
        - Stage encoded (ordinal or one-hot)
        - Composition encoded
        - All features normalized
        - No NaN values
        """
        result = self.build_normalized_dataset(df)

        # Encode Stage
        if "Stage" in result.columns:
            if self.stage_encoding == "ordinal":
                result["Stage_Encoded"] = result["Stage"].map(STAGE_ORDINAL).fillna(-1).astype(int)
            else:
                # One-hot encoding
                stage_dummies = pd.get_dummies(result["Stage"], prefix="Stage")
                result = pd.concat([result, stage_dummies], axis=1)

        # Encode Composition
        if "Composition" in result.columns:
            le = LabelEncoder()
            result["Composition_Encoded"] = le.fit_transform(result["Composition"].fillna("unknown"))

            # Also extract numeric ratio
            comp_map = {"5050": 50, "6040": 60, "7030": 70}
            result["Epoxy_Pct"] = result["Composition"].map(comp_map).fillna(50).astype(int)
            result["PA12_Pct"] = 100 - result["Epoxy_Pct"]

        # Drop string columns for ML (keep only numeric)
        ml_df = result.select_dtypes(include=[np.number]).copy()
        ml_df = ml_df.fillna(0)

        # Add back string metadata for reference (but ML models should use encoded versions)
        for col in ["Composition", "Stage"]:
            if col in result.columns:
                ml_df[col] = result[col].values

        return ml_df

    def save_datasets(
        self,
        df: pd.DataFrame,
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Save all three dataset versions.

        Returns dict of {name: filepath}.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        paths = {}

        # Raw
        raw = self.build_raw_dataset(df)
        raw_path = os.path.join(output_dir, "dataset.csv")
        raw.to_csv(raw_path, index=False)
        paths["raw"] = raw_path
        logger.info(f"Saved raw dataset: {raw.shape} → {raw_path}")

        # Normalized
        normalized = self.build_normalized_dataset(df)
        norm_path = os.path.join(output_dir, "dataset_normalized.csv")
        normalized.to_csv(norm_path, index=False)
        paths["normalized"] = norm_path
        logger.info(f"Saved normalized dataset: {normalized.shape} → {norm_path}")

        # ML-ready
        ml_ready = self.build_ml_ready_dataset(df)
        ml_path = os.path.join(output_dir, "dataset_ml_ready.csv")
        ml_ready.to_csv(ml_path, index=False)
        paths["ml_ready"] = ml_path
        logger.info(f"Saved ML-ready dataset: {ml_ready.shape} → {ml_path}")

        return paths
