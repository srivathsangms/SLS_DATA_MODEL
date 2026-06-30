"""
Feature aggregation: frame-level data → stage-level summaries.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """Aggregate per-frame feature data into stage-level summaries."""

    def aggregate_stage(
        self,
        frame_records: List[Dict],
        composition: str,
        temperature: int,
        stage: str,
    ) -> Dict:
        """
        Compute stage-level summary from a list of frame feature dicts.

        Parameters
        ----------
        frame_records : list of dict
            Each dict is one frame's features (output of to_dict() methods).
        composition : str
        temperature : int
        stage : str

        Returns
        -------
        dict
            Aggregated summary with Mean_*, Std_*, Min_*, Max_* for each feature.
        """
        if not frame_records:
            return {}

        df = pd.DataFrame(frame_records)
        summary = {
            "Composition": composition,
            "Temperature": temperature,
            "Stage": stage,
            "Num_Frames": len(df),
        }

        # Aggregate each numeric column
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].dropna()
            if len(values) > 0:
                summary[f"Stage_Mean_{col}"] = float(values.mean())
                summary[f"Stage_Std_{col}"] = float(values.std())
                summary[f"Stage_Min_{col}"] = float(values.min())
                summary[f"Stage_Max_{col}"] = float(values.max())
                summary[f"Stage_First_{col}"] = float(values.iloc[0])
                summary[f"Stage_Last_{col}"] = float(values.iloc[-1])
                summary[f"Stage_Delta_{col}"] = float(values.iloc[-1] - values.iloc[0])

        return summary

    def build_master_dataset(
        self,
        all_frame_records: List[Dict],
    ) -> pd.DataFrame:
        """
        Build the master dataset from all frame records across all experiments.

        Each row = one frame with metadata and all features.
        """
        if not all_frame_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_frame_records)

        # Ensure metadata columns come first
        meta_cols = ["Composition", "Temperature", "Stage", "Frame", "NumAtoms", "SimTime", "Timestep"]
        existing_meta = [c for c in meta_cols if c in df.columns]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df = df[existing_meta + sorted(feature_cols)]

        logger.info(f"Master dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
