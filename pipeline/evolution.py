"""
Cross-stage evolution tracker.

Tracks how every feature evolves across:
Equilibrium → Bed → Laser → Hold → Cooling
for each (composition, temperature) pair.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)

STAGE_ORDER = ["equilibrium", "bed", "laser", "hold", "cooling"]


class EvolutionTracker:
    """Track feature evolution across SLS stages."""

    def __init__(self, stage_order: List[str] = None):
        self.stage_order = stage_order or STAGE_ORDER

    def build_evolution_table(
        self,
        stage_summaries: List[Dict],
    ) -> pd.DataFrame:
        """
        Build evolution table from stage summaries.

        Parameters
        ----------
        stage_summaries : list of dict
            Each dict has Composition, Temperature, Stage, and Stage_Mean_* columns.

        Returns
        -------
        pd.DataFrame
            Wide-format table: one row per (Composition, Temperature, Feature),
            columns for each stage value.
        """
        if not stage_summaries:
            return pd.DataFrame()

        df = pd.DataFrame(stage_summaries)

        # Get feature columns (Stage_Mean_*)
        mean_cols = [c for c in df.columns if c.startswith("Stage_Mean_")]
        feature_names = [c.replace("Stage_Mean_", "") for c in mean_cols]

        records = []
        for (comp, temp), group in df.groupby(["Composition", "Temperature"]):
            for feat_name, mean_col in zip(feature_names, mean_cols):
                record = {
                    "Composition": comp,
                    "Temperature": temp,
                    "Feature": feat_name,
                }

                for stage in self.stage_order:
                    stage_rows = group[group["Stage"] == stage]
                    if len(stage_rows) > 0:
                        record[stage.capitalize()] = float(stage_rows[mean_col].iloc[0])
                    else:
                        record[stage.capitalize()] = np.nan

                # Compute deltas between consecutive stages
                stage_vals = [record.get(s.capitalize(), np.nan) for s in self.stage_order]
                for i in range(1, len(stage_vals)):
                    if not np.isnan(stage_vals[i]) and not np.isnan(stage_vals[i - 1]):
                        delta = stage_vals[i] - stage_vals[i - 1]
                        pct = (delta / abs(stage_vals[i - 1]) * 100) if stage_vals[i - 1] != 0 else 0
                        stage_name = self.stage_order[i].capitalize()
                        prev_name = self.stage_order[i - 1].capitalize()
                        record[f"Delta_{prev_name}_to_{stage_name}"] = delta
                        record[f"PctChange_{prev_name}_to_{stage_name}"] = pct

                # Overall change
                first_val = stage_vals[0]
                last_val = stage_vals[-1]
                if not np.isnan(first_val) and not np.isnan(last_val):
                    record["Total_Change"] = last_val - first_val
                    record["Total_PctChange"] = (
                        (last_val - first_val) / abs(first_val) * 100
                        if first_val != 0 else 0
                    )

                records.append(record)

        return pd.DataFrame(records)

    def build_per_experiment_evolution(
        self,
        stage_summaries: List[Dict],
        composition: str,
        temperature: int,
    ) -> pd.DataFrame:
        """Build evolution table for a single experiment."""
        filtered = [
            s for s in stage_summaries
            if s.get("Composition") == composition and s.get("Temperature") == temperature
        ]
        return self.build_evolution_table(filtered)
