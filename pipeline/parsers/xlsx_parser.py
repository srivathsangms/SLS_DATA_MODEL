"""
Pre-computed OVITO xlsx file parser.

Reads per-atom features exported from OVITO (coordination number,
atomic volume, cavity radius, displacement magnitude, etc.).
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class XlsxParser:
    """Parse OVITO-exported xlsx files with per-atom structural features."""

    # Column mapping from xlsx headers to standardised names
    COLUMN_MAP = {
        "particle identifier": "atom_id",
        "particle type": "atom_type",
        "position": "position",
        "coordination": "coordination_number",
        "atomic volume": "atomic_volume",
        "cavity radius": "cavity_radius",
        "per type coordination": "per_type_coordination",
        "displacement": "displacement_vector",
        "displacement magnitude": "displacement_magnitude",
    }

    def __init__(self, filepath: str):
        self.filepath = filepath

    def parse(self) -> pd.DataFrame:
        """
        Read xlsx file and return a DataFrame with standardised column names.

        Returns
        -------
        pd.DataFrame
            Per-atom features with columns matching COLUMN_MAP values.
        """
        try:
            df = pd.read_excel(self.filepath, engine="openpyxl")
        except Exception as e:
            logger.error(f"Failed to read {self.filepath}: {e}")
            return pd.DataFrame()

        # Rename columns
        rename_map = {}
        for orig, std in self.COLUMN_MAP.items():
            for col in df.columns:
                if col.strip().lower() == orig.lower():
                    rename_map[col] = std
                    break
        df = df.rename(columns=rename_map)

        logger.debug(f"Parsed {len(df)} atoms from {self.filepath}")
        return df

    def extract_summary(self) -> Dict[str, float]:
        """
        Extract aggregated statistics from xlsx per-atom data.

        Returns
        -------
        dict
            Summary statistics: mean/std/min/max for CN, atomic volume,
            cavity radius, displacement magnitude.
        """
        df = self.parse()
        if df.empty:
            return {}

        summary = {}

        for col, prefix in [
            ("coordination_number", "CN"),
            ("atomic_volume", "AtomicVolume"),
            ("cavity_radius", "CavityRadius"),
            ("displacement_magnitude", "Displacement"),
        ]:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(values) > 0:
                    summary[f"Mean_{prefix}"] = float(values.mean())
                    summary[f"Std_{prefix}"] = float(values.std())
                    summary[f"Min_{prefix}"] = float(values.min())
                    summary[f"Max_{prefix}"] = float(values.max())
                    summary[f"Median_{prefix}"] = float(values.median())

        return summary

    def get_distributions(self) -> Dict[str, np.ndarray]:
        """Get raw arrays for distribution plotting."""
        df = self.parse()
        distributions = {}

        for col, name in [
            ("coordination_number", "CN"),
            ("atomic_volume", "AtomicVolume"),
            ("cavity_radius", "CavityRadius"),
            ("displacement_magnitude", "Displacement"),
        ]:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").dropna().values
                distributions[name] = values

        return distributions
