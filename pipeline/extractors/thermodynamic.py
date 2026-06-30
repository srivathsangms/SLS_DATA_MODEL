"""
Thermodynamic feature extraction from log.lammps data.

Features: Temperature, Pressure, PE, KE, TotalEnergy, Density, Volume.
"""

import logging
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ThermodynamicFeatures:
    """Thermodynamic features for one frame / thermo output row."""
    temperature: float = 0.0
    pressure: float = 0.0
    potential_energy: float = 0.0
    kinetic_energy: float = 0.0
    total_energy: float = 0.0
    density: float = 0.0
    volume: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "Temperature": self.temperature,
            "Pressure": self.pressure,
            "PotEng": self.potential_energy,
            "KinEng": self.kinetic_energy,
            "TotEng": self.total_energy,
            "Density": self.density,
            "Volume": self.volume,
        }


class ThermodynamicExtractor:
    """Extract thermodynamic features from parsed log data."""

    def extract_from_row(self, row: Dict) -> ThermodynamicFeatures:
        """Extract features from a single thermo data row."""
        return ThermodynamicFeatures(
            temperature=float(row.get("Temp", 0)),
            pressure=float(row.get("Press", 0)),
            potential_energy=float(row.get("PotEng", 0)),
            kinetic_energy=float(row.get("KinEng", 0)),
            total_energy=float(row.get("TotEng", 0)),
            density=float(row.get("Density", 0)),
            volume=float(row.get("Volume", 0)),
        )

    def extract_stage_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute summary statistics for a stage's thermo data.

        Parameters
        ----------
        df : pd.DataFrame
            Thermo data for one stage with columns: Temp, Press, PotEng, etc.
        """
        summary = {}
        for col, prefix in [
            ("Temp", "Temperature"),
            ("Press", "Pressure"),
            ("PotEng", "PotEng"),
            ("KinEng", "KinEng"),
            ("TotEng", "TotEng"),
            ("Density", "Density"),
            ("Volume", "Volume"),
        ]:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    summary[f"Mean_{prefix}"] = float(values.mean())
                    summary[f"Std_{prefix}"] = float(values.std())
                    summary[f"Min_{prefix}"] = float(values.min())
                    summary[f"Max_{prefix}"] = float(values.max())
                    summary[f"First_{prefix}"] = float(values.iloc[0])
                    summary[f"Last_{prefix}"] = float(values.iloc[-1])

        return summary
