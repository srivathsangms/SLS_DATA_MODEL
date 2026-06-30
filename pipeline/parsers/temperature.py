"""
Temperature file parser.

Reads LAMMPS time-averaged temperature files (temp_*.txt).
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class TemperatureParser:
    """Parse LAMMPS temperature output files."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def parse(self) -> pd.DataFrame:
        """
        Parse temperature file.

        Returns
        -------
        pd.DataFrame
            Columns: TimeStep, Temperature
        """
        timesteps = []
        temperatures = []

        with open(self.filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        timesteps.append(int(parts[0]))
                        temperatures.append(float(parts[1]))
                    except ValueError:
                        continue

        df = pd.DataFrame({
            "TimeStep": timesteps,
            "Temperature": temperatures,
        })
        logger.debug(f"Parsed {len(df)} temperature records from {self.filepath}")
        return df
