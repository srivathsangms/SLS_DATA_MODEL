"""
LAMMPS log.lammps thermo data parser.

Extracts thermodynamic data (Step, Temp, Press, PE, KE, TotEng, Density, Vol)
and maps timestep ranges to simulation stages.
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Thermo header pattern
THERMO_HEADER = re.compile(r"^\s*Step\s+Temp\s+Press\s+PotEng\s+KinEng\s+TotEng\s+Density\s+Vol", re.IGNORECASE)

# Stage markers in the log (from comments and fix/unfix patterns)
STAGE_MARKERS = {
    "minimization": ["ENERGY MINIMIZATION", "minimize"],
    "nve_limit": ["NVE/LIMIT", "nve/limit"],
    "low_temp_relax": ["LOW TEMPERATURE RELAXATION"],
    "gradual_equil": ["GRADUAL EQUILIBRATION"],
    "equilibrium": ["NORMAL EQUILIBRATION", "EQUILIBRATION AT"],
    "bed": ["BED HEATING"],
    "laser": ["LASER HEATING"],
    "hold": ["HOLD AT", "NECK FORMATION"],
    "cooling": ["COOLING BACK"],
}


class LogParser:
    """Parse LAMMPS log files to extract thermodynamic data per stage."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def parse(self) -> Dict[str, pd.DataFrame]:
        """
        Parse log.lammps and return thermo data segmented by stage.

        Returns
        -------
        dict
            Mapping from stage name to DataFrame with columns:
            Step, Temp, Press, PotEng, KinEng, TotEng, Density, Volume
        """
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        # Find all thermo data blocks
        blocks = self._find_thermo_blocks(lines)
        logger.debug(f"Found {len(blocks)} thermo blocks in {self.filepath}")

        # Map blocks to stages using comments
        stage_blocks = self._map_blocks_to_stages(lines, blocks)

        return stage_blocks

    def parse_flat(self) -> pd.DataFrame:
        """Parse all thermo data into a single DataFrame with a Stage column."""
        stage_data = self.parse()
        frames = []
        for stage_name, df in stage_data.items():
            df = df.copy()
            df["Stage"] = stage_name
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _find_thermo_blocks(self, lines: List[str]) -> List[Tuple[int, int]]:
        """Find start/end line indices of thermo data blocks."""
        blocks = []
        i = 0
        while i < len(lines):
            if THERMO_HEADER.match(lines[i]):
                start = i + 1  # first data row
                end = start
                while end < len(lines):
                    stripped = lines[end].strip()
                    if not stripped:
                        break
                    # Check if line starts with a number (timestep)
                    parts = stripped.split()
                    try:
                        int(parts[0])
                        end += 1
                    except (ValueError, IndexError):
                        break
                blocks.append((start, end))
                i = end
            else:
                i += 1
        return blocks

    def _map_blocks_to_stages(
        self, lines: List[str], blocks: List[Tuple[int, int]]
    ) -> Dict[str, pd.DataFrame]:
        """Map thermo blocks to named stages based on preceding comments."""
        stage_data = {}

        # The key stages for SLS are blocks 5-9 (0-indexed):
        # 0: minimization, 1: nve_limit, 2: low_temp_relax,
        # 3: gradual_equil, 4: equilibrium, 5: bed, 6: laser, 7: hold, 8: cooling
        stage_order = [
            "minimization", "nve_limit", "low_temp_relax",
            "gradual_equil", "equilibrium", "bed", "laser", "hold", "cooling"
        ]

        for idx, (start, end) in enumerate(blocks):
            # Determine stage from block index
            if idx < len(stage_order):
                stage_name = stage_order[idx]
            else:
                stage_name = f"unknown_{idx}"

            # Parse data rows
            rows = []
            for line_idx in range(start, end):
                parts = lines[line_idx].split()
                if len(parts) >= 7:
                    try:
                        row = {
                            "Step": int(parts[0]),
                            "Temp": float(parts[1]),
                            "Press": float(parts[2]),
                            "PotEng": float(parts[3]),
                            "KinEng": float(parts[4]),
                            "TotEng": float(parts[5]),
                            "Density": float(parts[6]),
                            "Volume": float(parts[7]) if len(parts) > 7 else np.nan,
                        }
                        rows.append(row)
                    except (ValueError, IndexError):
                        continue

            if rows:
                stage_data[stage_name] = pd.DataFrame(rows)
                logger.debug(
                    f"Stage '{stage_name}': {len(rows)} thermo rows, "
                    f"steps {rows[0]['Step']}-{rows[-1]['Step']}"
                )

        return stage_data
