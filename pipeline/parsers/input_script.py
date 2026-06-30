"""
LAMMPS input script parser.

Extracts simulation parameters: timestep, run lengths, temperature targets,
dump frequency, and stage boundaries.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageInfo:
    """Information about a simulation stage from the input script."""
    name: str
    run_steps: int
    timestep: float  # fs
    temp_start: Optional[float] = None
    temp_end: Optional[float] = None
    dump_every: int = 500


@dataclass
class SimulationParams:
    """Parsed simulation parameters."""
    units: str = "real"
    atom_style: str = "charge"
    num_atom_types: int = 4
    pair_style: str = ""
    element_map: Dict[int, str] = field(default_factory=dict)
    dump_frequency: int = 500
    bond_dump_frequency: int = 500
    thermo_frequency: int = 100
    stages: List[StageInfo] = field(default_factory=list)


class InputScriptParser:
    """Parse LAMMPS input script for simulation parameters."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def parse(self) -> SimulationParams:
        """Parse input script and return simulation parameters."""
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        params = SimulationParams()
        current_timestep = 0.01  # default for minimization phase

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                # Check for stage comments
                continue

            parts = stripped.split()
            cmd = parts[0].lower() if parts else ""

            if cmd == "units" and len(parts) > 1:
                params.units = parts[1]

            elif cmd == "atom_style" and len(parts) > 1:
                params.atom_style = parts[1]

            elif cmd == "pair_style":
                params.pair_style = " ".join(parts[1:])

            elif cmd == "pair_coeff" and "*" in stripped:
                # Extract element mapping: pair_coeff * * ffield C H O N
                elements = [p for p in parts if p not in ["pair_coeff", "*", "ffield.reax.CHON_2017_weak"]
                            and not p.startswith("ffield") and p.isalpha() and len(p) <= 2]
                for idx, elem in enumerate(elements, 1):
                    params.element_map[idx] = elem

            elif cmd == "timestep" and len(parts) > 1:
                current_timestep = float(parts[1])

            elif cmd == "thermo" and len(parts) > 1:
                params.thermo_frequency = int(parts[1])

            elif cmd == "dump" and "custom" in stripped:
                # dump trj_all all custom 500 trajectory_all_stages.lammpstrj ...
                for j, p in enumerate(parts):
                    if p == "custom" and j + 1 < len(parts):
                        try:
                            params.dump_frequency = int(parts[j + 1])
                        except ValueError:
                            pass
                        break

            elif cmd == "fix" and "reax/c/bonds" in stripped:
                # fix bonds all reax/c/bonds 500 bonds_all_stages.reaxff
                for j, p in enumerate(parts):
                    if p == "reax/c/bonds" and j + 1 < len(parts):
                        try:
                            params.bond_dump_frequency = int(parts[j + 1])
                        except ValueError:
                            pass
                        break

            elif cmd == "fix" and "nvt" in stripped:
                # fix eq all nvt temp 300.0 300.0 100.0
                stage_name = self._identify_stage(lines, i)
                temp_start, temp_end = self._extract_nvt_temps(parts)
                run_steps = self._find_next_run(lines, i)
                if run_steps:
                    params.stages.append(StageInfo(
                        name=stage_name,
                        run_steps=run_steps,
                        timestep=current_timestep,
                        temp_start=temp_start,
                        temp_end=temp_end,
                        dump_every=params.dump_frequency,
                    ))

            elif cmd == "run" and len(parts) > 1:
                pass  # Handled by _find_next_run

        if not params.element_map:
            params.element_map = {1: "C", 2: "H", 3: "O", 4: "N"}

        logger.debug(
            f"Parsed {len(params.stages)} stages from input script. "
            f"Elements: {params.element_map}"
        )
        return params

    def _identify_stage(self, lines: List[str], fix_line: int) -> str:
        """Identify stage name from surrounding comments."""
        # Look backwards for a comment section header
        for j in range(fix_line - 1, max(fix_line - 10, 0), -1):
            line = lines[j].strip()
            if line.startswith("#") and any(
                kw in line.upper()
                for kw in ["EQUILIBRAT", "BED", "LASER", "HOLD", "COOL"]
            ):
                if "BED" in line.upper():
                    return "bed"
                elif "LASER" in line.upper():
                    return "laser"
                elif "HOLD" in line.upper() or "NECK" in line.upper():
                    return "hold"
                elif "COOL" in line.upper():
                    return "cooling"
                elif "NORMAL" in line.upper() or "EQUILIBRAT" in line.upper():
                    return "equilibrium"
                elif "GRADUAL" in line.upper():
                    return "gradual_equil"
                elif "LOW" in line.upper():
                    return "low_temp_relax"
        return "unknown"

    def _extract_nvt_temps(self, parts: List[str]) -> tuple:
        """Extract start and end temperatures from nvt fix line."""
        try:
            temp_idx = parts.index("temp")
            t_start = float(parts[temp_idx + 1])
            t_end = float(parts[temp_idx + 2])
            return t_start, t_end
        except (ValueError, IndexError):
            return None, None

    def _find_next_run(self, lines: List[str], start: int) -> Optional[int]:
        """Find the next 'run N' command after the given line."""
        for j in range(start + 1, min(start + 20, len(lines))):
            parts = lines[j].strip().split()
            if parts and parts[0].lower() == "run" and len(parts) > 1:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return None
