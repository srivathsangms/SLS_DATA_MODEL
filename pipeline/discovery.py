"""
Recursive experiment directory scanner.

Automatically discovers compositions, temperatures, and available files
within the data root directory.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex for folder names like "50_50 200 C", "60_40 100 C", "70_30 350 C"
EXPERIMENT_PATTERN = re.compile(
    r"(\d{2})_(\d{2})\s+(\d+)\s*C", re.IGNORECASE
)

# Mapping of composition folder names to codes
COMPOSITION_MAP = {
    "5050": "50/50",
    "6040": "60/40",
    "7030": "70/30",
}

# Stage keywords for matching trajectory file names
STAGE_KEYWORDS = {
    "equilibrium": ["equil", "equili"],
    "bed": ["bed"],
    "laser": ["laser"],
    "hold": ["hold"],
    "cooling": ["cool"],
}


@dataclass
class StageFiles:
    """Files associated with a single simulation stage."""
    name: str  # e.g. "equilibrium"
    trajectory: Optional[str] = None
    temperature: Optional[str] = None
    xlsx: Optional[str] = None


@dataclass
class ExperimentMetadata:
    """Metadata and file paths for one experiment (composition + temperature)."""
    composition: str          # e.g. "5050"
    composition_label: str    # e.g. "50/50"
    temperature: int          # e.g. 200
    directory: str            # absolute path
    log_file: Optional[str] = None
    bond_file: Optional[str] = None
    input_script: Optional[str] = None
    all_stages_trajectory: Optional[str] = None
    stages: Dict[str, StageFiles] = field(default_factory=dict)
    num_atoms: Optional[int] = None

    @property
    def label(self) -> str:
        return f"{self.composition}_{self.temperature}C"


class ExperimentDiscoverer:
    """Recursively scan root directory to discover all experiments."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def discover(self) -> List[ExperimentMetadata]:
        """Scan root and return list of discovered experiments."""
        experiments = []

        for comp_folder in sorted(os.listdir(self.root_dir)):
            comp_path = os.path.join(self.root_dir, comp_folder)
            if not os.path.isdir(comp_path):
                continue

            # Skip non-composition folders
            comp_code = comp_folder.replace("/", "").replace("\\", "").strip()
            if comp_code not in COMPOSITION_MAP and comp_code not in ["5050", "6040", "7030"]:
                # Try to extract composition from folder name
                if not any(c in comp_folder.lower() for c in ["50", "60", "70"]):
                    logger.debug(f"Skipping non-composition folder: {comp_folder}")
                    continue

            # Determine composition code
            comp_code = comp_folder.strip()
            if comp_code not in COMPOSITION_MAP:
                logger.debug(f"Skipping folder: {comp_folder}")
                continue

            comp_label = COMPOSITION_MAP[comp_code]

            # Scan sub-folders for experiments
            for exp_folder in sorted(os.listdir(comp_path)):
                exp_path = os.path.join(comp_path, exp_folder)
                if not os.path.isdir(exp_path):
                    continue

                match = EXPERIMENT_PATTERN.match(exp_folder)
                if not match:
                    logger.debug(f"Skipping non-experiment folder: {exp_folder}")
                    continue

                temperature = int(match.group(3))
                experiment = self._scan_experiment(
                    exp_path, comp_code, comp_label, temperature
                )
                experiments.append(experiment)
                logger.info(
                    f"Discovered: {experiment.label} "
                    f"({len(experiment.stages)} stages, "
                    f"log={experiment.log_file is not None}, "
                    f"bonds={experiment.bond_file is not None})"
                )

        logger.info(f"Total experiments discovered: {len(experiments)}")
        return experiments

    def _scan_experiment(
        self, directory: str, comp_code: str, comp_label: str, temperature: int
    ) -> ExperimentMetadata:
        """Scan a single experiment directory for all relevant files."""
        exp = ExperimentMetadata(
            composition=comp_code,
            composition_label=comp_label,
            temperature=temperature,
            directory=directory,
        )

        files = os.listdir(directory)
        files_lower = {f.lower(): f for f in files}

        # Find log file
        if "log.lammps" in files_lower:
            exp.log_file = os.path.join(directory, files_lower["log.lammps"])

        # Find bond file
        for f in files:
            if f.endswith(".reaxff") and "bonds" in f.lower():
                exp.bond_file = os.path.join(directory, f)
                break

        # Find input script
        for f in files:
            if f.startswith("in.") or f.startswith("in_"):
                exp.input_script = os.path.join(directory, f)
                break

        # Find all-stages trajectory
        for f in files:
            if "all_stages" in f.lower() and f.endswith(".lammpstrj"):
                exp.all_stages_trajectory = os.path.join(directory, f)
                break

        # Find stage-specific files
        for stage_name, keywords in STAGE_KEYWORDS.items():
            stage = StageFiles(name=stage_name)

            # Find trajectory
            for f in files:
                if f.endswith(".lammpstrj") and "all_stages" not in f.lower():
                    if any(kw in f.lower() for kw in keywords):
                        stage.trajectory = os.path.join(directory, f)
                        break

            # Find temperature file
            for f in files:
                if f.startswith("temp_") and f.endswith(".txt"):
                    if any(kw in f.lower() for kw in keywords):
                        stage.temperature = os.path.join(directory, f)
                        break

            # Find xlsx file
            for f in files:
                if f.endswith(".xlsx") and not f.startswith("~"):
                    fname = f.lower().replace(".xlsx", "")
                    if any(kw in fname for kw in keywords):
                        stage.xlsx = os.path.join(directory, f)
                        break

            exp.stages[stage_name] = stage

        return exp
