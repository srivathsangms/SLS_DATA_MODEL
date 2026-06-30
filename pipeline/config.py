"""
Pipeline configuration loader and dataclass.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    n_workers: int = 4
    frame_sampling: int = 1
    use_ovito: bool = True
    use_xlsx_if_available: bool = True


@dataclass
class StructuralConfig:
    neighbor_cutoff: float = 3.5
    rdf_cutoff: float = 12.0
    rdf_bins: int = 200
    voronoi_enabled: bool = True


@dataclass
class BondConfig:
    bond_order_cutoff: float = 0.3
    atom_types: Dict[int, str] = field(default_factory=lambda: {1: "C", 2: "H", 3: "O", 4: "N"})


@dataclass
class VisualizationConfig:
    dpi: int = 300
    format: str = "png"
    figsize: List[int] = field(default_factory=lambda: [10, 6])
    style: str = "seaborn-v0_8-whitegrid"
    colormap: str = "viridis"
    stage_colors: Dict[str, str] = field(default_factory=lambda: {
        "equilibrium": "#2196F3",
        "bed": "#FF9800",
        "laser": "#F44336",
        "hold": "#9C27B0",
        "cooling": "#4CAF50",
    })


@dataclass
class NormalizationConfig:
    method: str = "minmax"
    stage_encoding: str = "ordinal"


@dataclass
class PipelineConfig:
    root_dir: str = ""
    output_dir: str = "Results"
    stages: List[str] = field(default_factory=lambda: [
        "equilibrium", "bed", "laser", "hold", "cooling"
    ])
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    structural: StructuralConfig = field(default_factory=StructuralConfig)
    bond: BondConfig = field(default_factory=BondConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        cfg.root_dir = raw.get("root_dir", cfg.root_dir)
        cfg.output_dir = raw.get("output_dir", cfg.output_dir)
        cfg.stages = raw.get("stages", cfg.stages)

        if "processing" in raw:
            cfg.processing = ProcessingConfig(**raw["processing"])
        if "structural" in raw:
            cfg.structural = StructuralConfig(**raw["structural"])
        if "bond" in raw:
            bond_raw = raw["bond"]
            # Convert string keys to int for atom_types
            if "atom_types" in bond_raw:
                bond_raw["atom_types"] = {
                    int(k): v for k, v in bond_raw["atom_types"].items()
                }
            cfg.bond = BondConfig(**bond_raw)
        if "visualization" in raw:
            cfg.visualization = VisualizationConfig(**raw["visualization"])
        if "normalization" in raw:
            cfg.normalization = NormalizationConfig(**raw["normalization"])

        cfg.validate()
        return cfg

    def validate(self):
        """Validate that required paths exist."""
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        logger.info(f"Configuration validated. Root: {self.root_dir}")

    def get_output_path(self, *subdirs: str) -> str:
        """Get absolute path within the output directory, creating it if needed."""
        path = os.path.join(self.output_dir, *subdirs)
        os.makedirs(path, exist_ok=True)
        return path
