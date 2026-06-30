"""
Dynamic feature extraction from trajectory frames.

Features: Displacement Magnitude, Mean/Max Displacement, MSD.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DynamicFeatures:
    """Dynamic features for one trajectory frame."""
    mean_displacement: float = 0.0
    std_displacement: float = 0.0
    min_displacement: float = 0.0
    max_displacement: float = 0.0
    msd: float = 0.0  # Mean Square Displacement
    displacement_distribution: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "Mean_Displacement": self.mean_displacement,
            "Std_Displacement": self.std_displacement,
            "Min_Displacement": self.min_displacement,
            "Max_Displacement": self.max_displacement,
            "MSD": self.msd,
        }


class DynamicExtractor:
    """Extract dynamic features from trajectory frames."""

    def __init__(self):
        self._reference_positions = None

    def set_reference(self, positions: np.ndarray):
        """Set reference positions for displacement calculation."""
        self._reference_positions = positions.copy()

    def extract(
        self,
        positions: np.ndarray,
        box_lengths: np.ndarray,
        reference_positions: Optional[np.ndarray] = None,
    ) -> DynamicFeatures:
        """
        Extract dynamic features by comparing current positions to reference.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
            Current frame positions.
        box_lengths : np.ndarray, shape (3,)
            Box dimensions for minimum image convention.
        reference_positions : np.ndarray, optional
            If None, uses internally stored reference.
        """
        features = DynamicFeatures()

        ref = reference_positions if reference_positions is not None else self._reference_positions
        if ref is None:
            logger.debug("No reference positions set; skipping displacement")
            return features

        # Compute displacements with minimum image convention
        delta = positions - ref
        delta = delta - box_lengths * np.round(delta / box_lengths)
        magnitudes = np.linalg.norm(delta, axis=1)

        features.mean_displacement = float(np.mean(magnitudes))
        features.std_displacement = float(np.std(magnitudes))
        features.min_displacement = float(np.min(magnitudes))
        features.max_displacement = float(np.max(magnitudes))
        features.msd = float(np.mean(magnitudes ** 2))
        features.displacement_distribution = magnitudes

        return features
