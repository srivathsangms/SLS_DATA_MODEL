"""
Mechanical feature extraction from trajectory frames.

Features: Volumetric Strain, Shear Strain (from atomic displacement gradient).
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


@dataclass
class MechanicalFeatures:
    """Mechanical features for one trajectory frame."""
    volumetric_strain: float = 0.0        # (V - V_ref) / V_ref
    mean_volumetric_strain: float = 0.0   # per-atom mean
    mean_shear_strain: float = 0.0
    std_shear_strain: float = 0.0
    min_shear_strain: float = 0.0
    max_shear_strain: float = 0.0
    shear_strain_distribution: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "Volumetric_Strain": self.volumetric_strain,
            "Mean_Volumetric_Strain": self.mean_volumetric_strain,
            "Mean_Shear_Strain": self.mean_shear_strain,
            "Std_Shear_Strain": self.std_shear_strain,
            "Min_Shear_Strain": self.min_shear_strain,
            "Max_Shear_Strain": self.max_shear_strain,
        }


class MechanicalExtractor:
    """Extract mechanical features from trajectory frames."""

    def __init__(self, neighbor_cutoff: float = 3.5):
        self.neighbor_cutoff = neighbor_cutoff
        self._ref_volume = None
        self._ref_positions = None
        self._ref_box_lengths = None

    def set_reference(
        self,
        positions: np.ndarray,
        box_lengths: np.ndarray,
        volume: float,
    ):
        """Set reference configuration for strain calculation."""
        self._ref_volume = volume
        self._ref_positions = positions.copy()
        self._ref_box_lengths = box_lengths.copy()

    def extract(
        self,
        positions: np.ndarray,
        box_lengths: np.ndarray,
        volume: float,
    ) -> MechanicalFeatures:
        """
        Extract mechanical features.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
        box_lengths : np.ndarray, shape (3,)
        volume : float
        """
        features = MechanicalFeatures()

        if self._ref_volume is not None:
            features.volumetric_strain = (volume - self._ref_volume) / self._ref_volume

        if self._ref_positions is None:
            return features

        # Compute per-atom shear strain via atomic strain tensor
        try:
            shear_strains = self._compute_atomic_shear_strain(
                positions, box_lengths
            )
            if shear_strains is not None and len(shear_strains) > 0:
                features.mean_shear_strain = float(np.mean(shear_strains))
                features.std_shear_strain = float(np.std(shear_strains))
                features.min_shear_strain = float(np.min(shear_strains))
                features.max_shear_strain = float(np.max(shear_strains))
                features.mean_volumetric_strain = features.volumetric_strain
                features.shear_strain_distribution = shear_strains
        except Exception as e:
            logger.debug(f"Shear strain calculation failed: {e}")

        return features

    def _compute_atomic_shear_strain(
        self, positions: np.ndarray, box_lengths: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute per-atom von Mises shear strain using the local
        deformation gradient approach.

        Simplified version: uses displacement gradient tensor
        estimated from neighbor displacements.
        """
        N = len(positions)
        ref = self._ref_positions
        ref_box = self._ref_box_lengths

        if ref is None or len(ref) != N:
            return None

        # Displacements with MIC
        disp = positions - ref
        disp = disp - box_lengths * np.round(disp / box_lengths)

        # Build neighbor list in reference configuration
        wrapped_ref = ref % ref_box
        tree = cKDTree(wrapped_ref, boxsize=ref_box)

        shear_strains = np.zeros(N)

        # Sample atoms for efficiency (every 10th atom if N > 5000)
        sample_step = max(1, N // 1000)
        sampled_indices = np.arange(0, N, sample_step)

        for idx in sampled_indices:
            neighbors = tree.query_ball_point(wrapped_ref[idx], self.neighbor_cutoff)
            neighbors = [n for n in neighbors if n != idx]

            if len(neighbors) < 3:
                continue

            # Reference and current relative vectors
            dr_ref = wrapped_ref[neighbors] - wrapped_ref[idx]
            dr_ref = dr_ref - ref_box * np.round(dr_ref / ref_box)

            du = disp[neighbors] - disp[idx]

            # Solve for deformation gradient: F such that du ≈ F @ dr_ref.T
            try:
                # Least squares: F = du.T @ dr_ref @ inv(dr_ref.T @ dr_ref)
                A = dr_ref.T @ dr_ref
                B = du.T @ dr_ref
                F = B @ np.linalg.pinv(A)

                # Green-Lagrange strain: E = 0.5 * (F.T @ F - I)
                E = 0.5 * (F.T @ F + F + F.T)  # linearized strain

                # Von Mises shear strain invariant
                E_dev = E - (np.trace(E) / 3) * np.eye(3)
                eta = np.sqrt(0.5 * np.sum(E_dev ** 2))
                shear_strains[idx] = eta
            except Exception:
                shear_strains[idx] = 0.0

        # Interpolate for non-sampled atoms if subsampled
        if sample_step > 1:
            from scipy.interpolate import interp1d
            try:
                f_interp = interp1d(
                    sampled_indices, shear_strains[sampled_indices],
                    kind="nearest", fill_value="extrapolate"
                )
                shear_strains = f_interp(np.arange(N))
            except Exception:
                pass

        return shear_strains
