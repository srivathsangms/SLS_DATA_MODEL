"""
Bond feature extraction from ReaxFF bond data.

Features: Average Bond Order, Bond Length, Bond Formation/Breaking, Bond Type Counts.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Set

logger = logging.getLogger(__name__)


@dataclass
class BondFeatures:
    """Bond features for one timestep."""
    total_bonds: int = 0
    avg_bond_order: float = 0.0
    std_bond_order: float = 0.0
    avg_bonds_per_atom: float = 0.0
    broken_bonds: int = 0
    formed_bonds: int = 0
    net_bond_change: int = 0

    # Per-type counts
    bonds_CC: int = 0
    bonds_CH: int = 0
    bonds_CO: int = 0
    bonds_CN: int = 0
    bonds_HO: int = 0
    bonds_HN: int = 0
    bonds_NO: int = 0
    bonds_HH: int = 0
    bonds_OO: int = 0
    bonds_NN: int = 0

    # Bond order per type
    avg_bo_CC: float = 0.0
    avg_bo_CH: float = 0.0
    avg_bo_CO: float = 0.0
    avg_bo_CN: float = 0.0

    # Bond length stats (requires positions)
    mean_bond_length: float = 0.0
    std_bond_length: float = 0.0
    min_bond_length: float = 0.0
    max_bond_length: float = 0.0
    bond_length_distribution: Optional[np.ndarray] = None
    bond_order_distribution: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "Total_Bonds": self.total_bonds,
            "Avg_BondOrder": self.avg_bond_order,
            "Std_BondOrder": self.std_bond_order,
            "Avg_BondsPerAtom": self.avg_bonds_per_atom,
            "Broken_Bonds": self.broken_bonds,
            "Formed_Bonds": self.formed_bonds,
            "Net_BondChange": self.net_bond_change,
            "Bonds_CC": self.bonds_CC,
            "Bonds_CH": self.bonds_CH,
            "Bonds_CO": self.bonds_CO,
            "Bonds_CN": self.bonds_CN,
            "Bonds_HO": self.bonds_HO,
            "Bonds_HN": self.bonds_HN,
            "Bonds_NO": self.bonds_NO,
            "Bonds_HH": self.bonds_HH,
            "Bonds_OO": self.bonds_OO,
            "Bonds_NN": self.bonds_NN,
            "Avg_BO_CC": self.avg_bo_CC,
            "Avg_BO_CH": self.avg_bo_CH,
            "Avg_BO_CO": self.avg_bo_CO,
            "Avg_BO_CN": self.avg_bo_CN,
            "Mean_BondLength": self.mean_bond_length,
            "Std_BondLength": self.std_bond_length,
            "Min_BondLength": self.min_bond_length,
            "Max_BondLength": self.max_bond_length,
        }


class BondExtractor:
    """Extract bond features from parsed bond data and trajectory positions."""

    def __init__(self, atom_type_map: Dict[int, str] = None):
        self.atom_type_map = atom_type_map or {1: "C", 2: "H", 3: "O", 4: "N"}
        self._prev_bond_set: Optional[Set[Tuple[int, int]]] = None

    def extract(
        self,
        bond_stats: Dict[str, float],
        positions: Optional[np.ndarray] = None,
        atom_ids: Optional[np.ndarray] = None,
        bonds: Optional[List[Tuple[int, int, float]]] = None,
        box_lengths: Optional[np.ndarray] = None,
    ) -> BondFeatures:
        """
        Extract bond features from pre-computed bond statistics.

        Parameters
        ----------
        bond_stats : dict
            Output from BondParser.compute_bond_statistics()
        positions : np.ndarray, optional
            Atom positions for bond length calculation
        atom_ids : np.ndarray, optional
            Atom IDs corresponding to positions
        bonds : list, optional
            List of (atom_i, atom_j, bond_order) tuples
        box_lengths : np.ndarray, optional
            For minimum image convention
        """
        features = BondFeatures()

        features.total_bonds = int(bond_stats.get("total_bonds", 0))
        features.avg_bond_order = float(bond_stats.get("avg_bond_order", 0))
        features.std_bond_order = float(bond_stats.get("std_bond_order", 0))
        features.avg_bonds_per_atom = float(bond_stats.get("avg_bonds_per_atom", 0))

        # Per-type counts
        type_map = {
            "C-C": "bonds_CC", "C-H": "bonds_CH", "C-O": "bonds_CO",
            "C-N": "bonds_CN", "H-O": "bonds_HO", "H-N": "bonds_HN",
            "N-O": "bonds_NO", "H-H": "bonds_HH", "O-O": "bonds_OO",
            "N-N": "bonds_NN",
        }
        for bond_type, attr in type_map.items():
            setattr(features, attr, int(bond_stats.get(f"bonds_{bond_type}", 0)))

        # Per-type bond orders
        features.avg_bo_CC = float(bond_stats.get("avg_bo_C-C", 0))
        features.avg_bo_CH = float(bond_stats.get("avg_bo_C-H", 0))
        features.avg_bo_CO = float(bond_stats.get("avg_bo_C-O", 0))
        features.avg_bo_CN = float(bond_stats.get("avg_bo_C-N", 0))

        # Bond formation/breaking
        if bonds is not None:
            curr_set = set((a, b) for a, b, _ in bonds)
            if self._prev_bond_set is not None:
                broken = self._prev_bond_set - curr_set
                formed = curr_set - self._prev_bond_set
                features.broken_bonds = len(broken)
                features.formed_bonds = len(formed)
                features.net_bond_change = len(formed) - len(broken)
            self._prev_bond_set = curr_set

        # Bond lengths
        if positions is not None and atom_ids is not None and bonds is not None and box_lengths is not None:
            lengths = self._compute_bond_lengths(
                positions, atom_ids, bonds, box_lengths
            )
            if len(lengths) > 0:
                features.mean_bond_length = float(np.mean(lengths))
                features.std_bond_length = float(np.std(lengths))
                features.min_bond_length = float(np.min(lengths))
                features.max_bond_length = float(np.max(lengths))
                features.bond_length_distribution = lengths

        # Bond order distribution
        if bonds is not None:
            features.bond_order_distribution = np.array([bo for _, _, bo in bonds])

        return features

    def _compute_bond_lengths(
        self,
        positions: np.ndarray,
        atom_ids: np.ndarray,
        bonds: List[Tuple[int, int, float]],
        box_lengths: np.ndarray,
    ) -> np.ndarray:
        """Compute bond lengths from positions using minimum image convention."""
        id_to_idx = {aid: i for i, aid in enumerate(atom_ids)}
        lengths = []

        for aid, nid, bo in bonds:
            if aid in id_to_idx and nid in id_to_idx:
                pos_a = positions[id_to_idx[aid]]
                pos_b = positions[id_to_idx[nid]]
                delta = pos_b - pos_a
                delta = delta - box_lengths * np.round(delta / box_lengths)
                dist = np.linalg.norm(delta)
                lengths.append(dist)

        return np.array(lengths) if lengths else np.array([])

    def reset(self):
        """Reset the bond tracking state (for new stage)."""
        self._prev_bond_set = None
