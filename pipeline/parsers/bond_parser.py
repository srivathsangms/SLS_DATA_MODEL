"""
ReaxFF bond file parser.

Parses bonds_all_stages.reaxff to extract bond connectivity, bond orders,
and per-atom bonding information for each timestep.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Generator, Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BondTimestep:
    """Bond data for one timestep."""
    timestep: int
    num_atoms: int
    # Per-atom data
    atom_ids: np.ndarray          # (N,)
    atom_types: np.ndarray        # (N,)
    num_bonds: np.ndarray         # (N,) number of bonds per atom
    total_bond_order: np.ndarray  # (N,) abo - total bond order
    charges: np.ndarray           # (N,)
    # Bond connectivity: list of (atom_i, atom_j, bond_order) tuples
    bonds: List[Tuple[int, int, float]]


class BondParser:
    """Stream-parse ReaxFF bond files."""

    def __init__(self, filepath: str, atom_type_map: Dict[int, str] = None):
        self.filepath = filepath
        self.atom_type_map = atom_type_map or {1: "C", 2: "H", 3: "O", 4: "N"}

    def parse(self, every_n: int = 1) -> Generator[BondTimestep, None, None]:
        """Yield BondTimestep objects from file."""
        block_count = 0

        try:
            with open(self.filepath, "r") as f:
                while True:
                    ts = self._read_one_block(f)
                    if ts is None:
                        break
                    block_count += 1
                    if (block_count - 1) % every_n == 0:
                        yield ts
        except Exception as e:
            logger.error(f"Error parsing bond file at block {block_count}: {e}")
            raise

        logger.debug(f"Parsed {block_count} bond timesteps from {self.filepath}")

    def _read_one_block(self, f) -> Optional[BondTimestep]:
        """Read one timestep block from ReaxFF bond file."""
        # Find the timestep header
        timestep = None
        num_atoms = None

        while True:
            line = f.readline()
            if not line:
                return None

            line = line.strip()
            if line.startswith("# Timestep"):
                timestep = int(line.split()[-1])
            elif line.startswith("# Number of particles"):
                num_atoms = int(line.split()[-1])
            elif line.startswith("# id type nb"):
                # Header line - now read data
                break
            elif timestep is not None and not line.startswith("#") and line:
                # We hit data before seeing the column header; reprocess
                break

        if timestep is None:
            return None

        # Read atom data
        atom_ids_list = []
        atom_types_list = []
        num_bonds_list = []
        abo_list = []
        charge_list = []
        bonds_list = []

        atoms_read = 0
        while atoms_read < (num_atoms or 99999):
            pos = f.tell()
            line = f.readline()
            if not line:
                break

            stripped = line.strip()
            if stripped.startswith("# Timestep") or not stripped:
                # Start of next block or empty line
                if stripped.startswith("# Timestep"):
                    f.seek(pos)  # Put back
                break
            if stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 4:
                continue

            try:
                atom_id = int(parts[0])
                atom_type = int(parts[1])
                nb = int(parts[2])  # number of bonds
            except ValueError:
                continue

            atom_ids_list.append(atom_id)
            atom_types_list.append(atom_type)
            num_bonds_list.append(nb)

            # Parse neighbor IDs and bond orders
            # Format: id type nb id_1...id_nb mol bo_1...bo_nb abo nlp q
            # Neighbor IDs: parts[3:3+nb]
            # mol: parts[3+nb]
            # Bond orders: parts[3+nb+1:3+nb+1+nb]
            # abo: parts[3+nb+1+nb]
            # nlp: parts[3+nb+1+nb+1]
            # q: parts[3+nb+1+nb+2]

            neighbor_ids = []
            bond_orders = []
            for j in range(nb):
                idx = 3 + j
                if idx < len(parts):
                    neighbor_ids.append(int(parts[idx]))

            mol_idx = 3 + nb
            bo_start = mol_idx + 1
            for j in range(nb):
                idx = bo_start + j
                if idx < len(parts):
                    bond_orders.append(float(parts[idx]))

            abo_idx = bo_start + nb
            if abo_idx < len(parts):
                abo_list.append(float(parts[abo_idx]))
            else:
                abo_list.append(0.0)

            q_idx = abo_idx + 2  # skip nlp
            if q_idx < len(parts):
                charge_list.append(float(parts[q_idx]))
            else:
                charge_list.append(0.0)

            # Store bonds (only store where atom_id < neighbor to avoid double counting)
            for nid, bo in zip(neighbor_ids, bond_orders):
                if atom_id < nid:
                    bonds_list.append((atom_id, nid, bo))

            atoms_read += 1

        if not atom_ids_list:
            return None

        return BondTimestep(
            timestep=timestep,
            num_atoms=len(atom_ids_list),
            atom_ids=np.array(atom_ids_list, dtype=np.int32),
            atom_types=np.array(atom_types_list, dtype=np.int32),
            num_bonds=np.array(num_bonds_list, dtype=np.int32),
            total_bond_order=np.array(abo_list, dtype=np.float64),
            charges=np.array(charge_list, dtype=np.float64),
            bonds=bonds_list,
        )

    def compute_bond_statistics(
        self, ts: BondTimestep
    ) -> Dict[str, float]:
        """Compute bond statistics for a single timestep."""
        stats = {
            "total_bonds": len(ts.bonds),
            "avg_bond_order": float(np.mean(ts.total_bond_order)) if len(ts.total_bond_order) > 0 else 0,
            "std_bond_order": float(np.std(ts.total_bond_order)) if len(ts.total_bond_order) > 0 else 0,
            "avg_bonds_per_atom": float(np.mean(ts.num_bonds)) if len(ts.num_bonds) > 0 else 0,
        }

        # Count by bond type
        type_counts = {}
        bond_orders_by_type = {}
        for aid, nid, bo in ts.bonds:
            # Look up atom types
            idx_a = np.searchsorted(ts.atom_ids, aid)
            idx_b = np.searchsorted(ts.atom_ids, nid)
            if idx_a < len(ts.atom_ids) and ts.atom_ids[idx_a] == aid:
                type_a = self.atom_type_map.get(ts.atom_types[idx_a], "?")
            else:
                type_a = "?"
            if idx_b < len(ts.atom_ids) and ts.atom_ids[idx_b] == nid:
                type_b = self.atom_type_map.get(ts.atom_types[idx_b], "?")
            else:
                type_b = "?"

            bond_type = "-".join(sorted([type_a, type_b]))
            type_counts[bond_type] = type_counts.get(bond_type, 0) + 1
            if bond_type not in bond_orders_by_type:
                bond_orders_by_type[bond_type] = []
            bond_orders_by_type[bond_type].append(bo)

        for bt, count in type_counts.items():
            stats[f"bonds_{bt}"] = count
            orders = bond_orders_by_type[bt]
            stats[f"avg_bo_{bt}"] = float(np.mean(orders))

        return stats

    def compute_bond_changes(
        self, ts_prev: BondTimestep, ts_curr: BondTimestep
    ) -> Dict[str, int]:
        """Compute broken and newly formed bonds between two timesteps."""
        prev_bonds = set((a, b) for a, b, _ in ts_prev.bonds)
        curr_bonds = set((a, b) for a, b, _ in ts_curr.bonds)

        broken = prev_bonds - curr_bonds
        formed = curr_bonds - prev_bonds

        return {
            "broken_bonds": len(broken),
            "formed_bonds": len(formed),
            "net_bond_change": len(formed) - len(broken),
        }
