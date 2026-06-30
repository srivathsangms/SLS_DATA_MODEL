"""
Streaming LAMMPS trajectory (.lammpstrj) parser.

Memory-efficient: reads one frame at a time without loading the entire file.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Generator, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryFrame:
    """A single frame from a LAMMPS trajectory."""
    timestep: int
    num_atoms: int
    box_bounds: np.ndarray    # shape (3, 2) — [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    atom_ids: np.ndarray      # shape (N,)
    atom_types: np.ndarray    # shape (N,)
    positions: np.ndarray     # shape (N, 3)

    @property
    def box_lengths(self) -> np.ndarray:
        """Box dimensions [Lx, Ly, Lz]."""
        return self.box_bounds[:, 1] - self.box_bounds[:, 0]

    @property
    def volume(self) -> float:
        """Box volume in Å³."""
        L = self.box_lengths
        return float(L[0] * L[1] * L[2])


class TrajectoryParser:
    """Stream-parse LAMMPS dump files frame by frame."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def parse(self, every_n: int = 1) -> Generator[TrajectoryFrame, None, None]:
        """
        Yield TrajectoryFrame objects from file.

        Parameters
        ----------
        every_n : int
            Yield every Nth frame (1 = all frames).
        """
        frame_count = 0

        try:
            with open(self.filepath, "r") as f:
                while True:
                    frame = self._read_one_frame(f)
                    if frame is None:
                        break
                    frame_count += 1
                    if (frame_count - 1) % every_n == 0:
                        yield frame
        except Exception as e:
            logger.error(f"Error parsing {self.filepath} at frame {frame_count}: {e}")
            raise

        logger.debug(f"Parsed {frame_count} frames from {self.filepath}")

    def count_frames(self) -> int:
        """Count total frames without loading data."""
        count = 0
        with open(self.filepath, "r") as f:
            for line in f:
                if "ITEM: TIMESTEP" in line:
                    count += 1
        return count

    def _read_one_frame(self, f) -> Optional[TrajectoryFrame]:
        """Read a single frame from an open file handle."""
        # Read ITEM: TIMESTEP
        line = f.readline()
        if not line:
            return None
        if "ITEM: TIMESTEP" not in line:
            return None

        timestep = int(f.readline().strip())

        # Read ITEM: NUMBER OF ATOMS
        f.readline()  # ITEM: NUMBER OF ATOMS
        num_atoms = int(f.readline().strip())

        # Read ITEM: BOX BOUNDS
        f.readline()  # ITEM: BOX BOUNDS ...
        box_bounds = np.zeros((3, 2))
        for i in range(3):
            parts = f.readline().split()
            box_bounds[i, 0] = float(parts[0])
            box_bounds[i, 1] = float(parts[1])

        # Read ITEM: ATOMS header
        header_line = f.readline()  # ITEM: ATOMS id type x y z ...

        # Read atom data
        atom_ids = np.zeros(num_atoms, dtype=np.int32)
        atom_types = np.zeros(num_atoms, dtype=np.int32)
        positions = np.zeros((num_atoms, 3), dtype=np.float64)

        for i in range(num_atoms):
            parts = f.readline().split()
            atom_ids[i] = int(parts[0])
            atom_types[i] = int(parts[1])
            positions[i, 0] = float(parts[2])
            positions[i, 1] = float(parts[3])
            positions[i, 2] = float(parts[4])

        # Sort by atom ID for consistency
        sort_idx = np.argsort(atom_ids)
        atom_ids = atom_ids[sort_idx]
        atom_types = atom_types[sort_idx]
        positions = positions[sort_idx]

        return TrajectoryFrame(
            timestep=timestep,
            num_atoms=num_atoms,
            box_bounds=box_bounds,
            atom_ids=atom_ids,
            atom_types=atom_types,
            positions=positions,
        )
