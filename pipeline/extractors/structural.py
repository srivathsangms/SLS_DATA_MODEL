"""
Structural feature extraction from trajectory frames.

Features: Coordination Number, Atomic Volume, Cavity Radius, RDF.
Uses OVITO Python API for Voronoi (PBC-aware), vectorized RDF.

PERFORMANCE NOTES:
- OVITO Voronoi is ~100x faster than SciPy ghost-atom approach for PBC
- RDF uses vectorized histogram instead of Python loop over pairs
- CN uses cKDTree which handles PBC natively
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from scipy.spatial import cKDTree
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


@dataclass
class RDFResult:
    """Radial Distribution Function result."""
    r: np.ndarray            # bin centres (Å)
    g_r: np.ndarray          # g(r) values
    first_peak_r: float = 0.0
    first_peak_height: float = 0.0
    second_peak_r: float = 0.0
    second_peak_height: float = 0.0


@dataclass
class StructuralFeatures:
    """Structural features for one trajectory frame."""
    # Coordination Number
    mean_cn: float = 0.0
    std_cn: float = 0.0
    min_cn: float = 0.0
    max_cn: float = 0.0
    cn_distribution: Optional[np.ndarray] = None

    # Atomic Volume
    mean_atomic_volume: float = 0.0
    std_atomic_volume: float = 0.0
    min_atomic_volume: float = 0.0
    max_atomic_volume: float = 0.0
    atomic_volume_distribution: Optional[np.ndarray] = None

    # Cavity Radius
    mean_cavity_radius: float = 0.0
    std_cavity_radius: float = 0.0
    min_cavity_radius: float = 0.0
    max_cavity_radius: float = 0.0
    cavity_radius_distribution: Optional[np.ndarray] = None

    # RDF
    rdf: Optional[RDFResult] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for DataFrame row."""
        d = {
            "Mean_CN": self.mean_cn,
            "Std_CN": self.std_cn,
            "Min_CN": self.min_cn,
            "Max_CN": self.max_cn,
            "Mean_AtomicVolume": self.mean_atomic_volume,
            "Std_AtomicVolume": self.std_atomic_volume,
            "Min_AtomicVolume": self.min_atomic_volume,
            "Max_AtomicVolume": self.max_atomic_volume,
            "Mean_CavityRadius": self.mean_cavity_radius,
            "Std_CavityRadius": self.std_cavity_radius,
            "Min_CavityRadius": self.min_cavity_radius,
            "Max_CavityRadius": self.max_cavity_radius,
        }
        if self.rdf is not None:
            d["RDF_FirstPeak_r"] = self.rdf.first_peak_r
            d["RDF_FirstPeak_height"] = self.rdf.first_peak_height
            d["RDF_SecondPeak_r"] = self.rdf.second_peak_r
            d["RDF_SecondPeak_height"] = self.rdf.second_peak_height
        return d


class StructuralExtractor:
    """Extract structural features from trajectory frames."""

    def __init__(
        self,
        neighbor_cutoff: float = 3.5,
        rdf_cutoff: float = 12.0,
        rdf_bins: int = 200,
        voronoi_enabled: bool = True,
        use_ovito: bool = True,
    ):
        self.neighbor_cutoff = neighbor_cutoff
        self.rdf_cutoff = rdf_cutoff
        self.rdf_bins = rdf_bins
        self.voronoi_enabled = voronoi_enabled
        self.use_ovito = use_ovito
        self._ovito_available = False

        if use_ovito:
            try:
                import ovito
                self._ovito_available = True
                logger.debug("OVITO available — using for Voronoi")
            except ImportError:
                logger.warning("OVITO not available, Voronoi will use simple V/N estimate")

    def extract(
        self,
        positions: np.ndarray,
        box_bounds: np.ndarray,
        atom_types: Optional[np.ndarray] = None,
    ) -> StructuralFeatures:
        """
        Extract structural features from a single frame.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
        box_bounds : np.ndarray, shape (3, 2) — [[xlo, xhi], ...]
        atom_types : np.ndarray, shape (N,), optional
        """
        features = StructuralFeatures()
        box_lengths = box_bounds[:, 1] - box_bounds[:, 0]
        wrapped = positions % box_lengths

        # 1. Coordination Number (KDTree with PBC — fast)
        cn_values = self._compute_coordination_number(wrapped, box_lengths)
        if cn_values is not None and len(cn_values) > 0:
            features.mean_cn = float(np.mean(cn_values))
            features.std_cn = float(np.std(cn_values))
            features.min_cn = float(np.min(cn_values))
            features.max_cn = float(np.max(cn_values))
            features.cn_distribution = cn_values

        # 2. Atomic Volume + Cavity Radius
        if self.voronoi_enabled:
            vol, cav = self._compute_voronoi(wrapped, box_lengths)
            if vol is not None and len(vol) > 0:
                features.mean_atomic_volume = float(np.mean(vol))
                features.std_atomic_volume = float(np.std(vol))
                features.min_atomic_volume = float(np.min(vol))
                features.max_atomic_volume = float(np.max(vol))
                features.atomic_volume_distribution = vol
            if cav is not None and len(cav) > 0:
                features.mean_cavity_radius = float(np.mean(cav))
                features.std_cavity_radius = float(np.std(cav))
                features.min_cavity_radius = float(np.min(cav))
                features.max_cavity_radius = float(np.max(cav))
                features.cavity_radius_distribution = cav

        # 3. RDF (vectorized — fast)
        rdf = self._compute_rdf_vectorized(wrapped, box_lengths)
        features.rdf = rdf

        return features

    def _compute_coordination_number(
        self, wrapped: np.ndarray, box_lengths: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute coordination number using KDTree with PBC."""
        try:
            tree = cKDTree(wrapped, boxsize=box_lengths)
            cn = np.array(
                tree.query_ball_point(wrapped, self.neighbor_cutoff, return_length=True),
                dtype=np.float64,
            )
            cn = cn - 1  # Subtract self-count
            return cn
        except Exception as e:
            logger.warning(f"CN calculation failed: {e}")
            return None

    def _compute_voronoi(
        self, wrapped: np.ndarray, box_lengths: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute atomic volumes and cavity radii."""
        if self._ovito_available:
            return self._compute_voronoi_ovito(wrapped, box_lengths)
        else:
            return self._compute_voronoi_simple(wrapped, box_lengths)

    def _compute_voronoi_ovito(
        self, wrapped: np.ndarray, box_lengths: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute Voronoi using OVITO — fast, PBC-aware."""
        try:
            from ovito.data import DataCollection, SimulationCell, Particles
            from ovito.modifiers import VoronoiAnalysisModifier
            from ovito.pipeline import StaticSource, Pipeline

            data = DataCollection()

            cell = SimulationCell()
            cell_matrix = np.zeros((3, 4))
            cell_matrix[0, 0] = box_lengths[0]
            cell_matrix[1, 1] = box_lengths[1]
            cell_matrix[2, 2] = box_lengths[2]
            cell[:, :] = cell_matrix
            cell.pbc = (True, True, True)
            data.objects.append(cell)

            particles = Particles()
            particles.create_property("Position", data=wrapped.copy())
            data.objects.append(particles)

            pipeline = Pipeline(source=StaticSource(data=data))
            pipeline.modifiers.append(VoronoiAnalysisModifier(
                compute_indices=False,
                use_radii=False,
            ))

            result = pipeline.compute()
            volumes = np.array(result.particles["Atomic Volume"])

            # Cavity radius ~ (3V / 4π)^(1/3)
            cavity_radii = (3 * volumes / (4 * np.pi)) ** (1.0 / 3.0)

            return volumes, cavity_radii

        except Exception as e:
            logger.warning(f"OVITO Voronoi failed: {e}, falling back to simple estimate")
            return self._compute_voronoi_simple(wrapped, box_lengths)

    def _compute_voronoi_simple(
        self, wrapped: np.ndarray, box_lengths: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Simple per-atom volume estimate using nearest-neighbor distances.
        Much faster than full Voronoi tessellation with ghost atoms.
        """
        try:
            N = len(wrapped)
            tree = cKDTree(wrapped, boxsize=box_lengths)
            # Get distance to nearest neighbor for each atom
            dists, _ = tree.query(wrapped, k=2)  # k=2: self + nearest
            nn_dist = dists[:, 1]  # nearest neighbor distance

            # Approximate atomic volume as sphere with radius = nn_dist/2
            # This is crude but avoids the 27x ghost atom explosion
            volumes = (4.0 / 3.0) * np.pi * (nn_dist / 2) ** 3

            # Also compute a better estimate from total volume / N
            total_volume = float(np.prod(box_lengths))
            avg_vol = total_volume / N

            # Use average as baseline, modulate by nn_dist ratio
            mean_nn = np.mean(nn_dist)
            volumes = avg_vol * (nn_dist / mean_nn) ** 3

            cavity_radii = (3 * volumes / (4 * np.pi)) ** (1.0 / 3.0)

            return volumes, cavity_radii

        except Exception as e:
            logger.warning(f"Simple Voronoi estimate failed: {e}")
            return None, None

    def _compute_rdf_vectorized(
        self, wrapped: np.ndarray, box_lengths: np.ndarray
    ) -> RDFResult:
        """
        Compute RDF using vectorized distance computation.
        
        Uses KDTree sparse_distance_matrix for efficient pair enumeration,
        then np.histogram for binning (no Python loop over pairs).
        """
        N = len(wrapped)
        volume = float(np.prod(box_lengths))
        rho = N / volume

        r_edges = np.linspace(0, self.rdf_cutoff, self.rdf_bins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

        try:
            tree = cKDTree(wrapped, boxsize=box_lengths)
            # sparse_distance_matrix returns a dok_matrix of distances
            dist_matrix = tree.sparse_distance_matrix(tree, self.rdf_cutoff, output_type='ndarray')
            # dist_matrix is array of (i, j, dist) dtype structured array
            distances = dist_matrix['v']  # just the distances
            # Remove self-pairs (distance == 0)
            distances = distances[distances > 0]

            # Vectorized histogram
            hist, _ = np.histogram(distances, bins=r_edges)

            # Normalize
            shell_volumes = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
            ideal_counts = rho * shell_volumes
            g_r = np.zeros(self.rdf_bins)
            valid = ideal_counts > 0
            g_r[valid] = hist[valid] / (N * ideal_counts[valid])

        except Exception as e:
            logger.warning(f"Vectorized RDF failed: {e}, using fallback")
            g_r = np.zeros(self.rdf_bins)

        # Find peaks
        result = RDFResult(r=r_centers, g_r=g_r)
        try:
            peaks, _ = find_peaks(g_r, height=0.5, distance=5)
            if len(peaks) >= 1:
                result.first_peak_r = float(r_centers[peaks[0]])
                result.first_peak_height = float(g_r[peaks[0]])
            if len(peaks) >= 2:
                result.second_peak_r = float(r_centers[peaks[1]])
                result.second_peak_height = float(g_r[peaks[1]])
        except Exception:
            pass

        return result
