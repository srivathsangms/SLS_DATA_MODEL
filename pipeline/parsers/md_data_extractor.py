"""
Comprehensive MD Data Extractor for SLS ML Pipeline.

Extracts ALL available features from the Owais Data directory:
  1. log.lammps       → Thermodynamic data (Temp, Press, PotEng, KinEng, TotEng, Density, Volume)
  2. bond_evol/*.csv  → Chemical bond type evolution (C-C, C-H, C-O, C-N, H-O, H-N, N-O, ...)
  3. temp_*.txt       → LAMMPS fix time-averaged temperature profiles per stage
  4. *.data files     → Atomic charge distribution (q per atom type)

Each of these contributes a distinct set of physics-informed features:
  - Thermodynamics: energetics, density, pressure evolution across stages
  - Bond chemistry: which bonds form/break during sintering (C-C crosslinks = strength)
  - Temperature: actual vs. target T, thermostat stability, drift
  - Charges: electrostatic environment per element
"""

import os
import re
import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import skew, kurtosis, entropy

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

COMPOSITIONS = ["5050", "6040", "7030"]
TEMPERATURES = [100, 150, 200, 250, 300, 350]

# Folder name pattern inside each composition directory
# e.g.  "50_50 100 C",  "60_40 250 C",  "70_30 350 C"
EXP_FOLDER_RE = re.compile(r"(\d{2})_(\d{2})\s+(\d+)\s*C", re.IGNORECASE)

COMP_FOLDER_PREFIX = {"5050": "50_50", "6040": "60_40", "7030": "70_30"}

# SLS stage ordering (only the 5 main sintering stages, skipping prep stages)
SLS_STAGES = ["equilibrium", "bed", "laser", "hold", "cooling"]

# Bond CSV stage name mapping (different CSVs use different names)
BOND_STAGE_MAP = {
    "equilibration": "equilibrium",
    "equil":         "equilibrium",
    "equilibrium":   "equilibrium",
    "bed":           "bed",
    "bed_heating":   "bed",
    "bed heating":   "bed",
    "laser":         "laser",
    "laser_heating": "laser",
    "laser heating": "laser",
    "laser_sintering": "laser",
    "laser sintering": "laser",
    "hold":          "hold",
    "hold_at":       "hold",
    "neck_formation":"hold",
    "neck formation":"hold",
    "cooling":       "cooling",
    "cooling_back":  "cooling",
    "cooling back":  "cooling",
}

# Log stage order (block index → stage name)
LOG_STAGE_ORDER = [
    "minimization", "nve_limit", "low_temp_relax",
    "gradual_equil", "equilibrium", "bed", "laser", "hold", "cooling",
]

# Bond types present in the CSV files
BOND_TYPES = ["C-C", "C-H", "C-O", "C-N", "H-O", "H-N", "N-O", "H-H", "O-O", "N-N"]

# Temp file keyword → stage name
TEMP_FILE_STAGE_MAP = {
    "equil": "equilibrium",
    "bed":   "bed",
    "laser": "laser",
    "hold":  "hold",
    "cool":  "cooling",
}

# Atom type → element (from LAMMPS data files)
ATOM_TYPE_ELEM = {1: "C", 2: "H", 3: "O", 4: "N"}


# ── Main extractor class ───────────────────────────────────────────────────────

class MdDataExtractor:
    """
    Extracts physics features from all MD data sources in the Owais Data directory.

    Parameters
    ----------
    owais_root : str
        Path to 'IIT JAMMU/Owais Data' root directory.
    """

    def __init__(self, owais_root: str):
        self.root = Path(owais_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Owais Data root not found: {self.root}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_all(self, cache_path: Optional[str] = None) -> pd.DataFrame:
        """
        Extract ALL MD features from all experiments and merge into one DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per (Composition, Temperature, Stage).
            Shape: ~90 rows × 200+ features.
        """
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading MD features from cache: {cache_path}")
            return pd.read_csv(cache_path)

        logger.info("=== Extracting MD features from all sources ===")
        all_records = {}  # key: (comp, temp, stage) → feature dict

        for comp in COMPOSITIONS:
            for temp in TEMPERATURES:
                exp_dir = self._find_experiment_dir(comp, temp)
                if exp_dir is None:
                    continue

                for stage in SLS_STAGES:
                    key = (comp, temp, stage)
                    all_records[key] = {
                        "Composition": comp,
                        "Temperature": temp,
                        "Stage": stage,
                    }

        # 1. Thermodynamic features from log.lammps
        logger.info("  [1/4] Extracting thermodynamic features from log.lammps...")
        thermo_feats = self._extract_thermo_all()
        for key, feats in thermo_feats.items():
            if key in all_records:
                all_records[key].update(feats)

        # 2. Bond chemistry from bond evolution CSVs
        logger.info("  [2/4] Extracting bond chemistry features...")
        bond_feats = self._extract_bond_all()
        for key, feats in bond_feats.items():
            if key in all_records:
                all_records[key].update(feats)

        # 3. Temperature profiles from temp_*.txt
        logger.info("  [3/4] Extracting temperature profile features...")
        temp_feats = self._extract_temp_profiles_all()
        for key, feats in temp_feats.items():
            if key in all_records:
                all_records[key].update(feats)

        # 4. Atomic charge distribution from *.data files
        logger.info("  [4/4] Extracting atomic charge features...")
        charge_feats = self._extract_charges_all()
        for (comp, temp), feats in charge_feats.items():
            for stage in SLS_STAGES:
                key = (comp, temp, stage)
                if key in all_records:
                    all_records[key].update(feats)

        records_list = list(all_records.values())
        if not records_list:
            logger.error("No MD records extracted!")
            return pd.DataFrame()

        df = pd.DataFrame(records_list)
        logger.info(f"MD features extracted: {df.shape[0]} rows × {df.shape[1]} cols")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"MD feature cache saved: {cache_path}")

        return df

    # ── 1. Thermodynamic features from log.lammps ──────────────────────────────

    def _extract_thermo_all(self) -> Dict[Tuple, Dict]:
        results = {}
        for comp in COMPOSITIONS:
            for temp in TEMPERATURES:
                exp_dir = self._find_experiment_dir(comp, temp)
                if exp_dir is None:
                    continue
                log_path = exp_dir / "log.lammps"
                if not log_path.exists():
                    continue
                try:
                    stage_data = self._parse_log(log_path)
                    for stage in SLS_STAGES:
                        if stage in stage_data and not stage_data[stage].empty:
                            key = (comp, temp, stage)
                            results[key] = self._thermo_features(stage_data[stage], stage)
                except Exception as e:
                    logger.warning(f"  log.lammps failed {comp}/{temp}: {e}")
        return results

    def _parse_log(self, log_path: Path) -> Dict[str, pd.DataFrame]:
        """Parse log.lammps and return dict of stage → thermo DataFrame."""
        HEADER_RE = re.compile(
            r"^\s*Step\s+Temp\s+Press\s+PotEng\s+KinEng\s+TotEng\s+Density",
            re.IGNORECASE
        )
        with open(log_path, "r") as f:
            lines = f.readlines()

        blocks = []
        i = 0
        while i < len(lines):
            if HEADER_RE.match(lines[i]):
                start = i + 1
                end = start
                while end < len(lines):
                    parts = lines[end].strip().split()
                    try:
                        int(parts[0])
                        end += 1
                    except (ValueError, IndexError):
                        break
                blocks.append((start, end))
                i = end
            else:
                i += 1

        stage_data = {}
        for idx, (start, end) in enumerate(blocks):
            if idx < len(LOG_STAGE_ORDER):
                stage_name = LOG_STAGE_ORDER[idx]
            else:
                continue
            rows = []
            for li in range(start, end):
                parts = lines[li].split()
                if len(parts) >= 7:
                    try:
                        rows.append({
                            "Step":    int(parts[0]),
                            "Temp":    float(parts[1]),
                            "Press":   float(parts[2]),
                            "PotEng":  float(parts[3]),
                            "KinEng":  float(parts[4]),
                            "TotEng":  float(parts[5]),
                            "Density": float(parts[6]),
                            "Volume":  float(parts[7]) if len(parts) > 7 else np.nan,
                        })
                    except (ValueError, IndexError):
                        continue
            if rows:
                stage_data[stage_name] = pd.DataFrame(rows)
        return stage_data

    def _thermo_features(self, df: pd.DataFrame, stage: str) -> Dict:
        feats = {}
        for col in ["Temp", "Press", "PotEng", "KinEng", "TotEng", "Density", "Volume"]:
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue
            p = f"Thermo_{col}"
            feats[f"{p}_Mean"]  = float(np.mean(vals))
            feats[f"{p}_Std"]   = float(np.std(vals))
            feats[f"{p}_Min"]   = float(np.min(vals))
            feats[f"{p}_Max"]   = float(np.max(vals))
            feats[f"{p}_Range"] = float(np.max(vals) - np.min(vals))
            feats[f"{p}_First"] = float(vals[0])
            feats[f"{p}_Last"]  = float(vals[-1])
            feats[f"{p}_Delta"] = float(vals[-1] - vals[0])
            if abs(np.mean(vals)) > 1e-9:
                feats[f"{p}_CV"]    = float(np.std(vals) / abs(np.mean(vals)))
                feats[f"{p}_PctChange"] = float((vals[-1] - vals[0]) / abs(vals[0])) if abs(vals[0]) > 1e-9 else 0.0

        # Derived thermodynamic quantities
        if "PotEng" in df.columns and "KinEng" in df.columns and "Volume" in df.columns:
            pe_mean = float(df["PotEng"].mean())
            ke_mean = float(df["KinEng"].mean())
            vol_mean = float(df["Volume"].mean())
            tot_mean = pe_mean + ke_mean
            if abs(tot_mean) > 1e-9:
                feats["Thermo_ThermalFraction"] = ke_mean / abs(tot_mean)
            if abs(vol_mean) > 1e-9:
                feats["Thermo_EnergyDensity"] = tot_mean / vol_mean
                feats["Thermo_PEDensity"]     = pe_mean / vol_mean
        return feats

    # ── 2. Bond chemistry from bond evolution CSVs ─────────────────────────────

    def _extract_bond_all(self) -> Dict[Tuple, Dict]:
        results = {}
        for comp in COMPOSITIONS:
            bond_dir = self._find_bond_dir(comp)
            if bond_dir is None:
                continue
            for temp in TEMPERATURES:
                csv_path = self._find_bond_csv(bond_dir, comp, temp)
                if csv_path is None:
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    df.columns = [c.strip() for c in df.columns]
                    # Normalize stage column
                    if "Stage" in df.columns:
                        df["Stage_norm"] = df["Stage"].str.strip().str.lower().map(
                            lambda s: BOND_STAGE_MAP.get(s, s)
                        )
                    else:
                        df["Stage_norm"] = "unknown"
                    for stage in SLS_STAGES:
                        stage_df = df[df["Stage_norm"] == stage]
                        if stage_df.empty:
                            continue
                        key = (comp, temp, stage)
                        results[key] = self._bond_features(stage_df, comp, temp, stage)
                except Exception as e:
                    logger.warning(f"  Bond CSV failed {comp}/{temp}: {e}")
        return results

    def _bond_features(self, df: pd.DataFrame, comp: str, temp: int, stage: str) -> Dict:
        feats = {}
        bond_cols = [c for c in BOND_TYPES if c in df.columns]

        # Total bonds stats
        if "Total_bonds" in df.columns:
            vals = df["Total_bonds"].dropna().values
            if len(vals) > 0:
                feats["Bond_Total_Mean"]  = float(np.mean(vals))
                feats["Bond_Total_Std"]   = float(np.std(vals))
                feats["Bond_Total_Delta"] = float(vals[-1] - vals[0]) if len(vals) > 1 else 0.0
                feats["Bond_Total_PctChange"] = float((vals[-1]-vals[0])/max(abs(vals[0]),1)) if len(vals)>1 else 0.0

        # Per-bond-type features
        total_mean = df["Total_bonds"].mean() if "Total_bonds" in df.columns else 1.0
        for bt in bond_cols:
            vals = df[bt].dropna().values
            if len(vals) == 0:
                continue
            p = f"Bond_{bt.replace('-','')}"
            feats[f"{p}_Mean"]   = float(np.mean(vals))
            feats[f"{p}_Std"]    = float(np.std(vals))
            feats[f"{p}_Delta"]  = float(vals[-1] - vals[0]) if len(vals) > 1 else 0.0
            feats[f"{p}_First"]  = float(vals[0])
            feats[f"{p}_Last"]   = float(vals[-1])
            # Fraction of total bonds
            if total_mean > 0:
                feats[f"{p}_Frac"] = float(np.mean(vals)) / total_mean

        # Chemical network features
        if len(bond_cols) > 0:
            # Bond entropy (diversity of bond types)
            bond_means = np.array([df[bt].mean() for bt in bond_cols if bt in df.columns])
            bond_means = bond_means[bond_means > 0]
            if bond_means.sum() > 0:
                probs = bond_means / bond_means.sum()
                feats["Bond_Entropy"] = float(entropy(probs))
                feats["Bond_Diversity"] = float(len(bond_means[bond_means > 0.01 * bond_means.sum()]))

            # Crosslink density: C-N + C-O bonds (reactive crosslinks in epoxy/PA12)
            crosslink_bonds = [b for b in ["C-N", "C-O"] if b in df.columns]
            if crosslink_bonds and "Total_bonds" in df.columns:
                cross_mean = sum(df[b].mean() for b in crosslink_bonds)
                feats["Bond_CrosslinkDensity"] = cross_mean / max(total_mean, 1)

            # Backbone strength: C-C bonds
            if "C-C" in df.columns and "Total_bonds" in df.columns:
                feats["Bond_BackboneStrength"] = df["C-C"].mean() / max(total_mean, 1)

            # Hydrogen bonding: H-O + H-N
            h_bonds = [b for b in ["H-O", "H-N"] if b in df.columns]
            if h_bonds and "Total_bonds" in df.columns:
                hb_mean = sum(df[b].mean() for b in h_bonds)
                feats["Bond_HBondFraction"] = hb_mean / max(total_mean, 1)

            # Stability: sigma / total bonds (low = stable)
            if "Total_bonds" in df.columns:
                total_std = df["Total_bonds"].std()
                feats["Bond_Stability"] = 1.0 - float(total_std / max(total_mean, 1))

        return feats

    # ── 3. Temperature profiles from temp_*.txt ────────────────────────────────

    def _extract_temp_profiles_all(self) -> Dict[Tuple, Dict]:
        results = {}
        for comp in COMPOSITIONS:
            for temp in TEMPERATURES:
                exp_dir = self._find_experiment_dir(comp, temp)
                if exp_dir is None:
                    continue
                for txt_file in exp_dir.glob("temp_*.txt"):
                    stage = self._temp_file_to_stage(txt_file.name)
                    if stage is None:
                        continue
                    key = (comp, temp, stage)
                    try:
                        feats = self._parse_temp_file(txt_file, temp)
                        results[key] = feats
                    except Exception as e:
                        logger.debug(f"  temp file {txt_file.name} failed: {e}")
        return results

    def _temp_file_to_stage(self, filename: str) -> Optional[str]:
        fn_lower = filename.lower()
        for kw, stage in TEMP_FILE_STAGE_MAP.items():
            if kw in fn_lower:
                return stage
        return None

    def _parse_temp_file(self, path: Path, target_temp_C: int) -> Dict:
        """Parse LAMMPS fix time-average temperature output."""
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rows.append((int(parts[0]), float(parts[1])))
                    except (ValueError, IndexError):
                        continue
        if not rows:
            return {}

        steps, temps_K = zip(*rows)
        temps_K = np.array(temps_K)
        target_K = target_temp_C + 273.15

        feats = {
            "TempProfile_Mean":       float(np.mean(temps_K)),
            "TempProfile_Std":        float(np.std(temps_K)),
            "TempProfile_Min":        float(np.min(temps_K)),
            "TempProfile_Max":        float(np.max(temps_K)),
            "TempProfile_Range":      float(np.max(temps_K) - np.min(temps_K)),
            "TempProfile_First":      float(temps_K[0]),
            "TempProfile_Last":       float(temps_K[-1]),
            "TempProfile_Drift":      float(temps_K[-1] - temps_K[0]),
            "TempProfile_Stability":  float(1.0 - np.std(temps_K) / max(np.mean(temps_K), 1)),
            "TempProfile_Skew":       float(skew(temps_K)),
        }
        # Thermostat adherence
        if target_K > 0:
            adherence = 1.0 - abs(np.mean(temps_K) - target_K) / target_K
            feats["TempProfile_Adherence"] = float(max(0.0, adherence))
            feats["TempProfile_TargetDiff"] = float(np.mean(temps_K) - target_K)
        return feats

    # ── 4. Atomic charge from .data files ──────────────────────────────────────

    def _extract_charges_all(self) -> Dict[Tuple, Dict]:
        results = {}
        for comp in COMPOSITIONS:
            for temp in TEMPERATURES:
                exp_dir = self._find_experiment_dir(comp, temp)
                if exp_dir is None:
                    continue
                # Use the initial structure file for charges
                data_file = exp_dir / f"reax_ready_{comp.replace('5050','5050').replace('6040','6040').replace('7030','7030')}.data"
                # Try different naming patterns
                candidates = list(exp_dir.glob("reax_ready_*.data"))
                if not candidates:
                    candidates = list(exp_dir.glob("*.data"))
                    candidates = [f for f in candidates if "stage" not in f.name.lower()]
                if not candidates:
                    continue
                try:
                    feats = self._parse_charges(candidates[0])
                    results[(comp, temp)] = feats
                except Exception as e:
                    logger.debug(f"  charge parse failed {comp}/{temp}: {e}")
        return results

    def _parse_charges(self, path: Path) -> Dict:
        """Parse LAMMPS data file and extract per-type atomic charge statistics."""
        in_atoms = False
        type_charges: Dict[int, list] = {1: [], 2: [], 3: [], 4: []}
        all_charges = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if "Atoms" in line:
                    in_atoms = True
                    continue
                if in_atoms and line == "":
                    continue
                if in_atoms and line and not line[0].isdigit():
                    break
                if in_atoms and line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            atom_type = int(parts[1])
                            charge    = float(parts[2])
                            type_charges[atom_type].append(charge)
                            all_charges.append(charge)
                        except (ValueError, IndexError):
                            continue

        feats = {}
        all_q = np.array(all_charges)
        if len(all_q) > 0:
            feats["Charge_Overall_Mean"] = float(np.mean(all_q))
            feats["Charge_Overall_Std"]  = float(np.std(all_q))
            feats["Charge_Overall_Max"]  = float(np.max(all_q))
            feats["Charge_Overall_Min"]  = float(np.min(all_q))
            feats["Charge_TotalImbalance"] = float(np.sum(all_q))

        for atom_type, elem in ATOM_TYPE_ELEM.items():
            q_arr = np.array(type_charges[atom_type])
            if len(q_arr) > 0:
                feats[f"Charge_{elem}_Mean"] = float(np.mean(q_arr))
                feats[f"Charge_{elem}_Std"]  = float(np.std(q_arr))
                feats[f"Charge_{elem}_Max"]  = float(np.max(q_arr))
                feats[f"Charge_{elem}_Min"]  = float(np.min(q_arr))
                feats[f"Charge_{elem}_Frac"] = float(len(q_arr) / max(len(all_charges), 1))
        return feats

    # ── Directory helpers ──────────────────────────────────────────────────────

    def _find_experiment_dir(self, comp: str, temp: int) -> Optional[Path]:
        """Locate the experiment directory for a given composition and temperature."""
        prefix = COMP_FOLDER_PREFIX.get(comp, "")
        comp_dir = self.root / comp
        if not comp_dir.exists():
            return None
        expected = f"{prefix} {temp} C"
        p = comp_dir / expected
        if p.exists():
            return p
        # Fuzzy match
        for d in comp_dir.iterdir():
            if d.is_dir():
                m = EXP_FOLDER_RE.match(d.name)
                if m and int(m.group(3)) == temp:
                    return d
        return None

    def _find_bond_dir(self, comp: str) -> Optional[Path]:
        comp_dir = self.root / comp
        for name in ["bond evol", "Bond evol", "bond_evol", "Bond_evol"]:
            p = comp_dir / name
            if p.exists():
                return p
        return None

    def _find_bond_csv(self, bond_dir: Path, comp: str, temp: int) -> Optional[Path]:
        """Find bond evolution CSV for a given composition and temperature."""
        # Multiple naming conventions observed:
        # "5050 bond_type_evolution at 100.csv"
        # "6040_bond_type_evolution_at 100.csv"
        # "7030_bond_type_evolution_at 100.csv"
        for f in bond_dir.glob("*.csv"):
            name = f.name
            if str(temp) in name and ("bond" in name.lower() or "evolution" in name.lower()):
                return f
        return None
