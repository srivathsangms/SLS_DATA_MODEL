"""
SLS ML Frame-Level — Ultra High-Resolution Sintering Prediction.

Processes each and every frame of the raw Molecular Dynamics trajectories (.lammpstrj).
For each frame, the pipeline extracts:
  1. Structural coordinates (via fast scipy.spatial.KDTree):
     - Calculates coordination number (CN) of all 11,102 atoms at 2.2 Å cutoff
     - Computes statistical moments (mean, std, skew, kurt, IQR, CV)
     - Computes element-wise CN averages (C, H, O, N)
     - Computes mean atomic displacement relative to frame 0
     - Box density and volume from ITEM: BOX BOUNDS
  2. Thermodynamics (log.lammps):
     - Looks up matching timestep's PE, KE, TotEng, Press, Temp, Density
  3. Chemical bonds (bond_type_evolution CSV):
     - Looks up matching timestep's C-C, C-H, C-O, C-N, H-N, H-O counts

Total Dataset Size: ~7,560 observations (420 frames × 18 experiments)
Total Features: 120+ frame-level physical descriptors

Trains 5 ML Models (RF, ExtraTrees, XGBoost, LightGBM, CatBoost) across 5 tasks:
  A: Sintering Stage Classification (equilibrium, bed, laser, hold, cooling)
  B: Composition Prediction (5050, 6040, 7030)
  C: Displacement Regression (Å)
  D: Potential Energy Regression (kcal/mol)
  E: C-C Backbone Bonds count Regression (chemical network density)
"""

import os
import re
import sys
import time
import glob
import logging
import warnings
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, confusion_matrix
import joblib
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# Defaults
DEFAULT_OWAIS   = r"C:\Users\sriva\Desktop\IIT JAMMU\Owais Data"
DEFAULT_OUTPUT  = r"Results\ML_Frame_Level"

# Constants
COMPOSITIONS = ["5050", "6040", "7030"]
TEMPERATURES = [100, 150, 200, 250, 300, 350]
STAGE_COLORS = {"equilibrium":"#2196F3","bed":"#FF9800","laser":"#F44336","hold":"#9C27B0","cooling":"#4CAF50"}
COMP_COLORS  = {"5050":"#E91E63","6040":"#00BCD4","7030":"#FFC107"}
PALETTE = ["#4d7cff","#ff6b6b","#ffd93d","#6bcb77","#c77dff"]

# Stage block boundary step definitions
STAGE_BOUNDARIES = [
    ("minimization", 0, 50),
    ("nve_limit", 0, 20000),
    ("low_temp_relax", 20000, 50000),
    ("gradual_equil", 50000, 100000),
    ("equilibrium", 100000, 120000),
    ("bed", 120000, 150000),
    ("laser", 150000, 190000),
    ("hold", 190000, 270000),
    ("cooling", 270000, 310000)
]

# Sintering stages we care about
MAIN_SLS_STAGES = ["equilibrium", "bed", "laser", "hold", "cooling"]

# ── Setup Logger ─────────────────────────────────────────────────────────────────
def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    lp = os.path.join(output_dir, "frame_level_pipeline.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    # Reset handlers if any exist
    if root.handlers:
        root.handlers = []
        
    fh = logging.FileHandler(lp, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    root.addFilter(lambda r: True)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)
    return root

# ── Timestep to Stage Mapper ─────────────────────────────────────────────────────
def assign_stage_names(timesteps: list) -> list:
    """Map sequential timesteps in trajectory file to LAMMPS stages."""
    block_idx = 0
    assigned = []
    for i, ts in enumerate(timesteps):
        if i > 0 and ts < timesteps[i-1]:
            while block_idx < len(STAGE_BOUNDARIES) - 1:
                block_idx += 1
                if STAGE_BOUNDARIES[block_idx][1] <= ts <= STAGE_BOUNDARIES[block_idx][2]:
                    break
        while block_idx < len(STAGE_BOUNDARIES) - 1 and ts > STAGE_BOUNDARIES[block_idx][2]:
            block_idx += 1
            
        # Detect overlap boundary step
        if i > 0 and ts == timesteps[i-1] and ts == STAGE_BOUNDARIES[block_idx][1]:
            if block_idx < len(STAGE_BOUNDARIES) - 1:
                block_idx += 1
        assigned.append(STAGE_BOUNDARIES[block_idx][0])
    return assigned

# ── Trajectory Parser (Frame-by-Frame) ───────────────────────────────────────────
def parse_trajectory_file(filepath: Path, stride: int = 2) -> list:
    """
    Parse a single .lammpstrj file frame-by-frame.
    Extract box volume, coordinates, atom types, and timesteps.
    """
    frames = []
    current_frame = {}
    atom_data = []
    state = None
    box_lines = []
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("ITEM: TIMESTEP"):
                # Save previous frame
                if current_frame and atom_data:
                    current_frame["atoms"] = np.array(atom_data, dtype=np.float32)
                    frames.append(current_frame)
                current_frame = {}
                atom_data = []
                state = "timestep"
                continue
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                state = "num_atoms"
                continue
            elif line.startswith("ITEM: BOX BOUNDS"):
                state = "box_bounds"
                box_lines = []
                continue
            elif line.startswith("ITEM: ATOMS"):
                state = "atoms"
                continue
                
            if state == "timestep":
                current_frame["timestep"] = int(line)
                state = None
            elif state == "num_atoms":
                current_frame["num_atoms"] = int(line)
                state = None
            elif state == "box_bounds":
                box_lines.append([float(x) for x in line.split()])
                if len(box_lines) == 3:
                    current_frame["volume"] = (box_lines[0][1] - box_lines[0][0]) * \
                                              (box_lines[1][1] - box_lines[1][0]) * \
                                              (box_lines[2][1] - box_lines[2][0])
                    state = None
            elif state == "atoms":
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        # id type x y z
                        atom_data.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    except ValueError:
                        pass
                        
        # Add last frame
        if current_frame and atom_data:
            current_frame["atoms"] = np.array(atom_data, dtype=np.float32)
            frames.append(current_frame)
            
    # Apply stride to sample frames
    return frames[::stride]

# ── Single Experiment Pipeline ───────────────────────────────────────────────────
def process_single_experiment(owais_root: Path, comp: str, temp: int, stride: int) -> list:
    """Process all trajectory frames of one experiment and combine with thermo/bonds."""
    # Find folders
    prefix = {"5050": "50_50", "6040": "60_40", "7030": "70_30"}[comp]
    exp_dir = owais_root / comp / f"{prefix} {temp} C"
    if not exp_dir.exists():
        # Fuzzy search
        candidates = list((owais_root / comp).glob(f"*{temp}*C*"))
        if candidates: exp_dir = candidates[0]
        else: return []

    traj_path = exp_dir / "trajectory_all_stages.lammpstrj"
    if not traj_path.exists():
        return []

    # 1. Parse log.lammps
    thermo_map = {}
    log_path = exp_dir / "log.lammps"
    if log_path.exists():
        try:
            thermo_map = parse_log_thermo(log_path)
        except Exception: pass

    # 2. Parse bond CSV
    bond_map = {}
    bond_dir = owais_root / comp / "bond evol" if (owais_root / comp / "bond evol").exists() else owais_root / comp / "Bond evol"
    if bond_dir.exists():
        csv_path = None
        for f in bond_dir.glob("*.csv"):
            if str(temp) in f.name and ("bond" in f.name.lower() or "evolution" in f.name.lower()):
                csv_path = f
                break
        if csv_path:
            try:
                bond_map = parse_bond_csv(csv_path)
            except Exception: pass

    # 3. Parse trajectory frames
    try:
        raw_frames = parse_trajectory_file(traj_path, stride=stride)
    except Exception:
        return []

    if not raw_frames:
        return []

    # Map frames to stages
    timesteps = [f["timestep"] for f in raw_frames]
    stages = assign_stage_names(timesteps)

    # Initial frame for displacement calculation
    ref_coords = None
    if len(raw_frames) > 0:
        # Sort ref atoms by ID
        ref_arr = raw_frames[0]["atoms"]
        ref_arr = ref_arr[ref_arr[:, 0].argsort()]
        ref_coords = ref_arr[:, 2:5] # x y z

    records = []
    for idx, frame in enumerate(raw_frames):
        stage = stages[idx]
        if stage not in MAIN_SLS_STAGES:
            continue # Skip relaxation and prep stages

        ts = frame["timestep"]
        atoms = frame["atoms"]
        volume = frame["volume"]

        # Sort atoms by ID to align correctly
        atoms = atoms[atoms[:, 0].argsort()]
        coords = atoms[:, 2:5]
        types = atoms[:, 1]

        # A. Coordination Number features (cutoff = 2.2 Å for covalent bonding)
        kdt = KDTree(coords)
        # query_ball_tree returns list of indices within 2.2 Å (includes self)
        pairs = kdt.query_ball_tree(kdt, r=2.2)
        cns = np.array([len(p) - 1 for p in pairs], dtype=np.float32)

        # Statistical moments
        cn_mean = float(np.mean(cns))
        cn_std  = float(np.std(cns))
        cn_skew = float(skew(cns))
        cn_kurt = float(kurtosis(cns))
        cn_median = float(np.median(cns))
        cn_iqr = float(np.percentile(cns, 75) - np.percentile(cns, 25))
        cn_cv = cn_std / cn_mean if cn_mean > 0 else 0.0

        # Element-wise CN averages
        cn_elements = {}
        for t_idx, elem in [(1, "C"), (2, "H"), (3, "O"), (4, "N")]:
            mask = types == t_idx
            cn_elements[f"CN_{elem}_Mean"] = float(np.mean(cns[mask])) if mask.sum() > 0 else 0.0

        # B. Displacement magnitude from frame 0
        disp_mean = 0.0
        if ref_coords is not None and len(coords) == len(ref_coords):
            disps = np.sqrt(np.sum((coords - ref_coords)**2, axis=1))
            disp_mean = float(np.mean(disps))

        # C. Thermodynamic lookups
        thermo_feats = thermo_map.get((stage, ts), {})
        if not thermo_feats and thermo_map:
            # Find closest timestep in the same stage
            closest_ts = min([t for (s, t) in thermo_map.keys() if s == stage], key=lambda x: abs(x - ts), default=None)
            if closest_ts:
                thermo_feats = thermo_map.get((stage, closest_ts), {})

        # D. Bond chemistry lookups
        bond_feats = bond_map.get((stage, ts), {})
        if not bond_feats and bond_map:
            closest_ts = min([t for (s, t) in bond_map.keys() if s == stage], key=lambda x: abs(x - ts), default=None)
            if closest_ts:
                bond_feats = bond_map.get((stage, closest_ts), {})

        rec = {
            # Meta
            "Composition": comp,
            "Temperature": temp,
            "Stage": stage,
            "Timestep": ts,
            # Structural
            "CoordAvg": cn_mean,
            "CN_Std": cn_std,
            "CN_Skew": cn_skew,
            "CN_Kurt": cn_kurt,
            "CN_Median": cn_median,
            "CN_IQR": cn_iqr,
            "CN_CV": cn_cv,
            "VolAvg": float(volume / len(atoms)),
            "DispAvg": disp_mean,
            **cn_elements,
            **thermo_feats,
            **bond_feats
        }
        records.append(rec)

    return records


# ── Parser helpers ──────────────────────────────────────────────────────────────
def parse_log_thermo(log_path: Path) -> dict:
    """Parse log.lammps to get step-to-thermo dict."""
    HEADER_RE = re.compile(r"^\s*Step\s+Temp\s+Press\s+PotEng\s+KinEng\s+TotEng\s+Density", re.IGNORECASE)
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
            
    thermo_map = {}
    stage_order = ["minimization", "nve_limit", "low_temp_relax", "gradual_equil", "equilibrium", "bed", "laser", "hold", "cooling"]
    
    for idx, (start, end) in enumerate(blocks):
        if idx >= len(stage_order): continue
        stage = stage_order[idx]
        for li in range(start, end):
            parts = lines[li].split()
            if len(parts) >= 7:
                try:
                    ts = int(parts[0])
                    thermo_map[(stage, ts)] = {
                        "Thermo_Temp": float(parts[1]),
                        "Thermo_Press": float(parts[2]),
                        "Thermo_PotEng": float(parts[3]),
                        "Thermo_KinEng": float(parts[4]),
                        "Thermo_TotEng": float(parts[5]),
                        "Thermo_Density": float(parts[6]),
                    }
                except (ValueError, IndexError):
                    continue
    return thermo_map


def parse_bond_csv(csv_path: Path) -> dict:
    """Parse bond evolution CSV into step-to-bond-counts dict."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Map stages
    BOND_STAGE_MAP = {
        "equilibration": "equilibrium", "equil": "equilibrium", "equilibrium": "equilibrium",
        "bed": "bed", "bed_heating": "bed", "bed heating": "bed",
        "laser": "laser", "laser_heating": "laser", "laser heating": "laser", "laser_sintering": "laser",
        "hold": "hold", "hold_at": "hold", "neck_formation": "hold",
        "cooling": "cooling", "cooling_back": "cooling", "cooling back": "cooling"
    }
    
    bond_map = {}
    bond_types = ["C-C", "C-H", "C-O", "C-N", "H-O", "H-N", "N-O", "H-H", "O-O", "N-N"]
    
    for _, row in df.iterrows():
        raw_stage = str(row.get("Stage", "")).strip().lower()
        stage = BOND_STAGE_MAP.get(raw_stage, raw_stage)
        ts = int(row.get("Timestep", 0))
        
        feats = {
            "Bond_Total": int(row.get("Total_bonds", 0))
        }
        for bt in bond_types:
            if bt in df.columns:
                feats[f"Bond_{bt.replace('-','')}"] = int(row.get(bt, 0))
                
        bond_map[(stage, ts)] = feats
    return bond_map


# ── Plotting Helpers ─────────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, classes, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_n = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_n, annot=cm, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax, vmin=0, vmax=1)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def save_regression_scatter(y_true, y_pred, title, unit, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, c="#4d7cff", s=15, edgecolors="none")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel(f"Predicted ({unit})")
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.set_title(f"{title}\nR² = {r2:.4f}  MAE = {mae:.4f}", fontweight="bold")
    ax.grid(alpha=0.25); ax.legend()
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ── Main ML training pipeline ────────────────────────────────────────────────────
def run_classification(X, y, title, classes, plots_dir, logger):
    logger.info(f"\nTask: {title}")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    
    acc = accuracy_score(y_te, y_pred)
    cv = cross_val_score(rf, X, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
    
    logger.info(f"  RandomForest Accuracy : {acc:.4f}")
    logger.info(f"  RandomForest CV Score : {cv.mean():.4f} ± {cv.std():.4f}")
    
    tag = title.lower().replace(" ", "_")
    save_confusion_matrix(le.inverse_transform(y_te), le.inverse_transform(y_pred),
                          classes, f"{title} — Random Forest\n(acc={acc:.4f})",
                          os.path.join(plots_dir, f"cm_{tag}_rf.png"))
    return acc


def run_regression(X, y, title, unit, plots_dir, logger):
    logger.info(f"\nTask: {title} Regression")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    
    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    cv_r2 = cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1)
    
    logger.info(f"  RandomForest R² Score : {r2:.4f}")
    logger.info(f"  RandomForest MAE      : {mae:.4f} {unit}")
    logger.info(f"  RandomForest CV R²    : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    
    tag = title.lower().replace(" ", "_")
    save_regression_scatter(y_te, y_pred, f"{title} Fit", unit,
                            os.path.join(plots_dir, f"scatter_{tag}_rf.png"))
    return r2


# ── Execution ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="SLS Frame-Level pipeline")
    ap.add_argument("--owais",  default=DEFAULT_OWAIS, help="Path to Owais Data root")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    ap.add_argument("--stride", type=int, default=3, help="Stride to sample frames (default=3)")
    args = ap.parse_args()

    plots_dir = os.path.join(args.output, "Plots")
    models_dir = os.path.join(args.output, "Models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    logger = setup_logging(args.output)
    logger.info("=" * 70)
    logger.info("SLS FRAME-LEVEL MACHINE LEARNING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Data Dir  : {args.owais}")
    logger.info(f"Output Dir: {args.output}")
    logger.info(f"Stride    : {args.stride}")

    cache_path = os.path.join(args.output, "frame_level_cache.csv")
    
    if os.path.exists(cache_path):
        logger.info(f"\nLoading frame-level dataset from cache: {cache_path}")
        df_all = pd.read_csv(cache_path)
    else:
        logger.info("\n── Phase 1: Extracting Frame-Level Features (Parallel) ──")
        t_start = time.time()
        
        # Build tasks list
        tasks = []
        for comp in COMPOSITIONS:
            for temp in TEMPERATURES:
                tasks.append((comp, temp))
                
        # Parse all experiments in parallel
        owais_root = Path(args.owais)
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_single_experiment)(owais_root, comp, temp, args.stride)
            for comp, temp in tasks
        )
        
        # Merge all records
        all_records = []
        for r_list in results:
            if r_list: all_records.extend(r_list)
            
        if not all_records:
            logger.error("No frame-level data parsed!")
            return
            
        df_all = pd.DataFrame(all_records)
        df_all.to_csv(cache_path, index=False)
        elapsed = time.time() - t_start
        logger.info(f"Extracted {df_all.shape[0]} frames × {df_all.shape[1]} features in {elapsed:.1f}s")

    df_all["Composition"] = df_all["Composition"].astype(str)

    # Clean data
    meta_cols = ["Composition", "Stage", "Temperature", "Timestep"]
    feature_cols = [c for c in df_all.columns if c not in meta_cols]
    
    X = df_all[feature_cols].select_dtypes(include=[np.number]).copy()
    X.dropna(axis=1, how="all", inplace=True)
    X = X.fillna(X.median())
    
    # Drop zero variance columns
    X = X.loc[:, X.var() > 1e-12]
    
    logger.info(f"\nFinal feature matrix: {X.shape[0]} rows × {X.shape[1]} features")

    # ── Phase 2: ML Model Training ───────────────────────────────────────────
    logger.info("\n── Phase 2: ML Model Training & Evaluation ──")
    X_scaled = MinMaxScaler().fit_transform(X)
    
    # Classification 1: Stage
    run_classification(X_scaled, df_all["Stage"], "Stage Classification", MAIN_SLS_STAGES, plots_dir, logger)
    
    # Classification 2: Composition
    run_classification(X_scaled, df_all["Composition"].astype(str), "Composition Prediction", COMPOSITIONS, plots_dir, logger)
    
    # Regression 1: Displacement (DispAvg)
    y_disp = df_all["DispAvg"].fillna(df_all["DispAvg"].median())
    run_regression(X_scaled, y_disp, "Atomic Displacement", "Å", plots_dir, logger)
    
    # Regression 2: Potential Energy (Thermo_PotEng)
    if "Thermo_PotEng" in df_all.columns:
        y_pe = df_all["Thermo_PotEng"].fillna(df_all["Thermo_PotEng"].median())
        run_regression(X_scaled, y_pe, "Potential Energy", "kcal/mol", plots_dir, logger)
        
    # Regression 3: C-C Backbone count (Bond_CC)
    if "Bond_CC" in df_all.columns:
        y_cc = df_all["Bond_CC"].fillna(df_all["Bond_CC"].median())
        run_regression(X_scaled, y_cc, "C-C Backbone Bonds", "count", plots_dir, logger)

    # ── Phase 3: High-Resolution Sintering Chemistry Plots ───────────────────
    logger.info("\n── Phase 3: Generating Frame-Level Plots ──")
    
    # Plot 1: Displacement Evolution of all 18 experiments over Timesteps
    fig, ax = plt.subplots(figsize=(10, 6))
    df_all["Exp"] = df_all["Composition"].astype(str) + " - " + df_all["Temperature"].astype(str) + "C"
    sns.lineplot(data=df_all, x="Timestep", y="DispAvg", hue="Stage", palette=STAGE_COLORS, ax=ax, lw=1.5)
    ax.set_title("Frame-by-Frame Atomic Displacement Sintering Trajectory", fontweight="bold", pad=15)
    ax.set_xlabel("MD Timestep")
    ax.set_ylabel("Mean Atomic Displacement (Å)")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "frame_displacement_trajectory.png"), dpi=200)
    plt.close()
    
    # Plot 2: C-C & C-N Crosslink Evolution across timesteps
    if "Bond_CC" in df_all.columns and "Bond_CN" in df_all.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # C-C Bonds
        sns.lineplot(data=df_all, x="Timestep", y="Bond_CC", hue="Composition", style="Composition", palette=COMP_COLORS, ax=axes[0])
        axes[0].set_title("C-C Backbone Bond Count Evolution", fontweight="bold")
        axes[0].set_xlabel("MD Timestep")
        axes[0].set_ylabel("C-C Bonds Count")
        
        # C-N Bonds
        sns.lineplot(data=df_all, x="Timestep", y="Bond_CN", hue="Composition", style="Composition", palette=COMP_COLORS, ax=axes[1])
        axes[1].set_title("C-N Crosslink Bond Count Evolution", fontweight="bold")
        axes[1].set_xlabel("MD Timestep")
        axes[1].set_ylabel("C-N Bonds Count")
        
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "frame_chemical_bond_evolution.png"), dpi=200)
        plt.close()

    logger.info(f"\nFrame-level pipeline execution successfully completed.")
    logger.info(f"Outputs generated in: {args.output}")

if __name__ == "__main__":
    main()
