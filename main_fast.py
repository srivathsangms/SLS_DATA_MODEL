"""
Fast pipeline using pre-computed xlsx samples.

Uses the samples/ directory (xlsx + temp files) to build the ML dataset
WITHOUT parsing raw trajectories. Runs in seconds instead of hours.

The samples directory has:
  samples/{comp}/{temp}/{stage}.xlsx  — per-atom features from OVITO
  samples/{comp}/{comp}_final.xlsx    — pre-aggregated stage summaries
  samples/{comp}/{temp}/temp_*.txt    — temperature profiles
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from pipeline.config import PipelineConfig
from pipeline.parsers.xlsx_parser import XlsxParser
from pipeline.parsers.temperature import TemperatureParser
from pipeline.aggregator import FeatureAggregator
from pipeline.evolution import EvolutionTracker
from pipeline.statistics import StatisticalAnalyzer
from pipeline.visualization import PlotGenerator
from pipeline.dataset_builder import DatasetBuilder


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "pipeline_fast.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    return logger


SAMPLES_DIR = r"C:\Users\sriva\Desktop\samples"
COMPOSITIONS = ["5050", "6040", "7030"]
TEMPERATURES = [100, 150, 200, 250, 300, 350]
STAGES = ["equili", "bed", "laser", "hold", "cooling"]
STAGE_MAP = {
    "equili": "equilibrium",
    "bed": "bed",
    "laser": "laser",
    "hold": "hold",
    "cooling": "cooling",
}
TEMP_FILE_MAP = {
    "equili": "temp_equil_300K.txt",
    "bed": "temp_bed_60C.txt",
    "laser": "temp_laser_150C.txt",
    "hold": "temp_hold_150C.txt",
    "cooling": "temp_cooling_60C.txt",
}


def extract_per_atom_features(filepath: str) -> Dict[str, float]:
    """Extract aggregated features from one xlsx file."""
    parser = XlsxParser(filepath)
    return parser.extract_summary()


def extract_temperature_features(filepath: str) -> Dict[str, float]:
    """Extract temperature statistics from temp file."""
    if not os.path.exists(filepath):
        return {}

    parser = TemperatureParser(filepath)
    df = parser.parse()
    if df.empty:
        return {}

    temps = df["Temperature"].dropna()
    return {
        "Mean_Temperature": float(temps.mean()),
        "Std_Temperature": float(temps.std()),
        "Min_Temperature": float(temps.min()),
        "Max_Temperature": float(temps.max()),
        "First_Temperature": float(temps.iloc[0]),
        "Last_Temperature": float(temps.iloc[-1]),
        "Delta_Temperature": float(temps.iloc[-1] - temps.iloc[0]),
    }


def build_per_atom_dataset(samples_dir: str) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Build dataset from per-atom xlsx files.
    
    Returns
    -------
    master_df : pd.DataFrame
        One row per (composition, temperature, stage) with aggregated features.
    all_records : list of dict
    """
    logger = logging.getLogger(__name__)
    all_records = []

    for comp in COMPOSITIONS:
        for temp in TEMPERATURES:
            for stage_short, stage_full in STAGE_MAP.items():
                xlsx_path = os.path.join(samples_dir, comp, str(temp), f"{stage_short}.xlsx")
                if not os.path.exists(xlsx_path):
                    logger.warning(f"Missing: {xlsx_path}")
                    continue

                # Per-atom features
                features = extract_per_atom_features(xlsx_path)
                if not features:
                    continue

                # Temperature features
                temp_path = os.path.join(
                    samples_dir, comp, str(temp), TEMP_FILE_MAP.get(stage_short, "")
                )
                temp_features = extract_temperature_features(temp_path)

                record = {
                    "Composition": comp,
                    "Temperature": temp,
                    "Stage": stage_full,
                    **features,
                    **temp_features,
                }
                all_records.append(record)

                logger.debug(f"  {comp}/{temp}/{stage_short}: {len(features)} features")

    df = pd.DataFrame(all_records)
    logger.info(f"Built per-atom dataset: {df.shape}")
    return df, all_records


def load_final_xlsx(samples_dir: str) -> pd.DataFrame:
    """Load and combine all *_final.xlsx files."""
    logger = logging.getLogger(__name__)
    frames = []

    for comp in COMPOSITIONS:
        final_path = os.path.join(samples_dir, comp, f"{comp}_final.xlsx")
        if not os.path.exists(final_path):
            logger.warning(f"Missing final xlsx: {final_path}")
            continue

        df = pd.read_excel(final_path, engine="openpyxl")
        df["Composition"] = comp

        # Standardize stage names
        df["Stage"] = df["Stage"].str.strip().map({
            "equili": "equilibrium",
            "bed": "bed",
            "laser": "laser",
            "hold": "hold",
            "cooling": "cooling",
        })
        df.rename(columns={"Temp": "Temperature"}, inplace=True)
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded final xlsx data: {combined.shape}")
        return combined
    return pd.DataFrame()


def run_fast_pipeline(samples_dir: str = SAMPLES_DIR, output_dir: str = "Results"):
    """Run the fast analysis pipeline using pre-computed xlsx data."""
    start_time = time.time()
    logger = setup_logging(output_dir)

    logger.info("=" * 70)
    logger.info("SLS MD Analysis Pipeline (FAST MODE - xlsx)")
    logger.info("Using pre-computed OVITO features from samples/")
    logger.info("=" * 70)

    # --- Phase 1: Build master dataset from per-atom xlsx ---
    logger.info("\n--- Phase 1: Per-Atom Feature Extraction ---")
    master_df, all_records = build_per_atom_dataset(samples_dir)

    if master_df.empty:
        logger.error("No data extracted!")
        return

    logger.info(f"Master dataset: {master_df.shape[0]} rows × {master_df.shape[1]} columns")
    logger.info(f"Features: {list(master_df.columns)}")

    # --- Phase 2: Load pre-aggregated final xlsx ---
    logger.info("\n--- Phase 2: Final xlsx (Start/Middle/End) ---")
    final_df = load_final_xlsx(samples_dir)
    if not final_df.empty:
        final_path = os.path.join(output_dir, "Final_Dataset", "final_xlsx_combined.csv")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        final_df.to_csv(final_path, index=False)
        logger.info(f"Saved: {final_path}")

    # --- Phase 3: Evolution tracking ---
    logger.info("\n--- Phase 3: Evolution Analysis ---")
    tracker = EvolutionTracker()
    # Map features to Stage_Mean_* format for the tracker
    evo_records = []
    for r in all_records:
        evo_rec = {
            "Composition": r["Composition"],
            "Temperature": r["Temperature"],
            "Stage": r["Stage"]
        }
        for k, v in r.items():
            if k not in ["Composition", "Temperature", "Stage"]:
                evo_rec[f"Stage_Mean_{k}"] = v
        evo_records.append(evo_rec)
    evolution_df = tracker.build_evolution_table(evo_records)
    evolution_dir = os.path.join(output_dir, "Evolution")
    os.makedirs(evolution_dir, exist_ok=True)
    if not evolution_df.empty:
        evolution_df.to_csv(os.path.join(evolution_dir, "Evolution_Table.csv"), index=False)
        logger.info(f"Evolution table: {evolution_df.shape}")

    # --- Phase 4: Statistical analysis ---
    logger.info("\n--- Phase 4: Statistical Analysis ---")
    analyzer = StatisticalAnalyzer()
    stats_dir = os.path.join(output_dir, "Statistics")
    os.makedirs(stats_dir, exist_ok=True)

    summary = analyzer.compute_summary(master_df)
    summary.to_csv(os.path.join(stats_dir, "Summary.csv"))

    pearson = analyzer.compute_pearson_correlation(master_df)
    pearson.to_csv(os.path.join(stats_dir, "Correlation_Pearson.csv"))

    spearman = analyzer.compute_spearman_correlation(master_df)
    spearman.to_csv(os.path.join(stats_dir, "Correlation_Spearman.csv"))

    scores, loadings, explained_var = analyzer.compute_pca(master_df)
    if not scores.empty:
        scores.to_csv(os.path.join(stats_dir, "PCA_Scores.csv"))
        loadings.to_csv(os.path.join(stats_dir, "PCA_Loadings.csv"))
        pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(explained_var))],
            "Explained_Variance": explained_var,
            "Cumulative": np.cumsum(explained_var),
        }).to_csv(os.path.join(stats_dir, "PCA_Variance.csv"), index=False)

    ranking = pd.DataFrame()
    if "Stage" in master_df.columns:
        ranking = analyzer.compute_feature_ranking(master_df, target_col="Stage")
        if not ranking.empty:
            ranking.to_csv(os.path.join(stats_dir, "Feature_Ranking.csv"), index=False)

    # --- Phase 5: Visualization ---
    logger.info("\n--- Phase 5: Visualization ---")
    plotter = PlotGenerator(
        output_dir=os.path.join(output_dir, "Plots"),
        dpi=300,
        figsize=(10, 6),
    )

    # Evolution plots
    if not evolution_df.empty:
        key_features = ["Mean_CN", "Mean_AtomicVolume", "Mean_CavityRadius", "Mean_Displacement"]
        for feat in key_features:
            if feat in evolution_df["Feature"].values:
                plotter.plot_feature_evolution(
                    evolution_df, feat,
                    title=f"{feat} Evolution Across SLS Stages",
                    filename=f"evolution_{feat}",
                )
                for comp in COMPOSITIONS:
                    plotter.plot_evolution_line(
                        evolution_df, feat, comp,
                        title=f"{feat} Evolution — {comp} Epoxy/PA12",
                        filename=f"evolution_{feat}_{comp}",
                    )

    # Statistical plots
    plotter.plot_correlation_heatmap(pearson, "Pearson Correlation", "correlation_pearson")
    plotter.plot_correlation_heatmap(spearman, "Spearman Correlation", "correlation_spearman")
    if len(explained_var) > 0:
        plotter.plot_pca_scree(explained_var)
        if "Stage" in master_df.columns:
            plotter.plot_pca_biplot(scores, loadings, labels=master_df["Stage"])
    if not ranking.empty:
        plotter.plot_feature_ranking(ranking)

    # --- Phase 6: Save ML-ready datasets ---
    logger.info("\n--- Phase 6: Final Dataset ---")
    builder = DatasetBuilder(normalization="minmax", stage_encoding="ordinal")
    dataset_dir = os.path.join(output_dir, "Final_Dataset")
    paths = builder.save_datasets(master_df, dataset_dir)
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")

    # --- Done ---
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"Fast pipeline completed in {elapsed:.1f} seconds")
    logger.info(f"Output: {os.path.abspath(output_dir)}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    samples = sys.argv[1] if len(sys.argv) > 1 else SAMPLES_DIR
    run_fast_pipeline(samples)
