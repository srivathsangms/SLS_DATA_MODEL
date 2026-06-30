"""
Main pipeline orchestrator.

Coordinates discovery, parsing, feature extraction, aggregation,
evolution tracking, statistical analysis, visualization, and dataset output.
"""

import os
import sys
import time
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Pipeline modules
from pipeline.config import PipelineConfig
from pipeline.discovery import ExperimentDiscoverer, ExperimentMetadata
from pipeline.parsers.trajectory import TrajectoryParser
from pipeline.parsers.log_parser import LogParser
from pipeline.parsers.bond_parser import BondParser
from pipeline.parsers.temperature import TemperatureParser
from pipeline.parsers.xlsx_parser import XlsxParser
from pipeline.parsers.input_script import InputScriptParser
from pipeline.extractors.structural import StructuralExtractor
from pipeline.extractors.dynamic import DynamicExtractor
from pipeline.extractors.mechanical import MechanicalExtractor
from pipeline.extractors.thermodynamic import ThermodynamicExtractor
from pipeline.extractors.bond import BondExtractor
from pipeline.aggregator import FeatureAggregator
from pipeline.evolution import EvolutionTracker
from pipeline.statistics import StatisticalAnalyzer
from pipeline.visualization import PlotGenerator
from pipeline.dataset_builder import DatasetBuilder


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "pipeline.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    return logger


def process_stage_trajectory(
    experiment: ExperimentMetadata,
    stage_name: str,
    config: PipelineConfig,
) -> Tuple[List[Dict], Dict[str, np.ndarray], Optional[Dict]]:
    """
    Process a single stage trajectory and extract all frame-level features.

    Returns
    -------
    frame_records : list of dict
        One dict per frame with all features.
    distributions : dict
        Raw distribution arrays for plotting.
    rdf_data : dict or None
        RDF (r, g_r) for last frame.
    """
    logger = logging.getLogger(__name__)
    stage_files = experiment.stages.get(stage_name)
    if stage_files is None or stage_files.trajectory is None:
        logger.warning(f"No trajectory for {experiment.label} / {stage_name}")
        return [], {}, None

    logger.info(f"Processing {experiment.label} / {stage_name}: {stage_files.trajectory}")

    # Initialise extractors
    structural_ext = StructuralExtractor(
        neighbor_cutoff=config.structural.neighbor_cutoff,
        rdf_cutoff=config.structural.rdf_cutoff,
        rdf_bins=config.structural.rdf_bins,
        voronoi_enabled=config.structural.voronoi_enabled,
        use_ovito=config.processing.use_ovito,
    )
    dynamic_ext = DynamicExtractor()
    mechanical_ext = MechanicalExtractor(neighbor_cutoff=config.structural.neighbor_cutoff)

    # Parse trajectory
    parser = TrajectoryParser(stage_files.trajectory)
    frame_records = []
    distributions = {"CN": [], "AtomicVolume": [], "CavityRadius": [], "Displacement": []}
    rdf_data = None
    first_frame = True

    for frame_idx, frame in enumerate(parser.parse(every_n=config.processing.frame_sampling)):
        try:
            record = {
                "Composition": experiment.composition,
                "Temperature": experiment.temperature,
                "Stage": stage_name,
                "Frame": frame_idx,
                "Timestep": frame.timestep,
                "NumAtoms": frame.num_atoms,
                "SimTime": frame.timestep * 0.1,  # fs (timestep = 0.1 fs for ReaxFF)
            }

            box_lengths = frame.box_lengths

            # Set reference for first frame
            if first_frame:
                dynamic_ext.set_reference(frame.positions)
                mechanical_ext.set_reference(frame.positions, box_lengths, frame.volume)
                first_frame = False

            # Structural features
            struct_feats = structural_ext.extract(
                frame.positions, frame.box_bounds, frame.atom_types
            )
            record.update(struct_feats.to_dict())

            # Collect distributions from last frame for plotting
            if struct_feats.cn_distribution is not None:
                distributions["CN"] = struct_feats.cn_distribution
            if struct_feats.atomic_volume_distribution is not None:
                distributions["AtomicVolume"] = struct_feats.atomic_volume_distribution
            if struct_feats.cavity_radius_distribution is not None:
                distributions["CavityRadius"] = struct_feats.cavity_radius_distribution

            # RDF from last frame
            if struct_feats.rdf is not None:
                rdf_data = (struct_feats.rdf.r, struct_feats.rdf.g_r)

            # Dynamic features
            dyn_feats = dynamic_ext.extract(frame.positions, box_lengths)
            record.update(dyn_feats.to_dict())
            if dyn_feats.displacement_distribution is not None:
                distributions["Displacement"] = dyn_feats.displacement_distribution

            # Mechanical features
            mech_feats = mechanical_ext.extract(frame.positions, box_lengths, frame.volume)
            record.update(mech_feats.to_dict())

            frame_records.append(record)

        except Exception as e:
            logger.warning(f"Error processing frame {frame_idx} of {experiment.label}/{stage_name}: {e}")
            continue

    logger.info(f"  → Extracted {len(frame_records)} frames for {stage_name}")
    return frame_records, distributions, rdf_data


def process_experiment(
    experiment: ExperimentMetadata,
    config: PipelineConfig,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Process a full experiment (all stages).

    Returns
    -------
    all_frame_records : list of dict
    stage_summaries : list of dict
    plot_data : dict
        Data collected for plotting.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING EXPERIMENT: {experiment.label}")
    logger.info(f"  Directory: {experiment.directory}")
    logger.info(f"{'='*60}")

    all_frame_records = []
    stage_summaries = []
    aggregator = FeatureAggregator()
    plot_data = {
        "distributions": {},
        "rdf": {},
        "thermo": {},
        "bond_frames": [],
    }

    # --- 1. Parse thermodynamic data from log ---
    thermo_data = {}
    if experiment.log_file:
        try:
            log_parser = LogParser(experiment.log_file)
            thermo_data = log_parser.parse()
            logger.info(f"  Parsed log.lammps: {list(thermo_data.keys())}")
        except Exception as e:
            logger.warning(f"  Failed to parse log: {e}")

    # --- 2. Extract thermo features and merge ---
    thermo_ext = ThermodynamicExtractor()
    thermo_summaries = {}
    for stage_name in config.stages:
        if stage_name in thermo_data:
            thermo_summaries[stage_name] = thermo_ext.extract_stage_summary(thermo_data[stage_name])
            plot_data["thermo"][stage_name] = thermo_data[stage_name]

    # --- 3. Process each stage trajectory ---
    for stage_name in config.stages:
        frame_records, distributions, rdf_data = process_stage_trajectory(
            experiment, stage_name, config
        )

        # Merge thermo summary into frame records
        if stage_name in thermo_summaries:
            for record in frame_records:
                record.update(thermo_summaries[stage_name])

        all_frame_records.extend(frame_records)

        # Aggregate stage summary
        if frame_records:
            summary = aggregator.aggregate_stage(
                frame_records, experiment.composition, experiment.temperature, stage_name
            )
            # Add thermo summary
            if stage_name in thermo_summaries:
                summary.update(thermo_summaries[stage_name])
            stage_summaries.append(summary)

        # Collect plot data
        plot_data["distributions"][stage_name] = distributions
        if rdf_data:
            plot_data["rdf"][stage_name] = rdf_data

    # --- 4. Parse bond data (sampled for efficiency) ---
    if experiment.bond_file:
        try:
            bond_parser = BondParser(
                experiment.bond_file,
                atom_type_map=config.bond.atom_types,
            )
            bond_ext = BondExtractor(atom_type_map=config.bond.atom_types)

            bond_count = 0
            # Sample every 10th timestep for bond analysis (files are huge)
            for ts in bond_parser.parse(every_n=10):
                stats = bond_parser.compute_bond_statistics(ts)
                bond_feats = bond_ext.extract(stats, bonds=ts.bonds)
                bond_record = bond_feats.to_dict()
                bond_record["Timestep"] = ts.timestep
                bond_record["Composition"] = experiment.composition
                bond_record["Temperature"] = experiment.temperature
                plot_data["bond_frames"].append(bond_record)
                bond_count += 1

            logger.info(f"  Parsed {bond_count} bond timesteps")

            # Add bond summary to stage summaries
            if plot_data["bond_frames"]:
                bond_df = pd.DataFrame(plot_data["bond_frames"])
                for summary in stage_summaries:
                    stage = summary.get("Stage")
                    if stage and stage in thermo_data:
                        # Map bond timesteps to stages based on thermo step ranges
                        thermo_df = thermo_data[stage]
                        if not thermo_df.empty:
                            t_min = thermo_df["Step"].min()
                            t_max = thermo_df["Step"].max()
                            stage_bonds = bond_df[
                                (bond_df["Timestep"] >= t_min) & (bond_df["Timestep"] <= t_max)
                            ]
                            if not stage_bonds.empty:
                                for col in stage_bonds.select_dtypes(include=[np.number]).columns:
                                    if col != "Timestep":
                                        summary[f"Stage_Mean_{col}"] = float(stage_bonds[col].mean())

        except Exception as e:
            logger.warning(f"  Failed to parse bonds: {e}")

    # --- 5. Use xlsx data if available (5050 composition) ---
    if config.processing.use_xlsx_if_available:
        for stage_name in config.stages:
            stage_files = experiment.stages.get(stage_name)
            if stage_files and stage_files.xlsx:
                try:
                    xlsx_parser = XlsxParser(stage_files.xlsx)
                    xlsx_summary = xlsx_parser.extract_summary()
                    if xlsx_summary:
                        # Find matching stage summary and update
                        for summary in stage_summaries:
                            if summary.get("Stage") == stage_name:
                                for key, val in xlsx_summary.items():
                                    xlsx_key = f"XLSX_{key}"
                                    summary[xlsx_key] = val
                        logger.debug(f"  Added xlsx data for {stage_name}")
                except Exception as e:
                    logger.debug(f"  Failed to read xlsx for {stage_name}: {e}")

    logger.info(f"  Completed: {len(all_frame_records)} total frames, {len(stage_summaries)} stage summaries")
    return all_frame_records, stage_summaries, plot_data


def run_pipeline(config_path: str = "config.yaml"):
    """Run the full analysis pipeline."""
    start_time = time.time()

    # --- Load configuration ---
    config = PipelineConfig.from_yaml(config_path)
    logger = setup_logging(config.output_dir)
    logger.info("=" * 70)
    logger.info("SLS MD Analysis Pipeline")
    logger.info("Physics-Informed AI Framework for Molecular Evolution")
    logger.info("=" * 70)
    logger.info(f"Root directory: {config.root_dir}")
    logger.info(f"Output directory: {config.output_dir}")

    # --- Discover experiments ---
    logger.info("\n--- Phase 1: Discovery ---")
    discoverer = ExperimentDiscoverer(config.root_dir)
    experiments = discoverer.discover()

    if not experiments:
        logger.error("No experiments found!")
        return

    logger.info(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        logger.info(f"  {exp.label}: {len(exp.stages)} stages")

    # --- Process experiments ---
    logger.info("\n--- Phase 2: Feature Extraction ---")
    all_frame_records = []
    all_stage_summaries = []
    all_plot_data = {}

    for exp in tqdm(experiments, desc="Processing experiments"):
        try:
            frames, summaries, plot_data = process_experiment(exp, config)
            all_frame_records.extend(frames)
            all_stage_summaries.extend(summaries)
            all_plot_data[exp.label] = plot_data
        except Exception as e:
            logger.error(f"Failed to process {exp.label}: {e}")
            logger.debug(traceback.format_exc())
            continue

    logger.info(f"\nTotal frames extracted: {len(all_frame_records)}")
    logger.info(f"Total stage summaries: {len(all_stage_summaries)}")

    if not all_frame_records:
        logger.error("No data extracted! Check file paths and formats.")
        return

    # --- Build master dataset ---
    logger.info("\n--- Phase 3: Dataset Assembly ---")
    aggregator = FeatureAggregator()
    master_df = aggregator.build_master_dataset(all_frame_records)
    logger.info(f"Master dataset shape: {master_df.shape}")

    # --- Save feature-specific CSVs ---
    logger.info("\n--- Phase 4: Saving Feature Data ---")
    feature_dir = config.get_output_path("Feature_Data")

    # Group features by category
    feature_groups = {
        "CN": [c for c in master_df.columns if "CN" in c],
        "AtomicVolume": [c for c in master_df.columns if "AtomicVolume" in c],
        "CavityRadius": [c for c in master_df.columns if "CavityRadius" in c],
        "Displacement": [c for c in master_df.columns if "Displacement" in c or "MSD" in c],
        "Strain": [c for c in master_df.columns if "Strain" in c],
        "BondOrder": [c for c in master_df.columns if "BondOrder" in c or "BO_" in c],
        "BondLength": [c for c in master_df.columns if "BondLength" in c],
        "Bonds": [c for c in master_df.columns if "Bonds_" in c or "Total_Bond" in c],
        "RDF": [c for c in master_df.columns if "RDF" in c],
    }

    meta_cols = ["Composition", "Temperature", "Stage", "Frame", "Timestep"]
    existing_meta = [c for c in meta_cols if c in master_df.columns]

    for group_name, cols in feature_groups.items():
        available = [c for c in cols if c in master_df.columns]
        if available:
            group_df = master_df[existing_meta + available]
            csv_path = os.path.join(feature_dir, f"{group_name}.csv")
            group_df.to_csv(csv_path, index=False)
            logger.info(f"  Saved {group_name}.csv ({len(available)} features)")

    # Thermodynamic features
    thermo_cols = [c for c in master_df.columns if any(
        t in c for t in ["Temperature", "Pressure", "PotEng", "KinEng", "TotEng", "Density", "Volume"]
    )]
    if thermo_cols:
        thermo_available = [c for c in thermo_cols if c in master_df.columns]
        thermo_df = master_df[existing_meta + thermo_available]
        thermo_df.to_csv(os.path.join(feature_dir, "Thermodynamic.csv"), index=False)
        logger.info(f"  Saved Thermodynamic.csv ({len(thermo_available)} features)")

    # --- Evolution tracking ---
    logger.info("\n--- Phase 5: Evolution Analysis ---")
    evolution_tracker = EvolutionTracker()
    evolution_dir = config.get_output_path("Evolution")

    evolution_df = evolution_tracker.build_evolution_table(all_stage_summaries)
    if not evolution_df.empty:
        evolution_df.to_csv(os.path.join(evolution_dir, "Evolution_Table.csv"), index=False)
        logger.info(f"  Evolution table: {evolution_df.shape}")

        # Per-experiment evolution
        for exp in experiments:
            exp_evo = evolution_tracker.build_per_experiment_evolution(
                all_stage_summaries, exp.composition, exp.temperature
            )
            if not exp_evo.empty:
                exp_evo.to_csv(
                    os.path.join(evolution_dir, f"{exp.label}_evolution.csv"),
                    index=False,
                )

    # --- Statistical analysis ---
    logger.info("\n--- Phase 6: Statistical Analysis ---")
    analyzer = StatisticalAnalyzer()
    stats_dir = config.get_output_path("Statistics")

    # Summary statistics
    summary = analyzer.compute_summary(master_df)
    summary.to_csv(os.path.join(stats_dir, "Summary.csv"))
    logger.info(f"  Summary statistics: {summary.shape}")

    # Correlation matrices
    pearson = analyzer.compute_pearson_correlation(master_df)
    pearson.to_csv(os.path.join(stats_dir, "Correlation_Pearson.csv"))

    spearman = analyzer.compute_spearman_correlation(master_df)
    spearman.to_csv(os.path.join(stats_dir, "Correlation_Spearman.csv"))
    logger.info(f"  Correlation matrices: {pearson.shape}")

    # PCA
    scores, loadings, explained_var = analyzer.compute_pca(master_df)
    if not scores.empty:
        scores.to_csv(os.path.join(stats_dir, "PCA_Scores.csv"))
        loadings.to_csv(os.path.join(stats_dir, "PCA_Loadings.csv"))
        pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(explained_var))],
            "Explained_Variance": explained_var,
            "Cumulative": np.cumsum(explained_var),
        }).to_csv(os.path.join(stats_dir, "PCA_Variance.csv"), index=False)
        logger.info(f"  PCA: {len(explained_var)} components")

    # Feature ranking
    if "Stage" in master_df.columns:
        ranking = analyzer.compute_feature_ranking(master_df, target_col="Stage")
        if not ranking.empty:
            ranking.to_csv(os.path.join(stats_dir, "Feature_Ranking.csv"), index=False)
            logger.info(f"  Feature ranking: {len(ranking)} features ranked")

    # --- Visualization ---
    logger.info("\n--- Phase 7: Visualization ---")
    plotter = PlotGenerator(
        output_dir=os.path.join(config.output_dir, "Plots"),
        dpi=config.visualization.dpi,
        fmt=config.visualization.format,
        figsize=tuple(config.visualization.figsize),
        stage_colors=config.visualization.stage_colors,
    )

    # Histograms per experiment
    for exp_label, pdata in all_plot_data.items():
        for stage_name, dists in pdata.get("distributions", {}).items():
            if isinstance(dists, dict):
                for feat_name, values in dists.items():
                    if isinstance(values, np.ndarray) and len(values) > 0:
                        plotter.plot_histogram(
                            values,
                            title=f"{feat_name} Distribution - {exp_label} ({stage_name})",
                            xlabel=feat_name,
                            filename=f"{exp_label}_{stage_name}_{feat_name}_hist",
                        )

    # RDF plots per experiment
    for exp_label, pdata in all_plot_data.items():
        rdf_stages = pdata.get("rdf", {})
        if rdf_stages:
            plotter.plot_rdf_stages(
                rdf_stages,
                title=f"RDF Evolution - {exp_label}",
                filename=f"{exp_label}_rdf_stages",
            )
            # Individual RDF
            for stage_name, (r, g_r) in rdf_stages.items():
                plotter.plot_rdf(
                    r, g_r,
                    title=f"RDF - {exp_label} ({stage_name})",
                    filename=f"{exp_label}_{stage_name}_rdf",
                )

    # Thermodynamic plots per experiment
    for exp_label, pdata in all_plot_data.items():
        thermo_stages = pdata.get("thermo", {})
        if thermo_stages:
            for y_col, ylabel in [
                ("Temp", "Temperature (K)"),
                ("Press", "Pressure (atm)"),
                ("PotEng", "Potential Energy (kcal/mol)"),
                ("Density", "Density (g/cm³)"),
            ]:
                plotter.plot_thermo_all_stages(
                    thermo_stages, y_col,
                    title=f"{ylabel} vs Stage - {exp_label}",
                    ylabel=ylabel,
                    filename=f"{exp_label}_{y_col}_evolution",
                )

    # Evolution plots
    if not evolution_df.empty:
        key_features = [
            "Mean_CN", "Mean_AtomicVolume", "Mean_CavityRadius",
            "Mean_Displacement", "MSD", "Volumetric_Strain",
            "Mean_Shear_Strain",
        ]
        for feat in key_features:
            if feat in evolution_df["Feature"].values:
                plotter.plot_feature_evolution(
                    evolution_df, feat,
                    title=f"{feat} Evolution Across Stages",
                    filename=f"evolution_{feat}",
                )
                for comp in ["5050", "6040", "7030"]:
                    plotter.plot_evolution_line(
                        evolution_df, feat, comp,
                        title=f"{feat} Evolution - {comp}",
                        filename=f"evolution_{feat}_{comp}",
                    )

    # Bond evolution plots
    for exp_label, pdata in all_plot_data.items():
        bond_frames = pdata.get("bond_frames", [])
        if bond_frames:
            bond_df = pd.DataFrame(bond_frames)
            plotter.plot_bond_evolution(
                bond_df,
                title=f"Bond Type Evolution - {exp_label}",
                filename=f"{exp_label}_bond_evolution",
            )

    # Statistical plots
    plotter.plot_correlation_heatmap(
        pearson, "Pearson Correlation Matrix", "correlation_pearson"
    )
    plotter.plot_correlation_heatmap(
        spearman, "Spearman Correlation Matrix", "correlation_spearman"
    )
    if len(explained_var) > 0:
        plotter.plot_pca_scree(explained_var)
        if "Stage" in master_df.columns:
            plotter.plot_pca_biplot(
                scores, loadings,
                labels=master_df["Stage"] if "Stage" in master_df.columns else None,
            )
    if "Stage" in master_df.columns and not ranking.empty:
        plotter.plot_feature_ranking(ranking)

    logger.info(f"  All plots saved to {os.path.join(config.output_dir, 'Plots')}")

    # --- Save final datasets ---
    logger.info("\n--- Phase 8: Final Dataset ---")
    builder = DatasetBuilder(
        normalization=config.normalization.method,
        stage_encoding=config.normalization.stage_encoding,
    )
    dataset_dir = config.get_output_path("Final_Dataset")
    paths = builder.save_datasets(master_df, dataset_dir)
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")

    # --- Stage summary dataset ---
    if all_stage_summaries:
        summary_df = pd.DataFrame(all_stage_summaries)
        summary_df.to_csv(
            os.path.join(dataset_dir, "stage_summaries.csv"), index=False
        )
        logger.info(f"  Stage summaries: {summary_df.shape}")

    # --- Done ---
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"Pipeline completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Output directory: {os.path.abspath(config.output_dir)}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_pipeline(config_file)
