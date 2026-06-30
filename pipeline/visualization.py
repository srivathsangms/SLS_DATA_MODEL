"""
Publication-quality visualization module.

Generates histograms, evolution plots, RDF curves, thermodynamic trends,
bond distributions, correlation heatmaps, PCA biplots, and feature ranking charts.
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default stage order and colors
STAGE_ORDER = ["equilibrium", "bed", "laser", "hold", "cooling"]
STAGE_LABELS = {
    "equilibrium": "Equilibrium",
    "bed": "Bed Heating",
    "laser": "Laser Heating",
    "hold": "Hold",
    "cooling": "Cooling",
}

DEFAULT_COLORS = {
    "equilibrium": "#2196F3",
    "bed": "#FF9800",
    "laser": "#F44336",
    "hold": "#9C27B0",
    "cooling": "#4CAF50",
}


class PlotGenerator:
    """Generate publication-quality plots for MD analysis results."""

    def __init__(
        self,
        output_dir: str,
        dpi: int = 300,
        fmt: str = "png",
        figsize: Tuple[int, int] = (10, 6),
        style: str = "seaborn-v0_8-whitegrid",
        stage_colors: Dict[str, str] = None,
    ):
        self.output_dir = output_dir
        self.dpi = dpi
        self.fmt = fmt
        self.figsize = figsize
        self.stage_colors = stage_colors or DEFAULT_COLORS

        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("ggplot")

        # Set global font sizes
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        })

    def _save(self, fig, subdir: str, filename: str):
        """Save figure to output directory."""
        path = os.path.join(self.output_dir, subdir)
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{filename}.{self.fmt}")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.debug(f"Saved plot: {filepath}")

    # ----------------------------------------------------------------
    # HISTOGRAM PLOTS
    # ----------------------------------------------------------------

    def plot_histogram(
        self,
        data: np.ndarray,
        title: str,
        xlabel: str,
        filename: str,
        subdir: str = "Histograms",
        bins: int = 50,
        color: str = "#2196F3",
    ):
        """Plot a single histogram."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(data, bins=bins, color=color, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        self._save(fig, subdir, filename)

    def plot_stage_histograms(
        self,
        stage_data: Dict[str, np.ndarray],
        title: str,
        xlabel: str,
        filename: str,
        subdir: str = "Histograms",
        bins: int = 50,
    ):
        """Plot overlapping histograms for each stage."""
        fig, ax = plt.subplots(figsize=self.figsize)
        for stage in STAGE_ORDER:
            if stage in stage_data and len(stage_data[stage]) > 0:
                ax.hist(
                    stage_data[stage],
                    bins=bins,
                    alpha=0.5,
                    color=self.stage_colors.get(stage, "#888"),
                    label=STAGE_LABELS.get(stage, stage),
                    edgecolor="black",
                    linewidth=0.3,
                )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, subdir, filename)

    # ----------------------------------------------------------------
    # RDF PLOTS
    # ----------------------------------------------------------------

    def plot_rdf(
        self,
        r: np.ndarray,
        g_r: np.ndarray,
        title: str,
        filename: str,
        subdir: str = "RDF",
        peaks: Optional[List[Tuple[float, float]]] = None,
    ):
        """Plot single RDF curve."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(r, g_r, color="#1565C0", linewidth=1.5)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

        if peaks:
            for pr, ph in peaks:
                ax.annotate(
                    f"r={pr:.2f} Å",
                    xy=(pr, ph),
                    xytext=(pr + 0.5, ph + 0.2),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    fontsize=9,
                    color="red",
                )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        self._save(fig, subdir, filename)

    def plot_rdf_stages(
        self,
        stage_rdf: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str,
        filename: str,
        subdir: str = "RDF",
    ):
        """Plot RDF curves for all stages overlaid."""
        fig, ax = plt.subplots(figsize=self.figsize)
        for stage in STAGE_ORDER:
            if stage in stage_rdf:
                r, g_r = stage_rdf[stage]
                ax.plot(
                    r, g_r,
                    color=self.stage_colors.get(stage, "#888"),
                    label=STAGE_LABELS.get(stage, stage),
                    linewidth=1.5,
                    alpha=0.8,
                )
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, subdir, filename)

    # ----------------------------------------------------------------
    # THERMODYNAMIC TREND PLOTS
    # ----------------------------------------------------------------

    def plot_thermo_trend(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        ylabel: str,
        filename: str,
        subdir: str = "Thermodynamic",
        color: str = "#E53935",
    ):
        """Plot thermodynamic quantity vs timestep/frame."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(df[x_col], df[y_col], color=color, linewidth=0.8, alpha=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        self._save(fig, subdir, filename)

    def plot_thermo_all_stages(
        self,
        stage_dfs: Dict[str, pd.DataFrame],
        y_col: str,
        title: str,
        ylabel: str,
        filename: str,
        subdir: str = "Thermodynamic",
    ):
        """Plot a thermo quantity across all stages with color coding."""
        fig, ax = plt.subplots(figsize=(14, 6))
        offset = 0
        for stage in STAGE_ORDER:
            if stage in stage_dfs:
                df = stage_dfs[stage]
                if y_col in df.columns:
                    x = np.arange(len(df)) + offset
                    ax.plot(
                        x, df[y_col],
                        color=self.stage_colors.get(stage, "#888"),
                        label=STAGE_LABELS.get(stage, stage),
                        linewidth=0.8,
                        alpha=0.8,
                    )
                    # Add stage boundary marker
                    ax.axvline(x=offset, color="gray", linestyle=":", alpha=0.3)
                    offset += len(df)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Data Point Index")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        self._save(fig, subdir, filename)

    # ----------------------------------------------------------------
    # EVOLUTION PLOTS
    # ----------------------------------------------------------------

    def plot_feature_evolution(
        self,
        evolution_df: pd.DataFrame,
        feature: str,
        title: str,
        filename: str,
        subdir: str = "Evolution",
    ):
        """Plot feature evolution across stages as a grouped bar chart."""
        feat_data = evolution_df[evolution_df["Feature"] == feature]
        if feat_data.empty:
            return

        stage_cols = [s.capitalize() for s in STAGE_ORDER]
        available_cols = [c for c in stage_cols if c in feat_data.columns]

        if not available_cols:
            return

        fig, ax = plt.subplots(figsize=self.figsize)
        x = np.arange(len(available_cols))
        width = 0.25

        groups = feat_data.groupby(["Composition", "Temperature"])
        for i, ((comp, temp), group) in enumerate(groups):
            values = [float(group[c].iloc[0]) if c in group.columns and not group[c].isna().iloc[0] else 0
                      for c in available_cols]
            offset = (i - len(groups) // 2) * width
            bars = ax.bar(
                x + offset, values, width,
                label=f"{comp} @ {temp}°C",
                alpha=0.8,
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Stage")
        ax.set_ylabel(feature)
        ax.set_xticks(x)
        ax.set_xticklabels(available_cols, rotation=45, ha="right")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")
        self._save(fig, subdir, filename)

    def plot_evolution_line(
        self,
        evolution_df: pd.DataFrame,
        feature: str,
        composition: str,
        title: str,
        filename: str,
        subdir: str = "Evolution",
    ):
        """Plot feature evolution as lines for one composition, all temperatures."""
        feat_data = evolution_df[
            (evolution_df["Feature"] == feature) &
            (evolution_df["Composition"] == composition)
        ]
        if feat_data.empty:
            return

        stage_cols = [s.capitalize() for s in STAGE_ORDER]
        available_cols = [c for c in stage_cols if c in feat_data.columns]

        fig, ax = plt.subplots(figsize=self.figsize)
        for _, row in feat_data.iterrows():
            values = [float(row[c]) if c in row.index and not pd.isna(row[c]) else np.nan
                      for c in available_cols]
            ax.plot(
                available_cols, values,
                marker="o", linewidth=2, markersize=6,
                label=f"{int(row['Temperature'])}°C",
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Stage")
        ax.set_ylabel(feature)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        self._save(fig, subdir, filename)

    # ----------------------------------------------------------------
    # BOND PLOTS
    # ----------------------------------------------------------------

    def plot_bond_evolution(
        self,
        bond_df: pd.DataFrame,
        title: str,
        filename: str,
        subdir: str = "Bond",
    ):
        """Plot bond type counts evolution."""
        bond_cols = [c for c in bond_df.columns if c.startswith("Bonds_")]
        if not bond_cols or "Stage" not in bond_df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(bond_df))
        for col in bond_cols:
            if bond_df[col].sum() > 0:
                label = col.replace("Bonds_", "").replace("_", "-")
                ax.plot(x, bond_df[col], label=label, linewidth=1.5, alpha=0.8)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Bond Count")
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        self._save(fig, subdir, filename)

    # ----------------------------------------------------------------
    # STATISTICAL PLOTS
    # ----------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        corr_df: pd.DataFrame,
        title: str,
        filename: str,
        subdir: str = "Statistics",
    ):
        """Plot correlation matrix as heatmap."""
        # Limit to top features if too many
        if corr_df.shape[0] > 30:
            # Keep features with highest variance in correlations
            var = corr_df.var()
            top_features = var.nlargest(30).index
            corr_df = corr_df.loc[top_features, top_features]

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(corr_df.index, fontsize=7)
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self._save(fig, subdir, filename)

    def plot_pca_scree(
        self,
        explained_variance: np.ndarray,
        title: str = "PCA Scree Plot",
        filename: str = "pca_scree",
        subdir: str = "Statistics",
    ):
        """Plot PCA scree (explained variance) chart."""
        fig, ax = plt.subplots(figsize=self.figsize)
        n = len(explained_variance)
        x = np.arange(1, n + 1)
        cumulative = np.cumsum(explained_variance)

        ax.bar(x, explained_variance * 100, color="#1976D2", alpha=0.7, label="Individual")
        ax.plot(x, cumulative * 100, "ro-", linewidth=2, markersize=6, label="Cumulative")
        ax.axhline(y=90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance (%)")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        self._save(fig, subdir, filename)

    def plot_pca_biplot(
        self,
        scores: pd.DataFrame,
        loadings: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        title: str = "PCA Biplot",
        filename: str = "pca_biplot",
        subdir: str = "Statistics",
    ):
        """Plot PCA biplot (scores + top loadings)."""
        if scores.shape[1] < 2:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot scores
        if labels is not None:
            unique_labels = labels.unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                ax.scatter(
                    scores.loc[mask, "PC1"],
                    scores.loc[mask, "PC2"],
                    c=[color], label=label, alpha=0.6, s=30,
                )
        else:
            ax.scatter(scores["PC1"], scores["PC2"], alpha=0.5, s=20, c="#1976D2")

        # Plot top loadings as arrows
        top_n = min(10, len(loadings))
        loading_magnitude = np.sqrt(loadings["PC1"] ** 2 + loadings["PC2"] ** 2)
        top_features = loading_magnitude.nlargest(top_n).index

        scale = max(abs(scores["PC1"]).max(), abs(scores["PC2"]).max())
        for feat in top_features:
            ax.annotate(
                feat,
                xy=(loadings.loc[feat, "PC1"] * scale * 0.8,
                    loadings.loc[feat, "PC2"] * scale * 0.8),
                fontsize=7, color="red", alpha=0.7,
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.5),
                xytext=(0, 0),
            )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if labels is not None:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self._save(fig, subdir, filename)

    def plot_feature_ranking(
        self,
        ranking_df: pd.DataFrame,
        title: str = "Feature Importance Ranking",
        filename: str = "feature_ranking",
        subdir: str = "Statistics",
        top_n: int = 25,
    ):
        """Plot horizontal bar chart of feature importances."""
        df = ranking_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        ax.barh(df["Feature"], df["Importance"], color=colors, edgecolor="black", linewidth=0.3)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.3, axis="x")
        self._save(fig, subdir, filename)
