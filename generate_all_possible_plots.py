"""
Scientific plotting script for SLS MD Molecular Dynamics & Machine Learning.
Generates 12 high-resolution, physics-informed publication plots using features_ultimate.csv.
Saves outputs directly to Results/ML_Ultimate/Plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Output path
PLOTS_DIR = r"Results\ML_Ultimate\Plots"
FEATURES_PATH = r"Results\ML_Ultimate\features_ultimate.csv"

# Configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14
})

STAGE_ORDER = ["equilibrium", "bed", "laser", "hold", "cooling"]
STAGE_COLORS = {"equilibrium": "#2196F3", "bed": "#FF9800", "laser": "#F44336", "hold": "#9C27B0", "cooling": "#4CAF50"}
COMP_COLORS = {"5050": "#E91E63", "6040": "#00BCD4", "7030": "#FFC107"}
PALETTE = ["#4d7cff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff"]

def main():
    print("Loading ultimate features dataset...")
    if not os.path.exists(FEATURES_PATH):
        print(f"Error: {FEATURES_PATH} not found. Run pipeline first.")
        return
        
    df = pd.read_csv(FEATURES_PATH)
    df["Composition"] = df["Composition"].astype(str)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # ── Plot 1: Violin plot of Coordination Average by Stage & Composition ──────────
    print("Generating Plot 1: CN Violin Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df, x="Stage", y="CoordAvg", hue="Composition",
        order=STAGE_ORDER, palette=COMP_COLORS, ax=ax, split=False, inner="quart"
    )
    ax.set_title("Atomic Coordination Number Distribution across Sintering Stages", fontweight="bold", pad=15)
    ax.set_xlabel("Sintering Stage")
    ax.set_ylabel("Coordination Number (CN)")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scientific_CN_violin.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Scatter plot of Volume vs Cavity Radius coloured by Stage ─────────────
    print("Generating Plot 2: Volume vs Cavity Scatter...")
    fig, ax = plt.subplots(figsize=(9, 7))
    for stage in STAGE_ORDER:
        mask = df["Stage"] == stage
        ax.scatter(
            df.loc[mask, "VolAvg"], df.loc[mask, "CavAvg"],
            c=STAGE_COLORS[stage], label=stage.upper(), alpha=0.75, edgecolors="white", linewidths=0.3, s=60
        )
    ax.set_title("Atomic Volume vs. Cavity Radius Sintering Trajectory", fontweight="bold", pad=15)
    ax.set_xlabel("Mean Atomic Volume (Å³)")
    ax.set_ylabel("Mean Cavity Radius (Å)")
    ax.legend(title="Sintering Stage")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scientific_vol_vs_cavity.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Density Evolution by Temperature & Stage ────────────────────────────
    print("Generating Plot 3: Density Line Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df, x="Temperature", y="Thermo_Density_Mean", hue="Stage", style="Stage",
        hue_order=STAGE_ORDER, palette=STAGE_COLORS, markers=True, dashes=False, err_style="band", ax=ax, lw=2
    )
    ax.set_title("Thermodynamic Density vs. Build Temperature", fontweight="bold", pad=15)
    ax.set_xlabel("Build Temperature (°C)")
    ax.set_ylabel("Thermodynamic Density (g/cm³)")
    ax.legend(title="Sintering Stage", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scientific_density_vs_temp.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 4: Potential Energy vs. Kinetic Energy ─────────────────────────────────
    print("Generating Plot 4: Energy State Phase Space...")
    fig, ax = plt.subplots(figsize=(9, 7))
    for stage in STAGE_ORDER:
        mask = df["Stage"] == stage
        ax.scatter(
            df.loc[mask, "Thermo_PotEng_Mean"] / 1e6, df.loc[mask, "Thermo_KinEng_Mean"] / 1e3,
            c=STAGE_COLORS[stage], label=stage.upper(), alpha=0.8, edgecolors="white", linewidths=0.3, s=70
        )
    ax.set_title("Energy Phase Space Sintering Map (PE vs. KE)", fontweight="bold", pad=15)
    ax.set_xlabel("Potential Energy (×10⁶ kcal/mol)")
    ax.set_ylabel("Kinetic Energy (×10³ kcal/mol)")
    ax.legend(title="Sintering Stage")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scientific_energy_landscape.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 5: Average Displacement Magnitude by Position ──────────────────────────
    print("Generating Plot 5: Displacement Boxplot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df, x="Position", y="DispAvg", hue="Stage",
        hue_order=STAGE_ORDER, palette=STAGE_COLORS, ax=ax, width=0.6, fliersize=3
    )
    ax.set_title("Atomic Displacement Magnitude compared at S/M/E Positions", fontweight="bold", pad=15)
    ax.set_xlabel("Sample Frame Position")
    ax.set_ylabel("Average Displacement (Å)")
    ax.legend(title="Sintering Stage")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scientific_displacement_by_position.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 6: Heatmap of Crosslink Density ────────────────────────────────────────
    print("Generating Plot 6: Crosslink Density Heatmap...")
    if "Bond_CrosslinkDensity" in df.columns:
        pivot_df = df.groupby(["Composition", "Temperature"])["Bond_CrosslinkDensity"].mean().unstack()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Crosslink Density"})
        ax.set_title("Mean Crosslink Density (C-N & C-O fraction) by Comp & Temp", fontweight="bold", pad=15)
        ax.set_xlabel("Build Temperature (°C)")
        ax.set_ylabel("Polymer Blend (Composition)")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "scientific_crosslink_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # ── Plot 7: Backbone strength vs. Temperature ───────────────────────────────────
    print("Generating Plot 7: CC Backbone Strength...")
    if "Bond_BackboneStrength" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=df, x="Temperature", y="Bond_BackboneStrength", hue="Composition",
            palette=COMP_COLORS, ax=ax
        )
        ax.set_title("C-C Backbone Bond Fraction vs. Build Temperature", fontweight="bold", pad=15)
        ax.set_xlabel("Build Temperature (°C)")
        ax.set_ylabel("Backbone Bond Fraction (C-C / Total)")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "scientific_backbone_strength.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # ── Plot 8: Hydrogen Bonding Network ────────────────────────────────────────────
    print("Generating Plot 8: H-Bond Fraction...")
    if "Bond_HBondFraction" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df, x="Temperature", y="Bond_HBondFraction", hue="Composition", style="Composition",
            palette=COMP_COLORS, markers=True, err_style="bars", ax=ax, lw=2
        )
        ax.set_title("Hydrogen Bonding Network Fraction vs. Build Temperature", fontweight="bold", pad=15)
        ax.set_xlabel("Build Temperature (°C)")
        ax.set_ylabel("H-Bond Fraction (H-O & H-N / Total)")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "scientific_h_bonding.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # ── Plot 9: Thermostat Stability per Stage ──────────────────────────────────────
    print("Generating Plot 9: Thermostat Fluctuation Boxplot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df, x="Stage", y="Thermo_Temp_Std", order=STAGE_ORDER, palette=STAGE_COLORS, ax=ax
    )
    ax.set_title("Thermostatic Temperature Fluctuations (Fluctuation Std) per Stage", fontweight="bold", pad=15)
    ax.set_xlabel("Sintering Stage")
    ax.set_ylabel("Temperature Standard Deviation (K)")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scientific_thermostat_stability.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 10: 3D PCA Projection ──────────────────────────────────────────────────
    print("Generating Plot 10: 3D PCA Scatter...")
    meta_cols = ["Composition", "Stage", "Position", "Temperature",
                 "Stage_Ordinal", "Position_Ordinal", "Epoxy_Pct", "PA12_Pct"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    X = X.fillna(X.median())
    
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    
    for stage in STAGE_ORDER:
        mask = df["Stage"] == stage
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=STAGE_COLORS[stage], label=stage.upper(), alpha=0.7, edgecolors="white", linewidths=0.2, s=50
        )
    ax.set_title("3D PCA Cluster Map — SLS Molecular States", fontweight="bold", pad=15)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(title="Sintering Stage", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scientific_pca_3D.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 11: Mechanical Descriptors by Epoxy Fraction ────────────────────────────
    print("Generating Plot 11: Epoxy Fraction Boxplots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("CoordAvg", "Coordination Number"), ("VolAvg", "Atomic Volume (Å³)"), ("CavAvg", "Cavity Radius (Å)")]
    
    for idx, (col, title) in enumerate(metrics):
        sns.boxplot(
            data=df, x="Epoxy_Pct", y=col, ax=axes[idx], palette="Blues"
        )
        axes[idx].set_title(f"Epoxy Fraction vs. {title}", fontweight="bold")
        axes[idx].set_xlabel("Epoxy Percentage (%)")
        axes[idx].set_ylabel(title)
        
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scientific_epoxy_fraction_impact.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 12: Corner Pairplot of Core Parameters ─────────────────────────────────
    print("Generating Plot 12: Pairwise Corner Plot...")
    core_cols = ["CoordAvg", "VolAvg", "CavAvg", "DispAvg", "Stage"]
    pair_df = df[core_cols].copy()
    
    g = sns.pairplot(
        pair_df, hue="Stage", hue_order=STAGE_ORDER, palette=STAGE_COLORS,
        diag_kind="kde", plot_kws={"alpha": 0.65, "edgecolor": "white", "linewidth": 0.2, "s": 40}
    )
    g.fig.suptitle("Pairwise Correlation Matrix of Core Sintering Descriptors", fontweight="bold", y=1.02)
    g.savefig(os.path.join(PLOTS_DIR, "scientific_pairwise_pairplot.png"), dpi=300, bbox_inches="tight")
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
