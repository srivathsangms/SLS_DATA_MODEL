"""
SLS ML ULTIMATE — Maximum Feature Extraction & Multi-Model ML Pipeline.

Combines ALL data sources:
  Source 1: OVITO per-atom xlsx (samples/) → 80+ structural features
            - Coordination number, atomic volume, cavity radius, displacement
            - Per-element CN (C, H, O, N), distribution statistics (skew/kurt/IQR)
  Source 2: LAMMPS log.lammps → 50+ thermodynamic features
            - Temp, Pressure, PotEng, KinEng, TotEng, Density, Volume per stage
            - Derived: ThermalFraction, EnergyDensity, CV, PctChange
  Source 3: Bond evolution CSVs → 90+ chemical bonding features
            - C-C, C-H, C-O, C-N, H-O, H-N, N-O bond counts & fractions
            - Bond entropy, crosslink density, backbone strength, H-bond fraction
  Source 4: Temperature profile fix files → 15+ thermostat features
            - Mean/Std/Stability/Adherence/Drift per stage
  Source 5: LAMMPS .data files → 20+ atomic charge features
            - Per-element (C,H,O,N) charge mean/std, total imbalance

TOTAL: 300+ physics-informed features

ML Tasks:
  A: Stage Classification      — 5 classes (equilibrium/bed/laser/hold/cooling)
  B: Composition Prediction    — 3 classes (5050/6040/7030)
  C: Displacement Regression   — predict DispAvg (Å)
  D: Potential Energy Regression — predict mean PotEng (kcal/mol)
  E: Bond Network Stability    — binary: is C-C backbone > median?
  F: Temperature Adherence     — regression: TempProfile_Adherence

ML Models (7):
  1. Random Forest (n=500)
  2. XGBoost
  3. LightGBM
  4. CatBoost (if available)
  5. Extra Trees
  6. Voting Ensemble (soft vote: RF+XGB+LGB+ET)
  7. Stacking Ensemble (meta: Ridge)

Usage:
  python ml_ultimate.py
  python ml_ultimate.py --owais "C:/.../.../Owais Data" --samples "C:/.../samples" --output Results/ML_Ultimate
"""

import os
import sys
import warnings
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, StackingClassifier, StackingRegressor
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error, roc_auc_score,
)
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.parsers.samples_parser import SamplesParser
from pipeline.parsers.md_data_extractor import MdDataExtractor

# ── Defaults ────────────────────────────────────────────────────────────────────
DEFAULT_OWAIS   = r"C:\Users\sriva\Desktop\IIT JAMMU\Owais Data"
DEFAULT_SAMPLES = r"C:\Users\sriva\Desktop\samples"
DEFAULT_OUTPUT  = r"Results\ML_Ultimate"

# ── Color palette ────────────────────────────────────────────────────────────────
STAGE_COLORS = {"equilibrium":"#2196F3","bed":"#FF9800","laser":"#F44336","hold":"#9C27B0","cooling":"#4CAF50"}
COMP_COLORS  = {"5050":"#E91E63","6040":"#00BCD4","7030":"#FFC107"}
PALETTE = ["#4d7cff","#ff6b6b","#ffd93d","#6bcb77","#c77dff","#ff922b","#20c997","#f06595"]


# ── Logging ──────────────────────────────────────────────────────────────────────
def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    lp = os.path.join(output_dir, "ml_ultimate.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(lp, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)
    return root


# ── Feature preparation ──────────────────────────────────────────────────────────
def prepare_X(df: pd.DataFrame, meta_cols: list, logger) -> tuple:
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        X.drop(columns=all_nan, inplace=True)
    zero_var = X.columns[X.var() < 1e-12].tolist()
    if zero_var:
        X.drop(columns=zero_var, inplace=True)
    imp = SimpleImputer(strategy="median")
    X_arr = imp.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=X.columns, index=X.index)
    logger.info(f"  Features after cleaning: {X.shape[1]}")
    return X, list(X.columns)


# ── Models ───────────────────────────────────────────────────────────────────────
def build_classifiers():
    models = {}
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42
    )
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42
    )
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0
        )
    except ImportError: pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1
        )
    except ImportError: pass
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=42, verbose=0
        )
    except ImportError: pass
    return models


def build_regressors():
    models = {}
    models["RandomForest"] = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    models["ExtraTrees"]   = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbosity=0
        )
    except ImportError: pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbose=-1
        )
    except ImportError: pass
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=400, depth=6, learning_rate=0.05, random_seed=42, verbose=0
        )
    except ImportError: pass
    return models


# ── Plot helpers ─────────────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, classes, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_n = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)*1.5), max(5, len(classes)*1.4)))
    sns.heatmap(cm_n, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax, annot_kws={"size":11,"weight":"bold"}, vmin=0, vmax=1)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.tick_params(axis="x", rotation=30); ax.tick_params(axis="y", rotation=0)
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def save_feature_importance(importances, names, title, path, top_n=30):
    idx = np.argsort(importances)[::-1][:top_n]
    top_f = [names[i] for i in idx]; top_v = importances[idx]
    fig, ax = plt.subplots(figsize=(11, max(6, top_n*0.32)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_v)))[::-1]
    ax.barh(range(len(top_v)), top_v[::-1], color=colors[::-1], edgecolor="#333", linewidth=0.4)
    ax.set_yticks(range(len(top_f))); ax.set_yticklabels(top_f[::-1], fontsize=9)
    ax.set_xlabel("Importance", fontsize=11); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
    for i, v in enumerate(top_v[::-1]):
        ax.text(v+0.0005, i, f"{v:.4f}", va="center", fontsize=7)
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def save_pca_plot(X, labels, label_name, color_map, title, path):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(9, 7))
    unique = sorted(set(labels))
    for i, lbl in enumerate(unique):
        mask = np.array(labels) == lbl
        c = color_map.get(str(lbl), PALETTE[i % len(PALETTE)])
        ax.scatter(coords[mask,0], coords[mask,1], c=c, label=str(lbl),
                   alpha=0.75, edgecolors="white", linewidths=0.3, s=65)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title=label_name, framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.25); ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def save_regression_scatter(y_true, y_pred, model_name, target_name, r2, mae, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.65, c=PALETTE[0], edgecolors="white", linewidths=0.3, s=55)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel(f"Actual {target_name}", fontsize=12)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=12)
    ax.set_title(f"{target_name} Regression — {model_name}\nR²={r2:.4f}  MAE={mae:.4f}",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.25)
    ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def save_model_comparison(results, tasks, path):
    all_models = sorted({m for t in results.values() for m in t})
    x = np.arange(len(tasks)); w = 0.12; n = len(all_models)
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, model in enumerate(all_models):
        scores = [results.get(t, {}).get(model, {}).get("test_acc", 0) for t in tasks]
        offset = (i - n/2 + 0.5) * w
        bars = ax.bar(x + offset, scores, w, label=model,
                      color=PALETTE[i%len(PALETTE)], edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, scores):
            if v > 0.02:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=10, rotation=15)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Ultimate Model Comparison — All Classification Tasks", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15); ax.axhline(1.0, color="gray", ls="--", alpha=0.4, lw=0.8)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", alpha=0.25); ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def save_feature_source_pie(feature_names, path):
    """Pie chart of feature source breakdown."""
    sources = {"OVITO Structural": 0, "Bond Chemistry": 0, "Thermodynamics": 0,
               "Temperature Profile": 0, "Atomic Charges": 0, "Metadata/Derived": 0}
    for fn in feature_names:
        if fn.startswith("Bond_"):           sources["Bond Chemistry"] += 1
        elif fn.startswith("Thermo_"):       sources["Thermodynamics"] += 1
        elif fn.startswith("TempProfile_"): sources["Temperature Profile"] += 1
        elif fn.startswith("Charge_"):       sources["Atomic Charges"] += 1
        elif any(x in fn for x in ["_CN","_Vol","_Cav","_Disp","FracHigh","FracLow","Delta_Mean"]):
            sources["OVITO Structural"] += 1
        else:                                sources["Metadata/Derived"] += 1
    labels = [k for k, v in sources.items() if v > 0]
    sizes  = [sources[k] for k in labels]
    colors_pie = ["#4d7cff","#ff6b6b","#ffd93d","#6bcb77","#c77dff","#ff922b"]
    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(sizes, labels=None, colors=colors_pie[:len(labels)],
                                       autopct="%1.1f%%", startangle=140,
                                       pctdistance=0.82, wedgeprops=dict(linewidth=1.5, edgecolor="white"))
    for at in autotexts: at.set_fontsize(10); at.set_fontweight("bold")
    ax.legend(wedges, [f"{l} ({s})" for l,s in zip(labels,sizes)],
              loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11)
    ax.set_title("Feature Source Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def save_correlation_heatmap(df, title, path, top_n=35):
    num = df.select_dtypes(include=[np.number])
    top_cols = num.var().nlargest(top_n).index.tolist()
    corr = num[top_cols].corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.2, ax=ax, xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.75})
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def try_shap(model, X_train, X_test, feature_names, title, path, logger):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            sv = np.mean(np.abs(np.array(shap_values)), axis=0)
        else:
            sv = np.abs(shap_values)
        # Bar summary
        mean_shap = np.mean(sv, axis=0)
        idx = np.argsort(mean_shap)[::-1][:25]
        top_names = [feature_names[i] for i in idx]
        top_vals  = mean_shap[idx]
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_vals)))[::-1]
        ax.barh(range(len(top_vals)), top_vals[::-1], color=colors[::-1], edgecolor="#333", lw=0.4)
        ax.set_yticks(range(len(top_names))); ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
        plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
        logger.info(f"  SHAP saved: {os.path.basename(path)}")
    except ImportError:
        logger.warning("  shap not installed — run: pip install shap")
    except Exception as e:
        logger.warning(f"  SHAP failed: {e}")


# ── Classification task ──────────────────────────────────────────────────────────
def run_classification_task(X_scaled, y_series, feature_names, task_label, plots_dir, logger):
    logger.info(f"\n{'='*60}\nTASK: {task_label}\n{'='*60}")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series)
    classes = list(le.classes_)
    logger.info(f"  Classes: {classes}")
    logger.info(f"  Distribution:\n{y_series.value_counts().to_string()}")

    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_enc, test_size=0.2,
                                                random_state=42, stratify=y_enc)
    results = {}
    best_acc, best_model, best_name = -1, None, None
    classifiers = build_classifiers()

    for name, clf in classifiers.items():
        logger.info(f"\n  ── {name}")
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        cv  = cross_val_score(clf, X_scaled, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
        elapsed = time.time() - t0
        logger.info(f"     Accuracy : {acc:.4f}")
        logger.info(f"     CV       : {cv.mean():.4f} ± {cv.std():.4f}  [{elapsed:.1f}s]")
        results[name] = {"test_acc": acc, "cv_mean": cv.mean(), "cv_std": cv.std(), "time_s": elapsed}
        tag = task_label.replace(" ", "_").lower()
        save_confusion_matrix(le.inverse_transform(y_te), le.inverse_transform(y_pred),
                              classes, f"{task_label} — {name}\n(acc={acc:.4f})",
                              os.path.join(plots_dir, f"cm_{tag}_{name.lower()}.png"))
        base = clf.named_steps.get("svc", clf) if hasattr(clf, "named_steps") else clf
        if hasattr(base, "feature_importances_"):
            save_feature_importance(base.feature_importances_, feature_names,
                                    f"Feature Importance — {name} ({task_label})",
                                    os.path.join(plots_dir, f"fi_{tag}_{name.lower()}.png"))
        if acc > best_acc:
            best_acc, best_model, best_name = acc, clf, name

    # Voting ensemble
    est_list = [(n, m) for n, m in classifiers.items()]
    if len(est_list) >= 2:
        logger.info(f"\n  ── VotingEnsemble")
        t0 = time.time()
        ens = VotingClassifier(estimators=est_list, voting="soft", n_jobs=-1)
        ens.fit(X_tr, y_tr)
        y_pred_ens = ens.predict(X_te)
        acc_ens = accuracy_score(y_te, y_pred_ens)
        cv_ens  = cross_val_score(ens, X_scaled, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
        elapsed_ens = time.time() - t0
        logger.info(f"     Accuracy : {acc_ens:.4f}")
        logger.info(f"     CV       : {cv_ens.mean():.4f} ± {cv_ens.std():.4f}  [{elapsed_ens:.1f}s]")
        results["VotingEnsemble"] = {"test_acc": acc_ens, "cv_mean": cv_ens.mean(),
                                     "cv_std": cv_ens.std(), "time_s": elapsed_ens}
        tag = task_label.replace(" ", "_").lower()
        save_confusion_matrix(le.inverse_transform(y_te), le.inverse_transform(y_pred_ens),
                              classes, f"{task_label} — Voting Ensemble\n(acc={acc_ens:.4f})",
                              os.path.join(plots_dir, f"cm_{tag}_ensemble.png"))
        if acc_ens > best_acc:
            best_acc, best_model, best_name = acc_ens, ens, "VotingEnsemble"

    # SHAP for best RF
    rf = classifiers.get("RandomForest")
    if rf:
        try_shap(rf, X_tr, X_te, feature_names,
                 f"SHAP — {task_label} (Random Forest)",
                 os.path.join(plots_dir, f"shap_{task_label.replace(' ','_').lower()}_rf.png"),
                 logger)

    logger.info(f"\n  ★ Best: {best_name} (acc={best_acc:.4f})")
    return results, best_model, best_name, le


# ── Regression task ───────────────────────────────────────────────────────────────
def run_regression_task(X_scaled, y_series, feature_names, task_label, target_unit, plots_dir, logger):
    logger.info(f"\n{'='*60}\nREGRESSION: {task_label}\n{'='*60}")
    logger.info(f"  Target: mean={y_series.mean():.4f}, std={y_series.std():.4f} {target_unit}")
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_series.values,
                                                test_size=0.2, random_state=42)
    results = {}
    regressors = build_regressors()
    for name, reg in regressors.items():
        logger.info(f"\n  ── {name}")
        t0 = time.time()
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)
        r2   = r2_score(y_te, y_pred)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        cv_r2 = cross_val_score(reg, X_scaled, y_series.values, cv=5, scoring="r2", n_jobs=-1)
        elapsed = time.time() - t0
        logger.info(f"     R²   : {r2:.4f}  MAE: {mae:.4f} {target_unit}  [{elapsed:.1f}s]")
        logger.info(f"     CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        results[name] = {"r2": r2, "mae": mae, "rmse": rmse,
                         "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(), "time_s": elapsed}
        tag = task_label.replace(" ","_").lower()
        save_regression_scatter(pd.Series(y_te), pd.Series(y_pred), name,
                                f"{task_label} ({target_unit})", r2, mae,
                                os.path.join(plots_dir, f"scatter_{tag}_{name.lower()}.png"))
    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────────
def run_ultimate_pipeline(owais_dir: str, samples_dir: str, output_dir: str):
    t_start = time.time()
    plots_dir = os.path.join(output_dir, "Plots")
    models_dir = os.path.join(output_dir, "Models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("SLS ML ULTIMATE — Maximum Feature Extraction + 7 Models × 6 Tasks")
    logger.info("=" * 70)
    logger.info(f"Owais Data  : {owais_dir}")
    logger.info(f"Samples Dir : {samples_dir}")
    logger.info(f"Output Dir  : {output_dir}")

    # ── Phase 1: Parse OVITO xlsx (samples/) ─────────────────────────────────
    logger.info("\n── Phase 1: OVITO Structural Features (samples/) ──")
    cache_ovito = os.path.join(output_dir, "cache_ovito.csv")
    if os.path.exists(cache_ovito):
        logger.info(f"  Loading from cache: {cache_ovito}")
        df_ovito = pd.read_csv(cache_ovito)
    else:
        # Try both sample directories
        for src in [samples_dir, os.path.join(owais_dir, "samples")]:
            if os.path.exists(src):
                try:
                    sp = SamplesParser(src, verbose=True)
                    atom_cache = os.path.join(output_dir, "cache_per_atom.csv")
                    df_ovito = sp.build_rich_feature_matrix(max_atoms=8000)
                    df_ovito.to_csv(cache_ovito, index=False)
                    logger.info(f"  OVITO features: {df_ovito.shape}")
                    break
                except Exception as e:
                    logger.warning(f"  SamplesParser failed for {src}: {e}")
        else:
            df_ovito = pd.DataFrame()
    logger.info(f"  OVITO shape: {df_ovito.shape}")

    # ── Phase 2: Extract MD features (Owais Data) ─────────────────────────────
    logger.info("\n── Phase 2: MD Features (log, bonds, temp, charges) ──")
    cache_md = os.path.join(output_dir, "cache_md.csv")
    extractor = MdDataExtractor(owais_dir)
    df_md = extractor.extract_all(cache_path=cache_md)
    logger.info(f"  MD features shape: {df_md.shape}")

    # ── Phase 3: Merge all sources ────────────────────────────────────────────
    logger.info("\n── Phase 3: Merging All Feature Sources ──")
    if df_ovito.empty and df_md.empty:
        logger.error("No features extracted! Check data paths.")
        return

    merge_keys = ["Composition", "Temperature", "Stage"]

    if not df_ovito.empty and not df_md.empty:
        # Align merge key types
        for df in [df_ovito, df_md]:
            df["Composition"] = df["Composition"].astype(str)
            df["Temperature"] = df["Temperature"].astype(int)
            df["Stage"] = df["Stage"].astype(str)

        # df_ovito has 3 rows per (comp, temp, stage) — Position = Start/Middle/End
        # df_md has 1 row per (comp, temp, stage)
        # Merge MD onto each OVITO row
        df_all = df_ovito.merge(df_md, on=merge_keys, how="left", suffixes=("", "_md"))
        dup = [c for c in df_all.columns if c.endswith("_md")]
        df_all.drop(columns=dup, inplace=True)
    elif df_ovito.empty:
        df_all = df_md.copy()
    else:
        df_all = df_ovito.copy()

    logger.info(f"  Merged dataset: {df_all.shape[0]} rows × {df_all.shape[1]} columns")

    # Save full feature matrix
    full_feat_path = os.path.join(output_dir, "features_ultimate.csv")
    df_all.to_csv(full_feat_path, index=False)
    logger.info(f"  Saved: {full_feat_path}")

    # ── Phase 4: Feature prep & visualisations ────────────────────────────────
    logger.info("\n── Phase 4: Feature Preparation & Visualizations ──")
    meta_cols = ["Composition", "Stage", "Position", "Temperature",
                 "Stage_Ordinal", "Position_Ordinal", "Epoxy_Pct", "PA12_Pct"]

    X_df, feature_names = prepare_X(df_all, meta_cols, logger)
    X = X_df.values
    logger.info(f"  Final feature count: {len(feature_names)}")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=0.0)

    # Feature source pie chart
    save_feature_source_pie(feature_names, os.path.join(plots_dir, "feature_sources_pie.png"))

    # PCA plots
    for label_col, cmap in [("Stage", STAGE_COLORS), ("Composition", COMP_COLORS)]:
        if label_col in df_all.columns:
            save_pca_plot(X_scaled, list(df_all[label_col]), label_col, cmap,
                         f"PCA — Coloured by {label_col}",
                         os.path.join(plots_dir, f"pca_by_{label_col.lower()}.png"))

    # Correlation heatmap
    save_correlation_heatmap(X_df, "Feature Correlation (Top-35 by Variance)",
                             os.path.join(plots_dir, "correlation_heatmap.png"), top_n=35)

    # ── Phase 5: ML Training ──────────────────────────────────────────────────
    logger.info("\n── Phase 5: ML Model Training ──")
    all_clf_results = {}
    all_reg_results = {}
    best_model_bundle = None

    # --- Task A: Stage Classification ---
    if "Stage" in df_all.columns:
        y_stage = df_all["Stage"].fillna("unknown")
        res_A, best_A, best_A_name, le_A = run_classification_task(
            X_scaled, y_stage, feature_names,
            "Stage Classification", plots_dir, logger
        )
        all_clf_results["Stage_Classification"] = res_A
        best_acc_A = max(v["test_acc"] for v in res_A.values())
        best_model_bundle = {
            "model": best_A, "scaler": scaler, "label_encoder": le_A,
            "feature_names": feature_names, "model_name": best_A_name,
            "accuracy": best_acc_A, "task": "Stage Classification",
        }

    # --- Task B: Composition Prediction ---
    if "Composition" in df_all.columns:
        y_comp = df_all["Composition"].fillna("unknown")
        res_B, best_B, best_B_name, le_B = run_classification_task(
            X_scaled, y_comp, feature_names,
            "Composition Prediction", plots_dir, logger
        )
        all_clf_results["Composition_Prediction"] = res_B

    # --- Task C: Displacement Regression ---
    if "DispAvg" in df_all.columns and df_all["DispAvg"].std() > 1e-6:
        y_disp = df_all["DispAvg"].fillna(df_all["DispAvg"].median())
        res_C = run_regression_task(X_scaled, y_disp, feature_names,
                                    "Displacement", "Å", plots_dir, logger)
        all_reg_results["Displacement_Regression"] = res_C

    # --- Task D: Potential Energy Regression ---
    pe_col = "Thermo_PotEng_Mean"
    if pe_col in df_all.columns and df_all[pe_col].std() > 1e-6:
        y_pe = df_all[pe_col].fillna(df_all[pe_col].median())
        res_D = run_regression_task(X_scaled, y_pe, feature_names,
                                    "Potential Energy", "kcal/mol", plots_dir, logger)
        all_reg_results["PotEng_Regression"] = res_D

    # --- Task E: Bond Stability Classification (binary) ---
    cc_col = "Bond_CC_Mean"
    if cc_col in df_all.columns:
        median_cc = df_all[cc_col].median()
        y_bond_stable = (df_all[cc_col] >= median_cc).map(
            {True: "stable", False: "unstable"}
        )
        res_E, _, _, _ = run_classification_task(
            X_scaled, y_bond_stable, feature_names,
            "Bond Stability", plots_dir, logger
        )
        all_clf_results["Bond_Stability"] = res_E

    # --- Task F: Temperature Adherence Regression ---
    tadh_col = "TempProfile_Adherence"
    if tadh_col in df_all.columns and df_all[tadh_col].std() > 1e-6:
        y_tadh = df_all[tadh_col].fillna(df_all[tadh_col].median())
        res_F = run_regression_task(X_scaled, y_tadh, feature_names,
                                    "Temp Adherence", "(0-1)", plots_dir, logger)
        all_reg_results["TempAdherence_Regression"] = res_F

    # ── Phase 6: Feature importance (global RF n=500) ─────────────────────────
    logger.info("\n── Phase 6: Global Feature Importance (RF n=500) ──")
    if "Stage" in df_all.columns:
        le_tmp = LabelEncoder()
        y_enc_tmp = le_tmp.fit_transform(df_all["Stage"].fillna("unknown"))
        rf_global = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                            n_jobs=-1, random_state=42)
        rf_global.fit(X_scaled, y_enc_tmp)
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf_global.feature_importances_,
            "Source": [_feature_source(fn) for fn in feature_names],
        }).sort_values("Importance", ascending=False)
        fi_df.to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)
        save_feature_importance(rf_global.feature_importances_, feature_names,
                                "Top 30 Features — Stage Classification (RF n=500)",
                                os.path.join(plots_dir, "global_feature_importance.png"), top_n=30)
        # SHAP
        X_tr_g, X_te_g = train_test_split(X_scaled, test_size=0.2, random_state=42)
        try_shap(rf_global, X_tr_g, X_te_g, feature_names,
                 "SHAP Feature Importance — Stage Classification (RF n=500)",
                 os.path.join(plots_dir, "shap_global.png"), logger)

        # Source-level importance
        src_imp = fi_df.groupby("Source")["Importance"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_s = [PALETTE[i%len(PALETTE)] for i in range(len(src_imp))]
        src_imp.plot(kind="bar", ax=ax, color=colors_s, edgecolor="white")
        ax.set_title("Feature Source Contribution to Stage Classification", fontsize=13, fontweight="bold")
        ax.set_ylabel("Sum of RF Importance"); ax.tick_params(axis="x", rotation=30)
        ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "source_importance.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # ── Phase 7: Save best model ──────────────────────────────────────────────
    logger.info("\n── Phase 7: Saving Best Model ──")
    if best_model_bundle:
        mpath = os.path.join(models_dir, "best_model_bundle.pkl")
        joblib.dump(best_model_bundle, mpath)
        logger.info(f"  Saved: {mpath}")
        logger.info(f"  Task: {best_model_bundle['task']}")
        logger.info(f"  Model: {best_model_bundle['model_name']}")
        logger.info(f"  Accuracy: {best_model_bundle['accuracy']:.4f}")

    # ── Phase 8: Summary reports ──────────────────────────────────────────────
    logger.info("\n── Phase 8: Summary Reports ──")
    rows = []
    for task, tres in all_clf_results.items():
        for model, m in tres.items():
            rows.append({"Type":"Classification","Task":task,"Model":model,**m})
    for task, tres in all_reg_results.items():
        for model, m in tres.items():
            rows.append({"Type":"Regression","Task":task,"Model":model,**m})
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    logger.info(f"  Saved model_comparison.csv ({len(comp_df)} rows)")

    # Comparison bar chart (classification tasks only)
    if all_clf_results:
        save_model_comparison(all_clf_results, list(all_clf_results.keys()),
                              os.path.join(plots_dir, "model_comparison.png"))

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"ULTIMATE ML PIPELINE COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Output: {os.path.abspath(output_dir)}")
    logger.info(f"{'='*70}")

    logger.info("\n📊 RESULTS SUMMARY")
    logger.info("-" * 55)
    for task, tres in all_clf_results.items():
        logger.info(f"\n{task}:")
        for model, m in tres.items():
            logger.info(f"  {model:28s}  acc={m['test_acc']:.4f}  cv={m['cv_mean']:.4f}±{m['cv_std']:.4f}")
    for task, tres in all_reg_results.items():
        logger.info(f"\n{task}:")
        for model, m in tres.items():
            logger.info(f"  {model:28s}  R²={m['r2']:.4f}  MAE={m['mae']:.4f}")


def _feature_source(fn: str) -> str:
    if fn.startswith("Bond_"):             return "Bond Chemistry"
    if fn.startswith("Thermo_"):           return "Thermodynamics"
    if fn.startswith("TempProfile_"):      return "Temp Profile"
    if fn.startswith("Charge_"):           return "Atomic Charges"
    if any(x in fn for x in ["_CN","_Vol","_Cav","_Disp","FracHigh","FracLow","Delta_Mean"]):
        return "OVITO Structural"
    return "Metadata/Derived"


# ── Entry point ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="SLS ML ULTIMATE Pipeline")
    ap.add_argument("--owais",   default=DEFAULT_OWAIS,   help="Path to Owais Data root")
    ap.add_argument("--samples", default=DEFAULT_SAMPLES, help="Path to samples directory")
    ap.add_argument("--output",  default=DEFAULT_OUTPUT,  help="Output directory")
    args = ap.parse_args()
    run_ultimate_pipeline(args.owais, args.samples, args.output)

if __name__ == "__main__":
    main()
