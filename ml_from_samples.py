"""
SLS ML from Samples — Comprehensive Multi-Model ML Pipeline.

Reads all Excel files from Desktop/samples, engineers 80+ physics-informed
features, and trains 5 ML models across 3 prediction tasks:

  Task A: Stage Classification  (5 classes: equilibrium/bed/laser/hold/cooling)
  Task B: Composition Prediction (3 classes: 5050/6040/7030)
  Task C: Displacement Regression (predict DispAvg as sintering quality proxy)

Models:
  1. Random Forest (upgraded, tuned)
  2. XGBoost
  3. LightGBM
  4. Support Vector Machine
  5. Voting Ensemble (RF + XGB + LGB)

Output: Results/ML_Results/
  - features.csv               : engineered feature matrix
  - model_comparison.csv       : all model scores
  - best_model.pkl             : saved best model
  - Plots/                     : confusion matrices, feature importance, SHAP, PCA

Usage:
  python ml_from_samples.py
  python ml_from_samples.py --samples "C:/Users/.../Desktop/samples" --output Results/ML_Results
"""

import os
import sys
import warnings
import argparse
import logging
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Scikit-learn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    LabelEncoder,
    label_binarize,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.decomposition import PCA

import joblib

warnings.filterwarnings("ignore")

# ── Local import ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.parsers.samples_parser import SamplesParser

# ── Logging ─────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "ml_pipeline.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Color palette ────────────────────────────────────────────────────────────────

STAGE_COLORS = {
    "equilibrium": "#2196F3",
    "bed":         "#FF9800",
    "laser":       "#F44336",
    "hold":        "#9C27B0",
    "cooling":     "#4CAF50",
}
COMP_COLORS = {
    "5050": "#E91E63",
    "6040": "#00BCD4",
    "7030": "#FFC107",
}
PALETTE = ["#2196F3", "#FF9800", "#F44336", "#9C27B0", "#4CAF50", "#E91E63", "#00BCD4"]


# ── Feature preparation ──────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, logger: logging.Logger):
    """
    Separate feature matrix X from metadata, handle NaNs,
    return (X, feature_names, metadata).
    """
    meta_cols = ["Composition", "Stage", "Position", "Temperature",
                 "Stage_Ordinal", "Position_Ordinal", "Epoxy_Pct", "PA12_Pct"]

    # Drop cols that are purely meta or non-numeric
    drop_cols = [c for c in meta_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    # Drop columns that are entirely NaN
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        logger.info(f"  Dropping {len(all_nan)} all-NaN columns")
        X.drop(columns=all_nan, inplace=True)
    # Drop zero-variance columns
    var = X.var()
    zero_var = var[var < 1e-12].index.tolist()
    if zero_var:
        logger.info(f"  Dropping {len(zero_var)} zero-variance columns")
        X.drop(columns=zero_var, inplace=True)
    # Impute remaining NaN with column median (handles columns with some NaN)
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=X.columns, index=X.index)
    feature_names = list(X.columns)
    logger.info(f"  Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, feature_names, df[drop_cols] if drop_cols else pd.DataFrame()


# ── Model definitions ────────────────────────────────────────────────────────────

def get_classifiers(n_classes: int):
    """Return dict of (name → sklearn estimator) for classification."""
    models = {}

    # 1. Random Forest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # 2 & 3. XGBoost / LightGBM (graceful fallback)
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        logger.warning("XGBoost not installed — skipping")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        logger.warning("LightGBM not installed — skipping")

    # 4. SVM (with scaling inside pipeline)
    models["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ])

    return models


def get_regressors():
    """Return dict of regressors for Task C (displacement regression)."""
    models = {}
    models["RandomForest_Reg"] = RandomForestRegressor(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    )
    try:
        from xgboost import XGBRegressor
        models["XGBoost_Reg"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, n_jobs=-1, random_state=42, verbosity=0
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM_Reg"] = LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            n_jobs=-1, random_state=42, verbose=-1
        )
    except ImportError:
        pass
    return models


# ── Plot helpers ──────────────────────────────────────────────────────────────────

def save_confusion_matrix(
    y_true, y_pred, class_names, title: str, path: str
):
    """Save a publication-quality confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="#ddd", ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
        vmin=0, vmax=1,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(
    importances: np.ndarray, feature_names: list,
    title: str, path: str, top_n: int = 25
):
    """Save a horizontal bar chart of top-N feature importances."""
    idx = np.argsort(importances)[::-1][:top_n]
    top_feat = [feature_names[i] for i in idx]
    top_imp = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_imp)))[::-1]
    bars = ax.barh(range(len(top_imp)), top_imp[::-1], color=colors[::-1], edgecolor="#333", linewidth=0.5)
    ax.set_yticks(range(len(top_feat)))
    ax.set_yticklabels(top_feat[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Value labels on bars
    for bar, val in zip(bars, top_imp[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pca_plot(X: np.ndarray, labels, label_name: str, color_map: dict, title: str, path: str):
    """Save a PCA biplot coloured by label."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    unique_labels = sorted(set(labels))
    colors = list(color_map.values()) if color_map else PALETTE
    for i, lbl in enumerate(unique_labels):
        mask = np.array(labels) == lbl
        c = color_map.get(str(lbl), colors[i % len(colors)])
        ax.scatter(coords[mask, 0], coords[mask, 1], c=c, label=str(lbl),
                   alpha=0.7, edgecolors="white", linewidths=0.3, s=60)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title=label_name, framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison(results: dict, path: str):
    """Save a grouped bar chart comparing all model accuracies."""
    task_names = list(results.keys())
    all_model_names = sorted({m for t in results.values() for m in t})

    x = np.arange(len(task_names))
    width = 0.15
    n = len(all_model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(all_model_names):
        scores = [results[t].get(model_name, {}).get("test_acc", 0) for t in task_names]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=model_name,
                      color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, scores):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Model Comparison — All Tasks", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_stage_distribution(df: pd.DataFrame, path: str):
    """Save dataset class distribution plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, cmap in zip(axes,
                              ["Stage", "Composition", "Temperature"],
                              [STAGE_COLORS, COMP_COLORS, None]):
        counts = df[col].value_counts()
        colors = [cmap.get(str(k), "#90CAF9") for k in counts.index] if cmap else None
        counts.plot(kind="bar", ax=ax, color=colors or PALETTE[:len(counts)],
                    edgecolor="white", linewidth=0.5)
        ax.set_title(f"{col} Distribution", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    plt.suptitle("Dataset Overview", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, title: str, path: str, top_n: int = 30):
    """Save correlation heatmap for top-N most variable features."""
    numeric = df.select_dtypes(include=[np.number])
    # Pick top-N by variance
    top_cols = numeric.var().nlargest(top_n).index.tolist()
    corr = numeric[top_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, ax=ax,
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def try_shap(model, X_train, X_test, feature_names, title: str, path: str, logger):
    """Attempt SHAP summary plot — silently skip if shap not installed."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # For multi-class, shap_values is a list; use class with highest variance
        if isinstance(shap_values, list):
            # Pick class 0 for a summary view (or mean absolute across classes)
            sv = np.mean(np.abs(np.array(shap_values)), axis=0)
        else:
            sv = shap_values
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_test, feature_names=feature_names, show=False,
                          plot_type="bar", max_display=20)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  SHAP plot saved: {path}")
    except ImportError:
        logger.warning("  shap not installed — skipping SHAP plot (run: pip install shap)")
    except Exception as e:
        logger.warning(f"  SHAP failed: {e}")


# ── Task A: Stage Classification ─────────────────────────────────────────────────

def run_stage_classification(
    X: np.ndarray,
    y_stage: pd.Series,
    feature_names: list,
    plots_dir: str,
    logger: logging.Logger,
) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info("TASK A: Stage Classification (5 classes)")
    logger.info("=" * 60)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_stage)
    class_names = list(le.classes_)
    n_classes = len(class_names)
    logger.info(f"  Classes: {class_names}")
    logger.info(f"  Samples per class:\n{y_stage.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    results = {}
    best_acc = -1.0
    best_model_obj = None
    best_model_name = None

    classifiers = get_classifiers(n_classes)

    for name, clf in classifiers.items():
        logger.info(f"\n  ── {name} ──")
        t0 = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
        elapsed = time.time() - t0

        logger.info(f"     Test accuracy : {acc:.4f}")
        logger.info(f"     CV accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        logger.info(f"     Time          : {elapsed:.1f}s")

        # Try ROC-AUC (macro OVR)
        try:
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_test)
                y_bin = label_binarize(y_test, classes=list(range(n_classes)))
                auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
            else:
                auc = 0.0
        except Exception:
            auc = 0.0

        results[name] = {
            "test_acc": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "roc_auc": auc,
            "time_s": elapsed,
        }

        # Confusion matrix
        cm_path = os.path.join(plots_dir, f"stage_cm_{name.lower()}.png")
        y_pred_labels = le.inverse_transform(y_pred)
        y_test_labels = le.inverse_transform(y_test)
        save_confusion_matrix(
            y_test_labels, y_pred_labels, class_names,
            f"Stage Classification — {name}\n(Accuracy: {acc:.4f})", cm_path
        )

        # Feature importance
        base_model = clf.named_steps.get("svc", clf) if hasattr(clf, "named_steps") else clf
        if hasattr(base_model, "feature_importances_"):
            fi_path = os.path.join(plots_dir, f"stage_fi_{name.lower()}.png")
            save_feature_importance(
                base_model.feature_importances_, feature_names,
                f"Feature Importance — {name} (Stage Classification)", fi_path
            )

        if acc > best_acc:
            best_acc = acc
            best_model_obj = clf
            best_model_name = name

    # Voting ensemble (RF + XGB + LGB if available)
    logger.info("\n  ── VotingEnsemble ──")
    t0 = time.time()
    est_list = [(n, m) for n, m in classifiers.items() if n not in ("SVM",)]
    if len(est_list) >= 2:
        ensemble = VotingClassifier(estimators=est_list, voting="soft", n_jobs=-1)
        ensemble.fit(X_train, y_train)
        y_pred_ens = ensemble.predict(X_test)
        acc_ens = accuracy_score(y_test, y_pred_ens)
        cv_ens = cross_val_score(ensemble, X, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
        elapsed_ens = time.time() - t0
        logger.info(f"     Test accuracy : {acc_ens:.4f}")
        logger.info(f"     CV accuracy   : {cv_ens.mean():.4f} ± {cv_ens.std():.4f}")
        results["VotingEnsemble"] = {
            "test_acc": acc_ens, "cv_mean": cv_ens.mean(),
            "cv_std": cv_ens.std(), "time_s": elapsed_ens,
        }
        cm_path = os.path.join(plots_dir, "stage_cm_ensemble.png")
        save_confusion_matrix(
            le.inverse_transform(y_test),
            le.inverse_transform(y_pred_ens),
            class_names,
            f"Stage Classification — Voting Ensemble\n(Accuracy: {acc_ens:.4f})",
            cm_path,
        )
        if acc_ens > best_acc:
            best_acc = acc_ens
            best_model_obj = ensemble
            best_model_name = "VotingEnsemble"

    # SHAP for best tree model
    rf_model = classifiers.get("RandomForest")
    if rf_model is not None:
        shap_path = os.path.join(plots_dir, "stage_shap_rf.png")
        try_shap(rf_model, X_train, X_test, feature_names,
                 "SHAP Feature Importance — Random Forest (Stage Classification)",
                 shap_path, logger)

    logger.info(f"\n  ★ Best model: {best_model_name} (acc={best_acc:.4f})")
    return results, best_model_obj, best_model_name, le


# ── Task B: Composition Prediction ───────────────────────────────────────────────

def run_composition_prediction(
    X: np.ndarray,
    y_comp: pd.Series,
    feature_names: list,
    plots_dir: str,
    logger: logging.Logger,
) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info("TASK B: Composition Prediction (3 classes)")
    logger.info("=" * 60)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_comp)
    class_names = list(le.classes_)
    logger.info(f"  Classes: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    results = {}
    classifiers = get_classifiers(len(class_names))

    for name, clf in classifiers.items():
        logger.info(f"\n  ── {name} ──")
        t0 = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
        elapsed = time.time() - t0

        logger.info(f"     Test accuracy : {acc:.4f}")
        logger.info(f"     CV accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = {
            "test_acc": acc, "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(), "time_s": elapsed,
        }

        cm_path = os.path.join(plots_dir, f"comp_cm_{name.lower()}.png")
        save_confusion_matrix(
            le.inverse_transform(y_test), le.inverse_transform(y_pred),
            class_names,
            f"Composition Prediction — {name}\n(Accuracy: {acc:.4f})", cm_path
        )

    return results


# ── Task C: Displacement Regression ──────────────────────────────────────────────

def run_displacement_regression(
    X: np.ndarray,
    y_disp: pd.Series,
    feature_names: list,
    plots_dir: str,
    logger: logging.Logger,
) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info("TASK C: Displacement Regression (DispAvg proxy)")
    logger.info("=" * 60)
    logger.info(f"  Target stats: mean={y_disp.mean():.4f}, std={y_disp.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_disp.values, test_size=0.2, random_state=42
    )

    results = {}
    regressors = get_regressors()

    for name, reg in regressors.items():
        logger.info(f"\n  ── {name} ──")
        t0 = time.time()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        cv_r2 = cross_val_score(reg, X, y_disp.values, cv=5, scoring="r2", n_jobs=-1)
        elapsed = time.time() - t0

        logger.info(f"     R²   : {r2:.4f}")
        logger.info(f"     MAE  : {mae:.4f} Å")
        logger.info(f"     RMSE : {rmse:.4f} Å")
        logger.info(f"     CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

        results[name] = {
            "r2": r2, "mae": mae, "rmse": rmse,
            "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
            "time_s": elapsed,
        }

        # Scatter: actual vs predicted
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="white", linewidths=0.3,
                   c=PALETTE[0], s=50)
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual DispAvg (Å)", fontsize=12)
        ax.set_ylabel("Predicted DispAvg (Å)", fontsize=12)
        ax.set_title(f"Displacement Regression — {name}\nR²={r2:.4f}, MAE={mae:.4f} Å",
                     fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        fig.savefig(os.path.join(plots_dir, f"disp_scatter_{name.lower()}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    return results


# ── Main orchestrator ─────────────────────────────────────────────────────────────

def run_pipeline(samples_dir: str, output_dir: str):
    """End-to-end ML pipeline from samples xlsx → trained models + reports."""
    t_total = time.time()
    plots_dir = os.path.join(output_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("SLS ML FROM SAMPLES — Comprehensive ML Pipeline")
    logger.info("=" * 70)
    logger.info(f"Samples dir : {samples_dir}")
    logger.info(f"Output dir  : {output_dir}")

    # ── Phase 1: Parse & Feature Engineering ──────────────────────────────────
    logger.info("\n── Phase 1: Parsing Excel Files & Feature Engineering ──")
    parser = SamplesParser(samples_dir, verbose=True)
    df = parser.build_rich_feature_matrix(max_atoms=5000)

    if df.empty:
        logger.error("FATAL: Empty feature matrix. Check samples directory.")
        return

    logger.info(f"\nRaw feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")

    # Save raw features
    feat_path = os.path.join(output_dir, "features.csv")
    df.to_csv(feat_path, index=False)
    logger.info(f"Saved: {feat_path}")

    # Dataset distribution
    save_stage_distribution(df, os.path.join(plots_dir, "dataset_distribution.png"))

    # ── Phase 2: Prepare feature matrix ───────────────────────────────────────
    logger.info("\n── Phase 2: Preparing Feature Matrix ──")
    X_df, feature_names, meta_df = prepare_features(df, logger)
    X = X_df.values

    # Check targets
    if "Stage" not in df.columns:
        logger.error("'Stage' column not found — cannot run classification!")
        return
    if "Composition" not in df.columns:
        logger.error("'Composition' column not found — cannot run composition prediction!")
        return

    y_stage = df["Stage"].fillna("unknown")
    y_comp = df["Composition"].fillna("unknown")
    y_disp = df["DispAvg"].fillna(df["DispAvg"].median()) if "DispAvg" in df.columns else None

    # Normalise X for PCA/SVM — impute any residual NaN before scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # Safety guard: replace any remaining NaN/Inf with 0
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=0.0)

    # PCA visualizations
    logger.info("\n── Phase 3: PCA Visualizations ──")
    save_pca_plot(X_scaled, list(y_stage), "Stage", STAGE_COLORS,
                  "PCA — Coloured by Sintering Stage",
                  os.path.join(plots_dir, "pca_by_stage.png"))
    save_pca_plot(X_scaled, list(y_comp), "Composition", COMP_COLORS,
                  "PCA — Coloured by Composition",
                  os.path.join(plots_dir, "pca_by_composition.png"))

    # Correlation heatmap
    save_correlation_heatmap(
        X_df, "Feature Correlation Matrix (Top-30 by Variance)",
        os.path.join(plots_dir, "correlation_heatmap.png"), top_n=30
    )

    # ── Phase 4: ML Training ───────────────────────────────────────────────────
    all_results = {}

    # Task A
    stage_results, best_stage_model, best_stage_name, stage_le = run_stage_classification(
        X_scaled, y_stage, feature_names, plots_dir, logger
    )
    all_results["Stage_Classification"] = stage_results

    # Task B
    comp_results = run_composition_prediction(
        X_scaled, y_comp, feature_names, plots_dir, logger
    )
    all_results["Composition_Prediction"] = comp_results

    # Task C
    if y_disp is not None and y_disp.std() > 1e-6:
        disp_results = run_displacement_regression(
            X_scaled, y_disp, feature_names, plots_dir, logger
        )
        all_results["Displacement_Regression"] = disp_results

    # ── Phase 5: Save best model ───────────────────────────────────────────────
    logger.info("\n── Phase 5: Saving Best Model ──")
    model_dir = os.path.join(output_dir, "Models")
    os.makedirs(model_dir, exist_ok=True)

    best_stage_acc = max(v["test_acc"] for v in stage_results.values())
    bundle = {
        "model": best_stage_model,
        "scaler": scaler,
        "label_encoder": stage_le,
        "feature_names": feature_names,
        "best_model_name": best_stage_name,
        "best_accuracy": best_stage_acc,
    }
    model_path = os.path.join(model_dir, "best_stage_model.pkl")
    joblib.dump(bundle, model_path)
    logger.info(f"Saved best model bundle: {model_path}")
    logger.info(f"  Model: {best_stage_name}, Accuracy: {best_stage_acc:.4f}")

    # ── Phase 6: Summary Reports ───────────────────────────────────────────────
    logger.info("\n── Phase 6: Summary Reports ──")

    # Model comparison plot
    save_model_comparison(
        {k: v for k, v in all_results.items() if "Regression" not in k},
        os.path.join(plots_dir, "model_comparison.png")
    )

    # CSV summary
    rows = []
    for task, task_res in all_results.items():
        for model_name, metrics in task_res.items():
            row = {"Task": task, "Model": model_name}
            row.update(metrics)
            rows.append(row)
    comp_df = pd.DataFrame(rows)
    comp_path = os.path.join(output_dir, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Saved: {comp_path}")

    # Feature importance summary (from best RF model)
    try:
        from sklearn.ensemble import RandomForestClassifier as RFC
        stage_le_tmp = LabelEncoder()
        y_enc = stage_le_tmp.fit_transform(y_stage)
        rf_final = RandomForestClassifier(n_estimators=500, random_state=42,
                                           class_weight="balanced", n_jobs=-1)
        rf_final.fit(X_scaled, y_enc)
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf_final.feature_importances_,
        }).sort_values("Importance", ascending=False)
        fi_path = os.path.join(output_dir, "feature_importances.csv")
        fi_df.to_csv(fi_path, index=False)
        logger.info(f"Saved: {fi_path}")

        save_feature_importance(
            rf_final.feature_importances_, feature_names,
            "Top Features for SLS Stage Classification (RF, n=500)",
            os.path.join(plots_dir, "final_feature_importance.png"), top_n=30
        )

        # SHAP
        X_train_f, X_test_f = train_test_split(X_scaled, test_size=0.2, random_state=42)
        try_shap(rf_final, X_train_f, X_test_f, feature_names,
                 "SHAP Summary — Stage Classification (RF n=500)",
                 os.path.join(plots_dir, "final_shap.png"), logger)
    except Exception as e:
        logger.warning(f"Final RF feature importance failed: {e}")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    logger.info("\n" + "=" * 70)
    logger.info(f"ML pipeline complete in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    logger.info(f"Output: {os.path.abspath(output_dir)}")
    logger.info("=" * 70)

    # Print summary table
    logger.info("\n📊 RESULTS SUMMARY")
    logger.info("-" * 50)
    for task, task_res in all_results.items():
        logger.info(f"\n{task}:")
        for model_name, metrics in task_res.items():
            if "r2" in metrics:
                logger.info(f"  {model_name:25s}  R²={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}")
            else:
                logger.info(
                    f"  {model_name:25s}  acc={metrics['test_acc']:.4f}"
                    f"  cv={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}"
                )


# ── Entry point ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SLS ML from Samples — Comprehensive ML Pipeline"
    )
    parser.add_argument(
        "--samples",
        default=r"C:\Users\sriva\Desktop\samples",
        help="Path to samples directory (default: Desktop/samples)",
    )
    parser.add_argument(
        "--output",
        default=r"Results\ML_Results",
        help="Output directory for results (default: Results/ML_Results)",
    )
    args = parser.parse_args()

    run_pipeline(args.samples, args.output)


if __name__ == "__main__":
    main()
