"""
Machine Learning Sintering Stage Classifier for SLS MD Analysis.

Reads dataset_ml_ready.csv, trains a Random Forest classifier to predict the
SLS process stage from physics-informed descriptors, and outputs metrics/plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_sintering_predictor(results_dir: str = "Results"):
    print("=" * 70)
    print("SLS Sintering Stage ML Predictor")
    print("=" * 70)
    
    # Load dataset
    data_path = os.path.join(results_dir, "Final_Dataset", "dataset_ml_ready.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run the pipeline first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Select feature columns (physical descriptors)
    # Exclude non-feature or target metadata columns
    exclude_cols = [
        "Composition", "Stage", "Stage_Encoded", "Composition_Encoded"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Define features X and target y
    X = df[feature_cols]
    y = df["Stage"]
    
    print("\n--- Features Used for Sintering Stage Prediction ---")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:2d}. {col}")
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Initialize Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
    
    # Train
    clf.fit(X_train, y_train)
    
    # Cross Validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=clf.classes_, yticklabels=clf.classes_
    )
    plt.title("Sintering Stage Prediction Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Stage")
    plt.ylabel("True Stage")
    plt.tight_layout()
    
    plots_dir = os.path.join(results_dir, "Plots", "ML")
    os.makedirs(plots_dir, exist_ok=True)
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot: {cm_path}")
    
    # Save Feature Importance plot
    plt.figure(figsize=(10, 6))
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Top 15 features
    top_n = min(15, len(feature_cols))
    plt.bar(range(top_n), importances[indices[:top_n]], align="center", color="#4d7cff", edgecolor="black", alpha=0.8)
    plt.xticks(range(top_n), [feature_cols[i] for i in indices[:top_n]], rotation=45, ha="right")
    plt.title("Top 15 Physical Descriptors for Sintering State Identification", fontsize=14, fontweight="bold")
    plt.ylabel("RF Importance")
    plt.tight_layout()
    
    fi_path = os.path.join(plots_dir, "rf_feature_importances.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"Saved feature importances plot: {fi_path}")
    
    # Save the trained model
    models_dir = os.path.join(results_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "sintering_stage_rf.pkl")
    joblib.dump(clf, model_path)
    print(f"Saved model: {model_path}")
    
    print("\n" + "=" * 70)
    print("Machine Learning training complete!")
    print("=" * 70)

if __name__ == "__main__":
    train_sintering_predictor()
