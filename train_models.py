"""
Model Training & Evaluation Script for IntelliTraffic.
Trains Decision Tree, Random Forest, and XGBoost models for Congestion Level Classification
and Congestion Index Regression, then persists artifacts into models/ directory.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from data.generate_data import generate_synthetic_traffic_data
from utils.preprocessing import run_preprocessing_pipeline
from utils.model_utils import (
    train_classification_models,
    train_regression_models,
    save_artifact
)


def train_and_evaluate_all():
    print("==================================================================")
    print("IntelliTraffic ML Pipeline: Training & Evaluation")
    print("==================================================================")
    
    raw_csv = RAW_DATA_DIR / "traffic_data_raw.csv"
    processed_csv = PROCESSED_DATA_DIR / "traffic_data_processed.csv"
    
    # Step 1: Ensure dataset exists
    if not os.path.exists(raw_csv):
        print("[1/5] Generating synthetic raw traffic dataset...")
        raw_df = generate_synthetic_traffic_data(num_days=30, num_junctions=4)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        raw_df.to_csv(raw_csv, index=False)
    else:
        print(f"[1/5] Using raw traffic dataset at: {raw_csv}")
        
    # Step 2: Run Preprocessing & Feature Engineering
    print("[2/5] Running preprocessing & feature extraction...")
    processed_df, transformers = run_preprocessing_pipeline(str(raw_csv), str(processed_csv))
    
    # Step 3: Prepare Features & Targets
    print("[3/5] Preparing feature matrix (X) and targets (y)...")
    target_class_col = "congestion_level"
    target_reg_col = "congestion_index"
    
    # Label encode target classification
    label_encoder = LabelEncoder()
    # Order levels logically: Low=0, Moderate=1, High=2, Severe=3 if present
    level_order = ["Low", "Moderate", "High", "Severe"]
    label_encoder.fit(level_order)
    
    y_class = label_encoder.transform(processed_df[target_class_col])
    y_reg = processed_df[target_reg_col].values
    
    # Exclude non-feature columns
    exclude_cols = ["timestamp", target_class_col, target_reg_col]
    feature_cols = [c for c in processed_df.columns if c not in exclude_cols]
    
    X = processed_df[feature_cols].values
    
    # Train / Test Split
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    print(f"-> Train Samples: {len(X_train)} | Test Samples: {len(X_test)} | Features: {len(feature_cols)}")
    
    # Step 4: Train & Evaluate Classification Models
    print("\n[4/5] Training Classification Models (Low / Moderate / High / Severe)...")
    class_results = train_classification_models(X_train, y_train_class, X_test, y_test_class)
    
    print("\n--- Classification Performance Comparison ---")
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 70)
    for model_name, res in class_results.items():
        m = res["metrics"]
        print(f"{model_name:<20} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f}")
        
    # Step 5: Train & Evaluate Regression Models
    print("\n[5/5] Training Regression Models (Congestion Index 0 - 100)...")
    reg_results = train_regression_models(X_train, y_train_reg, X_test, y_test_reg)
    
    print("\n--- Regression Performance Comparison ---")
    print(f"{'Model':<20} | {'R2 Score':<10} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 55)
    for model_name, res in reg_results.items():
        m = res["metrics"]
        print(f"{model_name:<20} | {m['r2']:<10.4f} | {m['mae']:<10.4f} | {m['rmse']:<10.4f}")
        
    # Save artifacts into models/ directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save models
    if "Random Forest" in class_results:
        save_artifact(class_results["Random Forest"]["model"], str(MODELS_DIR / "random_forest_classifier.joblib"))
    if "XGBoost" in class_results:
        save_artifact(class_results["XGBoost"]["model"], str(MODELS_DIR / "xgboost_classifier.joblib"))
    if "Decision Tree" in class_results:
        save_artifact(class_results["Decision Tree"]["model"], str(MODELS_DIR / "decision_tree_classifier.joblib"))
        
    if "Random Forest" in reg_results:
        save_artifact(reg_results["Random Forest"]["model"], str(MODELS_DIR / "random_forest_regressor.joblib"))
    if "XGBoost" in reg_results:
        save_artifact(reg_results["XGBoost"]["model"], str(MODELS_DIR / "xgboost_regressor.joblib"))
    if "Decision Tree" in reg_results:
        save_artifact(reg_results["Decision Tree"]["model"], str(MODELS_DIR / "decision_tree_regressor.joblib"))
    
    # Save transformers & metadata
    save_artifact(label_encoder, str(MODELS_DIR / "label_encoder.joblib"))
    save_artifact(feature_cols, str(MODELS_DIR / "feature_names.joblib"))
    if "scaler" in transformers:
        save_artifact(transformers["scaler"], str(MODELS_DIR / "scaler.joblib"))
    if "encoder" in transformers:
        save_artifact(transformers["encoder"], str(MODELS_DIR / "encoder.joblib"))
        
    # Save metrics JSON summary
    summary_metrics = {
        "classification": {k: v["metrics"] for k, v in class_results.items()},
        "regression": {k: v["metrics"] for k, v in reg_results.items()},
        "feature_names": feature_cols,
        "classes": list(label_encoder.classes_)
    }
    metrics_path = MODELS_DIR / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)
        
    print(f"\n-> All models, transformers, and metrics saved to: {MODELS_DIR}")
    print("ML Pipeline execution completed successfully!")


if __name__ == "__main__":
    train_and_evaluate_all()
