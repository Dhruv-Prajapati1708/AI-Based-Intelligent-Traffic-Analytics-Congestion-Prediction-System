"""
Model utility functions for IntelliTraffic.
Handles model training (Decision Tree, Random Forest, XGBoost), evaluation,
cross-validation, and serialization using Joblib.
"""

import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier, XGBRegressor = None, None

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)


def evaluate_classifier(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates a classification model using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics


def evaluate_regressor(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluates a regression model using R2, MAE, MSE, and RMSE.
    """
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse))
    }
    return metrics


def train_classification_models(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Trains Decision Tree, Random Forest, and XGBoost Classifiers.
    Returns trained models and their evaluation performance metrics.
    """
    # 1. Decision Tree Classifier
    dt_clf = DecisionTreeClassifier(max_depth=10, random_state=random_state)
    dt_clf.fit(X_train, y_train)
    dt_metrics = evaluate_classifier(dt_clf, X_test, y_test)
    
    # 2. Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    rf_metrics = evaluate_classifier(rf_clf, X_test, y_test)
    
    results = {
        "Decision Tree": {"model": dt_clf, "metrics": dt_metrics},
        "Random Forest": {"model": rf_clf, "metrics": rf_metrics}
    }
    
    # 3. XGBoost Classifier
    if HAS_XGBOOST:
        xgb_clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state, n_jobs=-1)
        xgb_clf.fit(X_train, y_train)
        xgb_metrics = evaluate_classifier(xgb_clf, X_test, y_test)
        results["XGBoost"] = {"model": xgb_clf, "metrics": xgb_metrics}
        
    return results


def train_regression_models(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Trains Decision Tree, Random Forest, and XGBoost Regressors.
    Returns trained models and their evaluation performance metrics.
    """
    # 1. Decision Tree Regressor
    dt_reg = DecisionTreeRegressor(max_depth=10, random_state=random_state)
    dt_reg.fit(X_train, y_train)
    dt_metrics = evaluate_regressor(dt_reg, X_test, y_test)
    
    # 2. Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
    rf_reg.fit(X_train, y_train)
    rf_metrics = evaluate_regressor(rf_reg, X_test, y_test)
    
    results = {
        "Decision Tree": {"model": dt_reg, "metrics": dt_metrics},
        "Random Forest": {"model": rf_reg, "metrics": rf_metrics}
    }
    
    # 3. XGBoost Regressor
    if HAS_XGBOOST:
        xgb_reg = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state, n_jobs=-1)
        xgb_reg.fit(X_train, y_train)
        xgb_metrics = evaluate_regressor(xgb_reg, X_test, y_test)
        results["XGBoost"] = {"model": xgb_reg, "metrics": xgb_metrics}
        
    return results


def save_artifact(obj: Any, filepath: str) -> str:
    """
    Saves a model or transformer object to disk using Joblib.
    """
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(obj, filepath)
    return filepath


def load_artifact(filepath: str) -> Any:
    """
    Loads a model or transformer object from disk using Joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at: {filepath}")
    return joblib.load(filepath)
