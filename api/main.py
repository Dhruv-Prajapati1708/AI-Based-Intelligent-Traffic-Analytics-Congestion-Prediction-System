"""
FastAPI Microservice for IntelliTraffic
Exposes RESTful API endpoints for traffic congestion predictions and analytical queries.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query

from config.settings import MODELS_DIR, RAW_DATA_DIR

app = FastAPI(
    title="IntelliTraffic API",
    description="AI-Based Intelligent Traffic Analytics & Congestion Prediction REST API",
    version="1.0.0"
)

# Global artifacts storage
artifacts: Dict[str, Any] = {}


def load_api_artifacts():
    global artifacts
    model_files = {
        "rf_clf": MODELS_DIR / "random_forest_classifier.joblib",
        "xgb_clf": MODELS_DIR / "xgboost_classifier.joblib",
        "dt_clf": MODELS_DIR / "decision_tree_classifier.joblib",
        "rf_reg": MODELS_DIR / "random_forest_regressor.joblib",
        "xgb_reg": MODELS_DIR / "xgboost_regressor.joblib",
        "dt_reg": MODELS_DIR / "decision_tree_regressor.joblib",
        "label_encoder": MODELS_DIR / "label_encoder.joblib",
        "feature_names": MODELS_DIR / "feature_names.joblib"
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            artifacts[name] = joblib.load(path)
            
    metrics_path = MODELS_DIR / "metrics_summary.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            artifacts["metrics"] = json.load(f)


@app.on_event("startup")
def startup_event():
    load_api_artifacts()


class PredictionRequest(BaseModel):
    junction_id: str = Field("Junction_1", description="Junction Identifier e.g. Junction_1")
    hour: int = Field(8, ge=0, le=23, description="Hour of Day (0-23)")
    day_of_week: str = Field("Monday", description="Day of Week")
    weather_condition: str = Field("Clear", description="Weather e.g. Clear, Rain, Fog, Heavy Rain")
    temperature_c: float = Field(25.0, description="Temperature in Celsius")
    vehicle_count: int = Field(250, ge=0, description="Vehicle count per hour")
    average_speed_kmh: float = Field(45.0, ge=0.0, description="Average speed in km/h")
    model_name: Optional[str] = Field("XGBoost", description="Model architecture: XGBoost, Random Forest, or Decision Tree")


class PredictionResponse(BaseModel):
    junction_id: str
    predicted_congestion_level: str
    predicted_congestion_index: float
    model_used: str
    recommendation: str


@app.get("/")
def read_root():
    return {
        "title": "IntelliTraffic API",
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    loaded_models = [k for k in artifacts.keys() if k != "metrics"]
    return {
        "status": "healthy",
        "loaded_models_count": len(loaded_models),
        "available_models": loaded_models
    }


@app.get("/analytics/summary")
def get_analytics_summary():
    raw_csv = RAW_DATA_DIR / "traffic_data_raw.csv"
    if not os.path.exists(raw_csv):
        raise HTTPException(status_code=444, detail="Traffic raw data not found")
        
    df = pd.read_csv(raw_csv)
    summary = {
        "total_records": len(df),
        "avg_vehicle_count": float(df["vehicle_count"].mean()),
        "avg_speed_kmh": float(df["average_speed_kmh"].mean()),
        "avg_congestion_index": float(df["congestion_index"].mean()),
        "junctions": list(df["junction_id"].unique()),
        "weather_conditions": list(df["weather_condition"].unique())
    }
    return summary


@app.post("/predict/congestion", response_model=PredictionResponse)
def predict_congestion(req: PredictionRequest):
    if "feature_names" not in artifacts or "label_encoder" not in artifacts:
        load_api_artifacts()
        
    if "feature_names" not in artifacts:
        raise HTTPException(status_code=500, detail="ML Model artifacts not trained yet. Please run train_models.py.")

    # Convert inputs to model feature vector
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_idx = days.index(req.day_of_week) if req.day_of_week in days else 0
    is_weekend = 1 if day_idx >= 5 else 0
    is_rush_hour = 1 if ((7 <= req.hour <= 10) or (17 <= req.hour <= 20)) else 0
    
    hour_sin = np.sin(2 * np.pi * req.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * req.hour / 24.0)
    day_sin = np.sin(2 * np.pi * day_idx / 7.0)
    day_cos = np.cos(2 * np.pi * day_idx / 7.0)
    
    feature_names = artifacts["feature_names"]
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    if "hour" in feature_names: input_df["hour"] = req.hour
    if "day_of_week" in feature_names: input_df["day_of_week"] = day_idx
    if "is_weekend" in feature_names: input_df["is_weekend"] = is_weekend
    if "is_rush_hour" in feature_names: input_df["is_rush_hour"] = is_rush_hour
    if "hour_sin" in feature_names: input_df["hour_sin"] = hour_sin
    if "hour_cos" in feature_names: input_df["hour_cos"] = hour_cos
    if "day_sin" in feature_names: input_df["day_sin"] = day_sin
    if "day_cos" in feature_names: input_df["day_cos"] = day_cos
    
    if "vehicle_count" in feature_names: input_df["vehicle_count"] = req.vehicle_count
    if "average_speed_kmh" in feature_names: input_df["average_speed_kmh"] = req.average_speed_kmh
    if "temperature_c" in feature_names: input_df["temperature_c"] = req.temperature_c
    if "vehicle_count_lag_1h" in feature_names: input_df["vehicle_count_lag_1h"] = req.vehicle_count
    if "rolling_avg_vehicles_3h" in feature_names: input_df["rolling_avg_vehicles_3h"] = req.vehicle_count
    if "rolling_avg_speed_3h" in feature_names: input_df["rolling_avg_speed_3h"] = req.average_speed_kmh
    
    weather_col = f"weather_condition_{req.weather_condition}"
    junction_col = f"junction_id_{req.junction_id}"
    if weather_col in feature_names: input_df[weather_col] = 1.0
    if junction_col in feature_names: input_df[junction_col] = 1.0

    # Select model
    if "XGBoost" in req.model_name and "xgb_clf" in artifacts:
        clf_model = artifacts["xgb_clf"]
        reg_model = artifacts.get("xgb_reg")
        selected_name = "XGBoost"
    elif "Random Forest" in req.model_name and "rf_clf" in artifacts:
        clf_model = artifacts["rf_clf"]
        reg_model = artifacts.get("rf_reg")
        selected_name = "Random Forest"
    else:
        clf_model = artifacts.get("dt_clf") or list(artifacts.values())[0]
        reg_model = artifacts.get("dt_reg")
        selected_name = "Decision Tree"

    pred_class_idx = clf_model.predict(input_df.values)[0]
    pred_label = artifacts["label_encoder"].inverse_transform([pred_class_idx])[0]
    pred_index = float(reg_model.predict(input_df.values)[0]) if reg_model else 50.0

    # Recommendations
    if pred_label == "Low":
        rec = "Traffic flow is optimal. No action required."
    elif pred_label == "Moderate":
        rec = "Moderate density. Monitor signals and adjust timing if required."
    else:
        rec = "High congestion detected! Recommend activating dynamic diversion routes and extending green cycle."

    return PredictionResponse(
        junction_id=req.junction_id,
        predicted_congestion_level=pred_label,
        predicted_congestion_index=round(pred_index, 1),
        model_used=selected_name,
        recommendation=rec
    )
