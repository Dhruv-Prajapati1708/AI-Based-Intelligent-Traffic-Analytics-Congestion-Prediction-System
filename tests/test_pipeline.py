"""
Automated Test Suite for IntelliTraffic System using standard unittest.
Tests data generation, preprocessing feature shapes, model inference, and API endpoints.
"""

import os
import unittest
import pandas as pd
import numpy as np
try:
    from fastapi.testclient import TestClient
    HAS_TESTCLIENT = True
except Exception:
    HAS_TESTCLIENT = False
    TestClient = None

from data.generate_data import generate_synthetic_traffic_data
from utils.preprocessing import run_preprocessing_pipeline
from utils.model_utils import load_artifact
from config.settings import MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from api.main import app


class TestIntelliTrafficPipeline(unittest.TestCase):

    def test_01_data_generation(self):
        df = generate_synthetic_traffic_data(num_days=2, num_junctions=2, seed=42)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2 * 24 * 2)  # 96 rows
        expected_cols = [
            "timestamp", "junction_id", "vehicle_count", "average_speed_kmh",
            "weather_condition", "temperature_c", "is_holiday", "congestion_index", "congestion_level"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_02_preprocessing_pipeline(self):
        raw_path = os.path.join(RAW_DATA_DIR, "test_raw.csv")
        proc_path = os.path.join(PROCESSED_DATA_DIR, "test_proc.csv")
        
        df_raw = generate_synthetic_traffic_data(num_days=2, num_junctions=2, seed=42)
        df_raw.to_csv(raw_path, index=False)
        
        processed_df, transformers = run_preprocessing_pipeline(raw_path, proc_path)
        
        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertTrue(os.path.exists(proc_path))
        self.assertIn("hour_sin", processed_df.columns)
        self.assertIn("hour_cos", processed_df.columns)
        self.assertIn("vehicle_count_lag_1h", processed_df.columns)
        self.assertIn("encoder", transformers)
        self.assertIn("scaler", transformers)

        # Cleanup temporary test files
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(proc_path): os.remove(proc_path)

    def test_03_model_artifacts(self):
        clf_path = os.path.join(MODELS_DIR, "random_forest_classifier.joblib")
        reg_path = os.path.join(MODELS_DIR, "random_forest_regressor.joblib")
        feat_path = os.path.join(MODELS_DIR, "feature_names.joblib")
        
        if os.path.exists(clf_path):
            clf_model = load_artifact(clf_path)
            self.assertIsNotNone(clf_model)
            
        if os.path.exists(reg_path):
            reg_model = load_artifact(reg_path)
            self.assertIsNotNone(reg_model)

        if os.path.exists(feat_path):
            feature_names = load_artifact(feat_path)
            self.assertIsInstance(feature_names, list)
            self.assertGreater(len(feature_names), 0)

    def test_04_fastapi_endpoints(self):
        if not HAS_TESTCLIENT:
            self.skipTest("fastapi.testclient (httpx) not installed")
        client = TestClient(app)
        
        # 1. Root Endpoint
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "online")
        
        # 2. Health Endpoint
        resp_health = client.get("/health")
        self.assertEqual(resp_health.status_code, 200)
        self.assertEqual(resp_health.json()["status"], "healthy")
        
        # 3. Analytics Summary Endpoint
        resp_analytics = client.get("/analytics/summary")
        self.assertIn(resp_analytics.status_code, [200, 444])
        
        # 4. Predict Endpoint
        payload = {
            "junction_id": "Junction_1",
            "hour": 8,
            "day_of_week": "Monday",
            "weather_condition": "Clear",
            "temperature_c": 25.0,
            "vehicle_count": 300,
            "average_speed_kmh": 35.0,
            "model_name": "Random Forest"
        }
        resp_pred = client.post("/predict/congestion", json=payload)
        self.assertEqual(resp_pred.status_code, 200)
        data = resp_pred.json()
        self.assertIn("predicted_congestion_level", data)
        self.assertIn("predicted_congestion_index", data)
        self.assertEqual(data["junction_id"], "Junction_1")


if __name__ == "__main__":
    unittest.main()
