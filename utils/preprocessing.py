"""
Preprocessing & Feature Engineering utilities for IntelliTraffic.
Handles traffic data cleaning, temporal extraction, lag & rolling features, encoding, and scaling.
"""


import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw traffic CSV data and parses timestamp column.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clean_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans traffic data by sorting timestamps, dropping duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    # Drop exact duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Sort by junction and timestamp
    if "timestamp" in cleaned_df.columns and "junction_id" in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values(by=["junction_id", "timestamp"]).reset_index(drop=True)
        
    # Forward fill missing values for time series continuity, followed by backward fill
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].ffill().bfill()
    
    return cleaned_df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features and cyclical sine/cosine transformations.
    """
    df_feat = df.copy()
    if "timestamp" not in df_feat.columns:
        raise KeyError("'timestamp' column is required for temporal feature extraction.")
        
    dt = df_feat["timestamp"].dt
    df_feat["hour"] = dt.hour
    df_feat["day_of_week"] = dt.dayofweek
    df_feat["day_of_month"] = dt.day
    df_feat["month"] = dt.month
    df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)
    df_feat["is_rush_hour"] = (((df_feat["hour"] >= 7) & (df_feat["hour"] <= 10)) | 
                               ((df_feat["hour"] >= 17) & (df_feat["hour"] <= 20))).astype(int)
                               
    # Cyclical Sine/Cosine encodings
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24.0)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24.0)
    df_feat["day_sin"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7.0)
    df_feat["day_cos"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7.0)
    
    return df_feat


def create_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates lag and rolling statistics for vehicle count and average speed per junction.
    """
    df_lag = df.copy()
    
    if "junction_id" in df_lag.columns:
        # Group by junction to prevent data leakage across junctions
        grouped = df_lag.groupby("junction_id")
        
        # Lag features
        df_lag["vehicle_count_lag_1h"] = grouped["vehicle_count"].shift(1)
        df_lag["vehicle_count_lag_2h"] = grouped["vehicle_count"].shift(2)
        df_lag["avg_speed_lag_1h"] = grouped["average_speed_kmh"].shift(1)
        
        # Rolling averages & standard deviations (3-hour window)
        df_lag["rolling_avg_vehicles_3h"] = grouped["vehicle_count"].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        df_lag["rolling_std_vehicles_3h"] = grouped["vehicle_count"].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).std()
        ).fillna(0)
        
        df_lag["rolling_avg_speed_3h"] = grouped["average_speed_kmh"].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
    else:
        df_lag["vehicle_count_lag_1h"] = df_lag["vehicle_count"].shift(1)
        df_lag["vehicle_count_lag_2h"] = df_lag["vehicle_count"].shift(2)
        df_lag["avg_speed_lag_1h"] = df_lag["average_speed_kmh"].shift(1)
        df_lag["rolling_avg_vehicles_3h"] = df_lag["vehicle_count"].shift(1).rolling(window=3, min_periods=1).mean()
        df_lag["rolling_std_vehicles_3h"] = df_lag["vehicle_count"].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
        df_lag["rolling_avg_speed_3h"] = df_lag["average_speed_kmh"].shift(1).rolling(window=3, min_periods=1).mean()

    # Fill NaN values created by lag operations
    lag_cols = [c for c in df_lag.columns if "lag" in c or "rolling" in c]
    df_lag[lag_cols] = df_lag[lag_cols].bfill().fillna(0)
    
    return df_lag


def encode_and_scale_features(
    df: pd.DataFrame, 
    categorical_cols: list = None,
    numeric_cols: list = None,
    scale_numeric: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    One-hot encodes categorical columns and optionally standard-scales numerical features.
    """
    df_proc = df.copy()
    if categorical_cols is None:
        categorical_cols = ["weather_condition", "junction_id"]
    if numeric_cols is None:
        numeric_cols = [
            "vehicle_count", "average_speed_kmh", "temperature_c",
            "vehicle_count_lag_1h", "vehicle_count_lag_2h", "avg_speed_lag_1h",
            "rolling_avg_vehicles_3h", "rolling_std_vehicles_3h", "rolling_avg_speed_3h"
        ]

    # Filter present columns
    present_cat_cols = [c for c in categorical_cols if c in df_proc.columns]
    present_num_cols = [c for c in numeric_cols if c in df_proc.columns]

    transformers = {}

    # One-hot encoding
    if present_cat_cols:
        encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        encoded_array = encoder.fit_transform(df_proc[present_cat_cols])
        encoded_col_names = encoder.get_feature_names_out(present_cat_cols)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df_proc.index)
        
        df_proc = pd.concat([df_proc.drop(columns=present_cat_cols), encoded_df], axis=1)
        transformers["encoder"] = encoder

    # Standard scaling
    if present_num_cols:
        scaler = StandardScaler()
        if scale_numeric:
            df_proc[present_num_cols] = scaler.fit_transform(df_proc[present_num_cols])
        else:
            scaler.fit(df_proc[present_num_cols])
        transformers["scaler"] = scaler

    return df_proc, transformers


def run_preprocessing_pipeline(input_path: str, output_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full end-to-end preprocessing pipeline for raw traffic data.
    """
    print(f"Loading raw data from: {input_path}")
    raw_df = load_raw_data(input_path)
    
    print("Cleaning traffic data...")
    cleaned_df = clean_traffic_data(raw_df)
    
    print("Extracting temporal features...")
    temporal_df = extract_temporal_features(cleaned_df)
    
    print("Generating lag and rolling features...")
    featured_df = create_lag_and_rolling_features(temporal_df)
    
    print("Encoding categorical & scaling numeric features...")
    processed_df, transformers = encode_and_scale_features(featured_df)
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    processed_df.to_csv(output_path, index=False)
    print(f"Preprocessing completed. Processed dataset saved to: {output_path} ({len(processed_df)} records, {processed_df.shape[1]} features).")
    
    return processed_df, transformers