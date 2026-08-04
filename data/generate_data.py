"""
Synthetic Traffic Data Generator for IntelliTraffic project.
Generates spatio-temporal traffic sensor observations across city junctions.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_traffic_data(
    num_days: int = 30,
    num_junctions: int = 4,
    start_date: str = "2026-01-01 00:00:00",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic traffic data with realistic diurnal patterns, rush hours,
    weekend variations, weather impacts, and congestion metrics.
    """
    np.random.seed(seed)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    timestamps = [start_dt + timedelta(hours=i) for i in range(num_days * 24)]
    junction_ids = [f"Junction_{i+1}" for i in range(num_junctions)]
    weather_options = ["Clear", "Rain", "Fog", "Heavy Rain"]
    weather_probs = [0.70, 0.18, 0.08, 0.04]
    
    data_rows = []
    
    for dt in timestamps:
        hour = dt.hour
        day_of_week = dt.weekday()
        is_weekend = day_of_week >= 5
        is_rush_hour = (7 <= hour <= 10) or (17 <= hour <= 20)
        
        # Base diurnal volume profile
        base_volume = 120 + 200 * np.sin((hour - 4) * np.pi / 12) ** 2
        if is_rush_hour:
            base_volume *= 1.6
        if is_weekend:
            base_volume *= 0.75
            
        for j_idx, junction in enumerate(junction_ids):
            # Junction capacity variation multiplier
            j_multiplier = 0.85 + (j_idx * 0.15)
            
            # Random variation
            noise = np.random.normal(0, 15)
            vehicle_count = int(np.clip((base_volume * j_multiplier) + noise, 20, 600))
            
            # Weather random assignment per timestamp
            weather = np.random.choice(weather_options, p=weather_probs)
            weather_speed_penalty = 1.0
            if weather == "Rain":
                weather_speed_penalty = 0.85
            elif weather == "Fog":
                weather_speed_penalty = 0.75
            elif weather == "Heavy Rain":
                weather_speed_penalty = 0.60
                
            # Speed inverse to vehicle count
            free_flow_speed = 70.0 + (j_idx * 5)
            speed_drop = (vehicle_count / 600.0) * 45.0
            avg_speed = float(np.clip((free_flow_speed - speed_drop) * weather_speed_penalty + np.random.normal(0, 3), 10.0, 90.0))
            
            # Temperature estimation based on hour and season
            temp = float(np.clip(18.0 + 8.0 * np.sin((hour - 9) * np.pi / 12) + np.random.normal(0, 1.5), 5.0, 42.0))
            
            # Congestion index calculation (0 - 100)
            capacity = 450.0 * j_multiplier
            v_c_ratio = vehicle_count / capacity
            congestion_index = float(np.clip((v_c_ratio * 70.0) + ((70.0 - avg_speed) / 70.0 * 30.0), 0.0, 100.0))
            
            # Congestion Level categorization
            if congestion_index < 30.0:
                congestion_level = "Low"
            elif congestion_index < 55.0:
                congestion_level = "Moderate"
            elif congestion_index < 75.0:
                congestion_level = "High"
            else:
                congestion_level = "Severe"
                
            data_rows.append({
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "junction_id": junction,
                "vehicle_count": vehicle_count,
                "average_speed_kmh": round(avg_speed, 2),
                "weather_condition": weather,
                "temperature_c": round(temp, 1),
                "is_holiday": 1 if (day_of_week == 6 and hour % 4 == 0) else 0,
                "congestion_index": round(congestion_index, 2),
                "congestion_level": congestion_level
            })
            
    df = pd.DataFrame(data_rows)
    return df


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "traffic_data_raw.csv")
    
    print("Generating synthetic traffic dataset...")
    df_synthetic = generate_synthetic_traffic_data(num_days=30, num_junctions=4)
    df_synthetic.to_csv(output_filepath, index=False)
    print(f"Synthetic dataset saved successfully to '{output_filepath}' ({len(df_synthetic)} records).")
