"""
IntelliTraffic Streamlit Dashboard
AI-Based Intelligent Traffic Analytics & Congestion Prediction System
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Add parent directory to sys.path to allow absolute imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from utils.visualization import (
    plot_peak_hours,
    plot_daily_trends,
    plot_junction_comparison,
    plot_weather_impact,
    plot_model_comparison,
    generate_folium_map
)

# Page Setup
st.set_page_config(
    page_title="IntelliTraffic Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #0F172A;
        font-weight: 700;
    }
    .badge-green { background-color: #DEF7EC; color: #03543F; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-yellow { background-color: #FEF08A; color: #713F12; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-red { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_datasets():
    raw_path = RAW_DATA_DIR / "traffic_data_raw.csv"
    processed_path = PROCESSED_DATA_DIR / "traffic_data_processed.csv"
    
    raw_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    processed_df = pd.read_csv(processed_path) if os.path.exists(processed_path) else None
    return raw_df, processed_df


@st.cache_resource
def load_models_and_artifacts():
    artifacts = {}
    model_files = {
        "rf_clf": MODELS_DIR / "random_forest_classifier.joblib",
        "xgb_clf": MODELS_DIR / "xgboost_classifier.joblib",
        "dt_clf": MODELS_DIR / "decision_tree_classifier.joblib",
        "rf_reg": MODELS_DIR / "random_forest_regressor.joblib",
        "xgb_reg": MODELS_DIR / "xgboost_regressor.joblib",
        "dt_reg": MODELS_DIR / "decision_tree_regressor.joblib",
        "label_encoder": MODELS_DIR / "label_encoder.joblib",
        "scaler": MODELS_DIR / "scaler.joblib",
        "encoder": MODELS_DIR / "encoder.joblib",
        "feature_names": MODELS_DIR / "feature_names.joblib"
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            artifacts[name] = joblib.load(path)
            
    metrics_path = MODELS_DIR / "metrics_summary.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            artifacts["metrics"] = json.load(f)
            
    return artifacts


def main():
    raw_df, processed_df = load_datasets()
    artifacts = load_models_and_artifacts()
    
    if raw_df is None:
        st.error("Raw traffic data not found! Please run `python main.py` or `python train_models.py` first.")
        return

    # Header
    st.markdown('<div class="main-header">🚦 IntelliTraffic Analytics & Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Real-Time Traffic Analytics, Congestion Prediction & GIS Mapping</div>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/traffic-jam.png", width=70)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Dashboard Module:",
        [
            "📊 Executive Overview & EDA",
            "🗺️ Interactive GIS Traffic Map",
            "🔮 AI Congestion Predictor",
            "📈 ML Model Performance"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Filters")
    junction_filter = st.sidebar.multiselect(
        "Select Junctions:",
        options=sorted(raw_df["junction_id"].unique()),
        default=sorted(raw_df["junction_id"].unique())
    )
    weather_filter = st.sidebar.multiselect(
        "Select Weather:",
        options=sorted(raw_df["weather_condition"].unique()),
        default=sorted(raw_df["weather_condition"].unique())
    )
    
    # Filter dataset
    filtered_df = raw_df[
        (raw_df["junction_id"].isin(junction_filter)) &
        (raw_df["weather_condition"].isin(weather_filter))
    ]

    # PAGE 1: EDA & Overview
    if page == "📊 Executive Overview & EDA":
        st.header("Executive Traffic Analytics & EDA")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        total_records = len(filtered_df)
        avg_vehicles = int(filtered_df["vehicle_count"].mean())
        avg_speed = filtered_df["average_speed_kmh"].mean()
        avg_congestion = filtered_df["congestion_index"].mean()
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Records</div>
                <div class="metric-value">{total_records:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Vehicle Volume</div>
                <div class="metric-value">{avg_vehicles} <span style="font-size: 1rem; font-weight: normal;">cars/hr</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Speed</div>
                <div class="metric-value">{avg_speed:.1f} <span style="font-size: 1rem; font-weight: normal;">km/h</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Congestion Index</div>
                <div class="metric-value">{avg_congestion:.1f} / 100</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.plotly_chart(plot_peak_hours(filtered_df), use_container_width=True)
        with row1_col2:
            st.plotly_chart(plot_daily_trends(filtered_df), use_container_width=True)
            
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.plotly_chart(plot_junction_comparison(filtered_df), use_container_width=True)
        with row2_col2:
            st.plotly_chart(plot_weather_impact(filtered_df), use_container_width=True)

    # PAGE 2: GIS Map
    elif page == "🗺️ Interactive GIS Traffic Map":
        st.header("Interactive GIS Traffic Map")
        st.write("Real-time junction traffic density and road segment congestion status.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            folium_map = generate_folium_map(filtered_df)
            components.html(folium_map._repr_html_(), height=550)
            
        with col2:
            st.subheader("Legend")
            st.markdown("""
            - 🟢 **Low Congestion** (Index < 30.0)
            - 🟡 **Moderate Congestion** (Index 30 - 55)
            - 🔴 **High / Severe Congestion** (Index > 55.0)
            """)
            st.markdown("---")
            st.subheader("Latest Junction Summary")
            latest_status = filtered_df.sort_values(by="timestamp").groupby("junction_id").last().reset_index()
            for _, r in latest_status.iterrows():
                lvl = r["congestion_level"]
                badge_class = "badge-green" if lvl == "Low" else ("badge-yellow" if lvl == "Moderate" else "badge-red")
                st.markdown(f"**{r['junction_id']}**: <span class='{badge_class}'>{lvl} ({r['congestion_index']:.1f})</span>", unsafe_allow_html=True)

    # PAGE 3: AI Predictor
    elif page == "🔮 AI Congestion Predictor":
        st.header("Real-Time AI Congestion Predictor")
        st.write("Input current traffic & environmental parameters to predict Congestion Level and Index.")
        
        if "xgb_clf" not in artifacts or "scaler" not in artifacts:
            st.warning("ML Models or Scalers not loaded! Please run `python train_models.py` first.")
            return

        col_form, col_pred = st.columns([1, 1])
        
        with col_form:
            st.subheader("Input Traffic Features")
            selected_algorithm = st.selectbox(
                "Select Machine Learning Model:",
                ["XGBoost (Recommended)", "Random Forest", "Decision Tree"]
            )
            
            selected_junction = st.selectbox("Select Junction:", ["Junction_1", "Junction_2", "Junction_3", "Junction_4"])
            selected_hour = st.slider("Hour of Day (0-23):", 0, 23, 8)
            selected_day = st.selectbox("Day of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            selected_weather = st.selectbox("Weather Condition:", ["Clear", "Rain", "Fog", "Heavy Rain"])
            selected_temp = st.slider("Temperature (°C):", 5.0, 45.0, 25.0)
            vehicle_count_input = st.number_input("Current Vehicle Count (cars/hr):", min_value=10, max_value=800, value=250)
            avg_speed_input = st.number_input("Current Avg Speed (km/h):", min_value=5.0, max_value=100.0, value=45.0)
            
            predict_btn = st.button("🚀 Predict Traffic Congestion", type="primary", use_container_width=True)

        with col_pred:
            st.subheader("Prediction Results")
            if predict_btn:
                # Map inputs to features
                day_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(selected_day)
                is_weekend = 1 if day_idx >= 5 else 0
                is_rush_hour = 1 if ((7 <= selected_hour <= 10) or (17 <= selected_hour <= 20)) else 0
                
                hour_sin = np.sin(2 * np.pi * selected_hour / 24.0)
                hour_cos = np.cos(2 * np.pi * selected_hour / 24.0)
                day_sin = np.sin(2 * np.pi * day_idx / 7.0)
                day_cos = np.cos(2 * np.pi * day_idx / 7.0)
                
                # Construct feature dict matching training columns
                feature_names = artifacts["feature_names"]
                input_data = pd.DataFrame(0.0, index=[0], columns=feature_names)
                
                # Assign numeric & cyclical features
                if "hour" in feature_names: input_data["hour"] = selected_hour
                if "day_of_week" in feature_names: input_data["day_of_week"] = day_idx
                if "is_weekend" in feature_names: input_data["is_weekend"] = is_weekend
                if "is_rush_hour" in feature_names: input_data["is_rush_hour"] = is_rush_hour
                if "hour_sin" in feature_names: input_data["hour_sin"] = hour_sin
                if "hour_cos" in feature_names: input_data["hour_cos"] = hour_cos
                if "day_sin" in feature_names: input_data["day_sin"] = day_sin
                if "day_cos" in feature_names: input_data["day_cos"] = day_cos
                
                # Set numeric features (scaled using saved scaler if needed)
                if "vehicle_count" in feature_names: input_data["vehicle_count"] = vehicle_count_input
                if "average_speed_kmh" in feature_names: input_data["average_speed_kmh"] = avg_speed_input
                if "temperature_c" in feature_names: input_data["temperature_c"] = selected_temp
                if "vehicle_count_lag_1h" in feature_names: input_data["vehicle_count_lag_1h"] = vehicle_count_input
                if "rolling_avg_vehicles_3h" in feature_names: input_data["rolling_avg_vehicles_3h"] = vehicle_count_input
                if "rolling_avg_speed_3h" in feature_names: input_data["rolling_avg_speed_3h"] = avg_speed_input
                
                # One-hot features
                weather_col = f"weather_condition_{selected_weather}"
                junction_col = f"junction_id_{selected_junction}"
                if weather_col in feature_names: input_data[weather_col] = 1.0
                if junction_col in feature_names: input_data[junction_col] = 1.0

                # Select model
                if "XGBoost" in selected_algorithm:
                    clf_model = artifacts.get("xgb_clf")
                    reg_model = artifacts.get("xgb_reg")
                elif "Random Forest" in selected_algorithm:
                    clf_model = artifacts.get("rf_clf")
                    reg_model = artifacts.get("rf_reg")
                else:
                    clf_model = artifacts.get("dt_clf")
                    reg_model = artifacts.get("dt_reg")
                    
                # Run inference
                pred_class_idx = clf_model.predict(input_data.values)[0]
                pred_label = artifacts["label_encoder"].inverse_transform([pred_class_idx])[0]
                pred_index = float(reg_model.predict(input_data.values)[0]) if reg_model else 50.0
                
                # Display Prediction Output
                badge_class = "badge-green" if pred_label == "Low" else ("badge-yellow" if pred_label == "Moderate" else "badge-red")
                
                st.markdown(f"""
                <div style="background:#F1F5F9; border-radius:12px; padding:20px; text-align:center; border:1px solid #CBD5E1;">
                    <h4 style="margin:0; color:#475569;">Predicted Congestion Level</h4>
                    <h1 style="margin:10px 0;"><span class="{badge_class}">{pred_label}</span></h1>
                    <h3>Estimated Congestion Index: <b>{pred_index:.1f} / 100</b></h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("💡 Dynamic Traffic Recommendation")
                if pred_label == "Low":
                    st.success("🟢 Traffic flow is optimal. No action or re-routing required.")
                elif pred_label == "Moderate":
                    st.info("🟡 Moderate traffic density. Monitor junction signals & adjust signal timing.")
                elif pred_label in ["High", "Severe"]:
                    st.error("🔴 Heavy congestion detected! Recommend activating dynamic diversion routes and extending green light duration.")
            else:
                st.info("Click '🚀 Predict Traffic Congestion' to view AI predictions.")

    # PAGE 4: Model Metrics
    elif page == "📈 ML Model Performance":
        st.header("Machine Learning Model Evaluation & Metrics")
        
        metrics = artifacts.get("metrics")
        if metrics:
            st.plotly_chart(plot_model_comparison(metrics), use_container_width=True)
            
            st.subheader("Classification Models Breakdown")
            clf_df = pd.DataFrame(metrics["classification"]).T
            st.dataframe(clf_df.style.format("{:.4f}").highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)
            
            st.subheader("Regression Models Breakdown")
            reg_df = pd.DataFrame(metrics["regression"]).T
            st.dataframe(reg_df.style.format("{:.4f}").highlight_min(subset=["mae", "rmse"], color="#D1FAE5"), use_container_width=True)
        else:
            st.warning("Metrics summary JSON not found. Please run `python train_models.py`.")


if __name__ == "__main__":
    main()
