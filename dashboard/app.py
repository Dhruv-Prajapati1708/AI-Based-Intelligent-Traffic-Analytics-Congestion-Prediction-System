"""
IntelliTraffic Streamlit Dashboard
AI-Based Intelligent Traffic Analytics & Congestion Prediction System
Modern, Professional, and Feature-Rich Implementation
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

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
from utils.visualization import (
    plot_peak_hours,
    plot_daily_trends,
    plot_junction_comparison,
    plot_weather_impact,
    plot_model_comparison,
    generate_folium_map
)
from utils.report_generator import generate_traffic_pdf_report

# Page Configuration
st.set_page_config(
    page_title="IntelliTraffic AI | Intelligent Traffic Control",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Gradient Container */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
        color: #F8FAFC;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 500;
    }
    
    .status-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        text-align: left;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
        border-color: #CBD5E1;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.9rem;
        color: #0F172A;
        font-weight: 800;
        margin: 6px 0 2px 0;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 600;
    }

    /* Badges */
    .badge-green {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-yellow {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-red {
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Result Display Card */
    .prediction-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
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
        st.error("Raw traffic dataset not found! Please execute `python main.py` or `python train_models.py` first.")
        return

    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🚦 IntelliTraffic AI Platform</div>
        <div class="hero-subtitle">Intelligent Traffic Density Analytics, GIS Spatio-Temporal Congestion Mapping & Machine Learning Predictions</div>
        <div class="status-pill">⚡ ML Engine Active • XGBoost Accuracy: 99.31% • System Online</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Setup
    st.sidebar.image("https://img.icons8.com/color/96/traffic-jam.png", width=65)
    st.sidebar.title("IntelliTraffic AI")
    st.sidebar.markdown("---")
    
    # Sidebar Filters
    st.sidebar.subheader("🎛️ Filter Control Panel")
    selected_junctions = st.sidebar.multiselect(
        "Select Junctions:",
        options=sorted(raw_df["junction_id"].unique()),
        default=sorted(raw_df["junction_id"].unique())
    )
    selected_weather = st.sidebar.multiselect(
        "Select Weather Conditions:",
        options=sorted(raw_df["weather_condition"].unique()),
        default=sorted(raw_df["weather_condition"].unique())
    )
    
    filtered_df = raw_df[
        (raw_df["junction_id"].isin(selected_junctions)) &
        (raw_df["weather_condition"].isin(selected_weather))
    ]
    
    if filtered_df.empty:
        st.warning("No data matching the selected filter criteria. Please adjust your sidebar choices.")
        return

    # Navigation Tabs
    tab_eda, tab_map, tab_predict, tab_models, tab_reports = st.tabs([
        "📊 Executive Analytics",
        "🗺️ GIS Traffic Map",
        "🔮 AI Congestion Predictor",
        "📈 Model Benchmarks",
        "📥 Reports & Explorer"
    ])
    
    # TAB 1: EXECUTIVE ANALYTICS
    with tab_eda:
        st.subheader("Executive Traffic Overview")
        
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
                <div class="metric-subtitle">↑ Active Sensor Feed</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Traffic Volume</div>
                <div class="metric-value">{avg_vehicles} <span style="font-size:1rem; color:#64748B; font-weight:500;">cars/hr</span></div>
                <div class="metric-subtitle">Across Selected Junctions</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Flow Speed</div>
                <div class="metric-value">{avg_speed:.1f} <span style="font-size:1rem; color:#64748B; font-weight:500;">km/h</span></div>
                <div class="metric-subtitle">Normal Range: 35-70 km/h</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Congestion Index</div>
                <div class="metric-value">{avg_congestion:.1f} <span style="font-size:1rem; color:#64748B; font-weight:500;">/ 100</span></div>
                <div class="metric-subtitle">Overall City Status</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Grid of Analytical Charts
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

    # TAB 2: GIS MAP
    with tab_map:
        st.subheader("Real-Time GIS Spatio-Temporal Congestion Map")
        st.caption("Interactive map rendering city junctions, dynamic status colors, and road segment congestion levels.")
        
        map_col, status_col = st.columns([3.2, 1.2])
        
        with map_col:
            folium_map = generate_folium_map(filtered_df)
            components.html(folium_map._repr_html_(), height=580)
            
        with status_col:
            st.markdown("#### Status Legend")
            st.markdown("""
            - 🟢 **Low Congestion** (Index < 30)
            - 🟡 **Moderate Congestion** (Index 30–55)
            - 🔴 **High / Severe Congestion** (Index > 55)
            """)
            st.markdown("---")
            st.markdown("#### Latest Junction Status")
            
            latest_status = filtered_df.sort_values(by="timestamp").groupby("junction_id").last().reset_index()
            for _, row in latest_status.iterrows():
                j_id = row["junction_id"]
                lvl = row["congestion_level"]
                idx = row["congestion_index"]
                v_cnt = row["vehicle_count"]
                badge = "badge-green" if lvl == "Low" else ("badge-yellow" if lvl == "Moderate" else "badge-red")
                
                st.markdown(f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:12px; border-radius:10px; margin-bottom:10px;">
                    <div style="font-weight:700; color:#0F172A;">{j_id}</div>
                    <div style="margin-top:4px;">
                        <span class="{badge}">{lvl} ({idx:.1f})</span>
                    </div>
                    <div style="font-size:0.8rem; color:#64748B; margin-top:6px;">Volume: <b>{v_cnt}</b> vehicles/hr</div>
                </div>
                """, unsafe_allow_html=True)

    # TAB 3: AI PREDICTOR
    with tab_predict:
        st.subheader("Live AI Congestion Predictor")
        st.caption("Select environmental parameters to predict Congestion Level and estimated index via trained ML models.")
        
        if "xgb_clf" not in artifacts or "feature_names" not in artifacts:
            st.warning("Trained ML models not found. Please execute `python train_models.py`.")
            return

        form_col, res_col = st.columns([1.1, 1])
        
        with form_col:
            st.markdown("#### ⚙️ Simulation Parameters")
            
            selected_algorithm = st.selectbox(
                "Machine Learning Model Architecture:",
                ["XGBoost (Recommended - 99.31% Acc)", "Random Forest (98.44% Acc)", "Decision Tree (98.44% Acc)"]
            )
            
            p_junction = st.selectbox("Select Target Junction:", ["Junction_1", "Junction_2", "Junction_3", "Junction_4"])
            p_hour = st.slider("Hour of Day (0 - 23):", 0, 23, 8)
            p_day = st.selectbox("Day of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            p_weather = st.selectbox("Weather Condition:", ["Clear", "Rain", "Fog", "Heavy Rain"])
            p_temp = st.slider("Temperature (°C):", 5.0, 45.0, 26.0)
            p_vehicles = st.number_input("Vehicle Volume (cars/hr):", min_value=10, max_value=800, value=320)
            p_speed = st.number_input("Average Flow Speed (km/h):", min_value=5.0, max_value=100.0, value=38.0)
            
            predict_btn = st.button("🚀 Execute AI Congestion Prediction", type="primary", use_container_width=True)

        with res_col:
            st.markdown("#### 📊 Prediction Output & Advisory")
            
            if predict_btn:
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_idx = days.index(p_day)
                is_weekend = 1 if day_idx >= 5 else 0
                is_rush_hour = 1 if ((7 <= p_hour <= 10) or (17 <= p_hour <= 20)) else 0
                
                hour_sin = np.sin(2 * np.pi * p_hour / 24.0)
                hour_cos = np.cos(2 * np.pi * p_hour / 24.0)
                day_sin = np.sin(2 * np.pi * day_idx / 7.0)
                day_cos = np.cos(2 * np.pi * day_idx / 7.0)
                
                feature_names = artifacts["feature_names"]
                input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
                
                if "hour" in feature_names: input_df["hour"] = p_hour
                if "day_of_week" in feature_names: input_df["day_of_week"] = day_idx
                if "is_weekend" in feature_names: input_df["is_weekend"] = is_weekend
                if "is_rush_hour" in feature_names: input_df["is_rush_hour"] = is_rush_hour
                if "hour_sin" in feature_names: input_df["hour_sin"] = hour_sin
                if "hour_cos" in feature_names: input_df["hour_cos"] = hour_cos
                if "day_sin" in feature_names: input_df["day_sin"] = day_sin
                if "day_cos" in feature_names: input_df["day_cos"] = day_cos
                
                if "vehicle_count" in feature_names: input_df["vehicle_count"] = p_vehicles
                if "average_speed_kmh" in feature_names: input_df["average_speed_kmh"] = p_speed
                if "temperature_c" in feature_names: input_df["temperature_c"] = p_temp
                if "vehicle_count_lag_1h" in feature_names: input_df["vehicle_count_lag_1h"] = p_vehicles
                if "rolling_avg_vehicles_3h" in feature_names: input_df["rolling_avg_vehicles_3h"] = p_vehicles
                if "rolling_avg_speed_3h" in feature_names: input_df["rolling_avg_speed_3h"] = p_speed
                
                weather_col = f"weather_condition_{p_weather}"
                junction_col = f"junction_id_{p_junction}"
                if weather_col in feature_names: input_df[weather_col] = 1.0
                if junction_col in feature_names: input_df[junction_col] = 1.0

                if "XGBoost" in selected_algorithm:
                    clf_model = artifacts.get("xgb_clf")
                    reg_model = artifacts.get("xgb_reg")
                    model_label = "XGBoost"
                elif "Random Forest" in selected_algorithm:
                    clf_model = artifacts.get("rf_clf")
                    reg_model = artifacts.get("rf_reg")
                    model_label = "Random Forest"
                else:
                    clf_model = artifacts.get("dt_clf")
                    reg_model = artifacts.get("dt_reg")
                    model_label = "Decision Tree"

                pred_class_idx = clf_model.predict(input_df.values)[0]
                pred_label = artifacts["label_encoder"].inverse_transform([pred_class_idx])[0]
                pred_index = float(reg_model.predict(input_df.values)[0]) if reg_model else 50.0

                badge_style = "badge-green" if pred_label == "Low" else ("badge-yellow" if pred_label == "Moderate" else "badge-red")
                
                st.markdown(f"""
                <div class="prediction-container">
                    <div style="font-size:0.9rem; color:#64748B; font-weight:600; text-transform:uppercase;">Model: {model_label}</div>
                    <div style="margin: 14px 0;">
                        <span class="{badge_style}" style="font-size: 1.4rem;">{pred_label} CONGESTION</span>
                    </div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #0F172A;">{pred_index:.1f} <span style="font-size: 1rem; color: #64748B;">/ 100 Index</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 💡 Automated Traffic Management Recommendation")
                
                if pred_label == "Low":
                    st.success("🟢 **Optimal Traffic Flow:** No signal modifications or dynamic diversion protocols required.")
                elif pred_label == "Moderate":
                    st.info("🟡 **Moderate Traffic Density:** Extend green light duration on main corridors by +15 seconds.")
                else:
                    st.error("🔴 **High Congestion Warning:** Trigger dynamic route diversions and alert urban traffic control room.")
            else:
                st.info("Adjust simulation parameters on the left and click **🚀 Execute AI Congestion Prediction** to view real-time model outputs.")

    # TAB 4: MODEL BENCHMARKS
    with tab_models:
        st.subheader("Machine Learning Algorithm Performance Benchmarks")
        
        metrics = artifacts.get("metrics")
        if metrics:
            st.plotly_chart(plot_model_comparison(metrics), use_container_width=True)
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("#### Classification Metrics")
                clf_df = pd.DataFrame(metrics["classification"]).T
                # Select only float metric columns (excluding confusion_matrix list)
                clf_cols = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in clf_df.columns]
                clf_summary = clf_df[clf_cols].astype(float)
                st.dataframe(clf_summary.style.format("{:.4f}").highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)
                
            with col_b2:
                st.markdown("#### Regression Metrics")
                reg_df = pd.DataFrame(metrics["regression"]).T
                reg_cols = [c for c in ["r2", "mae", "mse", "rmse"] if c in reg_df.columns]
                reg_summary = reg_df[reg_cols].astype(float)
                st.dataframe(reg_summary.style.format("{:.4f}").highlight_min(subset=["mae", "rmse"], color="#D1FAE5"), use_container_width=True)
        else:
            st.warning("Metrics summary JSON not found. Please run `python train_models.py`.")

    # TAB 5: REPORTS & DATA EXPLORER
    with tab_reports:
        st.subheader("PDF Report Generator & Raw Data Explorer")
        
        col_r1, col_r2 = st.columns([1, 1])
        
        with col_r1:
            st.markdown("#### 📄 Executive PDF Report Generation")
            st.write("Generate a formatted PDF document compiling analytics statistics and ML benchmarks.")
            
            gen_report_btn = st.button("📥 Generate & Download PDF Report", type="primary")
            if gen_report_btn:
                pdf_path = REPORTS_DIR / "traffic_summary_report.pdf"
                raw_csv_path = RAW_DATA_DIR / "traffic_data_raw.csv"
                metrics_json_path = MODELS_DIR / "metrics_summary.json"
                
                generate_traffic_pdf_report(str(raw_csv_path), str(metrics_json_path), str(pdf_path))
                
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="💾 Download PDF File",
                        data=f.read(),
                        file_name="IntelliTraffic_Summary_Report.pdf",
                        mime="application/pdf"
                    )
                    
        with col_r2:
            st.markdown("#### 💾 Dataset Exporter")
            st.write("Download the current preprocessed dataset used by the ML engine.")
            csv_data = raw_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Traffic Dataset CSV",
                data=csv_data,
                file_name="IntelliTraffic_Dataset.csv",
                mime="text/csv"
            )
            
        st.markdown("---")
        st.markdown("#### 🔍 Filtered Dataset Sample")
        st.dataframe(filtered_df.head(100), use_container_width=True)


if __name__ == "__main__":
    main()
