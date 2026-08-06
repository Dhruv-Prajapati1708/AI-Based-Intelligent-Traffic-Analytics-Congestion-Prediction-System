"""
Visualization utilities for IntelliTraffic.
Provides Plotly interactive charts for traffic trends, peak hours, weather impact,
model metrics, and Folium geospatial congestion heatmaps.
"""

import json
import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium


def plot_peak_hours(df: pd.DataFrame) -> go.Figure:
    """
    Plots average vehicle count and congestion index across hours of the day (0-23).
    """
    df_temp = df.copy()
    if "timestamp" in df_temp.columns and not pd.api.types.is_datetime64_any_dtype(df_temp["timestamp"]):
        df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
        
    if "hour" not in df_temp.columns and "timestamp" in df_temp.columns:
        df_temp["hour"] = df_temp["timestamp"].dt.hour
        
    hourly = df_temp.groupby("hour").agg({
        "vehicle_count": "mean",
        "congestion_index": "mean"
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["hour"],
        y=hourly["vehicle_count"],
        name="Avg Vehicle Count",
        marker_color="#3366CC"
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour"],
        y=hourly["congestion_index"],
        name="Avg Congestion Index",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#DC3912", width=3)
    ))
    
    fig.update_layout(
        title="<b>Hourly Traffic Density & Peak Hour Analysis</b>",
        xaxis=dict(title="Hour of Day (0 - 23)", tickmode="linear", tick0=0, dtick=1),
        yaxis=dict(title="Average Vehicle Count"),
        yaxis2=dict(title="Congestion Index (0-100)", overlaying="y", side="right", range=[0, 100]),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_daily_trends(df: pd.DataFrame) -> go.Figure:
    """
    Plots daily total vehicle count and average congestion trend over time.
    """
    df_temp = df.copy()
    if "timestamp" in df_temp.columns and not pd.api.types.is_datetime64_any_dtype(df_temp["timestamp"]):
        df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
        
    df_temp["date"] = df_temp["timestamp"].dt.date
    daily = df_temp.groupby("date").agg({
        "vehicle_count": "sum",
        "congestion_index": "mean",
        "average_speed_kmh": "mean"
    }).reset_index()
    
    fig = px.line(
        daily, x="date", y="vehicle_count",
        title="<b>Daily Traffic Volume Trends</b>",
        labels={"date": "Date", "vehicle_count": "Total Vehicles / Day"},
        template="plotly_white",
        color_discrete_sequence=["#109618"]
    )
    fig.update_traces(mode="lines+markers")
    return fig


def plot_junction_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Compares traffic vehicle count and average speed across city junctions.
    """
    junction_summary = df.groupby("junction_id").agg({
        "vehicle_count": "mean",
        "average_speed_kmh": "mean",
        "congestion_index": "mean"
    }).reset_index()
    
    fig = px.bar(
        junction_summary,
        x="junction_id",
        y="vehicle_count",
        color="average_speed_kmh",
        title="<b>Junction Comparison: Traffic Volume vs Speed</b>",
        labels={"junction_id": "Junction", "vehicle_count": "Avg Vehicle Count", "average_speed_kmh": "Avg Speed (km/h)"},
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    return fig


def plot_weather_impact(df: pd.DataFrame) -> go.Figure:
    """
    Visualizes weather condition impact on vehicle speed and congestion.
    """
    weather_summary = df.groupby("weather_condition").agg({
        "average_speed_kmh": "mean",
        "congestion_index": "mean"
    }).reset_index()
    
    fig = go.Figure(data=[
        go.Bar(name="Avg Speed (km/h)", x=weather_summary["weather_condition"], y=weather_summary["average_speed_kmh"], marker_color="#FF9900"),
        go.Bar(name="Congestion Index", x=weather_summary["weather_condition"], y=weather_summary["congestion_index"], marker_color="#990099")
    ])
    fig.update_layout(
        barmode="group",
        title="<b>Weather Impact on Traffic Speed & Congestion</b>",
        xaxis_title="Weather Condition",
        template="plotly_white"
    )
    return fig


def plot_model_comparison(metrics: Dict[str, Any]) -> go.Figure:
    """
    Bar chart comparing Decision Tree, Random Forest, and XGBoost accuracy and F1-score.
    """
    models = []
    accuracies = []
    f1_scores = []
    
    for model_name, m in metrics.get("classification", {}).items():
        models.append(model_name)
        accuracies.append(m["accuracy"])
        f1_scores.append(m["f1_score"])
        
    fig = go.Figure(data=[
        go.Bar(name="Accuracy", x=models, y=accuracies, marker_color="#4CAF50", text=[f"{v:.3f}" for v in accuracies], textposition="auto"),
        go.Bar(name="F1-Score", x=models, y=f1_scores, marker_color="#2196F3", text=[f"{v:.3f}" for v in f1_scores], textposition="auto")
    ])
    fig.update_layout(
        barmode="group",
        title="<b>ML Classification Model Performance Comparison</b>",
        yaxis=dict(title="Score (0.0 - 1.0)", range=[0.8, 1.05]),
        template="plotly_white"
    )
    return fig


def generate_folium_map(df: pd.DataFrame, target_junction: Optional[str] = None) -> folium.Map:
    """
    Generates an interactive Folium map displaying city junctions, color-coded congestion markers,
    and road connections based on latest congestion levels.
    """
    # Coordinates for synthetic city junctions
    junction_coords = {
        "Junction_1": (51.5074, -0.1278),  # Central
        "Junction_2": (51.5150, -0.1410),  # North-West
        "Junction_3": (51.4980, -0.1010),  # South-East
        "Junction_4": (51.5200, -0.0900)   # North-East
    }
    
    # Calculate latest status per junction
    latest_df = df.sort_values(by="timestamp").groupby("junction_id").last().reset_index()
    
    # Map center
    center_lat = np.mean([lat for lat, lon in junction_coords.values()])
    center_lon = np.mean([lon for lat, lon in junction_coords.values()])
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")
    
    # Color mapping helper
    def get_color(idx, level):
        if idx >= 75.0 or level == "Severe":
            return "darkred", "#dc3545"
        elif idx >= 55.0 or level == "High":
            return "red", "#e63946"
        elif idx >= 30.0 or level == "Moderate":
            return "orange", "#ffb703"
        else:
            return "green", "#2a9d8f"

    # Add Junction Markers
    points = []
    for _, row in latest_df.iterrows():
        j_id = row["junction_id"]
        if j_id not in junction_coords:
            continue
            
        lat, lon = junction_coords[j_id]
        points.append((lat, lon))
        
        c_idx = row.get("congestion_index", 50.0)
        c_lvl = row.get("congestion_level", "Moderate")
        v_cnt = row.get("vehicle_count", 0)
        a_spd = row.get("average_speed_kmh", 0)
        
        icon_color, hex_color = get_color(c_idx, c_lvl)
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 180px;">
            <h4 style="margin: 0 0 5px 0; color: #333;">{j_id}</h4>
            <hr style="margin: 4px 0;">
            <b>Congestion:</b> <span style="color:{hex_color}; font-weight:bold;">{c_lvl} ({c_idx:.1f})</span><br>
            <b>Vehicles:</b> {v_cnt} cars/hr<br>
            <b>Avg Speed:</b> {a_spd:.1f} km/h
        </div>
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{j_id}: {c_lvl} ({c_idx:.1f})",
            icon=folium.Icon(color=icon_color, icon="car", prefix="fa")
        ).add_to(m)
        
        # Circle indicator
        folium.CircleMarker(
            location=[lat, lon],
            radius=15,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.3
        ).add_to(m)

    # Add connecting road lines
    road_connections = [
        ("Junction_1", "Junction_2"),
        ("Junction_1", "Junction_3"),
        ("Junction_2", "Junction_4"),
        ("Junction_3", "Junction_4")
    ]
    
    for j1, j2 in road_connections:
        if j1 in junction_coords and j2 in junction_coords:
            p1 = junction_coords[j1]
            p2 = junction_coords[j2]
            
            # Avg congestion index between junctions
            row1 = latest_df[latest_df["junction_id"] == j1]
            row2 = latest_df[latest_df["junction_id"] == j2]
            idx1 = row1["congestion_index"].values[0] if not row1.empty else 40.0
            idx2 = row2["congestion_index"].values[0] if not row2.empty else 40.0
            avg_idx = (idx1 + idx2) / 2.0
            
            _, hex_color = get_color(avg_idx, "Moderate")
            
            folium.PolyLine(
                locations=[p1, p2],
                color=hex_color,
                weight=5,
                opacity=0.8,
                tooltip=f"Road Segment ({j1} ↔ {j2}): Avg Congestion {avg_idx:.1f}"
            ).add_to(m)

    return m
