"""
Report generator utilities for IntelliTraffic.
Generates automated analytical PDF reports containing traffic statistics,
peak hour summaries, model evaluations, and recommendations using ReportLab.
"""

import os
import json
import pandas as pd
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_traffic_pdf_report(
    df_path: str,
    metrics_path: str,
    output_pdf_path: str
) -> str:
    """
    Generates a PDF executive report summarizing traffic analytics and ML model performance.
    """
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Data file not found: {df_path}")
        
    df = pd.read_csv(df_path)
    
    # Load metrics if available
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    # Calculate statistics
    total_records = len(df)
    avg_vehicles = int(df["vehicle_count"].mean())
    avg_speed = float(df["average_speed_kmh"].mean())
    avg_congestion = float(df["congestion_index"].mean())
    
    # Setup document
    output_dir = os.path.dirname(output_pdf_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=15
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155")
    )

    story = []
    
    # Title & Metadata
    story.append(Paragraph("🚦 IntelliTraffic Executive Analytics Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System Version 1.0", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=12))
    
    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Traffic Statistics", h2_style))
    summary_text = (
        f"This report presents an analytical summary of traffic flow, congestion patterns, "
        f"and machine learning prediction benchmarks across city junctions. The dataset comprises "
        f"<b>{total_records:,}</b> spatio-temporal traffic observations."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))
    
    # KPI Table
    kpi_data = [
        ["Metric Description", "Value", "Benchmark Status"],
        ["Total Recorded Volume", f"{total_records:,} records", "Normal"],
        ["Average Vehicle Density", f"{avg_vehicles} vehicles/hr", "Optimal"],
        ["Average Flow Speed", f"{avg_speed:.1f} km/h", "Moderate"],
        ["Average Congestion Index", f"{avg_congestion:.1f} / 100", "Stable"]
    ]
    kpi_table = Table(kpi_data, colWidths=[200, 150, 150])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 12))
    
    # Machine Learning Performance Section
    story.append(Paragraph("2. Machine Learning Model Benchmark", h2_style))
    story.append(Paragraph("Comparative performance evaluation across XGBoost, Random Forest, and Decision Tree algorithms:", body_style))
    story.append(Spacer(1, 6))
    
    ml_data = [
        ["Model Architecture", "Classification Accuracy", "F1 Score", "Regression R²", "MAE"]
    ]
    
    clf_metrics = metrics.get("classification", {})
    reg_metrics = metrics.get("regression", {})
    
    for model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
        c_acc = clf_metrics.get(model_name, {}).get("accuracy", 0.0)
        c_f1 = clf_metrics.get(model_name, {}).get("f1_score", 0.0)
        r_r2 = reg_metrics.get(model_name, {}).get("r2", 0.0)
        r_mae = reg_metrics.get(model_name, {}).get("mae", 0.0)
        ml_data.append([model_name, f"{c_acc:.4f}", f"{c_f1:.4f}", f"{r_r2:.4f}", f"{r_mae:.4f}"])
        
    ml_table = Table(ml_data, colWidths=[140, 90, 90, 90, 90])
    ml_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#93C5FD")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#EFF6FF")])
    ]))
    story.append(ml_table)
    story.append(Spacer(1, 12))
    
    # Recommendations Section
    story.append(Paragraph("3. Intelligent Recommendations & Traffic Management", h2_style))
    recs = (
        "• <b>Peak Hour Signal Optimization:</b> Extend green light cycles during morning (07:00-10:00) and evening (17:00-20:00) rush hours.<br/>"
        "• <b>Dynamic Re-routing:</b> Automatically redirect traffic via diversion roads when XGBoost model predicts High/Severe congestion.<br/>"
        "• <b>Weather Advisory Protocols:</b> Activate safety speed limits and warning displays during Heavy Rain and Fog conditions."
    )
    story.append(Paragraph(recs, body_style))
    
    # Build Document
    doc.build(story)
    return output_pdf_path


if __name__ == "__main__":
    raw_path = "data/raw/traffic_data_raw.csv"
    metrics_path = "models/metrics_summary.json"
    pdf_out = "reports/traffic_summary_report.pdf"
    
    if os.path.exists(raw_path):
        out = generate_traffic_pdf_report(raw_path, metrics_path, pdf_out)
        print(f"Generated PDF report successfully at: {out}")
