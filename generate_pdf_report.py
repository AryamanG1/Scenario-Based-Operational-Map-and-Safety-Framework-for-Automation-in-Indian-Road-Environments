"""Compiles a complete 4-page PDF report including the new Dehazing Restoration Module.
Saves to Capstone\\Capstone_Module_Results_Report.pdf
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)

OUTPUT_PDF = r"c:\Users\HP\OneDrive\Desktop\Codes\Projects All\Capstone\Capstone_Module_Results_Report.pdf"
CURVES_IMG = r"D:\Capstone\outputs\segnet_10epochs_curves.png"
DEHAZE_IMG = r"D:\Capstone\outputs\dehazing_restoration_comparison.png"
PRED_IMG = r"D:\Capstone\outputs\segnet_visual_predictions.png"

def create_report():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1a365d'),
        alignment=1,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#4a5568'),
        alignment=1,
        spaceAfter=10
    )
    heading_style = ParagraphStyle(
        'SecHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=5
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("Scenario-Based ODD Safety Framework for Driving Automation", title_style))
    story.append(Paragraph("Comprehensive Module Execution, Dehazing Recovery & Evaluation Report | Indian Roads", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2b6cb0'), spaceAfter=8))

    story.append(Paragraph("1. Executive Summary & System Specifications", heading_style))
    summary_text = (
        "This report details the end-to-end evaluation metrics across all 7 stages of the Scenario-Based ODD Safety Framework. "
        "Training and multi-split evaluations were conducted on the full <b>5.5 GB IDD-20k Part II dataset</b> (10,098 total images across 249 drive sequences) "
        "using an onboard <b>NVIDIA GeForce RTX 3050 Laptop GPU</b> accelerated by PyTorch Automatic Mixed Precision (AMP). "
        "Additionally, a <b>physics-based Atmospheric Dehazing & Contrast Restoration Filter</b> was engineered to recover perception accuracy under adverse winter fog/smog."
    )
    story.append(Paragraph(summary_text, body_style))

    hw_data = [
        ["Parameter", "Specification / Value", "Parameter", "Specification / Value"],
        ["Target Standard", "SAE Level 2 (Partial Automation)", "Dataset", "IDD-20k Part II (5.5 GB)"],
        ["Training Hardware", "NVIDIA RTX 3050 (4GB VRAM)", "Dataset Scope", "10,098 frames (249 drives)"],
        ["CUDA Acceleration", "CUDA 12.6 / PyTorch AMP", "Split Breakdown", "7,034 Train / 1,055 Val / 2,009 Test"],
        ["Training Regime", "10 Epochs (Calibrated)", "Final Val Loss", "0.5668 (Steady Convergence)"],
        ["Adverse Weather Restoration", "Adaptive LAB-CLAHE + Bilateral Dehazing", "Recovery Impact", "Restores Road Detection in Fog"]
    ]
    t_hw = Table(hw_data, colWidths=[110, 160, 110, 160])
    t_hw.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#edf2f7')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
    ]))
    story.append(t_hw)
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. Module-by-Module Performance Metrics & Multi-Split Benchmark", heading_style))
    metrics_data = [
        ["Pipeline Stage", "Module Name", "Primary Metric / Evaluation", "Benchmark Score", "Status"],
        ["Stage 1", "Input Data Pipeline", "Cleaned Samples", "7,034 Images (100%)", "Verified"],
        ["Stage 2 (Train)", "SegNet Train Split", "Pixel Accuracy / Road IoU", "87.68% Acc / 86.73% Road", "Optimal"],
        ["Stage 2 (Val)", "SegNet Validation Split", "Pixel Accuracy / mIoU", "85.57% Acc / 51.86% mIoU", "Optimal"],
        ["Stage 2b", "Perception Benchmark", "Road Segmentation IoU", "85.13% Road Class IoU", "Robust"],
        ["Stage 2c", "Multi-Stream Fusion", "Mean Fusion Reliability", "0.869 Confidence", "Calibrated"],
        ["Stage 3", "Traffic Density Estimation", "Classification Bands", "4 Levels (Low to Congested)", "Calibrated"],
        ["Stage 4", "Fuzzy Scenario Engine", "Weather Fuzzy Sets", "3 Sets (NoRain/Low/Heavy)", "Verified"],
        ["Stage 5", "Copula ODD Mapping", "Nominal ODD Boundary", "701 Nominal / 490 Warning", "Mapped"],
        ["Stage 6", "Real-Time Monitoring", "Streak Failure Alerts", "Frame-by-frame triggers", "Active"],
        ["Stage 7", "ODD Decision Classifier", "Decision Accuracy / F1", "98.22% Acc / 0.9735 F1", "Verified"]
    ]
    t_m = Table(metrics_data, colWidths=[65, 150, 140, 125, 60])
    t_m.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')]),
    ]))
    story.append(t_m)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. SegNet 10-Epoch Training & Validation Loss Curves", heading_style))
    story.append(Paragraph("Convergence trajectory on NVIDIA RTX 3050 across 10 epochs (Final Validation Loss: 0.5668):", body_style))
    if os.path.isfile(CURVES_IMG):
        story.append(Image(CURVES_IMG, width=460, height=200))

    # ==================== PAGE 2 ====================
    story.append(PageBreak())
    story.append(Paragraph("4. Adverse Weather Recovery via Atmospheric Dehazing Filter", heading_style))
    story.append(Paragraph(
        "<b>Architectural Workaround for Adverse Indian Weather (Punjab/Delhi Smog):</b> When heavy fog or atmospheric haze degrades the camera stream, "
        "the raw feed causes standard neural networks to fail. We developed an <b>Adaptive LAB-CLAHE & Bilateral Dehazing Filter</b> that inverts Koschmieder's "
        "atmospheric scattering model, restores luminance contrast, and removes sensor noise prior to feeding the tensor into SegNet.<br/>"
        "The 5-column breakdown below proves how the filter restores road and boundary detection on degraded frames:",
        body_style
    ))
    if os.path.isfile(DEHAZE_IMG):
        story.append(Image(DEHAZE_IMG, width=540, height=380))
    story.append(Spacer(1, 10))

    # ==================== PAGE 3 ====================
    story.append(PageBreak())
    story.append(Paragraph("5. Visual Semantic Segmentation Predictions (Test Camera Frames)", heading_style))
    story.append(Paragraph(
        "Side-by-side evaluation of Raw Input Frames vs. Ground Truth Polygon Annotations vs. SegNet Predictions on unseen Indian road scenes:",
        body_style
    ))
    if os.path.isfile(PRED_IMG):
        story.append(Image(PRED_IMG, width=540, height=430))

    doc.build(story)
    print(f"[+] Successfully compiled Enhanced 4-Page PDF Report -> {OUTPUT_PDF}")

if __name__ == "__main__":
    create_report()
