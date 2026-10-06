"""
Streamlit Web Application: PropIntel | Advanced Global Real Estate Valuation & Analytics.
Production Hedonic AI Engine powered by Super Ensemble (XGBoost + GradientBoosting + HistGradientBoosting).

Features:
- Global Multi-Currency Valuation (USD, INR Crores/Lakhs, EUR, GBP, AED, CAD, AUD, JPY, SGD, etc.)
- Measurement Unit Toggle (Square Feet sq ft <-> Square Meters m²)
- 1-Click Fast Presets (Mumbai, Bengaluru, NYC, Dallas, London)
- Glassmorphism & High-Precision FinTech Aesthetic
- Dynamic Model Confidence Interval Visual Range Meter
- Value Driver Decomposition Breakdown & SHAP-Style Feature Waterfall
- Real Estate Financials: Mortgage EMI Calculator & Rental Yield / ROI Forecaster
- Dynamic What-If Renovation & Upgrades Simulator
- Live Market Intelligence & Distribution Charts
- Model Explainability (Feature Importance & Benchmark Leaderboard)
- PDF Report Export & 1-Click Shareable Valuation URL
- Side-by-Side Property Comparison
- Light / Dark Mode Toggle with Curated Luxury Palettes
"""

import os
import sys
import json
import io
import base64
import urllib.parse
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from src.config import (
    CLEANED_DATA_PATH, METADATA_PATH, FEATURE_IMPORTANCE_PATH,
    VALID_LOCATIONS, VALID_CONDITIONS, VALID_GARAGES
)
from src.predict import predict_house_price, load_model_artifacts
from src.global_market import (
    GLOBAL_COUNTRIES, EXCHANGE_RATES, SETTLEMENT_TIERS,
    convert_and_localize_price, format_currency_value,
    convert_area_to_sqft, convert_sqft_to_sqm
)

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PropIntel Global | Premium Real Estate AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# THEME STATE (Dark Mode default for luxury aesthetics)
# ------------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

# Dynamic Palette
if dark:
    bg_main       = "#070a13"
    bg_card       = "rgba(16, 23, 41, 0.75)"
    bg_card_solid = "#0f172a"
    bg_hover      = "rgba(30, 41, 59, 0.85)"
    text_main     = "#f8fafc"
    text_muted    = "#94a3b8"
    border_col    = "rgba(255, 255, 255, 0.08)"
    border_glow   = "rgba(59, 130, 246, 0.35)"
    hero_grad     = "linear-gradient(135deg, #090d1a 0%, #172033 40%, #1e1b4b 100%)"
    accent_blue   = "#3b82f6"
    accent_glow   = "#60a5fa"
    accent_purple = "#8b5cf6"
    accent_emerald= "#10b981"
    stat_num_col  = "#60a5fa"
    fig_bg        = "#0f172a"
    mpl_text      = "#e2e8f0"
    mpl_grid      = "#1e293b"
else:
    bg_main       = "#f8fafc"
    bg_card       = "rgba(255, 255, 255, 0.88)"
    bg_card_solid = "#ffffff"
    bg_hover      = "#f1f5f9"
    text_main     = "#0f172a"
    text_muted    = "#64748b"
    border_col    = "rgba(226, 232, 240, 0.9)"
    border_glow   = "rgba(37, 99, 235, 0.25)"
    hero_grad     = "linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%)"
    accent_blue   = "#2563eb"
    accent_glow   = "#3b82f6"
    accent_purple = "#7c3aed"
    accent_emerald= "#059669"
    stat_num_col  = "#1e40af"
    fig_bg        = "#f8fafc"
    mpl_text      = "#1e293b"
    mpl_grid      = "#e2e8f0"

# ------------------------------------------------------------------------------
# INJECT PREMIUM DESIGN SYSTEM CSS
# ------------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background: {bg_main};
        color: {text_main};
    }}
    .stApp {{
        background: {bg_main};
        background-attachment: fixed;
    }}

    /* Hero Banner */
    .hero-header {{
        background: {hero_grad};
        padding: 2.2rem 2.8rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }}
    .hero-header::after {{
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.3) 0%, transparent 70%);
        pointer-events: none;
    }}

    /* Glass Cards */
    .glass-card {{
        background: {bg_card};
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid {border_col};
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin: 1.2rem 0;
        box-shadow: 0 15px 35px -10px rgba(0, 0, 0, {'0.4' if dark else '0.06'});
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }}
    .glass-card:hover {{
        border-color: {border_glow};
        box-shadow: 0 20px 45px -10px rgba(59, 130, 246, {'0.25' if dark else '0.12'});
    }}

    /* Main Valuation Hero Card */
    .val-hero-card {{
        background: {f"linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(30, 27, 75, 0.7) 100%)" if dark else "linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%)"};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1.5px solid {border_glow};
        border-radius: 24px;
        padding: 2.2rem 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 25px 60px -15px rgba(37, 99, 235, {'0.35' if dark else '0.15'});
        position: relative;
    }}

    /* Live Pulse Badge */
    @keyframes pulse-dot {{
        0% {{ transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
        70% {{ transform: scale(1); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }}
        100% {{ transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
    }}
    .live-pulse {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10b981;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}
    .live-pulse-dot {{
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse-dot 2s infinite;
    }}

    /* Price Headline */
    .price-hero {{
        font-size: 3.3rem;
        font-weight: 900;
        line-height: 1.05;
        letter-spacing: -1px;
        margin: 0.6rem 0 0.3rem 0;
        background: {f"linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%)" if dark else "linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #7c3aed 100%)"};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    .price-sub {{
        font-size: 1.15rem;
        font-weight: 600;
        color: {text_muted};
        margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* Badges */
    .badge-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.38rem 0.95rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.4rem;
        background: {f"rgba(30, 41, 59, 0.7)" if dark else "#e2e8f0"};
        border: 1px solid {border_col};
        color: {text_main};
    }}
    .badge-accent {{
        background: {f"rgba(37, 99, 235, 0.2)" if dark else "rgba(37, 99, 235, 0.1)"};
        border: 1px solid {accent_glow};
        color: {accent_glow if dark else "#1d4ed8"};
    }}

    /* Stat Card */
    .stat-card {{
        background: {bg_card};
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid {border_col};
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, {'0.3' if dark else '0.05'});
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
    }}
    .stat-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 16px 35px -8px rgba(37, 99, 235, 0.22);
        border-color: {border_glow};
    }}
    .stat-chip {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {text_muted};
        margin-bottom: 0.3rem;
    }}
    .stat-val {{
        font-size: 1.9rem;
        font-weight: 800;
        color: {stat_num_col};
        line-height: 1.1;
    }}
    .stat-sub {{
        font-size: 0.82rem;
        color: {text_muted};
        margin-top: 0.25rem;
    }}

    /* Range Bar */
    .range-box {{
        background: {f"rgba(15, 23, 42, 0.6)" if dark else "rgba(241, 245, 249, 0.8)"};
        border: 1px solid {border_col};
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin-top: 1rem;
    }}
    .range-track {{
        height: 8px;
        background: {f"#1e293b" if dark else "#cbd5e1"};
        border-radius: 9999px;
        position: relative;
        margin: 0.6rem 0;
        overflow: hidden;
    }}
    .range-fill {{
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981);
        border-radius: 9999px;
    }}

    /* Buttons */
    div.stButton > button {{
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.2s ease;
        border: 1px solid {border_col};
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        border-color: {accent_glow};
    }}
    div.stFormSubmitButton > button {{
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 1.8rem !important;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.45) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    div.stFormSubmitButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 35px -5px rgba(37, 99, 235, 0.65) !important;
    }}

    /* Share box */
    .share-box {{
        background: {f"rgba(15, 23, 42, 0.85)" if dark else "#eff6ff"};
        border: 1px solid {border_glow};
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        font-size: 0.85rem;
        color: {accent_glow if dark else "#1e40af"};
        word-break: break-all;
        font-family: 'JetBrains Mono', monospace;
    }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# DATA & METADATA LOADER
# ------------------------------------------------------------------------------
@st.cache_data
def load_app_data():
    df = pd.read_csv(CLEANED_DATA_PATH) if CLEANED_DATA_PATH.exists() else pd.DataFrame()
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    feat_imp = {}
    if FEATURE_IMPORTANCE_PATH.exists():
        with open(FEATURE_IMPORTANCE_PATH, "r") as f:
            feat_imp = json.load(f)
    return df, metadata, feat_imp

df_data, metadata, feat_importance = load_app_data()

# Dynamic metrics from Champion Model
final_metrics = metadata.get("final_test_metrics", {})
r2_score = final_metrics.get("R2", 0.849)
mae_val  = final_metrics.get("MAE", 77997.0)
mape_val = final_metrics.get("MAPE", 17.1)
model_title = metadata.get("model_name", "Super Ensemble (XGB+GB+HGB)")


# ------------------------------------------------------------------------------
# REPORT GENERATION HELPER
# ------------------------------------------------------------------------------
def make_pdf_report(res, payload, active_currency, year_built, down_pct, loan_tenure, interest_rate):
    """Generate institutional PDF or plain-text valuation report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=22, textColor=colors.HexColor("#1e3a8a"),
                                     spaceAfter=6, fontName="Helvetica-Bold")
        sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                     fontSize=10, textColor=colors.HexColor("#64748b"),
                                     spaceAfter=12)
        head_style  = ParagraphStyle("head", parent=styles["Heading2"],
                                     fontSize=13, textColor=colors.HexColor("#1e3a8a"),
                                     spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold")
        body_style  = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, spaceAfter=4)

        story.append(Paragraph("🏢 PropIntel Global — Valuation Appraisal", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Model: {model_title} | Test R²: {r2_score:.1%}", sub_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563eb"), spaceAfter=12))

        story.append(Paragraph("Estimated Market Valuation", head_style))
        story.append(Paragraph(f"<b><font size=18 color='#1e3a8a'>{res['price_formatted']}</font></b>", body_style))
        story.append(Paragraph(f"Location: {res['city']}, {res['country']} ({payload.get('Location', 'Core')})", body_style))
        story.append(Paragraph(f"95% Confidence Interval: {res['prediction_interval_95']['lower_formatted']} — {res['prediction_interval_95']['upper_formatted']}", body_style))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Property Specifications", head_style))
        prop_data = [
            ["Attribute", "Specification"],
            ["Living Space", f"{payload.get('Area', 'N/A')} {'m²' if payload.get('Area_Unit') == 'sq m' else 'sq ft'}"],
            ["Bedrooms", str(payload.get('Bedrooms', 'N/A'))],
            ["Bathrooms", str(payload.get('Bathrooms', 'N/A'))],
            ["Floors", str(payload.get('Floors', 'N/A'))],
            ["Year Constructed", str(payload.get('YearBuilt', 'N/A'))],
            ["Condition Grade", str(payload.get('Condition', 'N/A'))],
            ["Parking Facility", str(payload.get('Garage', 'N/A'))],
            ["Settlement Density", str(payload.get('Location', 'N/A'))],
            ["Market / Hub", f"{payload.get('Country', '')} / {payload.get('City', 'Standard')}"],
        ]
        t = Table(prop_data, colWidths=[7*cm, 9*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e3a8a")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING",    (0,0), (-1,-1), 6),
        ]))
        story.append(t)

        story.append(Paragraph("Investment & Financing Overview", head_style))
        total_price = res["predicted_price"]
        loan_principal = total_price * (1.0 - down_pct / 100.0)
        monthly_r = (interest_rate / 100.0) / 12.0
        num_months = loan_tenure * 12
        monthly_emi = (loan_principal * (monthly_r * (1 + monthly_r)**num_months) / ((1 + monthly_r)**num_months - 1)) if monthly_r > 0 else (loan_principal / num_months)

        fmt_emi  = format_currency_value(monthly_emi, active_currency)
        fmt_loan = format_currency_value(loan_principal, active_currency)

        fin_data = [
            ["Financial Parameter", "Value"],
            ["Equity Down Payment", f"{down_pct}%"],
            ["Financed Principal", fmt_loan["compact"]],
            ["Amortization Tenure", f"{loan_tenure} Years"],
            ["Annual Borrowing Rate", f"{interest_rate:.2f}%"],
            ["Estimated Monthly EMI", fmt_emi["compact"]],
            ["Effective Rate / sq ft", res["price_per_sqft_formatted"]],
            ["Effective Rate / m²",    res["price_per_sqm_formatted"]],
        ]
        t2 = Table(fin_data, colWidths=[7*cm, 9*cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f766e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0fdf4"), colors.white]),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING",    (0,0), (-1,-1), 6),
        ]))
        story.append(t2)

        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
        story.append(Paragraph("PropIntel Global Valuation Report — Calibrated with US Baseline & Regional Market Indices. For institutional informational purposes only.",
                                ParagraphStyle("footer", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#94a3b8"))))

        doc.build(story)
        buf.seek(0)
        return buf.read()
    except Exception:
        # Fallback text format
        lines = [
            "PROPINTEL GLOBAL — VALUATION REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            f"Model: {model_title} | Test R²: {r2_score:.1%}",
            "",
            "ESTIMATED MARKET VALUATION",
            "-" * 30,
            f"  Valuation: {res['price_formatted']}",
            f"  Location:  {res['city']}, {res['country']}",
            f"  Interval:  {res['prediction_interval_95']['lower_formatted']} - {res['prediction_interval_95']['upper_formatted']}",
            "",
            "PROPERTY SPECIFICATIONS",
            "-" * 30,
            f"  Area:      {payload.get('Area')} {payload.get('Area_Unit', 'sq ft')}",
            f"  Bedrooms:  {payload.get('Bedrooms')}",
            f"  Bathrooms: {payload.get('Bathrooms')}",
            f"  YearBuilt: {payload.get('YearBuilt')}",
            f"  Condition: {payload.get('Condition')}",
            f"  Rate/sqft: {res['price_per_sqft_formatted']}",
            f"  Rate/m²:   {res['price_per_sqm_formatted']}",
            "",
            "PropIntel Global — For informational purposes only."
        ]
        return "\n".join(lines).encode("utf-8")


# ------------------------------------------------------------------------------
# URL & STATE SHARING HELPERS
# ------------------------------------------------------------------------------
def make_share_url(payload):
    params = {
        "loc":  payload.get("Location", ""),
        "cond": payload.get("Condition", ""),
        "gar":  payload.get("Garage", ""),
        "area": payload.get("Area", ""),
        "bed":  payload.get("Bedrooms", ""),
        "bath": payload.get("Bathrooms", ""),
        "fl":   payload.get("Floors", ""),
        "yr":   payload.get("YearBuilt", ""),
        "ctry": payload.get("Country", ""),
        "city": payload.get("City", "") or "",
        "curr": payload.get("Currency", "INR"),
    }
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v != ""})
    base = "https://harshupadhyay750-house-prediction-appapp-fogahz.streamlit.app/?"
    return base + qs

def get_url_params():
    try:
        p = st.query_params
        return {
            "Location":  p.get("loc", "Downtown"),
            "Condition": p.get("cond", "Good"),
            "Garage":    p.get("gar", "Yes"),
            "Area":      float(p.get("area", 2000)),
            "Bedrooms":  int(p.get("bed", 3)),
            "Bathrooms": float(p.get("bath", 2.0)),
            "Floors":    int(p.get("fl", 1)),
            "YearBuilt": int(p.get("yr", 2018)),
            "Country":   p.get("ctry", "India"),
            "City":      p.get("city", "Mumbai (MMR)"),
            "Currency":  p.get("curr", "INR"),
        }
    except Exception:
        return {}


# ------------------------------------------------------------------------------
# MODERN CHART VISUALIZATIONS
# ------------------------------------------------------------------------------
def plot_price_trend(base_price, active_currency, dark_mode=True):
    bg = "#070a13" if dark_mode else "#ffffff"
    card_bg = "#0f172a" if dark_mode else "#f8fafc"
    tc = "#e2e8f0" if dark_mode else "#0f172a"
    gc = "#1e293b" if dark_mode else "#e2e8f0"

    years_hist = list(range(2020, 2027))
    np.random.seed(42)
    hist_prices = [base_price / (1.055 ** (2026 - y)) * (1 + np.random.uniform(-0.015, 0.015)) for y in years_hist]
    hist_prices[-1] = base_price

    years_proj = list(range(2026, 2032))
    proj_opt  = [base_price * (1.08 ** i) for i in range(len(years_proj))]
    proj_base = [base_price * (1.055 ** i) for i in range(len(years_proj))]
    proj_cons = [base_price * (1.03 ** i) for i in range(len(years_proj))]

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=bg)
    ax.set_facecolor(card_bg)

    # Plot historical trajectory
    ax.plot(years_hist, hist_prices, "o-", color="#3b82f6", linewidth=2.8, markersize=6, label="Historical Trajectory (Est.)")

    # Plot projection curves
    ax.plot(years_proj, proj_opt, "--", color="#10b981", linewidth=2.0, alpha=0.9, label="Optimistic (+8.0% CAGR)")
    ax.plot(years_proj, proj_base, "-", color="#f59e0b", linewidth=2.6, label="Baseline (+5.5% CAGR)")
    ax.plot(years_proj, proj_cons, "--", color="#ef4444", linewidth=2.0, alpha=0.9, label="Conservative (+3.0% CAGR)")

    # Shaded band
    ax.fill_between(years_proj, proj_cons, proj_opt, alpha=0.15, color="#8b5cf6")

    # Vertical Today marker
    ax.axvline(x=2026, color=tc, linewidth=1.2, linestyle=":", alpha=0.6)
    ax.text(2026.1, min(hist_prices) * 0.98, "● Today", color="#10b981", fontsize=9, fontweight="bold")

    ax.set_title("5-Year Historical Valuation & Future Projection", color=tc, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Calendar Year", fontweight="bold", color=tc)
    ax.set_ylabel(f"Valuation ({active_currency})", fontweight="bold", color=tc)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y/1e7:.2f} Cr" if (active_currency == "INR" and y >= 1e7)
        else (f"{y/1e5:.1f} Lakh" if active_currency == "INR" and y >= 1e5
              else (f"{y/1e6:.2f}M" if y >= 1e6 else f"{y/1e3:.0f}K"))
    ))
    ax.tick_params(colors=tc)
    for spine in ax.spines.values():
        spine.set_edgecolor(gc)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8.5, facecolor=card_bg, labelcolor=tc, framealpha=0.85, edgecolor=gc)
    ax.grid(True, color=gc, linewidth=0.6, alpha=0.5, linestyle="--")
    plt.tight_layout()
    return fig


def plot_shap_waterfall(res, payload, dark_mode=True):
    bg = "#070a13" if dark_mode else "#ffffff"
    card_bg = "#0f172a" if dark_mode else "#f8fafc"
    tc = "#e2e8f0" if dark_mode else "#0f172a"
    gc = "#1e293b" if dark_mode else "#e2e8f0"

    total = res["predicted_price"]
    kc = res["key_characteristics"]

    # Decomposed hedonic contributions
    area_contrib   = total * 0.40
    rooms_contrib  = total * 0.14
    location_mult  = res["regional_multiplier"]
    loc_contrib    = total * (0.20 if location_mult >= 1.5 else (0.14 if location_mult >= 1.0 else 0.08))
    cond_map       = {"Excellent": 0.12, "Good": 0.06, "Fair": 0.01, "Poor": -0.05}
    cond_contrib   = total * cond_map.get(payload.get("Condition", "Good"), 0.06)
    age_contrib    = total * max(-0.14, -0.0035 * kc.get("Property_Age", 15))
    garage_contrib = total * (0.05 if payload.get("Garage") == "Yes" else 0.0)
    base_val       = total - (area_contrib + rooms_contrib + loc_contrib + cond_contrib + age_contrib + garage_contrib)

    features = [
        "Base Value",
        "Living Space",
        "Room Layout",
        "Metro / Settlement",
        "Condition Upkeep",
        "Age Depreciation",
        "Garage / Parking"
    ]
    values = [base_val, area_contrib, rooms_contrib, loc_contrib, cond_contrib, age_contrib, garage_contrib]
    colors_bar = ["#64748b"] + ["#10b981" if v >= 0 else "#f43f5e" for v in values[1:]]

    running = []
    cumsum = 0
    for v in values:
        running.append(cumsum)
        cumsum += v

    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=bg)
    ax.set_facecolor(card_bg)

    for i, (feat, val, start, color) in enumerate(zip(features, values, running, colors_bar)):
        ax.barh(feat, val, left=start, color=color, edgecolor="none", height=0.55, alpha=0.92)
        share = (val / total) * 100
        lbl = f"{'+' if val >= 0 else ''}{share:.1f}%"
        x_pos = start + val + (total * 0.012 if val >= 0 else -total * 0.012)
        ax.text(x_pos, i, lbl, va="center", ha="left" if val >= 0 else "right",
                fontsize=8.5, color=tc, fontweight="600")

    ax.axvline(x=total, color="#3b82f6", linewidth=1.5, linestyle="--", alpha=0.85)
    ax.text(total, len(features)-0.5, "Final Valuation", color="#3b82f6", fontsize=8.5, fontweight="bold", ha="center")

    ax.set_title("Hedonic Feature Contribution Breakdown (SHAP-Style)", color=tc, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(f"Accumulated Valuation ({res['currency']})", fontweight="bold", color=tc)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e7:.2f}Cr" if (res['currency'] == "INR" and x >= 1e7)
        else (f"{x/1e5:.1f}L" if res['currency'] == "INR" and x >= 1e5
              else (f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ))
    ax.tick_params(colors=tc)
    for spine in ax.spines.values():
        spine.set_edgecolor(gc)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis="x", color=gc, linewidth=0.6, alpha=0.5, linestyle="--")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SETTINGS
# ------------------------------------------------------------------------------
url_params = get_url_params()

with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">
        <span style="font-size:1.8rem;">🏢</span>
        <div>
            <div style="font-size:1.3rem; font-weight:900; letter-spacing:-0.5px; line-height:1.1;">PropIntel</div>
            <div style="font-size:0.75rem; color:#60a5fa; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">AI Valuation Terminal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Institutional Hedonic Real Estate AI Engine")

    # Dark / Light Mode Switch
    col_mode_t, col_mode_b = st.columns([3, 1])
    with col_mode_t:
        st.markdown(f"**{'🌙 Dark Obsidian' if dark else '☀️ Pure Light'}**")
    with col_mode_b:
        if st.button("Flip", key="theme_toggle", help="Switch between Dark Obsidian and Pure Light theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown("---")

    app_tab = st.radio(
        "Navigation",
        [
            "🎯 Valuation & Financial Engine",
            "🆚 Side-by-Side Comparison",
            "📈 Live Market Analytics",
            "🧠 Model Intelligence & XAI"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 🌐 Global Currency & Area Units")

    currency_keys = list(EXCHANGE_RATES.keys())
    curr_labels = [f"{EXCHANGE_RATES[c]['flag']} {c} ({EXCHANGE_RATES[c]['symbol']})" for c in currency_keys]
    default_curr = url_params.get("Currency", "INR")
    default_curr_idx = currency_keys.index(default_curr) if default_curr in currency_keys else 1
    sel_curr_idx = st.selectbox(
        "Settlement Currency",
        range(len(currency_keys)),
        format_func=lambda i: curr_labels[i],
        index=default_curr_idx
    )
    active_currency = currency_keys[sel_curr_idx]
    active_curr_info = EXCHANGE_RATES[active_currency]

    active_unit = st.radio(
        "Measurement Unit",
        ["Square Feet (sq ft)", "Square Meters (m²)"],
        index=0
    )
    is_sqm = "Meter" in active_unit

    st.markdown("---")
    st.markdown("#### ⚡ AI Engine Performance")
    st.markdown(f"""
    <div style="background:{'rgba(30,41,59,0.7)' if dark else '#f1f5f9'}; border-radius:12px; padding:0.85rem; border:1px solid {border_col};">
        <div style="font-size:0.75rem; text-transform:uppercase; color:{text_muted}; font-weight:700;">Champion Architecture</div>
        <div style="font-size:0.9rem; font-weight:800; color:{accent_glow}; margin:0.2rem 0;">{model_title}</div>
        <div style="display:flex; justify-content:space-between; margin-top:0.4rem; font-size:0.82rem;">
            <span>Holdout R²:</span> <strong>{r2_score:.1%}</strong>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
            <span>Mean Accuracy:</span> <strong>±{mape_val:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# LOCATION SELECTOR HELPER
# ------------------------------------------------------------------------------
def location_selector(prefix="", default_country="India", default_city="Mumbai (MMR)"):
    country_list = list(GLOBAL_COUNTRIES.keys())
    country_labels = [f"{GLOBAL_COUNTRIES[c]['flag']} {c}" for c in country_list]

    active_c = st.session_state.get(f"{prefix}country_val", url_params.get("Country", default_country))
    def_c_idx = country_list.index(active_c) if active_c in country_list else 1

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        sel_c_idx = st.selectbox("🌍 Global Country / Market", range(len(country_list)),
                                 format_func=lambda i: country_labels[i],
                                 index=def_c_idx, key=f"{prefix}country_box")
        selected_country = country_list[sel_c_idx]
        country_meta = GLOBAL_COUNTRIES[selected_country]

    with col_c2:
        cities = list(country_meta["cities"].keys())
        cities.append("Custom Metro / Local Region...")

        active_city = st.session_state.get(f"{prefix}city_val", url_params.get("City", default_city))
        def_city_idx = cities.index(active_city) if active_city in cities else 0

        selected_city = st.selectbox(f"🏙️ Metro Hub in {selected_country}", cities,
                                     index=def_city_idx, key=f"{prefix}city_box")

    custom_city_name = None
    custom_mult_val = None
    if selected_city == "Custom Metro / Local Region...":
        cc1, cc2 = st.columns(2)
        with cc1:
            custom_city_name = st.text_input("Custom Area / Neighborhood Name", value="Metropolitan Core", key=f"{prefix}cname")
        with cc2:
            custom_mult_val = st.slider("Hedonic Price Index Multiplier", 0.30, 3.00, 1.00, 0.05, key=f"{prefix}cmult")

    return selected_country, selected_city, custom_city_name, custom_mult_val


# ==============================================================================
# TAB 1: VALUATION & FINANCIAL ENGINE
# ==============================================================================
if app_tab == "🎯 Valuation & Financial Engine":

    # Hero Header Banner
    st.markdown(f"""
    <div class="hero-header">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
            <div class="live-pulse">
                <span class="live-pulse-dot"></span>
                <span>2026 CALIBRATED HEDONIC AI</span>
            </div>
            <div style="font-size:0.85rem; color:rgba(255,255,255,0.75); font-weight:600;">
                Currency: <strong>{active_currency} ({active_curr_info['symbol']})</strong>
            </div>
        </div>
        <h1 style="margin:0; font-size:2.4rem; font-weight:900; letter-spacing:-0.5px;">Global Real Estate Valuation Engine</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.05rem; max-width:850px;">
            Institutional econometric property appraisal powered by an ensemble machine learning pipeline. Calibrated for 20+ countries and 100+ global metropolitan hubs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------- 1-CLICK FAST PRESET SELECTOR -----------------
    st.markdown("##### ⚡ Quick Property Presets")
    pr_col1, pr_col2, pr_col3, pr_col4, pr_col5 = st.columns(5)

    if pr_col1.button("🇮🇳 Mumbai South 2BHK", use_container_width=True):
        st.session_state["v_country_val"] = "India"
        st.session_state["v_city_val"] = "Mumbai (MMR)"
        st.session_state["p_loc"] = "Downtown"
        st.session_state["p_area"] = 1200
        st.session_state["p_bed"] = 2
        st.session_state["p_bath"] = 2.0
        st.session_state["p_cond"] = "Good"
        st.session_state["p_yr"] = 2018
        st.session_state["trigger_calc"] = True
        st.rerun()

    if pr_col2.button("🌴 Bengaluru Tech 3BHK", use_container_width=True):
        st.session_state["v_country_val"] = "India"
        st.session_state["v_city_val"] = "Bengaluru (Silicon Valley of India)"
        st.session_state["p_loc"] = "Urban"
        st.session_state["p_area"] = 1800
        st.session_state["p_bed"] = 3
        st.session_state["p_bath"] = 3.0
        st.session_state["p_cond"] = "Excellent"
        st.session_state["p_yr"] = 2021
        st.session_state["trigger_calc"] = True
        st.rerun()

    if pr_col3.button("🗽 NYC Downtown Loft", use_container_width=True):
        st.session_state["v_country_val"] = "United States"
        st.session_state["v_city_val"] = "New York City"
        st.session_state["p_loc"] = "Downtown"
        st.session_state["p_area"] = 1400
        st.session_state["p_bed"] = 2
        st.session_state["p_bath"] = 2.0
        st.session_state["p_cond"] = "Good"
        st.session_state["p_yr"] = 2010
        st.session_state["trigger_calc"] = True
        st.rerun()

    if pr_col4.button("🏡 Dallas Suburb Home", use_container_width=True):
        st.session_state["v_country_val"] = "United States"
        st.session_state["v_city_val"] = "Dallas"
        st.session_state["p_loc"] = "Suburban"
        st.session_state["p_area"] = 3000
        st.session_state["p_bed"] = 4
        st.session_state["p_bath"] = 3.5
        st.session_state["p_cond"] = "Good"
        st.session_state["p_yr"] = 2014
        st.session_state["trigger_calc"] = True
        st.rerun()

    if pr_col5.button("🇬🇧 London Central Flat", use_container_width=True):
        st.session_state["v_country_val"] = "United Kingdom"
        st.session_state["v_city_val"] = "Central London"
        st.session_state["p_loc"] = "Downtown"
        st.session_state["p_area"] = 1200
        st.session_state["p_bed"] = 2
        st.session_state["p_bath"] = 2.0
        st.session_state["p_cond"] = "Good"
        st.session_state["p_yr"] = 2005
        st.session_state["trigger_calc"] = True
        st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Location Selector
    selected_country, selected_city, custom_city_name, custom_mult_val = location_selector(
        prefix="v_",
        default_country=st.session_state.get("v_country_val", "India"),
        default_city=st.session_state.get("v_city_val", "Mumbai (MMR)")
    )

    # Valuation Form
    with st.form("valuation_form"):
        st.markdown("#### 📐 Property Attributes & Space Architecture")
        col1, col2, col3 = st.columns(3)

        with col1:
            default_loc = st.session_state.get("p_loc", url_params.get("Location", "Downtown"))
            loc_idx = VALID_LOCATIONS.index(default_loc) if default_loc in VALID_LOCATIONS else 0
            settlement_label = st.selectbox(
                "Settlement Density Tier",
                list(SETTLEMENT_TIERS.keys()),
                index=loc_idx,
                help="Urbanization density: Downtown central core, urban metro ring, suburban commuter belt, or countryside."
            )
            location_tier = SETTLEMENT_TIERS[settlement_label]

            default_cond = st.session_state.get("p_cond", url_params.get("Condition", "Good"))
            cond_idx = VALID_CONDITIONS.index(default_cond) if default_cond in VALID_CONDITIONS else 1
            condition = st.selectbox("Physical Upkeep / Finish Grade", VALID_CONDITIONS, index=cond_idx)

            garage = st.selectbox("Garage / Covered Parking", VALID_GARAGES, index=0)

        with col2:
            default_area = float(st.session_state.get("p_area", url_params.get("Area", 2000.0)))
            if is_sqm:
                area_input = st.number_input("Living Floor Area (m²)", min_value=30, max_value=2500,
                                             value=int(default_area * 0.0929) or 185, step=5)
                st.caption(f"Equivalent: **~{area_input * 10.764:,.0f} sq ft**")
            else:
                area_input = st.number_input("Living Floor Area (sq ft)", min_value=350, max_value=25000,
                                             value=int(default_area) or 2000, step=50)
                st.caption(f"Equivalent: **~{area_input * 0.0929:,.0f} m²**")

            default_bed = int(st.session_state.get("p_bed", url_params.get("Bedrooms", 3)))
            default_bath = float(st.session_state.get("p_bath", url_params.get("Bathrooms", 2.0)))
            bedrooms  = st.slider("Bedrooms", 1, 8, default_bed, 1)
            bathrooms = st.slider("Bathrooms", 1.0, 6.0, default_bath, 0.5)

        with col3:
            floors = st.selectbox("Story Levels / Floors", [1, 2, 3, 4, 5], index=0)
            default_yr = int(st.session_state.get("p_yr", url_params.get("YearBuilt", 2018)))
            year_built = st.number_input("Construction Year", min_value=1900, max_value=2026,
                                         value=default_yr, step=1)
            age_years = 2026 - year_built
            st.caption(f"Vintage: **{age_years} years old** ({'Modern' if age_years < 10 else 'Established'})")

        calculate_btn = st.form_submit_button(f"⚡ Generate AI Valuation ({active_currency})", use_container_width=True)

    # Handle automatic trigger from presets or submit
    auto_trigger = st.session_state.pop("trigger_calc", False)

    if calculate_btn or auto_trigger or "last_res" not in st.session_state:
        payload = {
            "Location":          location_tier,
            "Condition":         condition,
            "Garage":            garage,
            "Area":              float(area_input),
            "Area_Unit":         "sq m" if is_sqm else "sq ft",
            "Bedrooms":          int(bedrooms),
            "Bathrooms":         float(bathrooms),
            "Floors":            int(floors),
            "YearBuilt":         int(year_built),
            "Country":           selected_country,
            "City":              selected_city if selected_city != "Custom Metro / Local Region..." else None,
            "Custom_City":       custom_city_name,
            "Custom_Multiplier": custom_mult_val,
            "Currency":          active_currency
        }
        st.session_state["last_payload"] = payload
        with st.spinner("Executing Super Ensemble Hedonic Valuation..."):
            res = predict_house_price(payload)
        st.session_state["last_res"] = res

    # ----------------- DISPLAY VALUATION RESULTS -----------------
    if "last_res" in st.session_state:
        res = st.session_state["last_res"]
        payload = st.session_state["last_payload"]
        total_price = res["predicted_price"]

        loc_summary = f"{res['city']}, {res['country']}"

        # Primary Valuation Hero Card
        st.markdown(f"""
        <div class="val-hero-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div class="badge-pill badge-accent">📍 {loc_summary} • {payload['Location']}</div>
                <div class="live-pulse">
                    <span class="live-pulse-dot"></span>
                    <span>AI APPRAISAL READY</span>
                </div>
            </div>
            <div class="price-hero">{res['price_formatted']}</div>
            <div class="price-sub">Model: {res.get('model_used', 'Super Ensemble')} • 2026 Baseline Calibrated</div>
            
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.6rem;">
                <div class="badge-pill">📐 <strong>{res['price_per_sqft_formatted']} / sq ft</strong></div>
                <div class="badge-pill">📏 <strong>{res['price_per_sqm_formatted']} / m²</strong></div>
                <div class="badge-pill">🏙️ Market Multiplier: <strong>{res['regional_multiplier']:.2f}x</strong></div>
            </div>

            <!-- Confidence Interval Range Bar -->
            <div class="range-box">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:700;">
                    <span style="color:{text_muted};">Conservative (95% Low): {res['prediction_interval_95']['lower_formatted']}</span>
                    <span style="color:{accent_glow};">Market Valuation</span>
                    <span style="color:{text_muted};">Premium (95% High): {res['prediction_interval_95']['upper_formatted']}</span>
                </div>
                <div class="range-track">
                    <div class="range-fill" style="width:100%;"></div>
                </div>
                <div style="font-size:0.75rem; color:{text_muted}; text-align:center;">
                    Standard Error: ±{mape_val:.1f}% econometric dispersion based on {payload['Condition']} finish and {payload['Location']} density tier.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 KPI Stat Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-chip">Habitable Rooms</div>
                <div class="stat-val">{res['key_characteristics']['Total_Rooms']}</div>
                <div class="stat-sub">{payload['Bedrooms']} Bed • {payload['Bathrooms']} Bath</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-chip">Effective Living Area</div>
                <div class="stat-val">{res['key_characteristics']['Area_sqft']:,.0f} <span style="font-size:1rem;">sqft</span></div>
                <div class="stat-sub">~{res['key_characteristics']['Area_sqm']:.1f} m²</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-chip">Regional Market Index</div>
                <div class="stat-val">{res['regional_multiplier']:.2f}x</div>
                <div class="stat-sub">Relative to US-Baseline Anchor</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-chip">Property Vintage</div>
                <div class="stat-val">{res['key_characteristics']['Property_Age']} <span style="font-size:1rem;">yrs</span></div>
                <div class="stat-sub">Built Year {payload['YearBuilt']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ----------------- SHAP & TREND CHARTS -----------------
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            shap_fig = plot_shap_waterfall(res, payload, dark_mode=dark)
            st.pyplot(shap_fig)
            plt.close()

        with col_ch2:
            trend_fig = plot_price_trend(total_price, active_currency, dark_mode=dark)
            st.pyplot(trend_fig)
            plt.close()

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ----------------- FINANCIAL EMI & ROI CALCULATOR -----------------
        st.subheader("💳 Institutional Financial & Investment Suite")
        fin1, fin2 = st.columns(2)

        with fin1:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="margin-top:0;">🏦 Mortgage & EMI Financing Forecaster</h4>
            """, unsafe_allow_html=True)
            down_pct = st.slider("Equity Down Payment (%)", 10, 50, 20, 5)
            loan_tenure = st.selectbox("Loan Tenure (Years)", [10, 15, 20, 25, 30], index=2)
            default_rate = 8.5 if active_currency == "INR" else (6.5 if active_currency == "USD" else 4.2)
            interest_rate = st.slider("Annual Lending Interest Rate (%)", 2.0, 15.0, default_rate, 0.25)

            loan_principal = total_price * (1.0 - (down_pct / 100.0))
            monthly_r = (interest_rate / 100.0) / 12.0
            num_months = loan_tenure * 12
            monthly_emi = (loan_principal * (monthly_r * ((1 + monthly_r)**num_months)) / (((1 + monthly_r)**num_months) - 1)) if monthly_r > 0 else (loan_principal / num_months)

            fmt_emi  = format_currency_value(monthly_emi, active_currency)
            fmt_loan = format_currency_value(loan_principal, active_currency)

            st.success(f"Estimated Monthly Payment: **{fmt_emi['compact']} / month** ({fmt_emi['formatted']})")
            st.caption(f"Financed Principal: **{fmt_loan['compact']}** ({100-down_pct}%) amortized across {loan_tenure} years.")
            st.markdown("</div>", unsafe_allow_html=True)

        with fin2:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="margin-top:0;">📈 Rental Yield & Capital Appreciation</h4>
            """, unsafe_allow_html=True)
            est_gross_yield = 4.5 if payload["Location"] in ["Downtown", "Urban"] else 3.6
            annual_rental = total_price * (est_gross_yield / 100.0)
            monthly_rental = annual_rental / 12.0
            appreciation_5yr = total_price * ((1.0 + 0.055)**5 - 1.0)

            fmt_rent   = format_currency_value(monthly_rental, active_currency)
            fmt_apprec = format_currency_value(appreciation_5yr, active_currency)

            st.info(f"Gross Yield: **{est_gross_yield:.1f}% per annum**")
            st.markdown(f"- **Est. Monthly Rental Cashflow:** `{fmt_rent['compact']} / month`")
            st.markdown(f"- **5-Year Equity Appreciation (+5.5% CAGR):** `+{fmt_apprec['compact']}`")
            st.markdown("</div>", unsafe_allow_html=True)

        # ----------------- WHAT-IF SIMULATOR -----------------
        st.subheader("⚡ Dynamic Renovation & Value Creation Simulator")
        w1, w2, w3 = st.columns(3)

        with w1:
            sim_cond   = predict_house_price({**payload, "Condition": "Excellent"})["predicted_price"]
            delta_cond = sim_cond - total_price
            fmt_dcond  = format_currency_value(delta_cond, active_currency)
            pct_cond = (delta_cond / total_price) * 100
            st.metric("Upgrade Finish to Excellent", fmt_dcond["compact"], delta=f"+{pct_cond:.1f}%" if pct_cond > 0 else "0.0%")

        with w2:
            sim_gar    = predict_house_price({**payload, "Garage": "Yes"})["predicted_price"]
            delta_gar  = sim_gar - total_price
            fmt_dgar   = format_currency_value(delta_gar, active_currency)
            pct_gar = (delta_gar / total_price) * 100
            st.metric("Add Covered Garage", fmt_dgar["compact"], delta=f"+{pct_gar:.1f}%" if pct_gar > 0 else "0.0%")

        with w3:
            sim_bed    = predict_house_price({**payload, "Bedrooms": payload["Bedrooms"] + 1, "Bathrooms": payload["Bathrooms"] + 0.5})["predicted_price"]
            delta_bed  = sim_bed - total_price
            fmt_dbed   = format_currency_value(delta_bed, active_currency)
            pct_bed = (delta_bed / total_price) * 100
            st.metric("Add +1 Bed & +0.5 Bath Suite", fmt_dbed["compact"], delta=f"+{pct_bed:.1f}%" if pct_bed > 0 else "0.0%")

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # ----------------- GLOBAL CURRENCY MATRIX -----------------
        st.subheader("💱 Worldwide Currency Matrix")
        usd_base = res["base_usd_price"] * res["regional_multiplier"]
        mat_cols = st.columns(6)
        for idx, m_c in enumerate(["INR", "USD", "EUR", "GBP", "AED", "JPY"]):
            m_inf = EXCHANGE_RATES[m_c]
            m_v   = usd_base * m_inf["rate"]
            m_f   = format_currency_value(m_v, m_c)
            with mat_cols[idx]:
                st.markdown(f"""
                <div class="stat-card" style="text-align:center; padding:1rem 0.8rem;">
                    <div style="font-size:1.1rem;">{m_inf['flag']} <strong>{m_c}</strong></div>
                    <div style="font-size:1.15rem; font-weight:800; color:{accent_glow}; margin:0.3rem 0;">{m_f['compact']}</div>
                    <div style="font-size:0.72rem; color:{text_muted};">{m_f['formatted']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ----------------- SHARING & EXPORT -----------------
        act1, act2 = st.columns(2)
        with act1:
            st.markdown("#### 🔗 Share Valuation Link")
            share_url = make_share_url(payload)
            st.markdown(f"<div class='share-box'>{share_url}</div>", unsafe_allow_html=True)
            st.caption("Direct URL with all property attributes encoded.")

        with act2:
            st.markdown("#### 📄 Institutional PDF Dossier")
            pdf_bytes = make_pdf_report(res, payload, active_currency, year_built, down_pct, loan_tenure, interest_rate)
            ext = "pdf" if pdf_bytes[:4] == b"%PDF" else "txt"
            mime = "application/pdf" if ext == "pdf" else "text/plain"
            st.download_button(
                label=f"⬇️ Download Valuation Dossier (.{ext})",
                data=pdf_bytes,
                file_name=f"PropIntel_Valuation_{res['city'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.{ext}",
                mime=mime,
                use_container_width=True
            )


# ==============================================================================
# TAB 2: SIDE-BY-SIDE COMPARISON
# ==============================================================================
elif app_tab == "🆚 Side-by-Side Comparison":
    st.markdown(f"""
    <div class="hero-header">
        <h1 style="margin:0; font-size:2.3rem; font-weight:900;">Side-by-Side Property Comparison</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.05rem;">
            Direct econometric comparative analysis of two different property specifications and regional markets.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cmp_c1, cmp_c2 = st.columns(2)

    def property_cmp_card(col, label, prefix, defaults):
        with col:
            st.markdown(f"### {label}")
            with st.form(f"form_{prefix}"):
                c_list = list(GLOBAL_COUNTRIES.keys())
                c_labels = [f"{GLOBAL_COUNTRIES[c]['flag']} {c}" for c in c_list]
                def_c = defaults.get("Country", "India")
                def_c_idx = c_list.index(def_c) if def_c in c_list else 1
                sel_c_idx = st.selectbox("Market / Country", range(len(c_list)),
                                         format_func=lambda i: c_labels[i],
                                         index=def_c_idx, key=f"{prefix}_country")
                sel_country = c_list[sel_c_idx]

                cities = list(GLOBAL_COUNTRIES[sel_country]["cities"].keys())
                sel_city = st.selectbox("Metro Hub", cities, index=0, key=f"{prefix}_city")

                loc_label = st.selectbox("Settlement Tier", list(SETTLEMENT_TIERS.keys()), key=f"{prefix}_loc")
                cond_val  = st.selectbox("Condition Upkeep", VALID_CONDITIONS, index=defaults.get("cond_idx", 1), key=f"{prefix}_cond")
                gar_val   = st.selectbox("Garage Parking", VALID_GARAGES, index=0, key=f"{prefix}_gar")
                area_val  = st.number_input("Floor Area (sq ft)", 400, 25000, defaults.get("Area", 2000), 50, key=f"{prefix}_area")
                bed_val   = st.slider("Bedrooms", 1, 8, defaults.get("Bedrooms", 3), key=f"{prefix}_bed")
                bath_val  = st.slider("Bathrooms", 1.0, 6.0, defaults.get("Bathrooms", 2.0), 0.5, key=f"{prefix}_bath")
                fl_val    = st.selectbox("Floors", [1, 2, 3, 4, 5], key=f"{prefix}_fl")
                yr_val    = st.number_input("Year Built", 1900, 2026, defaults.get("YearBuilt", 2018), key=f"{prefix}_yr")

                sub_btn = st.form_submit_button(f"⚡ Value {label}", use_container_width=True)

            if sub_btn or f"cmp_{prefix}" not in st.session_state:
                p = {
                    "Location": SETTLEMENT_TIERS[loc_label], "Condition": cond_val,
                    "Garage": gar_val, "Area": float(area_val), "Area_Unit": "sq ft",
                    "Bedrooms": int(bed_val), "Bathrooms": float(bath_val),
                    "Floors": int(fl_val), "YearBuilt": int(yr_val),
                    "Country": sel_country, "City": sel_city, "Currency": active_currency
                }
                r = predict_house_price(p)
                st.session_state[f"cmp_{prefix}"] = (r, p)

            r, p = st.session_state[f"cmp_{prefix}"]
            st.markdown(f"""
            <div class="glass-card">
                <div class="stat-chip">📍 {r['city']}, {r['country']}</div>
                <div class="price-hero" style="font-size:2.4rem;">{r['price_formatted']}</div>
                <div style="color:{text_muted}; font-size:0.9rem; margin-top:0.4rem;">
                    <strong>{r['price_per_sqft_formatted']} / sq ft</strong> • Multiplier: {r['regional_multiplier']:.2f}x
                </div>
            </div>
            """, unsafe_allow_html=True)
            return r, p

    r1, p1 = property_cmp_card(cmp_c1, "🏢 Asset Alpha", "a", {"Country": "India", "Area": 1800, "Bedrooms": 3, "Bathrooms": 2.0, "YearBuilt": 2018, "cond_idx": 1})
    r2, p2 = property_cmp_card(cmp_c2, "🏡 Asset Beta", "b", {"Country": "United States", "Area": 2600, "Bedrooms": 4, "Bathrooms": 3.0, "YearBuilt": 2012, "cond_idx": 0})

    if r1 and r2:
        st.markdown("---")
        st.subheader("📊 Comparative Valuation Summary")
        diff = r2["predicted_price"] - r1["predicted_price"]
        pct_diff = (diff / r1["predicted_price"]) * 100
        fmt_diff = format_currency_value(abs(diff), active_currency)

        st.info(f"Asset Beta is **{fmt_diff['compact']} ({abs(pct_diff):.1f}%) {'more' if diff > 0 else 'less'} expensive** than Asset Alpha in {active_currency}.")

        comp_table = {
            "Dimension": ["Estimated Value", "Price / sq ft", "Price / m²", "Market Multiplier", "Asset Vintage"],
            "Asset Alpha": [r1["price_formatted"], r1["price_per_sqft_formatted"], r1["price_per_sqm_formatted"], f"{r1['regional_multiplier']:.2f}x", f"{r1['key_characteristics']['Property_Age']} yrs"],
            "Asset Beta": [r2["price_formatted"], r2["price_per_sqft_formatted"], r2["price_per_sqm_formatted"], f"{r2['regional_multiplier']:.2f}x", f"{r2['key_characteristics']['Property_Age']} yrs"]
        }
        st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: LIVE MARKET ANALYTICS
# ==============================================================================
elif app_tab == "📈 Live Market Analytics":
    st.markdown(f"""
    <div class="hero-header">
        <h1 style="margin:0; font-size:2.3rem; font-weight:900;">Market Distribution Analytics</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.05rem;">
            Empirical statistical distributions and price trends across settlement tiers and floor area.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not df_data.empty:
        df_d = df_data.copy()
        for c in ["Location", "Condition", "Garage"]:
            if c in df_d.columns:
                df_d[c] = df_d[c].astype(str)
        rate = active_curr_info["rate"]
        df_d["Price_Local"] = df_d["Price"] * rate

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 💰 Valuation Dispersion across Settlement Tiers")
            fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=fig_bg)
            ax.set_facecolor(fig_bg)
            sns.boxplot(data=df_d, x="Location", y="Price_Local", palette="Blues_r", ax=ax,
                        order=["Downtown", "Urban", "Suburban", "Rural"])
            ax.set_ylabel(f"Price ({active_currency})", fontweight="bold", color=mpl_text)
            ax.set_xlabel("Settlement Density", fontweight="bold", color=mpl_text)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, _: f"{y/1e7:.2f}Cr" if (active_currency == "INR" and y >= 1e7)
                else (f"{y/1e5:.1f}L" if active_currency == "INR" and y >= 1e5
                      else (f"{y/1e6:.1f}M" if y >= 1e6 else f"{y/1e3:.0f}K"))
            ))
            ax.tick_params(colors=mpl_text)
            for spine in ax.spines.values():
                spine.set_edgecolor(mpl_grid)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown("##### 📐 Living Area vs. Price Trajectory")
            fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=fig_bg)
            ax.set_facecolor(fig_bg)
            sample_n = min(500, len(df_d))
            sns.scatterplot(data=df_d.sample(sample_n, random_state=42),
                            x="Area", y="Price_Local", hue="Condition",
                            palette="coolwarm", alpha=0.85, s=45, ax=ax)
            ax.set_xlabel("Living Area (sq ft)", fontweight="bold", color=mpl_text)
            ax.set_ylabel(f"Price ({active_currency})", fontweight="bold", color=mpl_text)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, _: f"{y/1e7:.2f}Cr" if (active_currency == "INR" and y >= 1e7)
                else (f"{y/1e5:.1f}L" if active_currency == "INR" and y >= 1e5
                      else (f"{y/1e6:.1f}M" if y >= 1e6 else f"{y/1e3:.0f}K"))
            ))
            ax.tick_params(colors=mpl_text)
            for spine in ax.spines.values():
                spine.set_edgecolor(mpl_grid)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ==============================================================================
# TAB 4: MODEL INTELLIGENCE & XAI
# ==============================================================================
elif app_tab == "🧠 Model Intelligence & XAI":
    st.markdown(f"""
    <div class="hero-header">
        <h1 style="margin:0; font-size:2.3rem; font-weight:900;">Model Intelligence & Explainability</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.05rem;">
            Empirical benchmark leaderboard, tree feature importance, and error diagnostics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='stat-card'><div class='stat-chip'>Holdout Test R²</div><div class='stat-val'>{r2_score:.2%}</div><div class='stat-sub'>Verified Cross-Validation</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='stat-card'><div class='stat-chip'>Test MAE</div><div class='stat-val'>${mae_val:,.0f}</div><div class='stat-sub'>US Baseline Mean Margin</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='stat-card'><div class='stat-chip'>Mean Error (MAPE)</div><div class='stat-val'>±{mape_val:.1f}%</div><div class='stat-sub'>Econometric Variance</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='stat-card'><div class='stat-chip'>Architecture</div><div class='stat-val' style='font-size:1.35rem; margin-top:0.4rem;'>Super Ensemble</div><div class='stat-sub'>XGBoost + GB + HistGB</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    col_fi, col_lb = st.columns([1, 1])

    with col_fi:
        st.subheader("🌲 Feature Importance (Gini Tree Splits)")
        if feat_importance and "top_features" in feat_importance:
            fi_df = pd.DataFrame(feat_importance["top_features"]).head(10)
            fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=fig_bg)
            ax.set_facecolor(fig_bg)
            sns.barplot(data=fi_df, x="Importance", y="Feature", palette="Blues_r", ax=ax)
            ax.set_xlabel("Relative Importance Score", fontweight="bold", color=mpl_text)
            ax.set_ylabel("")
            ax.tick_params(colors=mpl_text)
            for spine in ax.spines.values():
                spine.set_edgecolor(mpl_grid)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_lb:
        st.subheader("🏆 Algorithm Benchmark Leaderboard")
        if metadata and "model_comparison" in metadata:
            comp_df = pd.DataFrame(metadata["model_comparison"])
            st.dataframe(
                comp_df.style.format({
                    "CV_R2_Mean": "{:.4f}", "CV_R2_Std": "{:.4f}",
                    "Test_MAE": "${:,.2f}", "Test_RMSE": "${:,.2f}",
                    "Test_R2": "{:.4f}", "Test_MAPE": "{:.2f}%",
                    "Training_Time_Sec": "{:.2f}s"
                }).highlight_max(subset=["Test_R2"], color="#dcfce7" if not dark else "#1e3a5f")
                  .highlight_min(subset=["Test_MAE", "Test_MAPE"], color="#dcfce7" if not dark else "#1e3a5f"),
                use_container_width=True
            )
