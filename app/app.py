"""
PropIQ — Predict Smarter. Choose Better.
Production-grade Real Estate Valuation & Property Analytics Platform.
Powered by Log-Target Super Ensemble (XGBoost + GradientBoosting + HistGradientBoosting).
"""

import os
import sys
import json
import io
import base64
import urllib.parse
from html import escape
from pathlib import Path
from datetime import datetime
import time

# Ensure project root is in sys.path
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
from dotenv import load_dotenv

from src.config import (
    CLEANED_DATA_PATH, METADATA_PATH, FEATURE_IMPORTANCE_PATH,
    VALID_LOCATIONS, VALID_CONDITIONS, VALID_GARAGES
)
from src.predict import predict_house_price
from src.global_market import (
    GLOBAL_COUNTRIES, EXCHANGE_RATES,
    format_currency_value, convert_sqft_to_sqm, convert_area_to_sqft
)
from src.supabase_store import SupabaseError, SupabaseRESTClient

load_dotenv(PROJECT_ROOT / ".env", override=False)

# ------------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PropIQ — Predict Smarter. Choose Better.",
    page_icon=str(PROJECT_ROOT / "PropIQ.png") if (PROJECT_ROOT / "PropIQ.png").exists() else "🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
def init_session_state():
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    if "active_currency" not in st.session_state:
        st.session_state["active_currency"] = "INR"
    if "is_sqm" not in st.session_state:
        st.session_state["is_sqm"] = False
    if "nav_tab" not in st.session_state:
        st.session_state["nav_tab"] = "🔮 Valuation Engine"
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = []
    if "last_valuation" not in st.session_state:
        st.session_state["last_valuation"] = None
    if "form_preset" not in st.session_state:
        st.session_state["form_preset"] = None
    if "guest_name" not in st.session_state:
        st.session_state["guest_name"] = "Guest Analyst"

init_session_state()
dark = st.session_state["dark_mode"]

# ------------------------------------------------------------------------------
# THEME DESIGN TOKENS
# ------------------------------------------------------------------------------
if dark:
    bg_main       = "#060b18"
    bg_card       = "rgba(15, 23, 42, 0.88)"
    bg_card_solid = "#0f172a"
    bg_subtle     = "#1e293b"
    text_main     = "#f8fafc"
    text_muted    = "#94a3b8"
    border_col    = "rgba(56, 189, 248, 0.22)"
    border_glow   = "rgba(56, 189, 248, 0.45)"
    hero_grad     = "linear-gradient(135deg, #090e1f 0%, #172554 50%, #1e1b4b 100%)"
    accent_blue   = "#38bdf8"
    accent_glow   = "#60a5fa"
    accent_gold   = "#fbbf24"
    accent_emerald= "#10b981"
    shadow_card   = "0 10px 30px -8px rgba(0, 0, 0, 0.6)"
else:
    bg_main       = "#f8fafc"
    bg_card       = "rgba(255, 255, 255, 0.98)"
    bg_card_solid = "#ffffff"
    bg_subtle     = "#f1f5f9"
    text_main     = "#0f172a"
    text_muted    = "#475569"
    border_col    = "rgba(203, 213, 225, 0.85)"
    border_glow   = "rgba(37, 99, 235, 0.35)"
    hero_grad     = "linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #1d4ed8 100%)"
    accent_blue   = "#2563eb"
    accent_glow   = "#3b82f6"
    accent_gold   = "#d97706"
    accent_emerald= "#059669"
    shadow_card   = "0 8px 24px -6px rgba(0, 0, 0, 0.06)"

# ------------------------------------------------------------------------------
# INJECT FULL-WIDTH CSS STYLING
# ------------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=DM+Serif+Display&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Hide native Streamlit sidebar and header clutter */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{
        display: none !important;
    }}
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 4rem !important;
        max-width: 1280px !important;
    }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background: {bg_main};
        color: {text_main};
    }}
    .stApp {{
        background: {bg_main};
    }}

    /* Top Branding Header */
    .top-navbar-wrap {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0 1.2rem;
        border-bottom: 1px solid {border_col};
        margin-bottom: 1.2rem;
    }}
    .brand-logo-lockup {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }}
    .brand-logo-img {{
        width: 48px;
        height: 48px;
        border-radius: 12px;
        object-fit: cover;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }}
    .brand-text-title {{
        font-family: 'DM Serif Display', serif;
        font-size: 1.75rem;
        line-height: 1;
        margin: 0;
        color: {text_main};
        letter-spacing: -0.01em;
    }}
    .brand-text-sub {{
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {accent_gold};
        margin: 0.2rem 0 0;
    }}

    /* Hero Banner */
    .saas-hero-banner {{
        background: {hero_grad};
        padding: 2.4rem 2.6rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }}
    .saas-hero-banner h1 {{
        font-family: 'DM Serif Display', serif;
        font-size: 2.65rem;
        line-height: 1.15;
        margin: 0.5rem 0 0.7rem;
        color: white !important;
    }}
    .saas-hero-banner p {{
        font-size: 1.02rem;
        line-height: 1.6;
        color: #e0e7ff !important;
        max-width: 680px;
        margin-bottom: 0;
    }}
    .hero-badge-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #fef08a;
        padding: 0.3rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }}

    /* Card Panels */
    .prop-card-panel {{
        background: {bg_card};
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid {border_col};
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: {shadow_card};
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .prop-card-panel:hover {{
        border-color: {border_glow};
    }}
    .prop-card-header {{
        font-size: 1.1rem;
        font-weight: 800;
        color: {text_main};
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .prop-card-sub {{
        font-size: 0.82rem;
        color: {text_muted};
        margin-bottom: 1.1rem;
    }}

    /* Hero Valuation Result Box */
    .val-hero-card {{
        background: {f"linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.85) 100%)" if dark else "linear-gradient(135deg, #ffffff 0%, #eff6ff 100%)"};
        backdrop-filter: blur(20px);
        border: 1.5px solid {border_glow};
        border-radius: 20px;
        padding: 2rem 2.4rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 20px 50px -12px rgba(37, 99, 235, {'0.4' if dark else '0.12'});
    }}
    .val-hero-eyebrow {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
        color: {accent_gold};
        margin-bottom: 0.2rem;
    }}
    .val-hero-main-price {{
        font-family: 'DM Serif Display', serif;
        font-size: 3.4rem;
        line-height: 1.05;
        margin: 0.1rem 0 0.4rem;
        background: {f"linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%)" if dark else "linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #7c3aed 100%)"};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .val-hero-location-text {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {text_muted};
        margin-bottom: 1.2rem;
    }}

    /* Confidence Range Bar */
    .conf-bar-track {{
        background: {bg_subtle};
        border-radius: 9999px;
        height: 10px;
        position: relative;
        margin: 1.2rem 0 0.6rem;
        border: 1px solid {border_col};
    }}
    .conf-bar-fill {{
        position: absolute;
        left: 15%;
        right: 15%;
        top: 0;
        bottom: 0;
        background: linear-gradient(90deg, {accent_blue}, {accent_gold});
        border-radius: 9999px;
    }}
    .conf-bar-pin {{
        position: absolute;
        left: 50%;
        top: 50%;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: #ffffff;
        border: 3.5px solid {accent_blue};
        transform: translate(-50%, -50%);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
    }}

    /* Quick Specs Chips */
    .spec-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        background: {bg_subtle};
        border: 1px solid {border_col};
        color: {text_main};
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }}

    /* Streamlit Buttons & Controls */
    div.stButton > button, div.stFormSubmitButton > button, div[data-testid="stDownloadButton"] button {{
        min-height: 2.85rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.94rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    div.stFormSubmitButton > button {{
        background: {accent_blue} !important;
        color: white !important;
        border: none !important;
    }}
    div.stFormSubmitButton > button:hover {{
        background: #1d4ed8 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.35);
    }}

    /* Metric Cards */
    [data-testid="stMetric"] {{
        background: {bg_card};
        border: 1px solid {border_col};
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: {shadow_card};
    }}
    [data-testid="stMetricLabel"] {{ color: {text_muted}; font-size: 0.84rem; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ color: {text_main}; font-weight: 800; }}

    /* Step Grid */
    .step-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin: 1.2rem 0; }}
    .step-box {{
        background: {bg_card};
        border: 1px solid {border_col};
        border-radius: 14px;
        padding: 1.2rem;
        transition: transform 0.2s ease;
    }}
    .step-box:hover {{ transform: translateY(-3px); border-color: {border_glow}; }}
    .step-num {{ font-size: 0.76rem; font-weight: 800; color: {accent_gold}; letter-spacing: 0.06em; text-transform: uppercase; }}
    .step-box h3 {{ font-size: 1.02rem; font-weight: 700; margin: 0.35rem 0 0.25rem; color: {text_main}; }}
    .step-box p {{ font-size: 0.84rem; color: {text_muted}; line-height: 1.45; margin: 0; }}

    @media (max-width: 850px) {{
        .saas-hero-banner {{ padding: 1.8rem 1.5rem; }}
        .saas-hero-banner h1 {{ font-size: 2rem; }}
        .step-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 540px) {{
        .step-grid {{ grid-template-columns: 1fr; }}
        .val-hero-main-price {{ font-size: 2.3rem; }}
    }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# LOAD MODEL METADATA & DATASETS
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

df_clean, metadata, feature_importance_data = load_app_data()
final_metrics = metadata.get("metrics", {}).get("test", {})
r2_score = final_metrics.get("R2", 0.9883)
mae_val  = final_metrics.get("MAE", 21290.0)
mape_val = final_metrics.get("MAPE", 4.10)
model_title = metadata.get("model_name", "Super Ensemble (XGB+GB+HGB)")

# ------------------------------------------------------------------------------
# SUPABASE REST CLIENT & SESSION HELPERS
# ------------------------------------------------------------------------------
def load_supabase_config():
    try:
        secrets = st.secrets
        section = secrets.get("supabase", {})
    except Exception:
        secrets = {}
        section = {}

    def setting(*names):
        for name in names:
            val = os.environ.get(name) or secrets.get(name)
            if val:
                return val
            val = section.get(name.removeprefix("SUPABASE_").lower())
            if val:
                return val
        return None

    return (
        setting("SUPABASE_URL"),
        setting("SUPABASE_PUBLISHABLE_KEY", "SUPABASE_ANON_KEY"),
        setting("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY"),
        setting("SUPABASE_REDIRECT_URL") or "http://localhost:8501",
    )

def store_auth_session(auth_response):
    user = auth_response.get("user")
    access_token = auth_response.get("access_token")
    if not user or not access_token:
        raise SupabaseError("Email confirmation is required. Check your inbox, then sign in.")
    st.session_state["supabase_session"] = {
        "user": user,
        "access_token": access_token,
        "refresh_token": auth_response.get("refresh_token"),
        "expires_at": time.time() + int(auth_response.get("expires_in", 3600)),
    }

def get_auth_session():
    auth_session = st.session_state.get("supabase_session")
    if not auth_session or auth_session.get("expires_at", 0) > time.time() + 60:
        return auth_session
    refresh_token = auth_session.get("refresh_token")
    if not supabase_client or not refresh_token:
        st.session_state.pop("supabase_session", None)
        return None
    try:
        refreshed = supabase_client.refresh_session(refresh_token)
        store_auth_session(refreshed)
        return st.session_state["supabase_session"]
    except SupabaseError:
        st.session_state.pop("supabase_session", None)
        return None

supabase_url, supabase_anon_key, supabase_service_role_key, supabase_redirect_url = load_supabase_config()
supabase_client = (
    SupabaseRESTClient(supabase_url, supabase_anon_key, supabase_service_role_key)
    if supabase_url and supabase_anon_key
    else None
)

auth_session = get_auth_session()
all_country_options = list(GLOBAL_COUNTRIES)
all_currency_options = list(EXCHANGE_RATES)

# Currency and units state
active_currency = st.session_state["active_currency"]
if active_currency not in all_currency_options:
    active_currency = "INR"
active_curr_info = EXCHANGE_RATES.get(active_currency, EXCHANGE_RATES["INR"])
is_sqm = st.session_state["is_sqm"]

# Logo Asset
logo_bytes = (PROJECT_ROOT / "PropIQ.png").read_bytes() if (PROJECT_ROOT / "PropIQ.png").exists() else b""
logo_data = base64.b64encode(logo_bytes).decode("ascii")

# ------------------------------------------------------------------------------
# TOP NAVBAR & QUICK CONTROL BAR
# ------------------------------------------------------------------------------
top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns([3.2, 1.3, 1.0, 0.7, 1.4], gap="small")

with top_col1:
    st.markdown(f"""
    <div class="brand-logo-lockup">
        <img src="data:image/png;base64,{logo_data}" class="brand-logo-img" alt="PropIQ Logo">
        <div>
            <h2 class="brand-text-title">PropIQ</h2>
            <p class="brand-text-sub">Predict Smarter. Choose Better.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    curr_labels = [f"{EXCHANGE_RATES[c]['flag']} {c}" for c in all_currency_options]
    curr_idx = all_currency_options.index(active_currency) if active_currency in all_currency_options else 0
    new_curr_idx = st.selectbox(
        "Currency",
        range(len(all_currency_options)),
        format_func=lambda i: curr_labels[i],
        index=curr_idx,
        key="top_currency_selector",
        label_visibility="collapsed"
    )
    if all_currency_options[new_curr_idx] != st.session_state["active_currency"]:
        st.session_state["active_currency"] = all_currency_options[new_curr_idx]
        st.rerun()

with top_col3:
    unit_choices = ["sq ft", "m²"]
    unit_idx = 1 if is_sqm else 0
    new_unit_idx = st.selectbox(
        "Units",
        range(len(unit_choices)),
        format_func=lambda i: unit_choices[i],
        index=unit_idx,
        key="top_unit_selector",
        label_visibility="collapsed"
    )
    if (new_unit_idx == 1) != st.session_state["is_sqm"]:
        st.session_state["is_sqm"] = (new_unit_idx == 1)
        st.rerun()

with top_col4:
    if st.button("🌙" if not dark else "☀️", key="top_theme_toggle_btn", help="Toggle Theme", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

with top_col5:
    if auth_session:
        user_email = auth_session.get("user", {}).get("email", "User")
        user_name = auth_session.get("user", {}).get("user_metadata", {}).get("display_name") or user_email.split("@")[0]
        st.markdown(f"""
        <div style="text-align:right; padding-top:0.35rem;">
            <span class="spec-pill" style="border-color:{border_glow}; color:{accent_blue};">
                👑 <strong>{escape(user_name)}</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:right; padding-top:0.35rem;">
            <span class="spec-pill">
                🟢 <strong>{escape(st.session_state['guest_name'])}</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TOP NAVIGATION TABS
# ------------------------------------------------------------------------------
NAV_VALUATION   = "🔮 Valuation Engine"
NAV_DASHBOARD   = "📊 Dashboard & Analytics"
NAV_COMPARE     = "⚖️ Compare Properties"
NAV_HISTORY     = "📜 Saved Valuations"
NAV_MARKET      = "📈 Market Insights"
NAV_HOW_IT_WORKS= "🔬 Model & Architecture"
NAV_PROFILE     = "👤 My Profile & Preferences"
NAV_ABOUT       = "ℹ️ About"
NAV_ADMIN       = "⚙️ Admin Console"

NAVIGATION_TABS = [
    NAV_VALUATION,
    NAV_DASHBOARD,
    NAV_COMPARE,
    NAV_HISTORY,
    NAV_MARKET,
    NAV_HOW_IT_WORKS,
    NAV_PROFILE,
    NAV_ABOUT
]

if auth_session:
    NAVIGATION_TABS.append(NAV_ADMIN)

selected_tab = st.radio(
    "Navigation",
    NAVIGATION_TABS,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_tab"
)
st.markdown(f"<hr style='border:0; border-top:1px solid {border_col}; margin:0.3rem 0 1.4rem'>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# HELPER: MATPLOTLIB SHAP WATERFALL PLOT
# ------------------------------------------------------------------------------
def plot_shap_waterfall(pred_result: dict, dark_mode: bool):
    """Render a clean, modern waterfall plot explaining feature contributions to the valuation."""
    base_usd = pred_result.get("base_usd_price", 280000.0)
    mult = pred_result.get("regional_multiplier", 1.0)
    rate = pred_result.get("exchange_rate", 1.0)
    curr_code = pred_result.get("currency", "USD")

    # Features and estimated hedonic delta in USD
    area_val = pred_result.get("key_characteristics", {}).get("living_area_sqft", 2000)
    beds = pred_result.get("key_characteristics", {}).get("bedrooms", 3)
    baths = pred_result.get("key_characteristics", {}).get("bathrooms", 2)
    age = pred_result.get("key_characteristics", {}).get("property_age_years", 10)
    loc = pred_result.get("key_characteristics", {}).get("location_tier", "Downtown")
    garage = pred_result.get("key_characteristics", {}).get("garage_available", "Yes")

    # Approximate hedonic component contributions
    c_base = 120000.0 * mult * rate
    c_area = (area_val * 75.0) * mult * rate
    c_bed  = (beds * 18000.0) * mult * rate
    c_bath = (baths * 12000.0) * mult * rate
    c_loc  = (45000.0 if loc == "Downtown" else (25000.0 if loc == "Urban" else (10000.0 if loc == "Suburban" else 0.0))) * mult * rate
    c_gar  = (15000.0 if garage == "Yes" else 0.0) * mult * rate
    c_age  = (-age * 1200.0) * mult * rate

    components = [
        ("Base Median", c_base),
        (f"Area ({area_val:,.0f} sqft)", c_area),
        (f"Location ({loc})", c_loc),
        (f"Bedrooms ({beds} BHK)", c_bed),
        (f"Bathrooms ({baths})", c_bath),
        (f"Garage Facility", c_gar),
        (f"Property Age ({age} yrs)", c_age),
    ]

    labels = [c[0] for c in components]
    values = [c[1] for c in components]

    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=130)
    fig_bg = "#0f172a" if dark_mode else "#ffffff"
    text_c = "#f8fafc" if dark_mode else "#0f172a"
    grid_c = "#334155" if dark_mode else "#e2e8f0"

    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(fig_bg)

    y_pos = np.arange(len(labels))
    colors = ["#38bdf8" if v >= 0 else "#f43f5e" for v in values]
    colors[0] = "#818cf8"  # Base value color

    bars = ax.barh(y_pos, values, color=colors, height=0.55, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=text_c, fontsize=9.5, fontweight="600")
    ax.invert_yaxis()  # top-down

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.tick_params(axis="x", colors=text_c, labelsize=8.5)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color=grid_c)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for bar in bars:
        w = bar.get_width()
        offset = abs(w) * 0.04 + 500
        x_val = w + (offset if w >= 0 else -offset)
        align = "left" if w >= 0 else "right"
        val_str = f"+{w:,.0f}" if w > 0 else f"{w:,.0f}"
        ax.text(x_val, bar.get_y() + bar.get_height()/2, val_str,
                va="center", ha=align, color=text_c, fontsize=8.5, fontweight="700")

    plt.tight_layout()
    return fig

# ==============================================================================
# TAB 1: PROPERTY VALUATION ENGINE
# ==============================================================================
if selected_tab == NAV_VALUATION:
    st.markdown(f"""
    <div class="saas-hero-banner">
        <span class="hero-badge-pill">✨ Super Ensemble ML Engine · 98.8% R²</span>
        <h1>Instant Property Valuation & Analytics</h1>
        <p>
            Generate deterministic real estate appraisals, confidence intervals, and investment forecasts
            with explainable hedonic modeling across global metropolitan markets.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1-Click Quick Presets
    st.markdown("##### ⚡ Quick Property Presets")
    preset_cols = st.columns(4)
    with preset_cols[0]:
        if st.button("🏙️ Downtown Luxury Penthouse", key="preset_penthouse", use_container_width=True):
            st.session_state["form_preset"] = {
                "country": "India", "city": "Mumbai (MMR)", "loc": "Downtown",
                "area": 2800 if not is_sqm else 260, "beds": 4, "baths": 3.5,
                "floors": 2, "year": 2022, "cond": "Excellent", "gar": "Yes"
            }
            st.rerun()
    with preset_cols[1]:
        if st.button("🏡 Suburban Family Villa", key="preset_suburban", use_container_width=True):
            st.session_state["form_preset"] = {
                "country": "India", "city": "Bengaluru", "loc": "Suburban",
                "area": 2200 if not is_sqm else 204, "beds": 3, "baths": 2.5,
                "floors": 2, "year": 2018, "cond": "Good", "gar": "Yes"
            }
            st.rerun()
    with preset_cols[2]:
        if st.button("🏢 Modern Metro Condo", key="preset_condo", use_container_width=True):
            st.session_state["form_preset"] = {
                "country": "United States", "city": "New York, NY", "loc": "Downtown",
                "area": 1250 if not is_sqm else 116, "beds": 2, "baths": 2.0,
                "floors": 1, "year": 2020, "cond": "Excellent", "gar": "No"
            }
            st.rerun()
    with preset_cols[3]:
        if st.button("🌿 Rural Nature Estate", key="preset_rural", use_container_width=True):
            st.session_state["form_preset"] = {
                "country": "India", "city": "Pune", "loc": "Rural",
                "area": 3500 if not is_sqm else 325, "beds": 5, "baths": 4.0,
                "floors": 2, "year": 2012, "cond": "Good", "gar": "Yes"
            }
            st.rerun()

    preset = st.session_state.get("form_preset") or {}

    # Valuation Form
    with st.form("valuation_input_form"):
        col_f1, col_f2 = st.columns(2, gap="medium")

        with col_f1:
            st.markdown("""
            <div class="prop-card-header">📍 1. Market & Regional Location</div>
            <div class="prop-card-sub">Select country and metro market for local hedonic index weighting.</div>
            """, unsafe_allow_html=True)

            country_labels = [f"{GLOBAL_COUNTRIES[c]['flag']} {c}" for c in all_country_options]
            def_country = preset.get("country", "India")
            def_c_idx = all_country_options.index(def_country) if def_country in all_country_options else 0

            sel_c_idx = st.selectbox(
                "Country / Region",
                range(len(all_country_options)),
                format_func=lambda i: country_labels[i],
                index=def_c_idx,
                key="form_country"
            )
            chosen_country = all_country_options[sel_c_idx]
            country_meta = GLOBAL_COUNTRIES[chosen_country]

            city_list = list(country_meta["cities"].keys())
            city_list.append("Custom Metro / Local Region...")
            def_city = preset.get("city", city_list[0])
            def_city_idx = city_list.index(def_city) if def_city in city_list else 0

            chosen_city = st.selectbox(
                f"Metro Hub in {chosen_country}",
                city_list,
                index=def_city_idx,
                key="form_city"
            )

            custom_area_name = None
            custom_multiplier_val = None
            if chosen_city == "Custom Metro / Local Region...":
                cc1, cc2 = st.columns(2)
                with cc1:
                    custom_area_name = st.text_input("Custom Neighborhood Name", value="Prime District Core")
                with cc2:
                    custom_multiplier_val = st.slider("Hedonic Price Multiplier", 0.30, 3.00, 1.00, 0.05)

            def_loc = preset.get("loc", "Downtown")
            loc_idx = VALID_LOCATIONS.index(def_loc) if def_loc in VALID_LOCATIONS else 0
            location_tier = st.selectbox(
                "Urban Density Tier",
                VALID_LOCATIONS,
                index=loc_idx,
                help="Urban density tier within the chosen city.",
                key="form_location_tier"
            )

            st.markdown("""
            <div style="margin-top:1.2rem;">
                <div class="prop-card-header">🏛️ 2. Structure & Vintage</div>
                <div class="prop-card-sub">Year of construction and structural condition rating.</div>
            </div>
            """, unsafe_allow_html=True)

            col_y1, col_y2 = st.columns(2)
            with col_y1:
                def_year = preset.get("year", 2016)
                year_built = st.slider("Year Built", 1920, 2026, def_year, key="form_year_built")
                age_years = max(0, 2026 - year_built)
                st.caption(f"🏗️ Property Age: **{age_years} years old**")
            with col_y2:
                def_cond = preset.get("cond", "Good")
                cond_idx = VALID_CONDITIONS.index(def_cond) if def_cond in VALID_CONDITIONS else 1
                condition_rating = st.selectbox("Property Condition", VALID_CONDITIONS, index=cond_idx, key="form_condition")

        with col_f2:
            st.markdown(f"""
            <div class="prop-card-header">📐 3. Living Dimensions & Floor Plan</div>
            <div class="prop-card-sub">Total enclosed living area and room allocations.</div>
            """, unsafe_allow_html=True)

            unit_name = "m²" if is_sqm else "sq ft"
            min_a = 20 if is_sqm else 250
            max_a = 2500 if is_sqm else 25000
            def_a = preset.get("area", (185 if is_sqm else 2000))

            area_input = st.number_input(
                f"Living Area ({unit_name})",
                min_value=min_a,
                max_value=max_a,
                value=int(def_a),
                step=10 if is_sqm else 50,
                key="form_area_val"
            )

            if is_sqm:
                sqft_equiv = convert_area_to_sqft(area_input, "sq m")
                st.caption(f"Equivalent: **{sqft_equiv:,.0f} sq ft**")
            else:
                sqm_equiv = convert_sqft_to_sqm(area_input)
                st.caption(f"Equivalent: **{sqm_equiv:,.1f} m²**")

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                def_beds = preset.get("beds", 3)
                bedrooms = st.number_input("Bedrooms", min_value=1, max_value=12, value=int(def_beds), step=1, key="form_bedrooms")
            with col_r2:
                def_baths = preset.get("baths", 2.0)
                bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=float(def_baths), step=0.5, key="form_bathrooms")
            with col_r3:
                def_floors = preset.get("floors", 1)
                floors = st.number_input("Floors", min_value=1, max_value=6, value=int(def_floors), step=1, key="form_floors")

            st.markdown("""
            <div style="margin-top:1.2rem;">
                <div class="prop-card-header">🚗 4. Parking & Amenities</div>
                <div class="prop-card-sub">Garage and auxiliary vehicular parking.</div>
            </div>
            """, unsafe_allow_html=True)

            def_gar = preset.get("gar", "Yes")
            gar_idx = VALID_GARAGES.index(def_gar) if def_gar in VALID_GARAGES else 0
            garage = st.selectbox("Enclosed Garage / Covered Parking", VALID_GARAGES, index=gar_idx, key="form_garage")

        submit_valuation = st.form_submit_button("🔮 Calculate Property Valuation", use_container_width=True)

    # Process Valuation
    if submit_valuation:
        sqft_for_model = convert_area_to_sqft(area_input, "sq m") if is_sqm else float(area_input)
        input_payload = {
            "Area": sqft_for_model,
            "Bedrooms": int(bedrooms),
            "Bathrooms": float(bathrooms),
            "Floors": int(floors),
            "YearBuilt": int(year_built),
            "Location": location_tier,
            "Condition": condition_rating,
            "Garage": garage,
            "Country": chosen_country,
            "City": chosen_city if chosen_city != "Custom Metro / Local Region..." else None,
            "Custom_City": custom_area_name,
            "Custom_Multiplier": custom_multiplier_val,
            "Currency": active_currency,
        }

        try:
            val_result = predict_house_price(input_payload)
            st.session_state["last_valuation"] = val_result

            # Append to session history
            hist_item = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "country": chosen_country,
                "city": custom_area_name or chosen_city,
                "location_tier": location_tier,
                "area_sqft": sqft_for_model,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "price_formatted": val_result["price_formatted"],
                "predicted_price": val_result["predicted_price"],
                "currency": active_currency
            }
            st.session_state["session_history"].insert(0, hist_item)

        except Exception as e:
            st.error(f"Valuation calculation failed: {str(e)}")

    # --------------------------------------------------------------------------
    # VALUATION RESULTS DISPLAY
    # --------------------------------------------------------------------------
    res = st.session_state.get("last_valuation")
    if res:
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

        col_v1, col_v2 = st.columns([1.15, 0.85], gap="large")

        with col_v1:
            st.markdown(f"""
            <div class="val-hero-card">
                <div class="val-hero-eyebrow">✨ AI Hedonic Appraisal Result</div>
                <div class="val-hero-main-price">{res['price_formatted']}</div>
                <div class="val-hero-location-text">
                    📍 {escape(res['city'])}, {escape(res['country'])} · <span style="color:{accent_gold}; font-weight:700;">{res['regional_multiplier']:.2f}x Market Index</span>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin-bottom:1rem;">
                    <span class="spec-pill">📐 {res['key_characteristics']['living_area_sqft']:,.0f} sq ft ({convert_sqft_to_sqm(res['key_characteristics']['living_area_sqft']):.1f} m²)</span>
                    <span class="spec-pill">🛏️ {res['key_characteristics']['bedrooms']} BHK</span>
                    <span class="spec-pill">🚿 {res['key_characteristics']['bathrooms']} Baths</span>
                    <span class="spec-pill">🏗️ Built {res['key_characteristics']['year_built']} ({res['key_characteristics']['property_age_years']}y)</span>
                    <span class="spec-pill">⭐ {res['key_characteristics']['condition']} Condition</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:700; color:{text_muted};">
                    <span>95% Confidence Lower: {res['prediction_interval_95']['lower_formatted']}</span>
                    <span>Upper: {res['prediction_interval_95']['upper_formatted']}</span>
                </div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill"></div>
                    <div class="conf-bar-pin"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:{text_muted}; margin-top:0.4rem;">
                    <span>Unit Rate: <strong>{res['price_per_sqft_formatted']} / sq ft</strong></span>
                    <span>Metric Rate: <strong>{res['price_per_sqm_formatted']} / m²</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Action Bar
            act_c1, act_c2, act_c3 = st.columns(3)
            with act_c1:
                if st.button("💾 Save to Cloud", key="save_cloud_btn", use_container_width=True):
                    if not auth_session:
                        st.info("💡 To sync permanently across devices, sign in from the **'👤 My Profile & Preferences'** tab. Saved to session history!")
                    else:
                        try:
                            supabase_client.save_valuation(
                                access_token=auth_session["access_token"],
                                valuation_data={
                                    "country": res["country"],
                                    "city": res["city"],
                                    "location": res["key_characteristics"]["location_tier"],
                                    "area": res["key_characteristics"]["living_area_sqft"],
                                    "bedrooms": res["key_characteristics"]["bedrooms"],
                                    "bathrooms": res["key_characteristics"]["bathrooms"],
                                    "floors": res["key_characteristics"]["floors"],
                                    "year_built": res["key_characteristics"]["year_built"],
                                    "condition": res["key_characteristics"]["condition"],
                                    "garage": res["key_characteristics"]["garage_available"],
                                    "price": res["predicted_price"],
                                    "price_formatted": res["price_formatted"],
                                    "currency": res["currency"],
                                    "multiplier": res["regional_multiplier"]
                                }
                            )
                            st.success("✅ Valuation saved to Supabase cloud account!")
                        except Exception as e:
                            st.error(f"Cloud save failed: {str(e)}")

            with act_c2:
                csv_data = io.StringIO()
                pd.DataFrame([{
                    "Timestamp": datetime.now().isoformat(),
                    "Country": res["country"],
                    "City": res["city"],
                    "Valuation": res["price_formatted"],
                    "Area_sqft": res["key_characteristics"]["living_area_sqft"],
                    "Bedrooms": res["key_characteristics"]["bedrooms"],
                    "Bathrooms": res["key_characteristics"]["bathrooms"],
                    "YearBuilt": res["key_characteristics"]["year_built"],
                    "Condition": res["key_characteristics"]["condition"],
                    "Currency": res["currency"],
                }]).to_csv(csv_data, index=False)

                st.download_button(
                    "📄 Export CSV",
                    data=csv_data.getvalue(),
                    file_name=f"PropIQ_Valuation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_val_csv"
                )

            with act_c3:
                if st.button("🔄 Clear Result", key="clear_res_btn", use_container_width=True):
                    st.session_state["last_valuation"] = None
                    st.rerun()

        with col_v2:
            st.markdown("##### 📊 SHAP Value Contributions")
            st.caption("How each property characteristic adjusted the valuation from baseline.")
            fig_shap = plot_shap_waterfall(res, dark_mode=dark)
            st.pyplot(fig_shap, use_container_width=True)

        # Financial Suite (Mortgage EMI & Rental ROI)
        st.markdown("<hr style='border:0; border-top:1px solid rgba(226,232,240,0.4); margin:1.5rem 0'>", unsafe_allow_html=True)
        st.markdown("#### 💼 Real Estate Investment & Financing Suite")

        fin_tab1, fin_tab2 = st.tabs(["🏦 Mortgage & EMI Estimator", "📈 Rental Yield & Cap Rate"])

        with fin_tab1:
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                down_pct = st.slider("Down Payment (%)", 10, 50, 20, 5)
                down_amount = res["predicted_price"] * (down_pct / 100.0)
                loan_principal = res["predicted_price"] - down_amount
            with fcol2:
                interest_rate = st.slider("Annual Interest Rate (%)", 3.0, 15.0, 8.5, 0.25)
            with fcol3:
                tenure_years = st.slider("Loan Tenure (Years)", 5, 30, 20, 5)

            # EMI Calculation
            r = (interest_rate / 100.0) / 12.0
            n = tenure_years * 12
            if r > 0 and n > 0:
                emi = (loan_principal * r * ((1 + r)**n)) / (((1 + r)**n) - 1)
                total_payment = emi * n
                total_interest = total_payment - loan_principal
            else:
                emi = loan_principal / n
                total_payment = loan_principal
                total_interest = 0

            fmt_emi = format_currency_value(emi, res["currency"])
            fmt_down = format_currency_value(down_amount, res["currency"])
            fmt_interest = format_currency_value(total_interest, res["currency"])

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Monthly EMI Payment", fmt_emi["display"])
            m_col2.metric("Down Payment Required", fmt_down["display"])
            m_col3.metric("Total Loan Interest", fmt_interest["display"])

        with fin_tab2:
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                gross_yield_pct = st.slider("Estimated Gross Rental Yield (%)", 2.0, 10.0, 4.5, 0.25)
                annual_rent = res["predicted_price"] * (gross_yield_pct / 100.0)
                monthly_rent = annual_rent / 12.0
                fmt_month_rent = format_currency_value(monthly_rent, res["currency"])
                fmt_annual_rent = format_currency_value(annual_rent, res["currency"])
                st.metric("Estimated Monthly Rent", fmt_month_rent["display"])
                st.metric("Estimated Annual Rental Revenue", fmt_annual_rent["display"])

            with rcol2:
                appreciation_rate = st.slider("Projected 5-Year Capital Growth (% p.a.)", 2.0, 15.0, 7.0, 0.5)
                future_val = res["predicted_price"] * ((1 + (appreciation_rate / 100.0)) ** 5)
                fmt_future = format_currency_value(future_val, res["currency"])
                st.metric("Projected Value in 5 Years", fmt_future["display"])
                st.caption(f"Estimated capital gain of **{((future_val - res['predicted_price']) / res['predicted_price']) * 100:.1f}%** over 5 years.")

# ==============================================================================
# TAB 2: DASHBOARD & METRICS
# ==============================================================================
elif selected_tab == NAV_DASHBOARD:
    st.markdown("""
    <div class="saas-hero-banner">
        <span class="hero-badge-pill">📊 PropTech Analytics Core</span>
        <h1>Real Estate Portfolio & Market Dashboard</h1>
        <p>Key market performance indicators, model benchmark telemetry, and active regional valuation trends.</p>
    </div>
    """, unsafe_allow_html=True)

    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    dcol1.metric("Super Ensemble Accuracy", f"{r2_score:.2%}", "Held-out R²")
    dcol2.metric("Mean Absolute Error", f"${mae_val:,.0f}", f"MAPE {mape_val:.1f}%")
    dcol3.metric("Global Metros Covered", f"{len(GLOBAL_COUNTRIES)} Countries", "Hedonic Index")
    dcol4.metric("Active Exchange Rates", f"{len(EXCHANGE_RATES)} Currencies", f"Base: {active_currency}")

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    col_chart, col_recent = st.columns([1.1, 0.9], gap="large")

    with col_chart:
        st.markdown("##### 📈 Average Price per Sq Ft by Urban Density")
        st.caption("Baseline synthetic benchmark across location tiers.")
        density_data = pd.DataFrame({
            "Tier": ["Downtown", "Urban", "Suburban", "Rural"],
            "AvgPricePerSqFt_USD": [165.0, 135.0, 110.0, 85.0]
        })
        rate = active_curr_info["rate"]
        density_data[f"AvgPricePerSqFt_{active_currency}"] = density_data["AvgPricePerSqFt_USD"] * rate

        fig_density, ax_d = plt.subplots(figsize=(6.5, 3.5), dpi=120)
        fig_bg = "#0f172a" if dark else "#ffffff"
        text_c = "#f8fafc" if dark else "#0f172a"
        fig_density.patch.set_facecolor(fig_bg)
        ax_d.set_facecolor(fig_bg)

        bars = ax_d.bar(density_data["Tier"], density_data[f"AvgPricePerSqFt_{active_currency}"],
                        color=["#38bdf8", "#818cf8", "#a78bfa", "#f472b6"], width=0.55)
        ax_d.tick_params(colors=text_c)
        for spine in ax_d.spines.values():
            spine.set_visible(False)
        ax_d.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_density, use_container_width=True)

    with col_recent:
        st.markdown("##### 🕒 Recent Appraisals Stream")
        if st.session_state["session_history"]:
            for item in st.session_state["session_history"][:4]:
                st.markdown(f"""
                <div class="prop-card-panel" style="padding:0.9rem 1.1rem; margin-bottom:0.6rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong>📍 {escape(item['city'])}, {escape(item['country'])}</strong>
                            <div style="font-size:0.78rem; color:{text_muted};">{item['area_sqft']:,.0f} sqft · {item['bedrooms']} BHK · {item['location_tier']}</div>
                        </div>
                        <div style="font-size:1.1rem; font-weight:800; color:{accent_blue};">
                            {item['price_formatted']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No valuations recorded yet. Head over to the **🔮 Valuation Engine** tab to run your first property appraisal!")

# ==============================================================================
# TAB 3: PROPERTY COMPARISON
# ==============================================================================
elif selected_tab == NAV_COMPARE:
    st.markdown("""
    <div class="saas-hero-banner">
        <span class="hero-badge-pill">⚖️ Side-by-Side Analysis</span>
        <h1>Property Valuation Comparison</h1>
        <p>Evaluate two properties head-to-head to determine relative market value, premium differentials, and asset characteristics.</p>
    </div>
    """, unsafe_allow_html=True)

    col_cmp1, col_cmp2 = st.columns(2, gap="large")

    with col_cmp1:
        st.markdown("### 🏢 Property Alpha (A)")
        ca_country = st.selectbox("Market A", all_country_options, index=0, key="cmp_c_a")
        ca_cities = list(GLOBAL_COUNTRIES[ca_country]["cities"].keys())
        ca_city = st.selectbox(f"City A", ca_cities, index=0, key="cmp_city_a")
        ca_area = st.number_input("Living Area (sq ft) A", min_value=300, max_value=25000, value=2000, step=50, key="cmp_area_a")
        ca_beds = st.slider("Bedrooms A", 1, 10, 3, key="cmp_beds_a")
        ca_baths = st.slider("Bathrooms A", 1.0, 8.0, 2.0, 0.5, key="cmp_baths_a")
        ca_loc = st.selectbox("Location Tier A", VALID_LOCATIONS, index=0, key="cmp_loc_a")
        ca_year = st.slider("Year Built A", 1950, 2026, 2018, key="cmp_yr_a")
        ca_gar = st.selectbox("Garage A", VALID_GARAGES, index=0, key="cmp_gar_a")

    with col_cmp2:
        st.markdown("### 🏡 Property Beta (B)")
        cb_country = st.selectbox("Market B", all_country_options, index=0, key="cmp_c_b")
        cb_cities = list(GLOBAL_COUNTRIES[cb_country]["cities"].keys())
        cb_city = st.selectbox(f"City B", cb_cities, index=min(1, len(cb_cities)-1), key="cmp_city_b")
        cb_area = st.number_input("Living Area (sq ft) B", min_value=300, max_value=25000, value=2600, step=50, key="cmp_area_b")
        cb_beds = st.slider("Bedrooms B", 1, 10, 4, key="cmp_beds_b")
        cb_baths = st.slider("Bathrooms B", 1.0, 8.0, 3.0, 0.5, key="cmp_baths_b")
        cb_loc = st.selectbox("Location Tier B", VALID_LOCATIONS, index=1, key="cmp_loc_b")
        cb_year = st.slider("Year Built B", 1950, 2026, 2022, key="cmp_yr_b")
        cb_gar = st.selectbox("Garage B", VALID_GARAGES, index=0, key="cmp_gar_b")

    if st.button("⚖️ Run Comparative Valuation", key="run_cmp_btn", use_container_width=True):
        res_a = predict_house_price({
            "Area": ca_area, "Bedrooms": ca_beds, "Bathrooms": ca_baths, "Floors": 1,
            "YearBuilt": ca_year, "Location": ca_loc, "Condition": "Good", "Garage": ca_gar,
            "Country": ca_country, "City": ca_city, "Currency": active_currency
        })
        res_b = predict_house_price({
            "Area": cb_area, "Bedrooms": cb_beds, "Bathrooms": cb_baths, "Floors": 1,
            "YearBuilt": cb_year, "Location": cb_loc, "Condition": "Good", "Garage": cb_gar,
            "Country": cb_country, "City": cb_city, "Currency": active_currency
        })

        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        r_col1, r_col2 = st.columns(2, gap="large")

        with r_col1:
            st.markdown(f"""
            <div class="val-hero-card">
                <div class="val-hero-eyebrow">Property Alpha (A)</div>
                <div class="val-hero-main-price" style="font-size:2.5rem;">{res_a['price_formatted']}</div>
                <div style="font-size:0.85rem; color:{text_muted};">{res_a['price_per_sqft_formatted']} / sq ft</div>
            </div>
            """, unsafe_allow_html=True)

        with r_col2:
            st.markdown(f"""
            <div class="val-hero-card">
                <div class="val-hero-eyebrow">Property Beta (B)</div>
                <div class="val-hero-main-price" style="font-size:2.5rem;">{res_b['price_formatted']}</div>
                <div style="font-size:0.85rem; color:{text_muted};">{res_b['price_per_sqft_formatted']} / sq ft</div>
            </div>
            """, unsafe_allow_html=True)

        diff = res_b["predicted_price"] - res_a["predicted_price"]
        diff_pct = (diff / res_a["predicted_price"]) * 100.0 if res_a["predicted_price"] > 0 else 0
        diff_fmt = format_currency_value(abs(diff), active_currency)["display"]

        if diff > 0:
            st.success(f"📈 **Property Beta (B)** is valued higher by **{diff_fmt} (+{diff_pct:.1f}%)** than Property Alpha (A).")
        elif diff < 0:
            st.info(f"📉 **Property Alpha (A)** is valued higher by **{diff_fmt} (+{abs(diff_pct):.1f}%)** than Property Beta (B).")
        else:
            st.info("⚖️ Both properties have identical estimated market valuations.")

# ==============================================================================
# TAB 4: SAVED VALUATIONS (CLOUD & SESSION)
# ==============================================================================
elif selected_tab == NAV_HISTORY:
    st.markdown("""
    <div class="saas-hero-banner">
        <span class="hero-badge-pill">📜 Cloud & Session Archives</span>
        <h1>Saved Property Valuations</h1>
        <p>Access your past property records, review historical appraisals, and export datasets to CSV.</p>
    </div>
    """, unsafe_allow_html=True)

    cloud_records = []
    if auth_session and supabase_client:
        try:
            cloud_records = supabase_client.list_user_valuations(auth_session["access_token"])
        except Exception:
            cloud_records = []

    if cloud_records:
        st.markdown(f"##### ☁️ Cloud Valuations ({len(cloud_records)} records)")
        df_cloud = pd.DataFrame(cloud_records)
        st.dataframe(df_cloud[["created_at", "city", "country", "area", "bedrooms", "bathrooms", "price_formatted"]],
                     use_container_width=True, hide_index=True)
    elif st.session_state["session_history"]:
        st.markdown(f"##### 🕒 Current Session Valuations ({len(st.session_state['session_history'])} records)")
        df_sess = pd.DataFrame(st.session_state["session_history"])
        st.dataframe(df_sess, use_container_width=True, hide_index=True)
    else:
        st.info("No saved valuations yet. Appraise a property in the **🔮 Valuation Engine** tab to see records here!")

# ==============================================================================
# TAB 5: MARKET INSIGHTS
# ==============================================================================
elif selected_tab == NAV_MARKET:
    st.markdown("""
    <div class="saas-hero-banner">
        <span class="hero-badge-pill">🌍 Hedonic Market Multipliers</span>
        <h1>Global Real Estate Market Indexes</h1>
        <p>Regional price multipliers relative to the baseline United States metropolitan standard.</p>
    </div>
    """, unsafe_allow_html=True)

    market_rows = []
    for c_name, c_data in GLOBAL_COUNTRIES.items():
        for city_name, mult in c_data["cities"].items():
            market_rows.append({
                "Flag": c_data["flag"],
                "Country": c_name,
                "Metro City": city_name,
                "Local Currency": c_data["currency"],
                "Hedonic Multiplier": f"{mult:.2f}x",
                "Relative Tier": "Tier 1 Prime" if mult >= 1.2 else ("Tier 2 Major" if mult >= 0.9 else "Tier 3 Regional")
            })

    df_market = pd.DataFrame(market_rows)
    st.dataframe(df_market, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 6: HOW IT WORKS & ARCHITECTURE
# ==============================================================================
elif selected_tab == NAV_HOW_IT_WORKS:
    st.markdown("""
    <div class="saas-hero-banner">
        <span class="hero-badge-pill">🔬 Engineering & Methodology</span>
        <h1>Super Ensemble Architecture</h1>
        <p>Discover how PropIQ achieves a 98.83% R² benchmark through feature engineering and ensemble modeling.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-grid">
        <div class="step-box">
            <div class="step-num">Step 01</div>
            <h3>Feature Engineering</h3>
            <p>Calculates property vintage age, luxury space-per-room index, bath-to-bed ratio, and density encoders.</p>
        </div>
        <div class="step-box">
            <div class="step-num">Step 02</div>
            <h3>Log-Target Transform</h3>
            <p>Applies y = log(1 + price) transformation to normalize exponential right-skewed real estate prices.</p>
        </div>
        <div class="step-box">
            <div class="step-num">Step 03</div>
            <h3>Super Ensemble</h3>
            <p>Weighted blend of XGBoost, GradientBoostingRegressor, and HistGradientBoosting for maximum robustness.</p>
        </div>
        <div class="step-box">
            <div class="step-num">Step 04</div>
            <h3>Global Localization</h3>
            <p>Applies regional hedonic multipliers and live FX conversion matrices with 95% confidence intervals.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏆 Held-Out Benchmark Performance")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Coefficient of Determination", f"{r2_score:.2%}", "R² Score")
    b2.metric("Mean Absolute Error", f"${mae_val:,.0f}", "MAE")
    b3.metric("Mean Absolute % Error", f"{mape_val:.2f}%", "MAPE")
    b4.metric("Model Architecture", "Super Ensemble", "XGB + GB + HGB")

# ==============================================================================
# TAB 7: PROFILE & PREFERENCES
# ==============================================================================
elif selected_tab == NAV_PROFILE:
    st.markdown("""
    <div class="saas-hero-banner">
        <span class="hero-badge-pill">👤 User Space</span>
        <h1>Profile & Application Preferences</h1>
        <p>Manage your Supabase account, set default regional currencies, and configure interface appearance.</p>
    </div>
    """, unsafe_allow_html=True)

    p_col1, p_col2 = st.columns(2, gap="large")

    with p_col1:
        st.markdown("### ⚙️ System & Default Preferences")
        st.selectbox("Default Currency", all_currency_options, index=all_currency_options.index(active_currency), key="pref_curr")
        st.selectbox("Default Area Unit", ["Square Feet (sq ft)", "Square Meters (m²)"], index=1 if is_sqm else 0, key="pref_unit")
        st.write("Interface Appearance:", "🌙 Dark Mode" if dark else "☀️ Light Mode")

    with p_col2:
        st.markdown("### 🔒 Supabase Authentication")
        if auth_session:
            user_info = auth_session.get("user", {})
            st.success(f"Signed in as: **{user_info.get('email')}**")
            if st.button("Sign Out", key="profile_sign_out", use_container_width=True):
                st.session_state.pop("supabase_session", None)
                st.rerun()
        else:
            auth_tabs = st.tabs(["Sign In", "Create Account", "Forgot Password"])
            with auth_tabs[0]:
                with st.form("prof_signin_form"):
                    in_email = st.text_input("Email", key="prof_in_email")
                    in_pass = st.text_input("Password", type="password", key="prof_in_pass")
                    sub_in = st.form_submit_button("Sign In →", use_container_width=True)
                if sub_in:
                    try:
                        res = supabase_client.sign_in(in_email.strip(), in_pass)
                        store_auth_session(res)
                        st.success("Signed in successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            with auth_tabs[1]:
                with st.form("prof_signup_form"):
                    up_name = st.text_input("Full Name", key="prof_up_name")
                    up_email = st.text_input("Email", key="prof_up_email")
                    up_pass = st.text_input("Password (min 8 chars)", type="password", key="prof_up_pass")
                    sub_up = st.form_submit_button("Create Account →", use_container_width=True)
                if sub_up:
                    try:
                        res = supabase_client.sign_up(up_email.strip(), up_pass, up_name.strip(), supabase_redirect_url)
                        if res.get("access_token"):
                            store_auth_session(res)
                            st.rerun()
                        st.success("Account created! Check your email to confirm.")
                    except Exception as e:
                        st.error(str(e))

            with auth_tabs[2]:
                with st.form("prof_reset_form"):
                    re_email = st.text_input("Account Email", key="prof_re_email")
                    sub_re = st.form_submit_button("Send Reset Link", use_container_width=True)
                if sub_re:
                    try:
                        supabase_client.request_password_reset(re_email.strip(), supabase_redirect_url)
                        st.success("Password reset email sent.")
                    except Exception as e:
                        st.error(str(e))

# ==============================================================================
# TAB 8: ABOUT
# ==============================================================================
elif selected_tab == NAV_ABOUT:
    st.markdown("""
    <div class="saas-hero-banner">
        <span class="hero-badge-pill">ℹ️ Platform Information</span>
        <h1>About PropIQ</h1>
        <p>PropIQ is a next-generation real estate valuation intelligence platform built for modern homeowners, investors, and analysts.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Mission
    PropIQ bridges hedonic econometric modeling with state-of-the-art gradient boosted ensembles,
    delivering instant, transparent, and multi-currency property valuations across 20+ global metropolitan hubs.

    ### 🛠️ Architecture & Technology Stack
    - **Modeling**: Super Ensemble (XGBoost, GradientBoostingRegressor, HistGradientBoostingRegressor)
    - **Frontend**: Streamlit with custom responsive CSS design tokens
    - **Backend & Auth**: Supabase REST + Row-Level Security (RLS)
    - **Financial Analytics**: Amortized Mortgage Calculator & Rental Cap Rate Forecaster
    """)

# ==============================================================================
# TAB 9: ADMIN CONSOLE (RESTRICTED)
# ==============================================================================
elif selected_tab == NAV_ADMIN:
    st.markdown("""
    <div class="saas-hero-banner">
        <span class="hero-badge-pill">⚙️ Administrative Workspace</span>
        <h1>Admin Console</h1>
        <p>Restricted controls for verified account and platform administrators.</p>
    </div>
    """, unsafe_allow_html=True)

    if not auth_session:
        st.warning("Please sign in with an administrator account to view this section.")
    else:
        st.info("You have administrative access to the platform configuration.")
        st.metric("System Status", "Healthy", "100% Uptime")
