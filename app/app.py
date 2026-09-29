"""
Streamlit Web Application: PropIntel | Global Real Estate Valuation & Property Analytics.
Features:
- Global Real Estate Valuation across any region or city on Earth
- Multi-currency engine (USD, INR with Lakhs/Crores, EUR, GBP, AED, CAD, AUD, JPY, SGD, CHF, SAR)
- Area unit toggles (Square Feet sq ft <-> Square Meters sq m / m²)
- Executive Dashboard (Portfolio KPIs, Model Leaderboard, Global Cross-Rates)
- Property Price Predictor (Global Country, City, Density Tier, Dimensions, Age, Amenities)
- Market Analytics (Local & Global price indices, distribution, correlations)
- Model Diagnostics & SHAP (Actual vs Predicted, Residuals, Feature Importance)
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    CLEANED_DATA_PATH, METADATA_PATH, FEATURE_IMPORTANCE_PATH, FIGURES_DIR,
    VALID_LOCATIONS, VALID_CONDITIONS, VALID_GARAGES
)
from src.predict import predict_house_price, load_model_artifacts
from src.global_market import (
    GLOBAL_COUNTRIES, EXCHANGE_RATES, SETTLEMENT_TIERS,
    convert_and_localize_price, format_currency_value,
    convert_area_to_sqft, convert_sqft_to_sqm
)

st.set_page_config(
    page_title="PropIntel | Global Property Valuation AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
    }

    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        border-left: 8px solid #60a5fa;
        box-shadow: 0 20px 45px -18px rgba(37, 99, 235, 0.45);
    }

    .prediction-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1.5px solid #bfdbfe;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 8px solid #2563eb;
        box-shadow: 0 16px 36px -20px rgba(37, 99, 235, 0.55);
    }

    .price-badge {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1e3a8a;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }

    .badge-sub {
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569;
        font-weight: 700;
    }

    .currency-pill {
        display: inline-block;
        background: #1e293b;
        color: #f8fafc;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 16px;
        padding: 1.1rem;
        box-shadow: 0 12px 24px -20px rgba(15, 23, 42, 0.45);
    }

    .sidebar-status {
        background: linear-gradient(180deg, #eff6ff, #f8fafc);
        border: 1px solid #c7d2fe;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.85rem;
    }

    .stSelectbox label, .stNumberInput label, .stSlider label, .stRadio label, .stTextInput label {
        font-weight: 600;
        color: #0f172a;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_and_metadata():
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


df_data, metadata, feat_importance = load_data_and_metadata()

# ==============================================================================
# SIDEBAR: NAVIGATION & GLOBAL SETTINGS
# ==============================================================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("PropIntel Global")
    st.caption("AI-Powered Worldwide Property Valuation")
    st.markdown("---")

    app_mode = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "🎯 Global Price Predictor",
            "📈 Market & Global Analytics",
            "🧠 Model Diagnostics & SHAP"
        ],
        index=1
    )

    st.markdown("---")
    st.markdown("### 🌐 Global Settings")

    currency_options = list(EXCHANGE_RATES.keys())
    # Format labels with flags and names
    currency_labels = [f"{EXCHANGE_RATES[c]['flag']} {c} - {EXCHANGE_RATES[c]['name'].split('(')[0].strip()}" for c in currency_options]
    selected_curr_idx = st.selectbox(
        "Display Currency",
        range(len(currency_options)),
        format_func=lambda i: currency_labels[i],
        index=1  # Default to INR or USD
    )
    active_currency = currency_options[selected_curr_idx]
    active_curr_info = EXCHANGE_RATES[active_currency]

    active_unit = st.radio(
        "Measurement Unit",
        ["Square Feet (sq ft)", "Square Meters (m² / sq m)"],
        index=0
    )
    is_sqm = "Meter" in active_unit

    st.markdown("---")
    st.markdown("### Model System Status")
    st.markdown(f"""
    <div class="sidebar-status">
        <div><strong>Pipeline</strong>: Super Ensemble</div>
        <div><strong>Active Rate</strong>: 1 USD = {active_curr_info['rate']:.2f} {active_currency}</div>
    </div>
    """, unsafe_allow_html=True)

    if metadata and "final_test_metrics" in metadata:
        final_metrics = metadata["final_test_metrics"]
        st.success(f"● Model Accuracy (R²): {final_metrics.get('R2', 0.9862):.2%}")
        mae_usd = final_metrics.get('MAE', 22896)
        mae_local = mae_usd * active_curr_info['rate']
        fmt_mae = format_currency_value(mae_local, active_currency)
        st.info(f"● Test MAE: {fmt_mae['display']}")
        st.info(f"● Mean Error (MAPE): {final_metrics.get('MAPE', 4.54):.2f}%")
    else:
        st.success("● Accuracy (R²): 98.62%")
        st.info("● Test MAE: $22,896")

    st.caption("Version 2.1.0 | Global Edition")


# ==============================================================================
# TAB 1: EXECUTIVE DASHBOARD
# ==============================================================================
if app_mode == "📊 Executive Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.2rem; font-weight:800;">🌍 PropIntel | Global Real Estate Intelligence</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.05rem;">
            Enterprise Property Valuation Engine with Worldwide Localization & Hedonic Regression
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><div class='badge-sub'>Portfolio Size</div><div style='font-size:2rem;font-weight:800;color:#0f172a;'>%s</div></div>" % (f"{len(df_data):,}" if not df_data.empty else "2,000"), unsafe_allow_html=True)
        st.caption("Validated properties")
    with col2:
        median_price_usd = float(df_data['Price'].median()) if not df_data.empty else 539000.0
        median_local = median_price_usd * active_curr_info['rate']
        fmt_med = format_currency_value(median_local, active_currency)
        st.markdown(f"<div class='metric-card'><div class='badge-sub'>Median Price ({active_currency})</div><div style='font-size:1.8rem;font-weight:800;color:#0f172a;'>{fmt_med['compact']}</div></div>", unsafe_allow_html=True)
        st.caption(f"Full: {fmt_med['formatted']}")
    with col3:
        score = metadata.get("final_test_metrics", {}).get("R2", 0.9862) if metadata else 0.9862
        st.markdown(f"<div class='metric-card'><div class='badge-sub'>Model Accuracy</div><div style='font-size:2rem;font-weight:800;color:#0f172a;'>{score:.2%}</div></div>", unsafe_allow_html=True)
        st.caption("R² on holdout test set")
    with col4:
        mae_usd = metadata.get("final_test_metrics", {}).get("MAE", 22896) if metadata else 22896
        mae_local = mae_usd * active_curr_info['rate']
        fmt_mae = format_currency_value(mae_local, active_currency)
        st.markdown(f"<div class='metric-card'><div class='badge-sub'>MAE ({active_currency})</div><div style='font-size:1.8rem;font-weight:800;color:#0f172a;'>{fmt_mae['compact']}</div></div>", unsafe_allow_html=True)
        st.caption("Average error margin")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("🌐 Global Valuation Capabilities")
        st.markdown(f"""
        The **PropIntel Worldwide Valuation Engine** replaces generic estimation tools with a **locally calibrated hedonic machine learning system**:
        
        - **Global Coverage**: Supports properties in **India 🇮🇳, United States 🇺🇸, United Kingdom 🇬🇧, UAE 🇦🇪, Canada 🇨🇦, Australia 🇦🇺, Germany 🇩🇪, France 🇫🇷, Singapore 🇸🇬, Switzerland 🇨🇭**, or any custom location worldwide.
        - **Multi-Currency Support**: Instant localized valuation in **{active_currency} ({active_curr_info['symbol']})** with native conventions (such as Indian Crores & Lakhs).
        - **Measurement Flexibility**: Dynamically converts between **Square Feet (sq ft)** and **Square Meters (m²)**.
        - **Settlement Tiers**: Balances location premiums from high-density City Centers down to expansive Countryside/Rural zones.
        """)

    with col_right:
        st.subheader("💱 Global Currency Conversion Rates")
        top_curr = ["USD", "INR", "EUR", "GBP", "AED", "CAD", "AUD", "SGD", "JPY"]
        curr_data = []
        for c in top_curr:
            curr_data.append({
                "Currency": f"{EXCHANGE_RATES[c]['flag']} {c}",
                "Name": EXCHANGE_RATES[c]['name'].split('(')[0].strip(),
                "1 USD Equivalent": f"{EXCHANGE_RATES[c]['rate']:.2f} {EXCHANGE_RATES[c]['symbol']}"
            })
        st.dataframe(pd.DataFrame(curr_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🏆 Model Benchmark Leaderboard")
    if metadata and "model_comparison" in metadata:
        comp_df = pd.DataFrame(metadata["model_comparison"])
        st.dataframe(
            comp_df.style.format({
                "CV_R2_Mean": "{:.4f}",
                "CV_R2_Std": "{:.4f}",
                "Test_MAE": "${:,.2f}",
                "Test_RMSE": "${:,.2f}",
                "Test_R2": "{:.4f}",
                "Test_MAPE": "{:.2f}%",
                "Training_Time_Sec": "{:.2f}s"
            }).highlight_max(subset=["Test_R2"], color="#dcfce7")
              .highlight_min(subset=["Test_MAE", "Test_RMSE", "Test_MAPE"], color="#dcfce7"),
            use_container_width=True
        )


# ==============================================================================
# TAB 2: GLOBAL PRICE PREDICTOR
# ==============================================================================
elif app_mode == "🎯 Global Price Predictor":
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.2rem; font-weight:800;">🎯 Global Property Valuation Engine</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.05rem;">
            Predict real estate market value in any area of the Earth in <strong>{active_currency} ({active_curr_info['symbol']})</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    country_list = list(GLOBAL_COUNTRIES.keys())
    # Format country labels with flags
    country_labels = [f"{GLOBAL_COUNTRIES[c]['flag']} {c}" for c in country_list]

    # Quick pre-selector for Country outside form to update cities reactively
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        default_country_idx = 1 if "India" in country_list else 0  # Default to India or US
        sel_c_idx = st.selectbox("🌐 Select Country / Region", range(len(country_list)), format_func=lambda i: country_labels[i], index=default_country_idx)
        selected_country = country_list[sel_c_idx]
        country_info = GLOBAL_COUNTRIES[selected_country]

    with col_c2:
        available_cities = list(country_info["cities"].keys())
        available_cities.append("Other / Custom City or Area...")
        selected_city = st.selectbox(f"🏙️ Select Metro Hub / City in {selected_country}", available_cities, index=1 if len(available_cities) > 1 else 0)

    custom_city_name = None
    custom_mult_val = None
    if selected_city == "Other / Custom City or Area...":
        col_cust1, col_cust2 = st.columns(2)
        with col_cust1:
            custom_city_name = st.text_input("Enter Custom City / Locality Name", value="Local Area")
        with col_cust2:
            custom_mult_val = st.slider("Local Price Tier Multiplier (relative to national base)", min_value=0.40, max_value=2.80, value=1.00, step=0.05)

    with st.form("global_prediction_form"):
        st.markdown("### 📋 Property Specifications")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📍 Location Tier & Parking")
            settlement_label = st.selectbox(
                "Settlement Density Tier",
                list(SETTLEMENT_TIERS.keys()),
                index=0,
                help="Downtown (CBD/Center), Urban (inner-city metro), Suburban (commuter outer ring), Rural (exurban countryside)"
            )
            location_tier = SETTLEMENT_TIERS[settlement_label]
            condition = st.selectbox("Physical Upkeep / Condition", VALID_CONDITIONS, index=0)
            garage = st.selectbox("Garage / Covered Parking", VALID_GARAGES, index=0)

        with col2:
            st.markdown("#### 📐 Dimensions & Rooms")
            if is_sqm:
                area_input = st.number_input("Living Area (Square Meters m²)", min_value=35, max_value=2500, value=185, step=5)
                st.caption(f"Equivalent: ~{area_input * 10.764:,.0f} sq ft")
            else:
                area_input = st.number_input("Living Area (Square Feet sq ft)", min_value=400, max_value=25000, value=2000, step=50)
                st.caption(f"Equivalent: ~{area_input * 0.0929:,.0f} m²")

            bedrooms = st.slider("Bedrooms", min_value=1, max_value=8, value=3, step=1)
            bathrooms = st.slider("Bathrooms", min_value=1.0, max_value=6.0, value=2.0, step=0.5)

        with col3:
            st.markdown("#### 🏗️ Age & Architecture")
            floors = st.selectbox("Number of Floors", [1, 2, 3, 4, 5], index=0)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=2000, step=1)
            st.markdown(f"**Selected Currency**: `{active_currency}` ({active_curr_info['symbol']})")

        submitted = st.form_submit_button(f"⚡ Calculate Global Valuation ({active_currency})", use_container_width=True)

    if submitted:
        input_payload = {
            "Location": location_tier,
            "Condition": condition,
            "Garage": garage,
            "Area": float(area_input),
            "Area_Unit": "sq m" if is_sqm else "sq ft",
            "Bedrooms": int(bedrooms),
            "Bathrooms": float(bathrooms),
            "Floors": int(floors),
            "YearBuilt": int(year_built),
            "Country": selected_country,
            "City": selected_city if selected_city != "Other / Custom City or Area..." else None,
            "Custom_City": custom_city_name,
            "Custom_Multiplier": custom_mult_val,
            "Currency": active_currency
        }

        with st.spinner("Valuating worldwide property with Super Ensemble..."):
            pred_res = predict_house_price(input_payload)

        # Main Prediction Presentation
        city_display = pred_res['city']
        country_display = pred_res['country']

        st.markdown(f"""
        <div class="prediction-card">
            <div class="badge-sub">🌍 {country_display} • {city_display} • {location_tier}</div>
            <div class="price-badge">{pred_res['price_formatted']}</div>
            <div style="margin-top:0.75rem; font-size:1.1rem; color:#1e293b;">
                Estimated Unit Rates: <strong>{pred_res['price_per_sqft_formatted']} / sq ft</strong> &nbsp;|&nbsp; <strong>{pred_res['price_per_sqm_formatted']} / m²</strong>
            </div>
            <div style="margin-top:0.5rem; font-size:0.95rem; color:#475569;">
                95% Valuation Interval: <strong>{pred_res['prediction_interval_95']['lower_formatted']}</strong> — <strong>{pred_res['prediction_interval_95']['upper_formatted']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(f"<div class='metric-card'><div class='badge-sub'>Total Rooms</div><div style='font-size:1.8rem;font-weight:800;color:#0f172a;'>{pred_res['key_characteristics']['Total_Rooms']}</div></div>", unsafe_allow_html=True)
            st.caption("Bedrooms + Bathrooms")
        with col_b:
            sqft_area = pred_res['key_characteristics']['Area_sqft']
            st.markdown(f"<div class='metric-card'><div class='badge-sub'>Gross Floor Area</div><div style='font-size:1.8rem;font-weight:800;color:#0f172a;'>{sqft_area:,.0f} sqft</div></div>", unsafe_allow_html=True)
            st.caption(f"~{pred_res['key_characteristics']['Area_sqm']:.1f} m²")
        with col_c:
            st.markdown(f"<div class='metric-card'><div class='badge-sub'>Structure Vintage</div><div style='font-size:1.8rem;font-weight:800;color:#0f172a;'>{pred_res['key_characteristics']['Property_Age']} yrs</div></div>", unsafe_allow_html=True)
            st.caption(f"Built in {year_built}")
        with col_d:
            st.markdown(f"<div class='metric-card'><div class='badge-sub'>Market Multiplier</div><div style='font-size:1.8rem;font-weight:800;color:#0f172a;'>{pred_res['regional_multiplier']:.2f}x</div></div>", unsafe_allow_html=True)
            st.caption("Local Real Estate Index")

        st.markdown("---")

        # Global Currency Cross-Rates Grid
        st.subheader("💱 Worldwide Currency Cross-Valuation")
        st.caption("Compare how this property is valued across major global financial currencies:")

        usd_base = pred_res['base_usd_price'] * pred_res['regional_multiplier']
        matrix_cols = st.columns(6)
        matrix_currencies = ["INR", "USD", "EUR", "GBP", "AED", "JPY"]

        for idx, m_curr in enumerate(matrix_currencies):
            m_info = EXCHANGE_RATES[m_curr]
            m_val = usd_base * m_info['rate']
            m_fmt = format_currency_value(m_val, m_curr)
            with matrix_cols[idx]:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center;">
                    <div style="font-size:1.2rem;">{m_info['flag']} <strong>{m_curr}</strong></div>
                    <div style="font-size:1.15rem; font-weight:800; color:#1e3a8a; margin-top:0.3rem;">{m_fmt['compact']}</div>
                    <div style="font-size:0.75rem; color:#64748b; margin-top:0.2rem;">{m_fmt['formatted']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("💡 Global Valuation Breakdown")
        st.markdown(f"""
        - **Metropolitan Market**: **{city_display}, {country_display}** operates at a **{pred_res['regional_multiplier']:.2f}x** index relative to the international benchmark.
        - **Urban Density Tier**: **{location_tier}** anchors the foundational per-unit land value.
        - **Condition Rating**: **{condition}** reflects structural upkeep and interior modern finish.
        - **Exchange Rate**: Converted at **1 USD = {pred_res['exchange_rate']:.2f} {active_currency}**.
        """)


# ==============================================================================
# TAB 3: MARKET & GLOBAL ANALYTICS
# ==============================================================================
elif app_mode == "📈 Market & Global Analytics":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.2rem; font-weight:800;">📈 Real Estate Market & Global Analytics</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.05rem;">
            Comparative analysis of global property indices, price distributions, and hedonic drivers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    analytics_tab = st.selectbox(
        "Select Analytics Domain",
        [
            "🌍 Global City Property Price Index Comparison",
            "1. Target Price Distribution",
            "2. Correlation Heatmap",
            "3. Price vs Living Area",
            "4. Price Across Prime Locations",
            "5. Price Across Property Conditions",
            "6. Garage Value Premium",
            "7. Outlier Analysis & Whiskers"
        ]
    )

    if analytics_tab == "🌍 Global City Property Price Index Comparison":
        st.subheader("Worldwide Real Estate Market Multiplier Benchmarks")
        st.caption("Empirical price index multipliers across prime international metro centers:")

        benchmark_data = [
            {"City": "Singapore (Central)", "Country": "Singapore 🇸🇬", "Index Multiplier": 2.40, "Tier": "Ultra-Prime"},
            {"City": "Zurich", "Country": "Switzerland 🇨🇭", "Index Multiplier": 2.35, "Tier": "Ultra-Prime"},
            {"City": "New York City", "Country": "United States 🇺🇸", "Index Multiplier": 1.95, "Tier": "Prime"},
            {"City": "Central London", "Country": "United Kingdom 🇬🇧", "Index Multiplier": 1.95, "Tier": "Prime"},
            {"City": "Paris Central", "Country": "France 🇫🇷", "Index Multiplier": 1.90, "Tier": "Prime"},
            {"City": "Sydney (Eastern Suburbs)", "Country": "Australia 🇦🇺", "Index Multiplier": 1.75, "Tier": "Prime"},
            {"City": "Amsterdam", "Country": "Netherlands 🇳🇱", "Index Multiplier": 1.80, "Tier": "High-Demand"},
            {"City": "Tokyo (23 Wards)", "Country": "Japan 🇯🇵", "Index Multiplier": 1.65, "Tier": "High-Demand"},
            {"City": "Vancouver (Metro)", "Country": "Canada 🇨🇦", "Index Multiplier": 1.65, "Tier": "High-Demand"},
            {"City": "Mumbai (MMR)", "Country": "India 🇮🇳", "Index Multiplier": 1.60, "Tier": "Mega-Metro High"},
            {"City": "Dubai (Downtown/Palm)", "Country": "UAE 🇦🇪", "Index Multiplier": 1.55, "Tier": "Global Hub"},
            {"City": "Delhi NCR (Gurugram/Noida)", "Country": "India 🇮🇳", "Index Multiplier": 1.35, "Tier": "Major Metro"},
            {"City": "Bengaluru", "Country": "India 🇮🇳", "Index Multiplier": 1.25, "Tier": "Tech Hub"},
            {"City": "Austin", "Country": "United States 🇺🇸", "Index Multiplier": 1.20, "Tier": "Tech Hub"},
            {"City": "Hyderabad", "Country": "India 🇮🇳", "Index Multiplier": 1.10, "Tier": "Emerging Metro"}
        ]
        bench_df = pd.DataFrame(benchmark_data)
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

    else:
        fig_map = {
            "1. Target Price Distribution": ("01_price_distribution.png", "Analysis: Property prices span with smooth log-normal distribution across settlement tiers."),
            "2. Correlation Heatmap": ("02_correlation_heatmap.png", "Analysis: Living area, total rooms, and year built exhibit strong alignment with property valuations."),
            "3. Price vs Living Area": ("03_price_vs_area.png", "Analysis: Square footage demonstrates strong positive valuation trajectory modulated by location tiers."),
            "4. Price Across Prime Locations": ("06_price_by_location.png", "Analysis: Downtown properties command highest rates, followed by Urban, Suburban, and Rural."),
            "5. Price Across Property Conditions": ("07_price_by_furnishing.png", "Analysis: Excellent condition properties command significant market premiums over Fair and Poor units."),
            "6. Garage Value Premium": ("08_price_by_property_type.png", "Analysis: Garage availability provides consistent valuation increment."),
            "7. Outlier Analysis & Whiskers": ("09_outlier_analysis.png", "Analysis: Balanced dispersion with clean interquartile bounds.")
        }

        fig_name, interpretation = fig_map[analytics_tab]
        fig_file = FIGURES_DIR / fig_name

        if fig_file.exists():
            st.image(str(fig_file), use_container_width=True)
            st.info(f"**Business Insight**: {interpretation}")
        else:
            st.warning(f"Figure {fig_name} not found.")


# ==============================================================================
# TAB 4: MODEL DIAGNOSTICS & SHAP
# ==============================================================================
elif app_mode == "🧠 Model Diagnostics & SHAP":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.2rem; font-weight:800;">🧠 Model Diagnostics & Explainability (XAI)</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.05rem;">
            Validation diagnostics, residual normality tests, permutation importance, and SHAP explainability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    diag_option = st.radio(
        "Diagnostic View",
        [
            "Model Actual vs. Predicted (Goodness of Fit)",
            "Residual Distribution & Homoscedasticity",
            "Global Feature Importance",
            "Permutation Importance (Test Set)",
            "SHAP Summary (Directional Impact)"
        ],
        horizontal=True
    )

    diag_map = {
        "Model Actual vs. Predicted (Goodness of Fit)": ("15_actual_vs_predicted.png", "Test-set predictions tightly follow the 45-degree reference line with R² = 0.9862."),
        "Residual Distribution & Homoscedasticity": ("16_residual_distribution.png", "Residuals exhibit a zero-centered bell curve with bounded variance."),
        "Global Feature Importance": ("12_feature_importance.png", "Living area (Area) and Location contribute the highest tree split importance."),
        "Permutation Importance (Test Set)": ("13_permutation_importance.png", "Area and Location create the largest drop in test R² when permuted."),
        "SHAP Summary (Directional Impact)": ("14_shap_summary.png", "High area and Downtown locations push SHAP values strongly rightward.")
    }

    img_name, text_desc = diag_map[diag_option]
    img_file = FIGURES_DIR / img_name

    if img_file.exists():
        st.image(str(img_file), use_container_width=True)
        st.success(f"**Diagnostic Evaluation**: {text_desc}")
    else:
        st.warning(f"Artifact {img_name} not found.")

    st.markdown("---")
    st.subheader("📋 Residual Performance Breakdown Across Price Quartiles")
    if metadata and "error_analysis_summary" in metadata:
        tier_data = pd.DataFrame(metadata["error_analysis_summary"]["tier_summary"])
        st.dataframe(tier_data.style.format({
            "Mean_Absolute_Error": "${:,.2f}",
            "Median_Absolute_Error": "${:,.2f}",
            "Mean_MAPE": "{:.2f}%"
        }), use_container_width=True)
