"""
Production Prediction Module for House Price Prediction System.
Loads persisted scikit-learn/ensemble pipeline, validates property inputs,
applies feature engineering, and calculates valuation with uncertainty intervals.
Supports global prediction for any area on Earth with multi-currency conversion
(USD, INR with Lakhs/Crores, EUR, GBP, AED, CAD, AUD, JPY, etc.) and unit toggles (sq ft / sq m).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_PATH, METADATA_PATH, FEATURE_IMPORTANCE_PATH,
    VALID_LOCATIONS, VALID_CONDITIONS, VALID_GARAGES
)
from src.feature_engineering import add_engineered_features
from src.global_market import (
    convert_and_localize_price,
    format_currency_value,
    convert_area_to_sqft,
    convert_sqft_to_sqm,
    GLOBAL_COUNTRIES,
    EXCHANGE_RATES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

_MODEL_PIPELINE = None
_MODEL_METADATA = None


def _train_missing_model_artifacts():
    """Build the model artifacts automatically when the repository is freshly cloned or the saved artifacts are missing."""
    try:
        from src import train as train_module

        original_model_path = getattr(train_module, "MODEL_PATH", None)
        original_metadata_path = getattr(train_module, "METADATA_PATH", None)
        original_feature_importance_path = getattr(train_module, "FEATURE_IMPORTANCE_PATH", None)

        train_module.MODEL_PATH = MODEL_PATH
        train_module.METADATA_PATH = METADATA_PATH
        train_module.FEATURE_IMPORTANCE_PATH = FEATURE_IMPORTANCE_PATH

        try:
            logger.warning("Model artifact missing; automatically training a fresh model from the available dataset...")
            train_module.run_pipeline()
        finally:
            if original_model_path is not None:
                train_module.MODEL_PATH = original_model_path
            if original_metadata_path is not None:
                train_module.METADATA_PATH = original_metadata_path
            if original_feature_importance_path is not None:
                train_module.FEATURE_IMPORTANCE_PATH = original_feature_importance_path
    except Exception as exc:
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH} and auto-training failed: {exc}") from exc


def load_model_artifacts():
    global _MODEL_PIPELINE, _MODEL_METADATA

    if not MODEL_PATH.exists():
        _MODEL_PIPELINE = None
        _train_missing_model_artifacts()

    if _MODEL_PIPELINE is None:
        logger.info(f"Loading model pipeline from {MODEL_PATH}...")
        _MODEL_PIPELINE = joblib.load(MODEL_PATH)

    if _MODEL_METADATA is None or not METADATA_PATH.exists():
        if not METADATA_PATH.exists():
            _train_missing_model_artifacts()
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as f:
                _MODEL_METADATA = json.load(f)

    return _MODEL_PIPELINE, _MODEL_METADATA


def validate_property_input(data: Dict[str, Any]) -> Dict[str, Any]:
    validated = data.copy()

    # Living Area: accept Area, Area_sqft, or Area_sqm
    unit_raw = str(validated.get("Area_Unit", "sq ft")).strip().lower()
    unit = unit_raw.replace(" ", "").replace("_", "")
    raw_area = validated.get("Area", validated.get("Area_sqft", validated.get("Area_sqm", None)))
    if raw_area is None:
        raise ValueError("Area (square footage or square meters) is required.")

    try:
        area_float = float(raw_area)
        # If input was explicitly in sq m, convert to sq ft
        if "sqm" in unit or "m2" in unit or "meter" in unit or "Area_sqm" in validated:
            area_sqft = convert_area_to_sqft(area_float, "sq m")
        else:
            area_sqft = area_float

        if area_sqft < 100 or area_sqft > 30000:
            raise ValueError(f"Area must be between 100 and 30,000 sq ft (got {area_sqft:.1f} sq ft).")
        validated["Area"] = area_sqft
        validated["original_area_input"] = area_float
        validated["area_unit"] = unit
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid Area: {e}")

    # Bedrooms
    bedrooms = int(validated.get("Bedrooms", 3))
    if bedrooms < 1 or bedrooms > 20:
        raise ValueError(f"Bedrooms must be between 1 and 20 (got {bedrooms}).")
    validated["Bedrooms"] = bedrooms

    # Bathrooms
    bathrooms = float(validated.get("Bathrooms", 2.0))
    if bathrooms < 0.5 or bathrooms > 20.0:
        raise ValueError(f"Bathrooms must be between 0.5 and 20 (got {bathrooms}).")
    validated["Bathrooms"] = bathrooms

    # Floors
    floors = int(validated.get("Floors", 1))
    if floors < 1 or floors > 20:
        raise ValueError(f"Floors must be between 1 and 20 (got {floors}).")
    validated["Floors"] = floors

    # YearBuilt
    year_built = int(validated.get("YearBuilt", 1995))
    if year_built < 1850 or year_built > 2026:
        raise ValueError(f"YearBuilt must be between 1850 and 2026 (got {year_built}).")
    validated["YearBuilt"] = year_built

    # Categoricals (Location Density Tier)
    loc = str(validated.get("Location", "Downtown")).strip().title()
    # Normalize friendly names
    if "Rural" in loc:
        loc = "Rural"
    elif "Suburban" in loc:
        loc = "Suburban"
    elif "Downtown" in loc or "Central" in loc:
        loc = "Downtown"
    elif "Urban" in loc:
        loc = "Urban"

    if loc not in VALID_LOCATIONS:
        loc = "Downtown"
    validated["Location"] = loc

    cond = str(validated.get("Condition", "Good")).strip().title()
    if cond not in VALID_CONDITIONS:
        cond = "Good"
    validated["Condition"] = cond

    gar = str(validated.get("Garage", "Yes")).strip().title()
    if gar not in VALID_GARAGES:
        gar = "Yes"
    validated["Garage"] = gar

    # Global attributes
    validated["Country"] = validated.get("Country", "United States")
    validated["City"] = validated.get("City", None)
    validated["Custom_City"] = validated.get("Custom_City", None)
    validated["Currency"] = str(validated.get("Currency", "USD")).upper()

    return validated


def predict_house_price(input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    pipeline, metadata = load_model_artifacts()

    if isinstance(input_data, dict):
        validated_dict = validate_property_input(input_data)
        df_input = pd.DataFrame([{
            "Area": validated_dict["Area"],
            "Bedrooms": validated_dict["Bedrooms"],
            "Bathrooms": validated_dict["Bathrooms"],
            "Floors": validated_dict["Floors"],
            "YearBuilt": validated_dict["YearBuilt"],
            "Location": validated_dict["Location"],
            "Condition": validated_dict["Condition"],
            "Garage": validated_dict["Garage"]
        }])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
        validated_dict = validate_property_input(df_input.iloc[0].to_dict())
    else:
        raise TypeError("Input data must be a dictionary or pandas DataFrame.")

    # Apply Feature Engineering
    df_transformed = add_engineered_features(df_input, is_training=False)

    # Base hedonic model inference in USD
    raw_pred = pipeline.predict(df_transformed)
    base_usd_val = float(raw_pred[0])
    base_usd_val = max(50000.0, round(base_usd_val, 2))

    # Base Uncertainty estimation using RMSE from test evaluation
    rmse = metadata["final_test_metrics"]["RMSE"] if metadata and "final_test_metrics" in metadata else 23000.0
    base_lower_bound = max(40000.0, round(base_usd_val - 1.96 * rmse, 2))
    base_upper_bound = round(base_usd_val + 1.96 * rmse, 2)

    # Global localization & Currency Conversion
    country_name = validated_dict.get("Country", "United States")
    city_name = validated_dict.get("City", None)
    custom_city = validated_dict.get("Custom_City", None)
    custom_mult = validated_dict.get("Custom_Multiplier", None)
    currency_code = validated_dict.get("Currency", "USD").upper()

    localized = convert_and_localize_price(
        usd_price=base_usd_val,
        currency_code=currency_code,
        country_name=country_name,
        city_name=city_name,
        custom_city=custom_city,
        custom_multiplier=custom_mult
    )

    mult = localized["regional_multiplier"]
    rate = localized["exchange_rate"]

    localized_predicted_price = localized["localized_price"]
    localized_lower = round(base_lower_bound * mult * rate, 2)
    localized_upper = round(base_upper_bound * mult * rate, 2)

    fmt_lower = format_currency_value(localized_lower, currency_code)
    fmt_upper = format_currency_value(localized_upper, currency_code)

    area_sqft = float(validated_dict["Area"])
    area_sqm = convert_sqft_to_sqm(area_sqft)

    price_per_sqft = round(localized_predicted_price / area_sqft, 2) if area_sqft > 0 else 0.0
    price_per_sqm = round(localized_predicted_price / area_sqm, 2) if area_sqm > 0 else 0.0

    fmt_sqft = format_currency_value(price_per_sqft, currency_code)
    fmt_sqm = format_currency_value(price_per_sqm, currency_code)

    # For default USD without country change, keep classic $ format for backward compatibility
    if currency_code == "USD" and mult == 1.0:
        price_formatted = f"${localized_predicted_price:,.2f}"
        lower_formatted = f"${localized_lower:,.2f}"
        upper_formatted = f"${localized_upper:,.2f}"
    else:
        price_formatted = localized["display_price"]
        lower_formatted = fmt_lower["display"]
        upper_formatted = fmt_upper["display"]

    return {
        "predicted_price": localized_predicted_price,
        "price_formatted": price_formatted,
        "currency": currency_code,
        "currency_symbol": localized["symbol"],
        "country": localized["country"],
        "city": localized["city"],
        "regional_multiplier": localized["regional_multiplier"],
        "base_usd_price": base_usd_val,
        "exchange_rate": rate,
        "price_per_sqft": price_per_sqft,
        "price_per_sqm": price_per_sqm,
        "price_per_sqft_formatted": fmt_sqft["formatted"],
        "price_per_sqm_formatted": fmt_sqm["formatted"],
        "prediction_interval_95": {
            "lower_bound": localized_lower,
            "upper_bound": localized_upper,
            "lower_formatted": lower_formatted,
            "upper_formatted": upper_formatted
        },
        "key_characteristics": {
            "Location": validated_dict["Location"],
            "Country": localized["country"],
            "City": localized["city"],
            "Condition": validated_dict["Condition"],
            "Garage": validated_dict["Garage"],
            "Area_sqft": round(area_sqft, 1),
            "Area_sqm": round(area_sqm, 1),
            "Area": round(area_sqft, 1),
            "Bedrooms": int(validated_dict["Bedrooms"]),
            "Bathrooms": float(validated_dict["Bathrooms"]),
            "Floors": int(validated_dict["Floors"]),
            "YearBuilt": int(validated_dict["YearBuilt"]),
            "Property_Age": int(df_transformed.iloc[0]["Property_Age"]),
            "Total_Rooms": float(df_transformed.iloc[0]["Total_Rooms"])
        },
        "model_used": metadata.get("model_name", "Super Ensemble (XGB+GB+HGB)") if metadata else "Super Ensemble"
    }


if __name__ == "__main__":
    sample = {
        "Area": 2500,
        "Bedrooms": 4,
        "Bathrooms": 3.0,
        "Floors": 2,
        "YearBuilt": 1990,
        "Location": "Downtown",
        "Condition": "Excellent",
        "Garage": "Yes",
        "Country": "India",
        "City": "Mumbai (MMR)",
        "Currency": "INR"
    }
    res = predict_house_price(sample)
    print("--- SAMPLE GLOBAL PREDICTION RESULT ---")
    print(json.dumps(res, indent=2, ensure_ascii=False))
