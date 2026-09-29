"""
Production Prediction Module for House Price Prediction System.
Loads persisted scikit-learn/ensemble pipeline, validates property inputs,
applies feature engineering, and calculates valuation with uncertainty intervals.
Adapted to the property dataset schema: Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage.
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

    # Living Area: accept Area or Area_sqft
    raw_area = validated.get("Area", validated.get("Area_sqft", None))
    if raw_area is None:
        raise ValueError("Area (square footage) is required.")
    try:
        area = float(raw_area)
        if area < 100 or area > 30000:
            raise ValueError(f"Area must be between 100 and 30,000 sq ft (got {area}).")
        validated["Area"] = area
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

    # Categoricals
    loc = str(validated.get("Location", "Downtown")).strip().title()
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

    return validated


def predict_house_price(input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    pipeline, metadata = load_model_artifacts()

    if isinstance(input_data, dict):
        validated_dict = validate_property_input(input_data)
        df_input = pd.DataFrame([validated_dict])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
        validate_property_input(df_input.iloc[0].to_dict())
    else:
        raise TypeError("Input data must be a dictionary or pandas DataFrame.")

    # Apply Feature Engineering
    df_transformed = add_engineered_features(df_input, is_training=False)

    # Model inference
    raw_pred = pipeline.predict(df_transformed)
    predicted_val = float(raw_pred[0])
    predicted_val = max(50000.0, round(predicted_val, 2))

    # Uncertainty estimation using RMSE from test evaluation
    rmse = metadata["final_test_metrics"]["RMSE"] if metadata and "final_test_metrics" in metadata else 23000.0
    lower_bound = max(40000.0, round(predicted_val - 1.96 * rmse, 2))
    upper_bound = round(predicted_val + 1.96 * rmse, 2)
    
    area = float(df_input.iloc[0]["Area"])
    price_per_sqft = round(predicted_val / area, 2) if area > 0 else 0.0

    return {
        "predicted_price": predicted_val,
        "price_formatted": f"${predicted_val:,.2f}",
        "price_per_sqft": price_per_sqft,
        "prediction_interval_95": {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "lower_formatted": f"${lower_bound:,.2f}",
            "upper_formatted": f"${upper_bound:,.2f}"
        },
        "key_characteristics": {
            "Location": df_input.iloc[0]["Location"],
            "Condition": df_input.iloc[0]["Condition"],
            "Garage": df_input.iloc[0]["Garage"],
            "Area": area,
            "Bedrooms": int(df_input.iloc[0]["Bedrooms"]),
            "Bathrooms": float(df_input.iloc[0]["Bathrooms"]),
            "Floors": int(df_input.iloc[0]["Floors"]),
            "YearBuilt": int(df_input.iloc[0]["YearBuilt"]),
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
        "Garage": "Yes"
    }
    res = predict_house_price(sample)
    print("--- SAMPLE PREDICTION RESULT ---")
    print(json.dumps(res, indent=2))
