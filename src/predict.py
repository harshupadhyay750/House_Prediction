"""
Production Prediction Module for House Price Prediction System.
Loads persisted scikit-learn/XGBoost pipeline, validates property inputs,
applies feature engineering, and calculates valuation with uncertainty intervals.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_PATH, METADATA_PATH,
    VALID_LOCATIONS, VALID_PROPERTY_TYPES,
    VALID_FURNISHING_STATUSES, VALID_AVAILABILITIES
)
from src.feature_engineering import add_engineered_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Cached model artifacts
_MODEL_PIPELINE = None
_MODEL_METADATA = None


def load_model_artifacts():
    """Lazy load and cache model artifacts from disk."""
    global _MODEL_PIPELINE, _MODEL_METADATA
    if _MODEL_PIPELINE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Train the model first.")
        logger.info(f"Loading model pipeline from {MODEL_PATH}...")
        _MODEL_PIPELINE = joblib.load(MODEL_PATH)
        
    if _MODEL_METADATA is None and METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            _MODEL_METADATA = json.load(f)

    return _MODEL_PIPELINE, _MODEL_METADATA


def validate_property_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates user or API input against realistic boundaries and allowed categories.
    Raises ValueError with descriptive diagnostic message if input is invalid.
    """
    validated = data.copy()

    # Numerical range checks
    if "Area_sqft" not in validated or validated["Area_sqft"] is None:
        raise ValueError("Area_sqft is required.")
    try:
        area = float(validated["Area_sqft"])
        if area < 100 or area > 30000:
            raise ValueError(f"Area_sqft must be between 100 and 30,000 sq ft (got {area}).")
        validated["Area_sqft"] = area
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid Area_sqft: {e}")

    # Bedrooms
    bedrooms = int(validated.get("Bedrooms", 1))
    if bedrooms < 1 or bedrooms > 20:
        raise ValueError(f"Bedrooms must be between 1 and 20 (got {bedrooms}).")
    validated["Bedrooms"] = bedrooms

    # Bathrooms
    bathrooms = float(validated.get("Bathrooms", 1.0))
    if bathrooms < 0.5 or bathrooms > 20.0:
        raise ValueError(f"Bathrooms must be between 0.5 and 20 (got {bathrooms}).")
    validated["Bathrooms"] = bathrooms

    # Optional defaults
    validated["Parking_Spaces"] = int(validated.get("Parking_Spaces", 1))
    validated["Floors"] = int(validated.get("Floors", 1))
    validated["Property_Age"] = float(validated.get("Property_Age", 5.0))
    validated["Balconies"] = int(validated.get("Balconies", 1))
    validated["Amenities_Count"] = int(validated.get("Amenities_Count", 4))
    validated["Nearby_Schools"] = int(validated.get("Nearby_Schools", 3))

    # Categorical validation
    loc = str(validated.get("Location", "Downtown Central")).strip().title()
    if loc not in VALID_LOCATIONS:
        # Fallback to closest or default
        logger.warning(f"Unknown location '{loc}', defaulting to Downtown Central.")
        loc = "Downtown Central"
    validated["Location"] = loc

    ptype = str(validated.get("Property_Type", "Apartment")).strip().title()
    if ptype not in VALID_PROPERTY_TYPES:
        ptype = "Apartment"
    validated["Property_Type"] = ptype

    furn = str(validated.get("Furnishing_Status", "Semi-Furnished")).strip().title()
    if furn not in VALID_FURNISHING_STATUSES:
        furn = "Semi-Furnished"
    validated["Furnishing_Status"] = furn

    avail = str(validated.get("Availability", "Ready to Move")).strip().title()
    if avail not in VALID_AVAILABILITIES:
        avail = "Ready to Move"
    validated["Availability"] = avail

    return validated


def predict_house_price(input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """
    Main prediction function.
    Accepts property characteristics dictionary or DataFrame,
    runs schema validation, feature engineering, and pipeline inference.
    Returns predicted valuation along with a 95% confidence prediction interval.
    """
    pipeline, metadata = load_model_artifacts()

    # Convert single dict to DataFrame
    if isinstance(input_data, dict):
        validated_dict = validate_property_input(input_data)
        df_input = pd.DataFrame([validated_dict])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
        # Validate first row
        validate_property_input(df_input.iloc[0].to_dict())
    else:
        raise TypeError("Input data must be a dictionary or pandas DataFrame.")

    # Apply Feature Engineering
    df_transformed = add_engineered_features(df_input, is_training=False)

    # Model inference
    raw_pred = pipeline.predict(df_transformed)
    predicted_val = float(raw_pred[0])
    predicted_val = max(50000.0, round(predicted_val, 2))

    # Uncertainty Estimation (approx 95% prediction interval using test RMSE)
    rmse = metadata["final_test_metrics"]["RMSE"] if metadata else 102000.0
    lower_bound = max(40000.0, round(predicted_val - 1.96 * rmse, 2))
    upper_bound = round(predicted_val + 1.96 * rmse, 2)
    
    # Valuation per sqft
    area = float(df_input.iloc[0]["Area_sqft"])
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
            "Property_Type": df_input.iloc[0]["Property_Type"],
            "Area_sqft": area,
            "Bedrooms": int(df_input.iloc[0]["Bedrooms"]),
            "Bathrooms": float(df_input.iloc[0]["Bathrooms"]),
            "Furnishing_Status": df_input.iloc[0]["Furnishing_Status"],
            "Total_Rooms": float(df_transformed.iloc[0]["Total_Rooms"]),
            "Luxury_Score": float(df_transformed.iloc[0]["Luxury_Score"])
        },
        "model_used": metadata.get("model_name", "Tuned XGBoost Regressor") if metadata else "Tuned Regressor"
    }


if __name__ == "__main__":
    sample_input = {
        "Location": "Downtown Central",
        "Property_Type": "Apartment",
        "Area_sqft": 1450,
        "Bedrooms": 3,
        "Bathrooms": 2.0,
        "Furnishing_Status": "Furnished",
        "Parking_Spaces": 1,
        "Floors": 12,
        "Property_Age": 3,
        "Balconies": 2,
        "Amenities_Count": 6,
        "Availability": "Ready to Move",
        "Nearby_Schools": 4
    }

    result = predict_house_price(sample_input)
    print("--- SAMPLE PREDICTION RESULT ---")
    print(json.dumps(result, indent=2))
