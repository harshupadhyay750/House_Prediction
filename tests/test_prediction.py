"""
Unit and Integration Test Suite for House Price Prediction System.
Tests data preprocessing, feature engineering, model persistence,
prediction logic, edge cases, invalid inputs, and API endpoints using pytest.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.config import (
    RAW_DATA_PATH, CLEANED_DATA_PATH, MODEL_PATH, METADATA_PATH,
    VALID_LOCATIONS, VALID_PROPERTY_TYPES
)
from src.data_preprocessing import clean_raw_data, inspect_dataset
from src.feature_engineering import add_engineered_features
from src.predict import predict_house_price, load_model_artifacts, validate_property_input
from src.api import app

client = TestClient(app)


# ---------------- DATA PREPROCESSING TESTS ----------------

def test_raw_data_exists():
    """Verify raw dataset exists and has records."""
    assert RAW_DATA_PATH.exists(), f"Raw data missing at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)
    assert len(df) > 1000
    assert "Price" in df.columns


def test_clean_raw_data_removes_duplicates_and_nans():
    """Ensure cleaning removes duplicates and imputes missing values."""
    mock_df = pd.DataFrame({
        "Property_ID": ["PROP_1", "PROP_2", "PROP_1"],
        "Location": ["Downtown Central", "Silicon Hills", "Downtown Central"],
        "Property_Type": ["Apartment", "Luxury Villa", "Apartment"],
        "Area_sqft": [-1200, 2500, -1200],  # negative area to test impossible value correction
        "Bedrooms": [0, 4, 0],              # 0 bedrooms to test correction
        "Bathrooms": [np.nan, 3.0, np.nan], # NaN to test imputation
        "Furnishing_Status": ["furnished", "Semi-Furnished", "furnished"], # dirty casing
        "Parking_Spaces": [1, 2, 1],
        "Floors": [5, 2, 5],
        "Property_Age": [np.nan, 2, np.nan],
        "Balconies": [1, 2, 1],
        "Amenities_Count": [4, 8, 4],
        "Availability": ["Ready to Move", "Under Construction", "Ready to Move"],
        "Nearby_Schools": [3, 4, 3],
        "Price": [600000.0, 1800000.0, 600000.0]
    })
    
    cleaned = clean_raw_data(mock_df)
    
    # Check duplicates removed
    assert len(cleaned) == 2
    # Check impossible negative area corrected
    assert (cleaned["Area_sqft"] > 0).all()
    # Check 0 bedrooms corrected
    assert (cleaned["Bedrooms"] >= 1).all()
    # Check missing values imputed
    assert cleaned["Bathrooms"].isnull().sum() == 0
    assert cleaned["Property_Age"].isnull().sum() == 0
    # Check casing standardized
    assert "Furnished" in cleaned["Furnishing_Status"].values


# ---------------- FEATURE ENGINEERING TESTS ----------------

def test_add_engineered_features():
    """Ensure engineered features are generated accurately."""
    df = pd.DataFrame({
        "Area_sqft": [1500.0],
        "Bedrooms": [3],
        "Bathrooms": [2.0],
        "Parking_Spaces": [1],
        "Balconies": [2],
        "Amenities_Count": [6],
        "Property_Age": [4.0]
    })
    
    feat = add_engineered_features(df)
    
    assert "Total_Rooms" in feat.columns
    assert feat["Total_Rooms"].iloc[0] == 5.0
    
    assert "Area_per_Bedroom" in feat.columns
    assert feat["Area_per_Bedroom"].iloc[0] == 500.0
    
    assert "Bathroom_to_Bedroom_Ratio" in feat.columns
    assert round(feat["Bathroom_to_Bedroom_Ratio"].iloc[0], 2) == 0.67
    
    assert "Luxury_Score" in feat.columns
    # 6 * 1.5 + 1 * 1.2 + 2 * 1.0 = 9.0 + 1.2 + 2.0 = 12.2
    assert feat["Luxury_Score"].iloc[0] == 12.2
    
    assert "Age_Category" in feat.columns
    assert feat["Age_Category"].iloc[0] == "New Construction (0-5 yrs)"


# ---------------- MODEL PERSISTENCE & LOADING TESTS ----------------

def test_model_artifact_loading():
    """Ensure saved model and metadata can be loaded without error."""
    pipeline, metadata = load_model_artifacts()
    assert pipeline is not None
    assert metadata is not None
    assert "final_test_metrics" in metadata
    assert metadata["final_test_metrics"]["R2"] > 0.90


# ---------------- PREDICTION FUNCTION TESTS ----------------

def test_predict_house_price_valid_payload():
    """Verify prediction returns valid numerical outputs and bounds."""
    sample = {
        "Area_sqft": 1600,
        "Bedrooms": 3,
        "Bathrooms": 2.0,
        "Location": "Silicon Hills",
        "Property_Type": "Apartment",
        "Furnishing_Status": "Furnished",
        "Parking_Spaces": 2,
        "Floors": 8,
        "Property_Age": 4,
        "Balconies": 2,
        "Amenities_Count": 7,
        "Availability": "Ready to Move",
        "Nearby_Schools": 4
    }
    
    result = predict_house_price(sample)
    
    assert "predicted_price" in result
    assert isinstance(result["predicted_price"], float)
    assert result["predicted_price"] > 100000.0
    assert "prediction_interval_95" in result
    assert result["prediction_interval_95"]["lower_bound"] < result["predicted_price"]
    assert result["prediction_interval_95"]["upper_bound"] > result["predicted_price"]
    assert result["price_per_sqft"] > 0


def test_predict_house_price_invalid_area():
    """Verify invalid area triggers descriptive ValueError."""
    bad_sample = {
        "Area_sqft": -500,  # invalid
        "Bedrooms": 2,
        "Bathrooms": 1.0
    }
    with pytest.raises(ValueError, match="Area_sqft must be between"):
        predict_house_price(bad_sample)


def test_predict_house_price_invalid_bedrooms():
    """Verify invalid bedroom count triggers ValueError."""
    bad_sample = {
        "Area_sqft": 1200,
        "Bedrooms": 0,  # invalid
        "Bathrooms": 1.0
    }
    with pytest.raises(ValueError, match="Bedrooms must be between"):
        predict_house_price(bad_sample)


# ---------------- FASTAPI ENDPOINT TESTS ----------------

def test_api_health_endpoint():
    """Verify /health returns 200 OK and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_api_predict_endpoint_success():
    """Verify POST /predict returns 200 with complete valuation response."""
    payload = {
        "Area_sqft": 1800,
        "Bedrooms": 3,
        "Bathrooms": 3.0,
        "Location": "Downtown Central",
        "Property_Type": "Apartment",
        "Furnishing_Status": "Furnished",
        "Parking_Spaces": 1,
        "Floors": 10,
        "Property_Age": 2,
        "Balconies": 2,
        "Amenities_Count": 6,
        "Availability": "Ready to Move",
        "Nearby_Schools": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "predicted_price" in res
    assert res["predicted_price"] > 0
    assert "price_formatted" in res
    assert "prediction_interval_95" in res


def test_api_predict_endpoint_validation_error():
    """Verify invalid JSON payload triggers 422 Unprocessable Entity."""
    bad_payload = {
        "Area_sqft": 20,  # less than ge=100
        "Bedrooms": 0     # less than ge=1
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
