"""
Unit and Integration Test Suite for House Price Prediction System.
Tests data preprocessing, feature engineering, model persistence,
prediction logic, edge cases, invalid inputs, and API endpoints using pytest.
Adapted to the property dataset schema.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.config import (
    RAW_DATA_PATH, CLEANED_DATA_PATH, MODEL_PATH, METADATA_PATH,
    VALID_LOCATIONS, VALID_CONDITIONS, VALID_GARAGES
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
    assert len(df) == 2000
    assert "Price" in df.columns
    assert "Area" in df.columns


def test_clean_raw_data_removes_duplicates_and_nans():
    """Ensure cleaning handles duplicates and bounds."""
    mock_df = pd.DataFrame({
        "Id": [1, 2, 1],
        "Area": [-1200, 2500, -1200],
        "Bedrooms": [0, 4, 0],
        "Bathrooms": [1, 3, 1],
        "Floors": [2, 1, 2],
        "YearBuilt": [1980, 2010, 1980],
        "Location": ["Downtown", "Suburban", "Downtown"],
        "Condition": ["Good", "Excellent", "Good"],
        "Garage": ["Yes", "No", "Yes"],
        "Price": [400000.0, 750000.0, 400000.0]
    })
    
    cleaned = clean_raw_data(mock_df)
    assert len(cleaned) == 2
    assert (cleaned["Area"] > 0).all()
    assert (cleaned["Bedrooms"] >= 1).all()


# ---------------- FEATURE ENGINEERING TESTS ----------------

def test_add_engineered_features():
    """Ensure engineered features are generated accurately."""
    df = pd.DataFrame({
        "Area": [2000.0],
        "Bedrooms": [4],
        "Bathrooms": [2.0],
        "Floors": [2],
        "YearBuilt": [2006]
    })
    
    feat = add_engineered_features(df)
    
    assert "Property_Age" in feat.columns
    assert feat["Property_Age"].iloc[0] == 20
    
    assert "Total_Rooms" in feat.columns
    assert feat["Total_Rooms"].iloc[0] == 6.0
    
    assert "Area_per_Bedroom" in feat.columns
    assert feat["Area_per_Bedroom"].iloc[0] == 500.0
    
    assert "Bed_Bath_Interaction" in feat.columns
    assert feat["Bed_Bath_Interaction"].iloc[0] == 8.0


# ---------------- MODEL PERSISTENCE & LOADING TESTS ----------------

def test_model_artifact_loading():
    """Ensure saved model and metadata can be loaded without error."""
    pipeline, metadata = load_model_artifacts()
    assert pipeline is not None
    assert metadata is not None
    assert "final_test_metrics" in metadata
    assert metadata["final_test_metrics"]["R2"] > 0.95


def test_model_artifact_loading_trains_when_missing(tmp_path, monkeypatch):
    """Ensure the system can train missing artifacts instead of crashing on a brand-new checkout."""
    import src.predict as predict_module

    missing_model = tmp_path / "missing_model.pkl"
    missing_metadata = tmp_path / "missing_metadata.json"

    monkeypatch.setattr(predict_module, "MODEL_PATH", missing_model)
    monkeypatch.setattr(predict_module, "METADATA_PATH", missing_metadata)

    pipeline, metadata = predict_module.load_model_artifacts()

    assert pipeline is not None
    assert metadata is not None
    assert missing_model.exists()
    assert missing_metadata.exists()


# ---------------- PREDICTION FUNCTION TESTS ----------------

def test_predict_house_price_valid_payload():
    """Verify prediction returns valid numerical outputs and bounds."""
    sample = {
        "Area": 2400,
        "Bedrooms": 3,
        "Bathrooms": 2.0,
        "Floors": 2,
        "YearBuilt": 2000,
        "Location": "Downtown",
        "Condition": "Good",
        "Garage": "Yes"
    }
    
    result = predict_house_price(sample)
    
    assert "predicted_price" in result
    assert isinstance(result["predicted_price"], float)
    assert result["predicted_price"] > 50000.0
    assert "prediction_interval_95" in result
    assert result["prediction_interval_95"]["lower_bound"] < result["predicted_price"]
    assert result["prediction_interval_95"]["upper_bound"] > result["predicted_price"]
    assert result["price_per_sqft"] > 0


def test_predict_house_price_invalid_area():
    """Verify invalid area triggers descriptive ValueError."""
    bad_sample = {
        "Area": -500,
        "Bedrooms": 2,
        "Bathrooms": 1.0
    }
    with pytest.raises(ValueError, match="Area must be between"):
        predict_house_price(bad_sample)


def test_predict_house_price_invalid_bedrooms():
    """Verify invalid bedroom count triggers ValueError."""
    bad_sample = {
        "Area": 1200,
        "Bedrooms": 0,
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
        "Area": 2800,
        "Bedrooms": 4,
        "Bathrooms": 3.0,
        "Floors": 2,
        "YearBuilt": 2010,
        "Location": "Downtown",
        "Condition": "Excellent",
        "Garage": "Yes"
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
        "Area": 20,
        "Bedrooms": 0
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
