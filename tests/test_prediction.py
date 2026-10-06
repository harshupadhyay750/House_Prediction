"""
Unit and Integration Test Suite for House Price Prediction System.
Tests data preprocessing, feature engineering, model persistence,
prediction logic, edge cases, invalid inputs, global market localization,
multi-currency conversions (INR, EUR, GBP, AED, etc.), unit conversion,
and API endpoints using pytest.
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
from src.global_market import (
    GLOBAL_COUNTRIES, EXCHANGE_RATES, convert_and_localize_price,
    format_indian_currency, convert_area_to_sqft
)
from src.api import app

client = TestClient(app)


# ---------------- DATA PREPROCESSING TESTS ----------------

def test_raw_data_exists():
    """Verify raw dataset exists and has records."""
    assert RAW_DATA_PATH.exists(), f"Raw data missing at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)
    assert len(df) >= 2000
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
    assert metadata["final_test_metrics"]["R2"] > 0.80


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
    """Verify prediction returns valid numerical outputs and bounds in default USD."""
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
    assert "$" in result["price_formatted"]


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


# ---------------- GLOBAL VALUATION & MULTI-CURRENCY TESTS ----------------

def test_global_prediction_inr():
    """Verify prediction in Indian Rupees with Mumbai market multiplier and Lakh/Crore formatting."""
    payload = {
        "Area": 2000,
        "Bedrooms": 3,
        "Bathrooms": 2.0,
        "Floors": 1,
        "YearBuilt": 2005,
        "Location": "Downtown",
        "Condition": "Excellent",
        "Garage": "Yes",
        "Country": "India",
        "City": "Mumbai (MMR)",
        "Currency": "INR"
    }
    res = predict_house_price(payload)
    assert res["currency"] == "INR"
    assert res["currency_symbol"] == "₹"
    assert "₹" in res["price_formatted"]
    assert "Cr" in res["price_formatted"] or "Lakh" in res["price_formatted"]
    assert res["predicted_price"] > 10000000  # Multi-Crore property in Mumbai
    assert res["regional_multiplier"] == 1.6
    assert res["price_per_sqft"] > 0
    assert res["price_per_sqm"] > 0


def test_global_prediction_euro_and_uk():
    """Verify predictions in EUR and GBP currencies with correct symbols."""
    payload_eur = {
        "Area": 1800,
        "Country": "France",
        "City": "Paris Central",
        "Currency": "EUR"
    }
    res_eur = predict_house_price(payload_eur)
    assert res_eur["currency"] == "EUR"
    assert res_eur["currency_symbol"] == "€"
    assert res_eur["predicted_price"] > 0

    payload_gbp = {
        "Area": 1500,
        "Country": "United Kingdom",
        "City": "Central London",
        "Currency": "GBP"
    }
    res_gbp = predict_house_price(payload_gbp)
    assert res_gbp["currency"] == "GBP"
    assert res_gbp["currency_symbol"] == "£"


def test_square_meter_unit_conversion():
    """Verify input in square meters converts seamlessly to square feet."""
    payload_sqm = {
        "Area": 185.0,  # ~1,991 sq ft
        "Area_Unit": "sq m",
        "Bedrooms": 3,
        "Bathrooms": 2.0,
        "YearBuilt": 2000
    }
    res = predict_house_price(payload_sqm)
    assert abs(res["key_characteristics"]["Area_sqft"] - (185.0 * 10.7639104)) < 2.0
    assert res["key_characteristics"]["Area_sqm"] == 185.0


def test_format_indian_currency():
    """Verify Indian currency comma grouping and Lakh/Crore representations."""
    compact_cr, full_cr = format_indian_currency(18500000)
    assert "1.85 Cr" in compact_cr
    assert full_cr == "₹1,85,00,000"

    compact_lakh, full_lakh = format_indian_currency(7500000)
    assert "75.00 Lakh" in compact_lakh
    assert full_lakh == "₹75,00,000"


# ---------------- FASTAPI ENDPOINT TESTS ----------------

def test_api_health_endpoint():
    """Verify /health returns 200 OK and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_api_currencies_and_locations_endpoints():
    """Verify /currencies and /locations return global market datasets."""
    res_curr = client.get("/currencies")
    assert res_curr.status_code == 200
    assert "INR" in res_curr.json()["currencies"]
    assert "EUR" in res_curr.json()["currencies"]

    res_loc = client.get("/locations")
    assert res_loc.status_code == 200
    assert "India" in res_loc.json()["countries"]
    assert "United States" in res_loc.json()["countries"]


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


def test_api_predict_global_inr_payload():
    """Verify POST /predict with INR and Indian metropolitan location."""
    payload = {
        "Area": 2200,
        "Area_Unit": "sq ft",
        "Bedrooms": 3,
        "Bathrooms": 3.0,
        "Floors": 1,
        "YearBuilt": 2015,
        "Location": "Downtown",
        "Condition": "Excellent",
        "Garage": "Yes",
        "Country": "India",
        "City": "Bengaluru (Silicon Valley of India)",
        "Currency": "INR"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["currency"] == "INR"
    assert "₹" in data["price_formatted"]
    assert data["predicted_price"] > 1000000


def test_api_predict_endpoint_validation_error():
    """Verify invalid JSON payload triggers 422 Unprocessable Entity."""
    bad_payload = {
        "Area": 1,
        "Bedrooms": 0
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
