"""
FastAPI REST API for House Price Prediction System.
Exposes endpoints for single and batch property valuations with Pydantic validation.
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import predict_house_price, load_model_artifacts
from src.config import (
    VALID_LOCATIONS, VALID_PROPERTY_TYPES,
    VALID_FURNISHING_STATUSES, VALID_AVAILABILITIES
)

app = FastAPI(
    title="🏠 House Price Prediction & Property Analytics API",
    description="Enterprise Machine Learning REST API for real estate valuation and analytics.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PropertyInput(BaseModel):
    Area_sqft: float = Field(..., ge=100, le=30000, description="Gross living area in square feet", json_schema_extra={"example": 1500.0})
    Bedrooms: int = Field(..., ge=1, le=15, description="Number of bedrooms", json_schema_extra={"example": 3})
    Bathrooms: float = Field(..., ge=1.0, le=15.0, description="Number of bathrooms", json_schema_extra={"example": 2.0})
    Location: str = Field("Downtown Central", description=f"Prime locality. Options: {VALID_LOCATIONS}", json_schema_extra={"example": "Downtown Central"})
    Property_Type: str = Field("Apartment", description=f"Type of asset. Options: {VALID_PROPERTY_TYPES}", json_schema_extra={"example": "Apartment"})
    Furnishing_Status: str = Field("Furnished", description=f"Furnishing tier. Options: {VALID_FURNISHING_STATUSES}", json_schema_extra={"example": "Furnished"})
    Parking_Spaces: int = Field(1, ge=0, le=10, description="Designated parking slots", json_schema_extra={"example": 1})
    Floors: int = Field(5, ge=1, le=50, description="Floor level or total stories", json_schema_extra={"example": 5})
    Property_Age: float = Field(3.0, ge=0, le=100, description="Age of structure in years", json_schema_extra={"example": 3.0})
    Balconies: int = Field(1, ge=0, le=10, description="Number of private balconies", json_schema_extra={"example": 2})
    Amenities_Count: int = Field(5, ge=0, le=20, description="Count of premium community amenities", json_schema_extra={"example": 6})
    Availability: str = Field("Ready to Move", description=f"Construction status: {VALID_AVAILABILITIES}", json_schema_extra={"example": "Ready to Move"})
    Nearby_Schools: int = Field(3, ge=0, le=10, description="Reputable schools within 2-mile radius", json_schema_extra={"example": 4})


class PredictionInterval(BaseModel):
    lower_bound: float
    upper_bound: float
    lower_formatted: str
    upper_formatted: str


class PredictionOutput(BaseModel):
    predicted_price: float
    price_formatted: str
    price_per_sqft: float
    prediction_interval_95: PredictionInterval
    model_used: str
    key_characteristics: Dict[str, Any]


class BatchPropertyInput(BaseModel):
    properties: List[PropertyInput]


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to the House Price Prediction & Property Analytics API",
        "documentation": "/docs",
        "health": "/health",
        "status": "operational"
    }


@app.get("/health", tags=["General"])
def health_check():
    try:
        pipeline, metadata = load_model_artifacts()
        return {
            "status": "healthy",
            "model_loaded": pipeline is not None,
            "model_name": metadata.get("model_name", "Unknown") if metadata else "Loaded",
            "last_trained": metadata.get("trained_timestamp", "Unknown") if metadata else "N/A"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact unavailable: {str(e)}"
        )


@app.get("/model-info", tags=["Analytics"])
def get_model_info():
    _, metadata = load_model_artifacts()
    if not metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    return metadata


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_price(property_data: PropertyInput):
    """
    Accepts property characteristics and returns the predicted market valuation
    along with a 95% confidence interval and valuation per sq ft.
    """
    try:
        result = predict_house_price(property_data.model_dump())
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict", tags=["Prediction"])
def batch_predict(batch: BatchPropertyInput):
    """Batch prediction endpoint for institutional portfolio valuation."""
    results = []
    for prop in batch.properties:
        try:
            res = predict_house_price(prop.model_dump())
            results.append(res)
        except Exception as e:
            results.append({"error": str(e)})
    return {"total": len(results), "predictions": results}
