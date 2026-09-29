"""
FastAPI REST API for House Price Prediction System.
Exposes endpoints for single and batch property valuations with Pydantic validation.
Adapted to the user dataset schema: Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage.
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import predict_house_price, load_model_artifacts
from src.config import (
    VALID_LOCATIONS, VALID_CONDITIONS, VALID_GARAGES
)

app = FastAPI(
    title="🏠 House Price Prediction & Property Analytics API",
    description="Enterprise Machine Learning REST API for real estate valuation and analytics.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PropertyInput(BaseModel):
    Area: float = Field(..., ge=100, le=30000, description="Gross living area in sq ft", json_schema_extra={"example": 2500.0})
    Bedrooms: int = Field(3, ge=1, le=15, description="Number of bedrooms", json_schema_extra={"example": 4})
    Bathrooms: float = Field(2.0, ge=0.5, le=15.0, description="Number of bathrooms", json_schema_extra={"example": 3.0})
    Floors: int = Field(1, ge=1, le=20, description="Number of floors", json_schema_extra={"example": 2})
    YearBuilt: int = Field(1995, ge=1850, le=2026, description="Year property was constructed", json_schema_extra={"example": 1990})
    Location: str = Field("Downtown", description=f"Location tier: {VALID_LOCATIONS}", json_schema_extra={"example": "Downtown"})
    Condition: str = Field("Good", description=f"Physical condition: {VALID_CONDITIONS}", json_schema_extra={"example": "Excellent"})
    Garage: str = Field("Yes", description=f"Garage parking availability: {VALID_GARAGES}", json_schema_extra={"example": "Yes"})


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
    try:
        result = predict_house_price(property_data.model_dump())
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict", tags=["Prediction"])
def batch_predict(batch: BatchPropertyInput):
    results = []
    for prop in batch.properties:
        try:
            res = predict_house_price(prop.model_dump())
            results.append(res)
        except Exception as e:
            results.append({"error": str(e)})
    return {"total": len(results), "predictions": results}
