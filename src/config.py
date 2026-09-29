"""
Configuration module for House Price Prediction & Property Analytics System.
Defines paths, feature definitions, hyperparameters, and global constants.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "housing_raw.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "housing_cleaned.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "house_price_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORT_MD_PATH = REPORTS_DIR / "model_report.md"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Target Variable
TARGET_COLUMN = "Price"

# Raw Feature Definitions
NUMERICAL_FEATURES = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Parking_Spaces",
    "Floors",
    "Property_Age",
    "Balconies",
    "Amenities_Count",
    "Nearby_Schools"
]

CATEGORICAL_FEATURES = [
    "Location",
    "Property_Type",
    "Furnishing_Status",
    "Availability"
]

# Engineered Features
ENGINEERED_NUMERICAL_FEATURES = [
    "Total_Rooms",
    "Area_per_Bedroom",
    "Bathroom_to_Bedroom_Ratio",
    "Luxury_Score",
    "Bed_Bath_Interaction",
    "Rooms_per_Floor",
    "Amenities_per_Room",
    "Est_Depreciation",
    "Log_Area",
    "Floor_Area_Ratio",
    "Room_Density",
    "Luxury_Density"
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "Age_Category"
]

ALL_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

# Valid Categories for Validation
VALID_LOCATIONS = [
    "Downtown Central", "Silicon Hills", "Greenwood Heights",
    "Riverside District", "Lakeside Estates", "Harbor Point",
    "Midtown Corridor", "Oakridge Valley", "Sunset Park", "Westend Terrace"
]

VALID_PROPERTY_TYPES = [
    "Apartment", "Independent House", "Luxury Villa", "Penthouse", "Studio Apartment"
]

VALID_FURNISHING_STATUSES = [
    "Furnished", "Semi-Furnished", "Unfurnished"
]

VALID_AVAILABILITIES = [
    "Ready to Move", "Under Construction"
]
