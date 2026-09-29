"""
Configuration module for House Price Prediction & Property Analytics System.
Configured for the user's property dataset schema:
Id, Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage, Price
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
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt"
]

CATEGORICAL_FEATURES = [
    "Location",
    "Condition",
    "Garage"
]

# Engineered Features
ENGINEERED_NUMERICAL_FEATURES = [
    "Property_Age",
    "Total_Rooms",
    "Area_per_Bedroom",
    "Area_per_Room",
    "Area_per_Floor",
    "Bathroom_to_Bedroom_Ratio",
    "Log_Area",
    "Bed_Bath_Interaction",
    "Rooms_per_Floor",
    "Est_Depreciation",
    "Effective_Area",
    "Condition_Score",
    "Location_Tier_Score",
    "Garage_Binary",
    "Quality_Space_Interaction",
    "Location_Area_Interaction"
]

ENGINEERED_CATEGORICAL_FEATURES = []

ALL_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

# Valid Categories for Validation
VALID_LOCATIONS = [
    "Downtown", "Urban", "Suburban", "Rural"
]

VALID_CONDITIONS = [
    "Excellent", "Good", "Fair", "Poor"
]

VALID_GARAGES = [
    "Yes", "No"
]
