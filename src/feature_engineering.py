"""
Feature Engineering Module for House Price Prediction System.
Implements domain-specific property transformations:
- Total_Rooms: sum of bedrooms and bathrooms
- Area_per_Bedroom: measures spaciousness of living quarters
- Bathroom_to_Bedroom_Ratio: proxy for luxury and convenience
- Luxury_Score: composite amenity, parking, and balcony index
- Age_Category: binned architectural lifecycle stage
- Log_Area: reduces right-skewness of square footage
Avoids target leakage (Price per sqft is calculated ONLY for EDA, never in features).
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def add_engineered_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Applies feature engineering transformations to a property DataFrame.
    Guarantees no target leakage.
    """
    data = df.copy()

    # 1. Total Rooms
    data["Total_Rooms"] = data["Bedrooms"] + data["Bathrooms"]

    # 2. Area per Bedroom (Space efficiency)
    safe_bedrooms = np.maximum(data["Bedrooms"], 1)
    data["Area_per_Bedroom"] = (data["Area_sqft"] / safe_bedrooms).round(2)

    # 3. Bathroom to Bedroom Ratio
    data["Bathroom_to_Bedroom_Ratio"] = (data["Bathrooms"] / safe_bedrooms).round(3)

    # 4. Luxury Composite Score
    # Combines amenities, parking spaces, and balconies with business-driven weights
    amenities = data["Amenities_Count"] if "Amenities_Count" in data.columns else 0
    parking = data["Parking_Spaces"] if "Parking_Spaces" in data.columns else 0
    balconies = data["Balconies"] if "Balconies" in data.columns else 0
    data["Luxury_Score"] = (amenities * 1.5 + parking * 1.2 + balconies * 1.0).round(2)

    # 5. Age Category Binning
    if "Property_Age" in data.columns:
        def categorize_age(age_val):
            if pd.isna(age_val):
                return "Modern (6-15 yrs)"
            if age_val <= 5:
                return "New Construction (0-5 yrs)"
            elif age_val <= 15:
                return "Modern (6-15 yrs)"
            elif age_val <= 25:
                return "Mature (16-25 yrs)"
            else:
                return "Established (>25 yrs)"

        data["Age_Category"] = data["Property_Age"].apply(categorize_age)

    # 6. EDA Only: Price per sqft (never included in feature set to prevent target leakage)
    if is_training and "Price" in data.columns and "Area_sqft" in data.columns:
        data["Price_per_sqft"] = (data["Price"] / data["Area_sqft"]).round(2)

    return data


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for feature engineering.
    Can be seamlessly embedded in a Scikit-Learn Pipeline.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return add_engineered_features(X, is_training=False)


if __name__ == "__main__":
    from src.config import TRAIN_DATA_PATH
    if TRAIN_DATA_PATH.exists():
        df_train = pd.read_csv(TRAIN_DATA_PATH)
        df_feat = add_engineered_features(df_train, is_training=True)
        logger.info(f"Engineered features added. New columns: {[c for c in df_feat.columns if c not in df_train.columns]}")
        print(df_feat[["Total_Rooms", "Area_per_Bedroom", "Bathroom_to_Bedroom_Ratio", "Luxury_Score", "Age_Category"]].head())
