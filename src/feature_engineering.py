"""
Feature Engineering Module for House Price Prediction System.
Generates domain-specific features adapted to the property dataset:
- Property_Age (from YearBuilt)
- Total_Rooms (Bedrooms + Bathrooms)
- Area_per_Bedroom
- Bathroom_to_Bedroom_Ratio
- Log_Area
- Bed_Bath_Interaction
- Rooms_per_Floor
- Est_Depreciation
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def add_engineered_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Applies feature engineering transformations to property DataFrame.
    Guarantees zero target leakage.
    """
    data = df.copy()

    # 1. Property Age
    if "YearBuilt" in data.columns:
        data["Property_Age"] = 2026 - data["YearBuilt"]
    else:
        data["Property_Age"] = data.get("Property_Age", 10.0)

    # 2. Total Rooms
    data["Total_Rooms"] = data["Bedrooms"] + data["Bathrooms"]

    # 3. Area per Bedroom (Space efficiency)
    safe_bedrooms = np.maximum(data["Bedrooms"], 1)
    data["Area_per_Bedroom"] = (data["Area"] / safe_bedrooms).round(2)

    # 4. Bathroom to Bedroom Ratio
    data["Bathroom_to_Bedroom_Ratio"] = (data["Bathrooms"] / safe_bedrooms).round(3)

    # 5. Log Area
    data["Log_Area"] = np.log1p(data["Area"]).round(4)

    # 6. Bed Bath Interaction
    data["Bed_Bath_Interaction"] = data["Bedrooms"] * data["Bathrooms"]

    # 7. Rooms per Floor
    safe_floors = np.maximum(data["Floors"] if "Floors" in data.columns else 1, 1)
    data["Rooms_per_Floor"] = (data["Total_Rooms"] / safe_floors).round(2)

    # 8. Estimated Depreciation
    data["Est_Depreciation"] = np.maximum(0.60, 1.0 - (data["Property_Age"] * 0.005)).round(4)

    # EDA Only: Price per sqft
    if is_training and "Price" in data.columns and "Area" in data.columns:
        data["Price_per_sqft"] = (data["Price"] / data["Area"]).round(2)

    return data


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
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
        print("Engineered features sample:\n", df_feat.head(2))
