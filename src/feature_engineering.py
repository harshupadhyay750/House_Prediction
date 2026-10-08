
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

    # 4. Area per Room (Volume per habitable room)
    safe_rooms = np.maximum(data["Total_Rooms"], 1.0)
    data["Area_per_Room"] = (data["Area"] / safe_rooms).round(2)

    # 5. Area per Floor (Footprint density)
    safe_floors = np.maximum(data["Floors"] if "Floors" in data.columns else 1, 1)
    data["Area_per_Floor"] = (data["Area"] / safe_floors).round(2)

    # 6. Bathroom to Bedroom Ratio
    data["Bathroom_to_Bedroom_Ratio"] = (data["Bathrooms"] / safe_bedrooms).round(3)

    # 7. Log Area
    data["Log_Area"] = np.log1p(data["Area"]).round(4)

    # 8. Bed Bath Interaction
    data["Bed_Bath_Interaction"] = data["Bedrooms"] * data["Bathrooms"]

    # 9. Rooms per Floor
    data["Rooms_per_Floor"] = (data["Total_Rooms"] / safe_floors).round(2)

    # 10. Estimated Depreciation & Effective Area
    data["Est_Depreciation"] = np.maximum(0.60, 1.0 - (data["Property_Age"] * 0.005)).round(4)
    data["Effective_Area"] = (data["Area"] * data["Est_Depreciation"]).round(2)

    # 11. Ordinal Scores for Monotonic Tree Splits
    cond_map = {"Poor": 1.0, "Fair": 2.0, "Good": 3.0, "Excellent": 4.0}
    if "Condition" in data.columns:
        data["Condition_Score"] = data["Condition"].astype(str).str.strip().str.title().map(cond_map).fillna(2.5)
    else:
        data["Condition_Score"] = 2.5

    loc_map = {"Rural": 1.0, "Suburban": 2.0, "Urban": 3.0, "Downtown": 4.0}
    if "Location" in data.columns:
        data["Location_Tier_Score"] = data["Location"].astype(str).str.strip().str.title().map(loc_map).fillna(2.5)
    else:
        data["Location_Tier_Score"] = 2.5

    if "Garage" in data.columns:
        data["Garage_Binary"] = (data["Garage"].astype(str).str.strip().str.title() == "Yes").astype(float)
    else:
        data["Garage_Binary"] = 1.0

    # 12. Spatial and Quality Cross-Interactions
    data["Quality_Space_Interaction"] = (data["Condition_Score"] * data["Area_per_Room"]).round(2)
    data["Location_Area_Interaction"] = (data["Location_Tier_Score"] * data["Area"]).round(2)

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
