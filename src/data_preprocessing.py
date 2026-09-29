"""
Data Preprocessing Module for House Price Prediction System.
Handles data loading, inspection, cleaning, consistency formatting,
outlier detection, and train/test splitting without data leakage.
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH, CLEANED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH,
    RANDOM_STATE, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform automated inspection on the dataset.
    Returns summary statistics, missing value percentages, and duplicate counts.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100
    missing_summary = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percent": missing_pct.round(2)
    })
    missing_summary = missing_summary[missing_summary["missing_count"] > 0]
    
    duplicate_count = int(df.duplicated().sum())
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Detected {duplicate_count} duplicate rows.")
    if not missing_summary.empty:
        logger.info(f"Missing values found in {len(missing_summary)} columns:\n{missing_summary}")
    else:
        logger.info("No missing values found.")
        
    return {
        "shape": df.shape,
        "duplicates": duplicate_count,
        "missing_summary": missing_summary.to_dict(orient="index"),
        "dtypes": df.dtypes.astype(str).to_dict()
    }


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw property data:
    1. Remove duplicate records
    2. Normalize categorical casing and strip whitespaces
    3. Fix/remove impossible physical values (negative area, 0 bedrooms)
    4. Handle extreme data entry errors and anomalies
    5. Handle missing values:
       - Numerical: Impute with median grouped by property type or global median
       - Categorical: Impute with mode
    """
    logger.info("Starting data cleaning pipeline...")
    df_clean = df.copy()

    # 1. Remove duplicates
    initial_len = len(df_clean)
    # Check duplicates excluding Property_ID if present, or across all feature columns
    feature_cols = [c for c in df_clean.columns if c != "Property_ID"]
    df_clean = df_clean.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    dropped_dups = initial_len - len(df_clean)
    logger.info(f"Dropped {dropped_dups} duplicate records.")

    # 2. Standardize Categorical Columns (fix casing, whitespace, inconsistent labels)
    for col in ["Furnishing_Status", "Property_Type", "Location", "Availability"]:
        if col in df_clean.columns:
            def _normalize_cat(val):
                if pd.isna(val) or str(val).lower() in ["nan", "none", ""]:
                    return np.nan
                s = str(val).strip().title()
                if col == "Furnishing_Status":
                    mapping = {
                        "Semi-Furnished": "Semi-Furnished",
                        "Semi - Furnished": "Semi-Furnished",
                        "Furnished": "Furnished",
                        "Unfurnished": "Unfurnished"
                    }
                    return mapping.get(s, s)
                return s

            df_clean[col] = df_clean[col].apply(_normalize_cat)

    # 3. Detect and handle impossible values
    # Impossible: Area <= 0
    if "Area_sqft" in df_clean.columns:
        invalid_area_mask = df_clean["Area_sqft"] <= 0
        invalid_count = invalid_area_mask.sum()
        if invalid_count > 0:
            logger.warning(f"Detected {invalid_count} records with impossible non-positive Area_sqft. Correcting with absolute value.")
            df_clean.loc[invalid_area_mask, "Area_sqft"] = df_clean.loc[invalid_area_mask, "Area_sqft"].abs()

    # Impossible: Bedrooms <= 0 for non-studio or in general
    if "Bedrooms" in df_clean.columns:
        invalid_bed_mask = df_clean["Bedrooms"] <= 0
        invalid_bed_count = invalid_bed_mask.sum()
        if invalid_bed_count > 0:
            logger.warning(f"Detected {invalid_bed_count} records with Bedrooms <= 0. Imputing to minimum 1 bedroom.")
            df_clean.loc[invalid_bed_mask, "Bedrooms"] = 1

    # 4. Outlier Filtering for extreme recording errors
    # Extreme area error: > 15,000 sqft or Price > $10,000,000
    if "Area_sqft" in df_clean.columns and "Price" in df_clean.columns:
        outlier_mask = (df_clean["Area_sqft"] > 12000) | (df_clean["Price"] > 10000000)
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            logger.warning(f"Removing {outlier_count} extreme entry error outliers.")
            df_clean = df_clean[~outlier_mask].reset_index(drop=True)

    # 5. Missing value imputation
    # Impute categorical missing values with mode
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            mode_val = df_clean[col].mode(dropna=True)[0]
            df_clean[col] = df_clean[col].fillna(mode_val)

    # Impute numeric missing values with group medians (by Property_Type if available, else median)
    if "Property_Type" in df_clean.columns:
        for col in ["Bathrooms", "Property_Age", "Nearby_Schools"]:
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col] = df_clean.groupby("Property_Type")[col].transform(
                    lambda s: s.fillna(s.median())
                )
                # Fallback global median if still NaN
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        for col in NUMERICAL_FEATURES:
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    logger.info(f"Cleaning complete. Retained {len(df_clean)} valid property records.")
    return df_clean


def split_and_save_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the cleaned dataset into train and test sets (80/20 split)
    with fixed random_state to ensure reproducibility and prevent data leakage.
    Saves cleaned, train, and test datasets into data/processed/.
    """
    logger.info(f"Splitting dataset into train ({int((1-test_size)*100)}%) and test ({int(test_size*100)}%)...")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    
    logger.info(f"Saved cleaned data to {CLEANED_DATA_PATH} ({len(df)} rows)")
    logger.info(f"Saved train set to {TRAIN_DATA_PATH} ({len(train_df)} rows)")
    logger.info(f"Saved test set to {TEST_DATA_PATH} ({len(test_df)} rows)")
    
    return train_df, test_df


if __name__ == "__main__":
    if not RAW_DATA_PATH.exists():
        logger.error(f"Raw data file not found at {RAW_DATA_PATH}")
    else:
        raw_df = pd.read_csv(RAW_DATA_PATH)
        inspect_dataset(raw_df)
        cleaned_df = clean_raw_data(raw_df)
        split_and_save_data(cleaned_df)
