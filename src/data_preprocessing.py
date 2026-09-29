"""
Data Preprocessing Module for House Price Prediction System.
Handles data loading, inspection, cleaning, and train/test splitting.
Adapted to the user property dataset.
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
    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100
    missing_summary = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percent": missing_pct.round(2)
    })
    missing_summary = missing_summary[missing_summary["missing_count"] > 0]
    
    duplicate_count = int(df.duplicated(subset=[c for c in df.columns if c != "Id"]).sum())
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Detected {duplicate_count} duplicate rows.")
    return {
        "shape": df.shape,
        "duplicates": duplicate_count,
        "missing_summary": missing_summary.to_dict(orient="index")
    }


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning raw property data...")
    df_clean = df.copy()

    # Drop duplicates excluding Id
    feature_cols = [c for c in df_clean.columns if c != "Id"]
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    logger.info(f"Deduplication complete: dropped {initial_len - len(df_clean)} records.")

    # Standardize string categories
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # Check numerical bounds
    if "Area" in df_clean.columns:
        df_clean["Area"] = df_clean["Area"].abs()
    if "Bedrooms" in df_clean.columns:
        df_clean["Bedrooms"] = np.maximum(df_clean["Bedrooms"], 1)
    if "Bathrooms" in df_clean.columns:
        df_clean["Bathrooms"] = np.maximum(df_clean["Bathrooms"], 1)

    # Impute missing numericals with median
    for col in NUMERICAL_FEATURES:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    logger.info(f"Retained {len(df_clean)} clean property records.")
    return df_clean


def split_and_save_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    if RAW_DATA_PATH.exists():
        raw_df = pd.read_csv(RAW_DATA_PATH)
        inspect_dataset(raw_df)
        cleaned_df = clean_raw_data(raw_df)
        split_and_save_data(cleaned_df)
    else:
        logger.error(f"Raw data file not found at {RAW_DATA_PATH}")
