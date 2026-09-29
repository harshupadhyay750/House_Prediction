"""
Model Evaluation and Error Analysis Module for House Price Prediction System.
Computes evaluation metrics (MAE, MSE, RMSE, R2, MAPE), produces comparison tables,
and performs residual and subgroup error diagnostics.
"""

import time
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, training_time: float = 0.0) -> Dict[str, float]:
    """
    Computes regression performance metrics for house price valuation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0  # as percentage
    
    return {
        "MAE": round(float(mae), 2),
        "MSE": round(float(mse), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape), 2),
        "Training_Time_Sec": round(float(training_time), 3)
    }


def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_features: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Conducts in-depth error and residual analysis:
    - Identifies top 10 largest prediction errors
    - Evaluates error dispersion across price quartiles
    - Evaluates error across locations if provided
    """
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)
    pct_errors = (abs_errors / y_true) * 100.0

    analysis_df = pd.DataFrame({
        "Actual_Price": y_true,
        "Predicted_Price": y_pred,
        "Residual": residuals,
        "Absolute_Error": abs_errors,
        "Pct_Error": pct_errors
    })

    if df_features is not None:
        for col in ["Location", "Property_Type", "Area_sqft", "Bedrooms"]:
            if col in df_features.columns:
                analysis_df[col] = df_features[col].values

    # Price quartiles error breakdown
    analysis_df["Price_Tier"] = pd.qcut(
        analysis_df["Actual_Price"],
        q=4,
        labels=["Budget (Q1)", "Mid-Tier (Q2)", "Upper-Tier (Q3)", "Luxury (Q4)"]
    )
    tier_summary = analysis_df.groupby("Price_Tier", observed=False).agg(
        Mean_Absolute_Error=("Absolute_Error", "mean"),
        Median_Absolute_Error=("Absolute_Error", "median"),
        Mean_MAPE=("Pct_Error", "mean"),
        Record_Count=("Actual_Price", "count")
    ).round(2).reset_index()

    # Top 10 worst prediction errors
    worst_10 = analysis_df.sort_values(by="Absolute_Error", ascending=False).head(10).to_dict(orient="records")

    return {
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "tier_summary": tier_summary.to_dict(orient="records"),
        "worst_10_errors": worst_10
    }
