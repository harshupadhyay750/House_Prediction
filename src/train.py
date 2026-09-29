"""
Model Training and Optimization Pipeline for House Price Prediction System.
Trains baseline, linear, regularized, and tree-based ensemble models.
Performs 5-Fold Cross-Validation, Hyperparameter Optimization,
Explainability Analysis (Feature Importance, Permutation Importance, SHAP),
Error Diagnostics, and persists the production pipeline using joblib.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import shap

from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, TARGET_COLUMN,
    ALL_NUMERICAL_FEATURES, ALL_CATEGORICAL_FEATURES,
    RANDOM_STATE, MODEL_PATH, METADATA_PATH, FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR
)
from src.feature_engineering import add_engineered_features
from src.evaluate import calculate_metrics, analyze_residuals

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an enterprise-grade scikit-learn ColumnTransformer.
    - Numerical: median imputation + standard scaling
    - Categorical: most_frequent imputation + one-hot encoding (ignores unseen)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, ALL_CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts output feature names after ColumnTransformer transformation."""
    num_cols = list(ALL_NUMERICAL_FEATURES)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = list(cat_encoder.get_feature_names_out(ALL_CATEGORICAL_FEATURES))
    return num_cols + cat_cols


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], str]:
    """
    Trains and benchmarks multiple regression algorithms across 5-Fold Cross Validation
    and independent Test set evaluation.
    """
    logger.info("Initializing model algorithms for benchmark comparison...")

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=50.0, random_state=RANDOM_STATE, max_iter=3000),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=14, min_samples_split=4,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=5,
            random_state=RANDOM_STATE
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=160, learning_rate=0.07, max_depth=5,
            subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }

    results = []
    trained_pipelines = {}
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, regressor in models.items():
        logger.info(f"--> Training & Cross-Validating: {name}")
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor)
        ])

        # 5-Fold Cross Validation R2 on training data
        start_time = time.time()
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_kfold, scoring="r2", n_jobs=-1)
        mean_cv_r2 = cv_scores.mean()
        std_cv_r2 = cv_scores.std()

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start_time

        # Test evaluation
        y_pred = pipeline.predict(X_test)
        metrics = calculate_metrics(y_test.values, y_pred, training_time=elapsed)

        res_entry = {
            "Model": name,
            "CV_R2_Mean": round(float(mean_cv_r2), 4),
            "CV_R2_Std": round(float(std_cv_r2), 4),
            "Test_MAE": metrics["MAE"],
            "Test_RMSE": metrics["RMSE"],
            "Test_R2": metrics["R2"],
            "Test_MAPE": metrics["MAPE"],
            "Training_Time_Sec": metrics["Training_Time_Sec"]
        }
        results.append(res_entry)
        trained_pipelines[name] = pipeline
        logger.info(f"    Finished {name} - CV R2: {mean_cv_r2:.4f} (±{std_cv_r2:.4f}), Test R2: {metrics['R2']:.4f}, Test MAE: ${metrics['MAE']:,.2f}")

    comparison_df = pd.DataFrame(results).sort_values(by="Test_R2", ascending=False).reset_index(drop=True)
    best_candidate_name = comparison_df.iloc[0]["Model"]
    logger.info(f"Top performing model prior to tuning: {best_candidate_name}")

    return comparison_df, trained_pipelines, best_candidate_name


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_model_name: str
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Performs RandomizedSearchCV on the selected champion algorithm to optimize performance.
    """
    logger.info(f"Initiating Hyperparameter Tuning for {best_model_name}...")
    
    if "XGBoost" in best_model_name:
        base_regressor = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {
            "regressor__n_estimators": [150, 200, 250, 300],
            "regressor__max_depth": [4, 5, 6, 7],
            "regressor__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "regressor__subsample": [0.75, 0.85, 0.95],
            "regressor__colsample_bytree": [0.75, 0.85, 0.95],
            "regressor__min_child_weight": [1, 3, 5]
        }
    else:
        base_regressor = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {
            "regressor__n_estimators": [120, 180, 250],
            "regressor__max_depth": [10, 14, 18, None],
            "regressor__min_samples_split": [2, 5, 8],
            "regressor__min_samples_leaf": [1, 2, 4],
            "regressor__max_features": ["sqrt", "log2", None]
        }

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", base_regressor)
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=15,
        scoring="neg_root_mean_squared_error",
        cv=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    logger.info(f"Tuning completed. Best parameters: {search.best_params_}")
    
    clean_params = {k.replace("regressor__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, clean_params


def generate_explainability_artifacts(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Generates Feature Importance, Permutation Importance, SHAP Analysis,
    and Error Diagnostic Visualizations saved to reports/figures/.
    """
    logger.info("Generating Explainability and Error Diagnostic artifacts...")
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    feature_names = get_feature_names(preprocessor)
    
    # 1. Tree Feature Importance
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    else:
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": np.ones(len(feature_names))})

    # Plot Top 15 Feature Importances
    plt.figure(figsize=(10, 6))
    top_15 = imp_df.head(15)
    sns.barplot(data=top_15, x="Importance", y="Feature", palette="Blues_r")
    plt.title("Top 15 Drivers of House Valuation (Model Gini Importance)", fontsize=13, fontweight="bold")
    plt.xlabel("Relative Importance")
    plt.ylabel("Engineered Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_feature_importance.png", dpi=300)
    plt.close()

    # 2. Permutation Importance on Test Set
    logger.info("Calculating Permutation Importance on test set...")
    perm_res = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    perm_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Permutation_Mean": perm_res.importances_mean,
        "Permutation_Std": perm_res.importances_std
    }).sort_values(by="Permutation_Mean", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=perm_df.head(12), x="Permutation_Mean", y="Feature", palette="Greens_r")
    plt.title("Permutation Importance on Test Set (Drop in R²)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Importance (R² Metric Drop)")
    plt.ylabel("Raw / Domain Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "13_permutation_importance.png", dpi=300)
    plt.close()

    # 3. SHAP Analysis
    logger.info("Computing SHAP values for local and global model explainability...")
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Use a sample of 250 records for swift, responsive SHAP computation
    sample_size = min(250, len(X_test_proc))
    X_sample = X_test_proc[:sample_size]
    
    try:
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(11, 7))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
        plt.title("SHAP Summary Plot: Directional Impact of Features on Valuation", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "14_shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"TreeExplainer exception: {e}. Fallback to permutation importance.")

    # 4. Error Diagnostics: Actual vs Predicted
    y_pred = pipeline.predict(X_test)
    residuals = y_test.values - y_pred

    plt.figure(figsize=(8, 7))
    sns.scatterplot(x=y_test.values, y=y_pred, alpha=0.5, color="#2563eb", s=35)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect 45° Line")
    
    formatter = ticker.FuncFormatter(lambda x, pos: f"${x*1e-6:.2f}M" if x>=1e6 else f"${x*1e-3:.0f}K")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title("Actual vs. Predicted Property Valuations (Test Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Market Price (USD)")
    plt.ylabel("Predicted Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15_actual_vs_predicted.png", dpi=300)
    plt.close()

    # 5. Residual Distribution & Homoscedasticity Check
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(residuals, kde=True, ax=axes[0], color="#8b5cf6", bins=35)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title("Residual Distribution (Error Bell Curve)", fontweight="bold")
    axes[0].set_xlabel("Prediction Error (Residual in USD)")
    axes[0].set_ylabel("Count")

    sns.scatterplot(x=y_pred, y=residuals, ax=axes[1], alpha=0.45, color="#059669", s=30)
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residuals vs. Fitted Values (Homoscedasticity)", fontweight="bold")
    axes[1].set_xlabel("Predicted Price (USD)")
    axes[1].set_ylabel("Residual (USD)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "16_residual_distribution.png", dpi=300)
    plt.close()

    return {
        "top_features": imp_df.head(20).to_dict(orient="records"),
        "top_permutation": perm_df.head(10).to_dict(orient="records")
    }


def run_pipeline() -> Dict[str, Any]:
    """Orchestrates end-to-end training, tuning, evaluation, and persistence."""
    logger.info("Initiating Full ML Pipeline Execution...")
    
    # 1. Load Data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # 2. Feature Engineering
    train_feat = add_engineered_features(train_df, is_training=False)
    test_feat = add_engineered_features(test_df, is_training=False)

    feature_cols = [c for c in train_feat.columns if c not in [TARGET_COLUMN, "Property_ID", "Price_per_sqft"]]
    X_train = train_feat[feature_cols]
    y_train = train_feat[TARGET_COLUMN]
    X_test = test_feat[feature_cols]
    y_test = test_feat[TARGET_COLUMN]

    # 3. Model Benchmark
    comparison_df, trained_pipelines, best_name = train_and_compare_models(
        X_train, y_train, X_test, y_test
    )

    # 4. Hyperparameter Tuning
    tuned_pipeline, best_params = tune_hyperparameters(X_train, y_train, best_name)
    
    # Final Test Set Evaluation of Tuned Pipeline
    y_pred_tuned = tuned_pipeline.predict(X_test)
    final_metrics = calculate_metrics(y_test.values, y_pred_tuned)
    logger.info(f"=== FINAL TUNED MODEL: {best_name} ===")
    logger.info(f"MAE:  ${final_metrics['MAE']:,.2f}")
    logger.info(f"RMSE: ${final_metrics['RMSE']:,.2f}")
    logger.info(f"R²:   {final_metrics['R2']:.4f}")
    logger.info(f"MAPE: {final_metrics['MAPE']:.2f}%")

    # 5. Error Diagnostics
    error_analysis = analyze_residuals(y_test.values, y_pred_tuned, df_features=test_feat)

    # 6. Explainability & Diagnostics Plots
    explainability_data = generate_explainability_artifacts(
        tuned_pipeline, X_train, X_test, y_test
    )

    # 7. Model Persistence
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_pipeline, MODEL_PATH)
    logger.info(f"Saved optimized model pipeline to {MODEL_PATH}")

    # 8. Save Metadata
    metadata = {
        "model_name": best_name,
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_hyperparameters": best_params,
        "final_test_metrics": final_metrics,
        "model_comparison": comparison_df.to_dict(orient="records"),
        "error_analysis_summary": {
            "mean_residual": error_analysis["mean_residual"],
            "std_residual": error_analysis["std_residual"],
            "tier_summary": error_analysis["tier_summary"]
        },
        "feature_columns": feature_cols,
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved pipeline metadata to {METADATA_PATH}")

    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(explainability_data, f, indent=4)
    logger.info(f"Saved feature importance data to {FEATURE_IMPORTANCE_PATH}")

    return metadata


if __name__ == "__main__":
    run_pipeline()
