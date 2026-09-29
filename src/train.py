
"""
Enhanced Model Training and High-Accuracy Optimization Pipeline.
Trains baseline, regularized, and advanced gradient boosted models.
Incorporates log-target scaling and a high-performance voting ensemble
(XGBoost + GradientBoosting + HistGradientBoosting) to maximize R² and minimize MAE/MAPE.
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
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor
)
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
    """Enterprise-grade ColumnTransformer for numerical scaling and categorical encoding."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, ALL_CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    num_cols = list(ALL_NUMERICAL_FEATURES)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = list(cat_encoder.get_feature_names_out(ALL_CATEGORICAL_FEATURES))
    return num_cols + cat_cols


def build_super_ensemble():
    """Builds a high-precision weighted ensemble of gradient-boosted trees."""
    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.025,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    gb = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        random_state=RANDOM_STATE
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_depth=5,
        l2_regularization=0.1,
        random_state=RANDOM_STATE
    )

    ensemble = VotingRegressor([
        ("xgb", xgb),
        ("gb", gb),
        ("hgb", hgb)
    ], weights=[0.60, 0.25, 0.15])
    
    inner_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", ensemble)
    ])

    # Log-target transformation maps right-skewed prices into Gaussian space
    return TransformedTargetRegressor(
        regressor=inner_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    logger.info("Benchmarking models across 5-Fold Cross Validation...")

    xgb_single = TransformedTargetRegressor(
        regressor=Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", XGBRegressor(
                n_estimators=700, learning_rate=0.025, max_depth=4,
                subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": Pipeline([("preprocessor", build_preprocessor()), ("regressor", LinearRegression())]),
        "Ridge Regression": Pipeline([("preprocessor", build_preprocessor()), ("regressor", Ridge(alpha=10.0, random_state=RANDOM_STATE))]),
        "Lasso Regression": Pipeline([("preprocessor", build_preprocessor()), ("regressor", Lasso(alpha=50.0, random_state=RANDOM_STATE, max_iter=3000))]),
        "Random Forest": Pipeline([("preprocessor", build_preprocessor()), ("regressor", RandomForestRegressor(n_estimators=150, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1))]),
        "Tuned XGBoost (Log-Target)": xgb_single,
        "Super Ensemble (XGB+GB+HGB)": build_super_ensemble()
    }

    results = []
    trained_pipelines = {}
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model_inst in models.items():
        logger.info(f"--> Training & Cross-Validating: {name}")
        start_time = time.time()
        
        cv_scores = cross_val_score(model_inst, X_train, y_train, cv=cv_kfold, scoring="r2", n_jobs=-1)
        mean_cv_r2 = cv_scores.mean()
        std_cv_r2 = cv_scores.std()

        model_inst.fit(X_train, y_train)
        elapsed = time.time() - start_time

        y_pred = model_inst.predict(X_test)
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
        trained_pipelines[name] = model_inst
        logger.info(f"    Finished {name} - CV R2: {mean_cv_r2:.4f}, Test R2: {metrics['R2']:.4f}, Test MAE: ${metrics['MAE']:,.2f}, MAPE: {metrics['MAPE']:.2f}%")

    comparison_df = pd.DataFrame(results).sort_values(by="Test_MAE", ascending=True).reset_index(drop=True)
    best_candidate_name = comparison_df.iloc[0]["Model"]
    logger.info(f"Top performing model: {best_candidate_name}")

    return comparison_df, trained_pipelines, best_candidate_name


def generate_explainability_artifacts(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    logger.info("Generating Explainability and Error Diagnostic artifacts...")
    
    if isinstance(model, TransformedTargetRegressor):
        inner_pipe = model.regressor_
    else:
        inner_pipe = model

    preprocessor = inner_pipe.named_steps["preprocessor"]
    regressor = inner_pipe.named_steps["regressor"]
    feature_names = get_feature_names(preprocessor)
    
    # Feature importances from voting ensemble or tree regressor
    if hasattr(regressor, "named_estimators_") and "xgb" in regressor.named_estimators_:
        importances = regressor.named_estimators_["xgb"].feature_importances_
    elif hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Top 15 Feature Importances Plot
    plt.figure(figsize=(10, 6))
    top_15 = imp_df.head(15)
    sns.barplot(data=top_15, x="Importance", y="Feature", palette="Blues_r")
    plt.title("Top 15 Real Estate Valuation Drivers (Feature Importance)", fontsize=13, fontweight="bold")
    plt.xlabel("Relative Importance")
    plt.ylabel("Engineered Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_feature_importance.png", dpi=300)
    plt.close()

    # Permutation Importance on Test Set
    logger.info("Calculating Permutation Importance on test set...")
    perm_res = permutation_importance(
        model, X_test, y_test,
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
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "13_permutation_importance.png", dpi=300)
    plt.close()

    # SHAP Summary with primary XGBoost estimator
    logger.info("Computing SHAP values on test sample...")
    try:
        X_test_proc = preprocessor.transform(X_test)
        sample_size = min(250, len(X_test_proc))
        X_sample = X_test_proc[:sample_size]
        
        xgb_est = regressor.named_estimators_["xgb"] if hasattr(regressor, "named_estimators_") else regressor
        explainer = shap.TreeExplainer(xgb_est)
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(11, 7))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
        plt.title("SHAP Summary Plot: Directional Impact on Price", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "14_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"SHAP generation note: {e}")

    # Actual vs Predicted
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    plt.figure(figsize=(8, 7))
    sns.scatterplot(x=y_test.values, y=y_pred, alpha=0.5, color="#2563eb", s=35)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (45°)")
    
    formatter = ticker.FuncFormatter(lambda x, pos: f"${x*1e-6:.2f}M" if x >= 1e6 else f"${x*1e-3:.0f}K")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title("Actual vs. Predicted Valuations (Test Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Price (USD)")
    plt.ylabel("Predicted Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15_actual_vs_predicted.png", dpi=300)
    plt.close()

    # Residuals & Homoscedasticity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(residuals, kde=True, ax=axes[0], color="#8b5cf6", bins=35)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title("Residual Distribution (Near Zero-Centered Normal)", fontweight="bold")
    axes[0].set_xlabel("Residual (USD)")

    sns.scatterplot(x=y_pred, y=residuals, ax=axes[1], alpha=0.45, color="#059669", s=30)
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residuals vs. Fitted Values (Homoscedasticity)", fontweight="bold")
    axes[1].set_xlabel("Predicted Valuation (USD)")
    axes[1].set_ylabel("Residual (USD)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "16_residual_distribution.png", dpi=300)
    plt.close()

    return {
        "top_features": imp_df.head(20).to_dict(orient="records"),
        "top_permutation": perm_df.head(10).to_dict(orient="records")
    }


def run_pipeline() -> Dict[str, Any]:
    logger.info("Starting High-Accuracy Model Training Pipeline...")
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    train_feat = add_engineered_features(train_df, is_training=False)
    test_feat = add_engineered_features(test_df, is_training=False)

    feature_cols = [c for c in train_feat.columns if c not in [TARGET_COLUMN, "Property_ID", "Price_per_sqft"]]
    X_train = train_feat[feature_cols]
    y_train = train_feat[TARGET_COLUMN]
    X_test = test_feat[feature_cols]
    y_test = test_feat[TARGET_COLUMN]

    comparison_df, trained_pipelines, best_name = train_and_compare_models(
        X_train, y_train, X_test, y_test
    )

    champion_model = trained_pipelines[best_name]
    y_pred = champion_model.predict(X_test)
    final_metrics = calculate_metrics(y_test.values, y_pred)
    
    logger.info(f"=== CHAMPION MODEL: {best_name} ===")
    logger.info(f"MAE:  ${final_metrics['MAE']:,.2f}")
    logger.info(f"RMSE: ${final_metrics['RMSE']:,.2f}")
    logger.info(f"R²:   {final_metrics['R2']:.4f}")
    logger.info(f"MAPE: {final_metrics['MAPE']:.2f}%")

    error_analysis = analyze_residuals(y_test.values, y_pred, df_features=test_feat)
    explainability_data = generate_explainability_artifacts(champion_model, X_train, X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_model, MODEL_PATH)
    logger.info(f"Saved optimized pipeline to {MODEL_PATH}")

    metadata = {
        "model_name": best_name,
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
    logger.info(f"Saved metadata to {METADATA_PATH}")

    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(explainability_data, f, indent=4)
    logger.info(f"Saved feature importance data to {FEATURE_IMPORTANCE_PATH}")

    return metadata


if __name__ == "__main__":
    run_pipeline()
