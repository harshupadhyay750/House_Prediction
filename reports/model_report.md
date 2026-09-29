# 🏠 Technical Model Report: House Price Prediction & Property Analytics System

**Author**: Lead Data Scientist  
**Date**: September 2026  
**Artifact Version**: `v1.0.0-prod`  
**Model Champion**: Optimized XGBoost Regressor Pipeline  
**Model Persistence**: `models/house_price_model.pkl`  

---

## 1. Executive Summary

Accurate property valuation is fundamental to real estate investments, mortgage underwriting, and portfolio risk management. Traditional manual appraisals suffer from human bias, geographic inconsistency, and lag times exceeding two to three weeks.

This report documents the design, development, and validation of an **Enterprise Hedonic Regression Engine** that estimates real estate market valuations instantaneously from physical, geographic, and amenity attributes. 

Across rigorous 5-Fold Cross-Validation and an unseen 1,200-property test holdout, the finalized **XGBoost Regressor** pipeline achieved:
* **Coefficient of Determination ($R^2$)**: **0.9875** (explaining 98.75% of market price variance)
* **Mean Absolute Error (MAE)**: **$63,774.69** (a 46.8% reduction compared to linear baselines)
* **Mean Absolute Percentage Error (MAPE)**: **5.85%** (industry benchmark target is < 10%)
* **Root Mean Squared Error (RMSE)**: **$102,306.02**

The model is encapsulated in a leak-free Scikit-Learn `Pipeline` with automated preprocessing, served via an interactive **Streamlit dashboard** and a production **FastAPI REST API**.

---

## 2. Dataset Architecture & Ingestion

The system was developed on a dataset comprising **6,025 property transactions** featuring 15 structural, geographic, and community attributes:

| Variable | Type | Description | Domain Range |
| :--- | :--- | :--- | :--- |
| `Property_ID` | String | Unique property transaction identifier | PROP_10001 to PROP_16000 |
| `Location` | Categorical | Prime urban/suburban real estate corridor | 10 Localities (e.g. Downtown Central, Silicon Hills) |
| `Property_Type` | Categorical | Architectural asset classification | Apartment, House, Villa, Penthouse, Studio |
| `Area_sqft` | Numerical | Gross livable interior square footage | 350 to 5,200 sq ft |
| `Bedrooms` | Discrete | Dedicated bedroom count | 1 to 6 |
| `Bathrooms` | Numerical | Bathroom count | 1.0 to 6.0 |
| `Furnishing_Status` | Categorical | Interior fitting status | Furnished, Semi-Furnished, Unfurnished |
| `Parking_Spaces` | Discrete | Dedicated covered parking slots | 0 to 4 |
| `Floors` | Discrete | Floor level or building height | 1 to 35 |
| `Property_Age` | Numerical | Age of physical structure in years | 0 to 40 years |
| `Balconies` | Discrete | Private outdoor balconies | 0 to 4 |
| `Amenities_Count` | Discrete | Count of amenities (pool, gym, clubhouse, etc.) | 1 to 10 |
| `Availability` | Categorical | Project completion state | Ready to Move, Under Construction |
| `Nearby_Schools` | Discrete | Reputable educational institutions within 2 mi | 1 to 5 |
| **`Price`** | **Target (USD)** | **Final observed market transaction price** | **$120,000 to $3,850,000** |

---

## 3. Data Cleaning & Integrity Safeguards

Real estate feeds contain systematic anomalies arising from automated MLS web scrapers and manual listing entries. The following data cleaning steps were implemented:

1. **Deduplication**: 25 exact duplicate listings were identified and eliminated.
2. **Missing Value Imputation**:
   - `Property_Age` (2.49% missing): Imputed using group medians conditioned on `Property_Type`.
   - `Bathrooms` (1.41% missing): Imputed using group medians conditioned on `Property_Type`.
   - `Furnishing_Status` (1.16% missing): Imputed using the modal category (`Semi-Furnished`).
   - `Nearby_Schools` (1.00% missing): Imputed using the regional median.
3. **Casing Normalization**: Standardized mixed casing variants (`furnished`, `FURNISHED`, `semi-furnished`) into unified title-cased labels.
4. **Physical Boundary Corrections**:
   - Detected 4 instances with non-positive living area (`Area_sqft <= 0`); corrected via absolute magnitude.
   - Detected 4 instances with zero bedrooms; clipped to minimum viable bound of 1 bedroom.
5. **Extreme Outlier Filtering**: 4 severe recording errors (>15,000 sq ft or >$10,000,000) were pruned, leaving **5,996 clean records**.
6. **Data Leakage Prevention**: Data splitting occurred strictly prior to transformer fitting; encoders and scalers compute parameters purely from the 80% training split (4,796 samples).

---

## 4. Exploratory Data Analysis & Business Insights

Eleven visualizations were generated and analyzed:

1. **Target Distribution**: Raw property prices exhibit a log-normal right skew. While tree-based algorithms (XGBoost/Random Forest) are invariant to monotonic transformations, linear baselines benefit significantly from log scaling.
2. **Correlation Structure**: `Area_sqft` ($r = 0.88$), `Total_Rooms` ($r = 0.79$), and `Luxury_Score` ($r = 0.62$) show the strongest positive Pearson correlation with transaction price.
3. **Price vs. Area by Asset Class**: Clear hierarchical pricing separation emerges across asset classes. Luxury Villas and Penthouses exhibit steeper price-per-square-foot trajectories than standard Apartments.
4. **Location Valuation Premia**: *Silicon Hills* and *Harbor Point* command highest baseline valuation rates ($490-$520 / sqft), whereas suburban corridors like *Oakridge Valley* and *Sunset Park* anchor the affordable tier ($320-$340 / sqft).
5. **Furnishing Value Add**: Fully furnished residences maintain an average premium of ~$35,000 over unfurnished units, after controlling for square footage.

---

## 5. Domain Feature Engineering

To capture property economics, five domain features were constructed:

* **Total Rooms** (`Bedrooms + Bathrooms`): Represents spatial compartmentalization.
* **Area per Bedroom** (`Area_sqft / max(Bedrooms, 1)`): Measures room spaciousness.
* **Bathroom-to-Bedroom Ratio** (`Bathrooms / max(Bedrooms, 1)`): Strong proxy for executive and luxury property standards.
* **Luxury Score** (`1.5 * Amenities + 1.2 * Parking + 1.0 * Balconies`): Composite lifestyle index.
* **Age Category** (`[0-5y: New]`, `[6-15y: Modern]`, `[16-25y: Mature]`, `[>25y: Established]`): Captures non-linear maintenance and depreciation phases.

> **Target Leakage Safeguard**: `Price_per_sqft` was evaluated strictly for EDA reporting and explicitly excluded from modeling feature sets.

---

## 6. Model Benchmark & Empirical Comparison

Seven regression candidates were evaluated using **5-Fold Cross-Validation** on the training partition (4,796 samples) and tested on the holdout partition (1,200 samples).

### Comprehensive Benchmark Table

| Model Architecture | CV $R^2$ Mean | CV $R^2$ Std | Test MAE ($) | Test RMSE ($) | Test $R^2$ | Test MAPE (%) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor (Tuned)** | **0.9741** | **±0.0231** | **$63,774.69** | **$102,306.02** | **0.9875** | **5.85%** | 0.85s |
| **XGBoost Regressor (Default)** | 0.9739 | ±0.0234 | $62,676.21 | $101,980.12 | 0.9876 | 5.81% | 0.62s |
| **Gradient Boosting Regressor** | 0.9726 | ±0.0230 | $63,501.46 | $105,420.10 | 0.9867 | 5.92% | 8.24s |
| **Random Forest Regressor** | 0.9681 | ±0.0234 | $74,285.76 | $118,310.45 | 0.9833 | 7.04% | 8.87s |
| **Linear Regression** | 0.9563 | ±0.0236 | $119,995.88 | $168,140.23 | 0.9662 | 11.23% | 0.18s |
| **Lasso Regression ($\alpha=50$)** | 0.9563 | ±0.0236 | $119,876.80 | $168,110.15 | 0.9662 | 11.21% | 0.47s |
| **Ridge Regression ($\alpha=10$)** | 0.9562 | ±0.0237 | $119,002.01 | $168,230.80 | 0.9661 | 11.15% | 0.26s |
| **Mean Baseline (Dummy)** | -0.0012 | ±0.0010 | $719,357.90 | $913,540.20 | -0.0000 | 88.45% | 0.05s |

### Metric Justification
* **MAE ($63,774)**: Direct dollar measure of typical appraisal discrepancy.
* **RMSE ($102,306)**: Penalizes large pricing misestimations.
* **$R^2$ (0.9875)**: Proves that 98.75% of pricing variance is accurately explained.
* **MAPE (5.85%)**: Demonstrates that relative estimation error remains bounded under 6% across price points.

---

## 7. Hyperparameter Optimization

A 4-fold `RandomizedSearchCV` exploring 60 candidate combinations was executed for the XGBoost pipeline.

**Optimal Configuration:**
```python
{
    'n_estimators': 250,
    'learning_rate': 0.08,
    'max_depth': 4,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.95
}
```
*Restricting tree depth to 4 with subsampling at 85% prevented overfitting while capturing non-linear interactions between square footage and neighborhood multipliers.*

---

## 8. Model Explainability & Interpretability (XAI)

Model decisions were analyzed using:

1. **Gini Tree Importance**: `Area_sqft` (48.2%), `Area_per_Bedroom` (14.5%), and `Property_Type_Luxury Villa` (10.2%) constitute the top three split contributors.
2. **Permutation Importance**: Shuffling `Area_sqft` results in an $R^2$ drop of over 0.72 on unseen test data.
3. **SHAP Summary Analysis**:
   - `Area_sqft`: Strong monotonic positive contribution.
   - `Location_Silicon Hills`: Positive SHAP value shift of +$120,000.
   - `Property_Age`: Consistent negative SHAP pull of -$8,500 per additional 5 years.

---

## 9. Error Diagnostics & Subgroup Performance

Residual diagnostic tests confirm model validity:
* **Residual Mean**: -$1,420 (statistically indistinguishable from zero).
* **Homoscedasticity**: Residual dispersion remains proportional across budget and luxury segments.

### Quartile Performance Breakdown

| Price Tier | Price Range | Mean Abs Error (MAE) | Mean MAPE |
| :--- | :--- | :---: | :---: |
| **Budget (Q1)** | $120K – $620K | $34,120.45 | 7.12% |
| **Mid-Tier (Q2)** | $620K – $1.15M | $51,480.12 | 5.82% |
| **Upper-Tier (Q3)** | $1.15M – $1.85M | $72,390.80 | 4.95% |
| **Luxury (Q4)** | $1.85M – $3.85M | $97,010.55 | 3.91% |

*The model demonstrates lower percentage error (MAPE < 4%) on high-value properties due to distinct spatial signals.*

---

## 10. Deployment & Integration

The system exposes two primary consumption interfaces:
1. **Interactive Streamlit App (`app/app.py`)**: Designed for property underwriters, portfolio analysts, and homebuyers. Provides instant valuation, 95% confidence intervals, and SHAP diagnostics.
2. **FastAPI REST Service (`src/api.py`)**: High-throughput REST API supporting single and batch property appraisals with strict Pydantic schema validation.
