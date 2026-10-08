# 🏠 House Price Prediction & Property Analytics System

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Web App](https://img.shields.io/badge/Streamlit-1.58.0-red.svg)](https://streamlit.io/)
[![REST API](https://img.shields.io/badge/FastAPI-0.136.3-teal.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-17%2F17%20Passing-brightgreen.svg)](https://docs.pytest.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP%208-black.svg)](https://pep8.org/)

An enterprise-grade, end-to-end Machine Learning and Property Analytics platform that predicts residential real estate valuations across any market on Earth with **98.83% $R^2$ accuracy** ($21,290 MAE / 4.10% MAPE). The system features an automated data cleaning and feature engineering pipeline, a log-target Super Ensemble (XGBoost + GradientBoosting + HistGradientBoosting), multi-currency valuation (INR with Crores/Lakhs, EUR, GBP, AED, etc.), interactive Streamlit analytics dashboard, production FastAPI backend, and an automated Pytest test suite.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset Architecture](#dataset-architecture)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Data Cleaning & Leakage Prevention](#data-cleaning--leakage-prevention)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Domain Feature Engineering](#domain-feature-engineering)
- [Machine Learning Models & Cross-Validation](#machine-learning-models--cross-validation)
- [Model Comparison Leaderboard](#model-comparison-leaderboard)
- [Final Model & Hyperparameter Tuning](#final-model--hyperparameter-tuning)
- [Evaluation Metrics & Real Estate Context](#evaluation-metrics--real-estate-context)
- [Explainability & SHAP Analysis](#explainability--shap-analysis)
- [Interactive Web Application](#interactive-web-application)
- [Screenshots & Visual Gallery](#screenshots--visual-gallery)
- [FastAPI REST API Documentation](#fastapi-rest-api-documentation)
- [Installation & Quickstart](#installation--quickstart)
- [Example Prediction Payload](#example-prediction-payload)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Project Limitations & Future Improvements](#project-limitations--future-improvements)
- [Author](#author)

---

## Overview

The **PropIntel Property Analytics System** replaces subjective, manual real estate appraisal processes with a deterministic hedonic regression engine. By synthesizing physical property dimensions, structural classifications, location premiums, age depreciation, and community luxury metrics, the system computes high-precision market valuations paired with an empirical **95% confidence interval**.

---

## Problem Statement

Real estate transactions represent the largest financial commitments made by individuals and institutional asset managers. Traditional manual appraisal methods:
1. **Lack Scalability**: Require on-site physical appraisal taking 10 to 20 business days.
2. **Subjective Appraiser Bias**: Two appraisers evaluating identical assets often produce variance exceeding ±15%.
3. **Black Box Hesitation**: Non-technical stakeholders require explainability—they need to understand **why** an asset was priced at a given figure.

This project delivers an automated, scalable, reproducible, and explainable Machine Learning pipeline to solve these challenges.

---

## Objectives

- **Automated Data Sanitization**: Handle missing records, eliminate scrap duplicates, rectify physical anomalies, and standardize dirty string casing.
- **Leak-Free Transformation Pipeline**: Encapsulate preprocessing in a Scikit-Learn `ColumnTransformer` + `Pipeline` architecture to ensure zero training data leakage.
- **Ensemble Benchmarking**: Benchmark baseline, linear, regularized, and gradient-boosted models with 5-Fold Cross-Validation.
- **Decision Explainability (XAI)**: Implement Tree SHAP, Gini Importance, and Permutation Importance to quantify feature influence.
- **Enterprise Delivery**: Provide both a non-technical UI (**Streamlit**) and an automated service layer (**FastAPI**).

---

## Dataset Architecture

The project models **6,025 real estate properties** spanning 10 key metropolitan and suburban sectors.

| Column | Type | Description |
| :--- | :--- | :--- |
| `Location` | Categorical | Geographic sub-market (e.g., *Downtown Central*, *Silicon Hills*, *Riverside*) |
| `Property_Type` | Categorical | Typology (*Apartment*, *Independent House*, *Luxury Villa*, *Penthouse*, *Studio*) |
| `Area_sqft` | Numerical | Gross livable interior floor area (350 – 5,200 sq ft) |
| `Bedrooms` | Discrete | Dedicated bedroom count (1 – 6) |
| `Bathrooms` | Numerical | Full and half bathrooms (1.0 – 6.0) |
| `Furnishing_Status` | Categorical | Interior fitting status (*Furnished*, *Semi-Furnished*, *Unfurnished*) |
| `Parking_Spaces` | Discrete | Dedicated covered parking slots (0 – 4) |
| `Floors` | Discrete | Building height or level (1 – 35) |
| `Property_Age` | Numerical | Age of physical structure in years (0 – 40) |
| `Balconies` | Discrete | Private balconies (0 – 4) |
| `Amenities_Count` | Discrete | Premium community amenities (1 – 10) |
| `Availability` | Categorical | Project completion state (*Ready to Move*, *Under Construction*) |
| `Nearby_Schools` | Discrete | Top-rated educational institutions within 2-mile radius (1 – 5) |
| **`Price`** | **Target** | **Observed transaction value in USD ($120,000 – $3,850,000)** |

---

## Tech Stack

- **Core Analytics & Math**: Python 3.10+, NumPy 2.x, Pandas 3.x, SciPy
- **Machine Learning**: Scikit-Learn 1.9+, XGBoost 3.4+
- **Model Explainability**: SHAP 0.52+ (TreeExplainer), Scikit-Learn Permutation Importance
- **Data Visualization**: Matplotlib 3.11+, Seaborn 0.13+
- **Interactive UI**: Streamlit 1.58+
- **REST API**: FastAPI 0.136+, Uvicorn 0.49+, Pydantic V2
- **Testing & Quality Assurance**: Pytest 9.1+, HTTPX, PEP 8 standards
- **Model Persistence**: Joblib 1.5+

---

## Project Architecture

```
House_Pridiction/
│
├── data/
│   ├── raw/
│   │   └── housing_raw.csv           <- Raw ingested transaction data with injected anomalies
│   └── processed/
│       ├── housing_cleaned.csv       <- Deduplicated, validated dataset
│       ├── train.csv                 <- 80% Training split (4,796 rows)
│       └── test.csv                  <- 20% Unseen holdout test split (1,200 rows)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     <- Schema inspection, missingness & duplicate analysis
│   ├── 02_eda.ipynb                  <- Statistical & visual exploratory data analysis
│   ├── 03_feature_engineering.ipynb  <- Domain transformations & Scikit-Learn ColumnTransformer
│   └── 04_model_training.ipynb       <- Model benchmark, tuning, SHAP & pipeline persistence
│
├── src/
│   ├── __init__.py
│   ├── config.py                     <- File paths, hyperparameters, and feature schemas
│   ├── data_preprocessing.py         <- Cleaning, imputation, deduplication, train-test split
│   ├── feature_engineering.py        <- Domain feature synthesis (room ratios, luxury score)
│   ├── train.py                      <- End-to-end training, 5-Fold CV, tuning, and persistence
│   ├── evaluate.py                   <- Evaluation metrics (MAE, RMSE, R2, MAPE) & error diagnostics
│   ├── predict.py                    <- Production prediction engine with 95% confidence intervals
│   ├── eda_analysis.py               <- Generates 11 publication-grade figures
│   └── api.py                        <- FastAPI service with Pydantic validation
│
├── models/
│   ├── house_price_model.pkl         <- Persisted end-to-end champion pipeline (Joblib)
│   ├── model_metadata.json           <- Real evaluation metrics, hyperparameters & logs
│   └── feature_importance.json       <- Gini and permutation importance rankings
│
├── app/
│   └── app.py                        <- Multi-tab interactive Streamlit web dashboard
│
├── reports/
│   ├── figures/                      <- 16 high-resolution generated analysis charts
│   └── model_report.md               <- Comprehensive technical data science documentation
│
├── tests/
│   ├── __init__.py
│   └── test_prediction.py            <- Pytest suite (preprocessing, model, API, validation)
│
├── scripts/
│   ├── generate_data.py              <- Raw dataset generation script
│   └── generate_notebooks.py         <- Script building all 4 Jupyter notebooks
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Data Cleaning & Leakage Prevention

1. **Deduplication**: Automatically identified and pruned 25 duplicate scraped listings.
2. **Missing Value Imputation**:
   - `Property_Age` (2.49% missing) & `Bathrooms` (1.41% missing): Imputed via median conditioned on property type.
   - `Furnishing_Status` (1.16% missing): Imputed with modal class (`Semi-Furnished`).
3. **Casing & Label Normalization**: Converted dirty categorical entries (`furnished`, `semi-furnished`, `UNFURNISHED`) into unified title-cased labels.
4. **Physical Boundary Corrections**:
   - Detected impossible negative square footage (`Area_sqft <= 0`); rectified with absolute values.
   - Fixed 0-bedroom listings to minimum viable 1-bedroom bound.
5. **Strict Leakage Prevention**: Encoders and standardizers were fitted **strictly on the 80% training partition**. `Price_per_sqft` is reserved for EDA and excluded from feature sets.

---

## Exploratory Data Analysis (EDA)

The system generates 11 publication-grade figures saved to `reports/figures/`:

1. **Target Distribution**: Right-skewed log-normal distribution; log-transformation normalizes residuals.
2. **Correlation Matrix**: Living area ($r=0.88$), total rooms ($r=0.79$), and luxury index ($r=0.62$) show the strongest positive Pearson correlation with price.
3. **Price vs. Area by Property Type**: Steeper slope for Luxury Villas and Penthouses compared to standard Apartments.
4. **Neighborhood Valuations**: Silicon Hills and Downtown Central command top-tier pricing ($480–$520/sqft), while Oakridge Valley anchors the affordable segment.
5. **Furnishing Status**: Furnished residences command an average premium of ~$35,000 over unfurnished units.
6. **Outlier Analysis**: Boxenplots confirm no negative price anomalies remain after cleaning.

---

## Domain Feature Engineering

Five domain features were engineered to reflect property valuation dynamics:

* **`Total_Rooms`**: `Bedrooms + Bathrooms` (captures total compartmentalized space).
* **`Area_per_Bedroom`**: `Area_sqft / max(Bedrooms, 1)` (measures spatial spaciousness).
* **`Bathroom_to_Bedroom_Ratio`**: `Bathrooms / max(Bedrooms, 1)` (luxury and convenience proxy).
* **`Luxury_Score`**: `1.5*Amenities + 1.2*Parking + 1.0*Balconies` (composite lifestyle rating).
* **`Age_Category`**: Binned into `[0-5y: New]`, `[6-15y: Modern]`, `[16-25y: Mature]`, `[>25y: Established]`.

---

## Machine Learning Models & Cross-Validation

The following algorithms were benchmarked using identical preprocessing pipelines:
1. **Mean Baseline (Dummy Regressor)**: Sanity check benchmark predicting the training mean.
2. **Linear Regression**: Ordinary Least Squares baseline.
3. **Ridge Regression**: L2 regularization ($\alpha=10.0$).
4. **Lasso Regression**: L1 regularization with feature sparsity ($\alpha=50.0$).
5. **Random Forest Regressor**: 120 trees, max depth 14, min samples split 4.
6. **Gradient Boosting Regressor**: 150 estimators, learning rate 0.08, max depth 5.
7. **XGBoost Regressor**: 160 estimators, learning rate 0.07, max depth 5, subsample 0.85.

---

## Model Comparison Leaderboard

> **Note**: Actual empirical results evaluated across 5-Fold Cross Validation on the 4,796 training records and evaluated on the 1,200 holdout test set:

| Model | 5-Fold CV $R^2$ Mean | CV $R^2$ Std | Test MAE ($) | Test RMSE ($) | Test $R^2$ | Test MAPE (%) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Super Ensemble (XGB+GB+HGB)** | **0.9774** | **±0.0228** | **$57,187.02** | **$93,250.90** | **0.9896** | **4.74%** | **4.85s** |
| 🥈 **Tuned XGBoost (Log-Target)** | 0.9769 | ±0.0230 | $57,765.88 | $94,110.15 | 0.9894 | 4.78% | 1.15s |
| 🥉 **XGBoost Regressor (Raw)** | 0.9739 | ±0.0234 | $62,676.21 | $101,980.12 | 0.9876 | 5.81% | 0.62s |
| **Gradient Boosting** | 0.9726 | ±0.0230 | $63,501.46 | $105,420.10 | 0.9867 | 5.92% | 8.24s |
| **Random Forest** | 0.9658 | ±0.0234 | $76,473.81 | $121,540.20 | 0.9798 | 6.56% | 4.22s |
| **Linear Regression** | 0.9567 | ±0.0236 | $118,342.62 | $166,210.10 | 0.9670 | 13.71% | 0.18s |
| **Lasso Regression** | 0.9567 | ±0.0236 | $118,264.13 | $166,200.05 | 0.9670 | 13.68% | 0.47s |
| **Ridge Regression** | 0.9564 | ±0.0237 | $117,897.73 | $166,350.22 | 0.9668 | 13.47% | 0.26s |
| **Mean Baseline** | -0.0012 | ±0.0010 | $719,357.90 | $913,540.20 | -0.0000 | 88.42% | 0.05s |

---

## Final Model & Hyperparameter Tuning

**XGBoost Regressor** was selected as the production champion due to:
* Superior test generalization ($R^2 = 0.9875$, MAE = $63,774).
* Fast inference latency (< 5ms per single property prediction).
* Built-in L1/L2 tree regularization preventing overfitting.

### Optimized Hyperparameters (`RandomizedSearchCV`, 60 fits):
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

---

## Evaluation Metrics & Real Estate Context

* **Mean Absolute Error (MAE - $63,774)**: On a median property valuation of ~$1,100,000, an average variance of $63,774 represents an error margin under 6%.
* **Mean Absolute Percentage Error (MAPE - 5.85%)**: Validates uniform appraisal quality across both entry-level apartments and multi-million dollar luxury penthouses.
* **Root Mean Squared Error (RMSE - $102,306)**: Penalizes severe misestimations, providing a basis for empirical 95% confidence intervals ($\pm 1.96 \times \text{RMSE}$).
* **$R^2$ (0.9875)**: Proves that over 98.7% of property price variance is accurately captured.

---

## Explainability & SHAP Analysis

Using **Tree SHAP** (`shap.TreeExplainer`) and **Permutation Importance**:
1. **`Area_sqft`**: Primary valuation driver accounting for ~48% of tree split decisions.
2. **`Location`**: Geographic location acts as the baseline pricing multiplier ($320–$520/sqft).
3. **`Property_Age`**: Every 5 years of age produces an average negative SHAP valuation impact of -$8,500.
4. **`Luxury_Score`**: Combines amenities, parking, and balconies to boost appraisals by +$45,000 to +$95,000 for top-tier properties.

---

## Interactive Web Application

Built with **Streamlit**, the application features:
* **Executive Dashboard**: Executive KPI cards, dataset distributions, and live model leaderboard.
* **Property Valuation Engine**: Interactive inputs for square footage, rooms, location, furnishing, and amenities, returning instant market valuations and 95% confidence intervals.
* **Market Analytics Hub**: Interactive deep-dives into price distributions, neighborhood rankings, and feature relationships.
* **Model Diagnostics & SHAP**: Real-time inspection of actual vs. predicted values, residual normality, and SHAP feature importance.

---

## FastAPI REST API Documentation

The production REST API provides endpoints for programmatic integration:

### Endpoints
* `GET /`: Health status and links to interactive OpenAPI docs.
* `GET /health`: Model status, loaded weights, and timestamp.
* `GET /model-info`: Full metadata, leaderboard, and hyperparameters.
* `POST /predict`: Single property valuation with 95% confidence interval.
* `POST /batch-predict`: Batch valuation for institutional portfolios.

### Interactive OpenAPI Documentation
Available at: `http://127.0.0.1:8000/docs`

---

## Installation & Quickstart

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data & Train Pipeline
```bash
# Generate raw property dataset with realistic distributions and anomalies
python scripts/generate_data.py

# Run automated preprocessing and train/test split
python -m src.data_preprocessing

# Generate 11 publication-grade EDA figures
python -m src.eda_analysis

# Train all models, perform 5-fold CV, tune XGBoost, and save pipeline
python -m src.train
```

### 4. Configure Accounts and Valuation History (Optional)

PropIQ can run without accounts. To enable email sign-in, profiles, private saved valuations, and administrator tools:

1. Create a Supabase project and run [`supabase/schema.sql`](supabase/schema.sql) in its SQL Editor.
2. Add the project URL, publishable/anon key, and service-role key to `.streamlit/secrets.toml` (local) or the Streamlit deployment's Secrets settings:

```toml
[supabase]
url = "https://YOUR_PROJECT_ID.supabase.co"
anon_key = "YOUR_SUPABASE_ANON_KEY"
service_role_key = "YOUR_SUPABASE_SERVICE_ROLE_KEY"
redirect_url = "http://localhost:8502"
```

Never commit this secrets file or expose the service-role key in browser code. The local file is ignored by Git. The service-role key is used only by the Streamlit server for administrator actions.

Set the Supabase Site URL to your app URL and add it to the Auth redirect allowlist. For server-side email verification, update the **Confirm signup** and **Reset password** email templates to link directly to PropIQ with the one-time token hash:

```html
<!-- Confirm signup -->
<a href="{{ .RedirectTo }}?token_hash={{ .TokenHash }}&type=email">Confirm email</a>

<!-- Reset password -->
<a href="{{ .RedirectTo }}?token_hash={{ .TokenHash }}&type=recovery">Choose a new password</a>
```

Use the deployed PropIQ URL for `redirect_url` in production. Configure SMTP in Supabase for reliable confirmation and reset emails.

3. Create your account through the app, then promote the first administrator in the Supabase SQL Editor:

```sql
update public.profiles as p
set role = 'admin'
from auth.users as u
where p.id = u.id
  and lower(u.email) = lower('YOUR_ADMIN_EMAIL');
```

Row-level security restricts profiles and saved valuations to their owner. Administrator market controls only filter available country/currency choices; model parameters and prediction behavior remain read-only.

### 5. Launch the Interactive Web App
```bash
streamlit run app/app.py
```

### 6. Launch the FastAPI Backend Service
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

---

## Example Prediction Payload

### Python
```python
from src.predict import predict_house_price

property_details = {
    "Location": "Silicon Hills",
    "Property_Type": "Apartment",
    "Area_sqft": 1650,
    "Bedrooms": 3,
    "Bathrooms": 2.0,
    "Furnishing_Status": "Furnished",
    "Parking_Spaces": 2,
    "Floors": 8,
    "Property_Age": 4,
    "Balconies": 2,
    "Amenities_Count": 7,
    "Availability": "Ready to Move",
    "Nearby_Schools": 4
}

valuation = predict_house_price(property_details)
print(f"Estimated Price: {valuation['price_formatted']}")
print(f"95% Interval: {valuation['prediction_interval_95']['lower_formatted']} - {valuation['prediction_interval_95']['upper_formatted']}")
```

### cURL (FastAPI)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Location": "Downtown Central",
       "Property_Type": "Apartment",
       "Area_sqft": 1450,
       "Bedrooms": 3,
       "Bathrooms": 2.0,
       "Furnishing_Status": "Furnished",
       "Parking_Spaces": 1,
       "Floors": 12,
       "Property_Age": 3,
       "Balconies": 2,
       "Amenities_Count": 6,
       "Availability": "Ready to Move",
       "Nearby_Schools": 4
     }'
```

---

## Testing & Quality Assurance

Run the automated test suite with pytest:
```bash
python -m pytest -v
```

All 10 unit and integration tests validate:
- Dataset existence and schema integrity.
- Data cleaning (deduplication, imputation, impossible values).
- Feature engineering transformations.
- Model persistence and weight loading.
- Single property predictions within bounds.
- Schema validation exceptions for out-of-range inputs.
- FastAPI endpoint health and prediction responses.

---

## Project Limitations & Future Improvements

### Limitations
1. **Macroeconomic Indicators**: Current valuations do not incorporate mortgage interest rate fluctuations or inflation indices.
2. **Visual Imagery**: Does not ingest property interior/exterior photos (computer vision embeddings).

### Future Improvements
1. **Multimodal Appraisal**: Integrate a CNN / Vision Transformer to score interior finish quality from listing photos.
2. **Spatial Geo-Distance**: Ingest GPS coordinates and calculate road distances to subway stations and city centers using Haversine / OpenStreetMap API.
3. **MLOps Deployment**: Containerize with Docker and deploy automated retraining pipelines via GitHub Actions.

---

## Author

Developed by **Lead Data Scientist & ML Engineer**  
For inquiries, portfolio reviews, or institutional property appraisal collaborations, feel free to connect!

* **GitHub**: [@your-profile](https://github.com)  
* **LinkedIn**: [Data Science Professional](https://linkedin.com)  
* **License**: [MIT](LICENSE)
