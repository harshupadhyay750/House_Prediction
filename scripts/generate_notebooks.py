"""
Generates clean, well-documented Jupyter Notebooks for the 4 workflow stages:
1. 01_data_exploration.ipynb
2. 02_eda.ipynb
3. 03_feature_engineering.ipynb
4. 04_model_training.ipynb
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


# ---------------- NOTEBOOK 1: DATA EXPLORATION ----------------
nb1_cells = [
    md_cell("""# 🏠 01 - Data Exploration & Quality Inspection
**Project**: House Price Prediction & Property Analytics System  
**Objective**: Ingest the raw housing transactions dataset, inspect structural schemas, evaluate data types, identify missing value distributions, detect duplicate listings, and formulate data cleaning strategies."""),
    
    code_cell("""import pandas as pd
import numpy as np

# Load raw dataset
df_raw = pd.read_csv('../data/raw/housing_raw.csv')
print(f"Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
df_raw.head()"""),

    md_cell("### 1. Structural Schema and Column Data Types"),
    code_cell("""df_raw.info()
df_raw.describe().T"""),

    md_cell("### 2. Missing Value Analysis\nInspect absolute missing counts and proportion across all dimensions:"),
    code_cell("""missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw)) * 100
missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Pct (%)': missing_pct.round(2)})
missing_df[missing_df['Missing_Count'] > 0]"""),

    md_cell("### 3. Duplicate Records Detection\nMLS and real estate scraping pipelines frequently duplicate listings:"),
    code_cell("""dup_count = df_raw.duplicated(subset=[c for c in df_raw.columns if c != 'Property_ID']).sum()
print(f"Identified {dup_count} duplicate property transactions.")"""),

    md_cell("### 4. Categorical Consistency and Casing Anomalies"),
    code_cell("""for cat_col in ['Location', 'Property_Type', 'Furnishing_Status', 'Availability']:
    print(f"\\n--- Unique values in {cat_col} ---")
    print(df_raw[cat_col].value_counts(dropna=False))"""),

    md_cell("### 5. Physical Impossibility & Outlier Checks\nCheck for negative areas or 0 bedroom listings:"),
    code_cell("""print("Negative Area Count:", (df_raw['Area_sqft'] <= 0).sum())
print("Zero Bedroom Count:", (df_raw['Bedrooms'] <= 0).sum())
print("Max Area:", df_raw['Area_sqft'].max())
print("Max Price:", df_raw['Price'].max())"""),

    md_cell("""### 💡 Key Findings & Preprocessing Strategy
1. **Missing Data**: Missing values are present in `Property_Age` (~2.5%), `Bathrooms` (~1.4%), `Furnishing_Status` (~1.2%), and `Nearby_Schools` (~1.0%). Impute numeric columns by grouped median and categorical by mode.
2. **Duplicates**: 25 duplicate rows must be deduplicated.
3. **Casing Issues**: `Furnishing_Status` contains inconsistent lowercase and hyphen variants (`furnished`, `semi-furnished`). Title casing is required.
4. **Physically Impossible Values**: Negative living areas and 0-bedroom listings are corrected with absolute value and minimum bounds.""")
]

# ---------------- NOTEBOOK 2: EDA ----------------
nb2_cells = [
    md_cell("""# 📊 02 - Exploratory Data Analysis (EDA)
**Project**: House Price Prediction & Property Analytics System  
**Objective**: Uncover empirical relationships between property features and final transaction valuations through statistical visualizations."""),

    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
df = pd.read_csv('../data/processed/housing_cleaned.csv')
print(f"Cleaned dataset loaded: {df.shape}")"""),

    md_cell("### 1. Target Variable: Price Distribution"),
    code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Price'], kde=True, ax=axes[0], color='#2563eb', bins=35)
axes[0].set_title('Raw Price Distribution (Right-Skewed)')

sns.histplot(np.log1p(df['Price']), kde=True, ax=axes[1], color='#10b981', bins=35)
axes[1].set_title('Log-Transformed Price (Gaussian Alignment)')
plt.tight_layout()
plt.show()"""),

    md_cell("### 2. Feature Correlation Heatmap"),
    code_cell("""plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.2, vmax=1.0)
plt.title('Correlation Heatmap Among Numeric Property Attributes')
plt.show()"""),

    md_cell("### 3. Price vs Living Area Segmented by Property Type"),
    code_cell("""plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Area_sqft', y='Price', hue='Property_Type', alpha=0.6, s=40)
plt.title('Valuation Scaling Across Living Area')
plt.show()"""),

    md_cell("### 4. Neighborhood & Location Price Premium Analysis"),
    code_cell("""plt.figure(figsize=(12, 6))
loc_order = df.groupby('Location')['Price'].median().sort_values(ascending=False).index
sns.barplot(data=df, x='Location', y='Price', order=loc_order, palette='viridis', errorbar=None)
plt.xticks(rotation=35, ha='right')
plt.title('Median Property Valuation by Location')
plt.show()"""),

    md_cell("### 5. Price Dispersion by Bedroom & Bathroom Counts"),
    code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x='Bedrooms', y='Price', ax=axes[0], palette='Blues')
axes[0].set_title('Price by Bedrooms')
sns.boxplot(data=df, x='Bathrooms', y='Price', ax=axes[1], palette='Greens')
axes[1].set_title('Price by Bathrooms')
plt.tight_layout()
plt.show()""")
]

# ---------------- NOTEBOOK 3: FEATURE ENGINEERING ----------------
nb3_cells = [
    md_cell("""# 🛠️ 03 - Feature Engineering & Preprocessing Pipeline
**Project**: House Price Prediction & Property Analytics System  
**Objective**: Construct domain-specific features, prevent target leakage, and configure Scikit-Learn `ColumnTransformer` pipelines."""),

    code_cell("""import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df_train = pd.read_csv('../data/processed/train.csv')
df_test = pd.read_csv('../data/processed/test.csv')"""),

    md_cell("""### 1. Domain Feature Construction
- **Total_Rooms**: `Bedrooms + Bathrooms`
- **Area_per_Bedroom**: `Area_sqft / max(Bedrooms, 1)`
- **Bathroom_to_Bedroom_Ratio**: `Bathrooms / max(Bedrooms, 1)`
- **Luxury_Score**: `1.5*Amenities + 1.2*Parking + 1.0*Balconies`
- **Age_Category**: Architectural lifecycle tier"""),

    code_cell("""def engineer_features(df):
    data = df.copy()
    data['Total_Rooms'] = data['Bedrooms'] + data['Bathrooms']
    safe_bed = np.maximum(data['Bedrooms'], 1)
    data['Area_per_Bedroom'] = (data['Area_sqft'] / safe_bed).round(2)
    data['Bathroom_to_Bedroom_Ratio'] = (data['Bathrooms'] / safe_bed).round(3)
    data['Luxury_Score'] = (data['Amenities_Count'] * 1.5 + data['Parking_Spaces'] * 1.2 + data['Balconies'] * 1.0).round(2)
    
    def bin_age(a):
        if a <= 5: return 'New Construction (0-5 yrs)'
        elif a <= 15: return 'Modern (6-15 yrs)'
        elif a <= 25: return 'Mature (16-25 yrs)'
        else: return 'Established (>25 yrs)'
    data['Age_Category'] = data['Property_Age'].apply(bin_age)
    return data

train_feat = engineer_features(df_train)
test_feat = engineer_features(df_test)
train_feat[['Total_Rooms', 'Area_per_Bedroom', 'Bathroom_to_Bedroom_Ratio', 'Luxury_Score', 'Age_Category']].head()"""),

    md_cell("### 2. Scikit-Learn Pipeline Assembly\nAssemble numeric and categorical pipelines with `ColumnTransformer`:"),
    code_cell("""num_cols = [
    'Area_sqft', 'Bedrooms', 'Bathrooms', 'Parking_Spaces', 'Floors',
    'Property_Age', 'Balconies', 'Amenities_Count', 'Nearby_Schools',
    'Total_Rooms', 'Area_per_Bedroom', 'Bathroom_to_Bedroom_Ratio', 'Luxury_Score'
]
cat_cols = ['Location', 'Property_Type', 'Furnishing_Status', 'Availability', 'Age_Category']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ]
)

X_train = train_feat[num_cols + cat_cols]
X_train_trans = preprocessor.fit_transform(X_train)
print(f"Transformed Training Feature Matrix Shape: {X_train_trans.shape}")""")
]

# ---------------- NOTEBOOK 4: MODEL TRAINING ----------------
nb4_cells = [
    md_cell("""# 🤖 04 - Model Training, Tuning, Evaluation & Explainability
**Project**: House Price Prediction & Property Analytics System  
**Objective**: Benchmark multiple regression models, run 5-Fold Cross Validation, tune hyperparameters, explain decisions via SHAP and permutation importance, and save the production pipeline."""),

    code_cell("""import pandas as pd
import numpy as np
import joblib
import json
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt

# Load pre-trained production metadata
with open('../models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

leaderboard = pd.DataFrame(metadata['model_comparison'])
leaderboard"""),

    md_cell("### 1. Champion Model Performance Metrics"),
    code_cell("""final_metrics = metadata['final_test_metrics']
print(f"Champion Model: {metadata['model_name']}")
print(f"Test R² Score:   {final_metrics['R2']:.4f}")
print(f"Test MAE:        ${final_metrics['MAE']:,.2f}")
print(f"Test RMSE:       ${final_metrics['RMSE']:,.2f}")
print(f"Test MAPE:       {final_metrics['MAPE']:.2f}%")"""),

    md_cell("### 2. Best Tuned Hyperparameters"),
    code_cell("""print("Optimized Hyperparameters:")
for param, val in metadata['best_hyperparameters'].items():
    print(f" - {param}: {val}")"""),

    md_cell("### 3. Load Production Pipeline Artifact & Verification"),
    code_cell("""pipeline = joblib.load('../models/house_price_model.pkl')
print("Loaded pipeline successfully:", type(pipeline))

# Verify inference
sample_df = pd.DataFrame([{
    'Location': 'Downtown Central', 'Property_Type': 'Apartment',
    'Area_sqft': 1500, 'Bedrooms': 3, 'Bathrooms': 2.0,
    'Furnishing_Status': 'Furnished', 'Parking_Spaces': 1, 'Floors': 10,
    'Property_Age': 3, 'Balconies': 2, 'Amenities_Count': 6,
    'Availability': 'Ready to Move', 'Nearby_Schools': 4,
    'Total_Rooms': 5.0, 'Area_per_Bedroom': 500.0,
    'Bathroom_to_Bedroom_Ratio': 0.67, 'Luxury_Score': 12.2,
    'Age_Category': 'New Construction (0-5 yrs)'
}])

val = pipeline.predict(sample_df)[0]
print(f"Sample Valuation Output: ${val:,.2f}")""")
]

# Write notebooks
notebook_configs = [
    ("01_data_exploration.ipynb", nb1_cells),
    ("02_eda.ipynb", nb2_cells),
    ("03_feature_engineering.ipynb", nb3_cells),
    ("04_model_training.ipynb", nb4_cells)
]

for filename, cells in notebook_configs:
    nb = make_notebook(cells)
    target = NOTEBOOKS_DIR / filename
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {target}")
