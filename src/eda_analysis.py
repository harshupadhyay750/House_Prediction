"""
Exploratory Data Analysis (EDA) Module for House Price Prediction System.
Generates 11 professional publication-quality figures with business interpretations.
Adapted to the user dataset schema: Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import TRAIN_DATA_PATH, FIGURES_DIR
from src.feature_engineering import add_engineered_features

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8


def format_currency(x, pos):
    if x >= 1e6:
        return f"${x*1e-6:.2f}M"
    elif x >= 1e3:
        return f"${x*1e-3:.0f}K"
    else:
        return f"${x:.0f}"


def run_eda():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_DATA_PATH)
    df_feat = add_engineered_features(df, is_training=True)
    
    currency_formatter = ticker.FuncFormatter(format_currency)
    
    # 1. Target Price Distribution & Log Price
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df_feat["Price"], kde=True, ax=axes[0], color="#2563eb", bins=35)
    axes[0].set_title("Property Price Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Property Price (USD)")
    axes[0].xaxis.set_major_formatter(currency_formatter)
    axes[0].set_ylabel("Frequency")
    
    log_price = np.log1p(df_feat["Price"])
    sns.histplot(log_price, kde=True, ax=axes[1], color="#10b981", bins=35)
    axes[1].set_title("Log-Transformed Price Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Log(Price)")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_price_distribution.png", dpi=300)
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = [
        "Price", "Area", "Bedrooms", "Bathrooms", "Floors",
        "YearBuilt", "Property_Age", "Total_Rooms", "Area_per_Bedroom"
    ]
    corr = df_feat[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-0.3, vmax=1.0, cbar_kws={'label': 'Pearson Correlation'}
    )
    plt.title("Numerical Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_correlation_heatmap.png", dpi=300)
    plt.close()

    # 3. Price vs Area by Location
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_feat, x="Area", y="Price", hue="Location",
        palette="deep", alpha=0.65, s=40
    )
    plt.title("Property Price vs. Area (sq ft) Segmented by Location", fontsize=13, fontweight="bold")
    plt.xlabel("Living Area (Square Feet)")
    plt.ylabel("Price (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.legend(title="Location", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_price_vs_area.png", dpi=300)
    plt.close()

    # 4. Price vs Bedrooms
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df_feat, x="Bedrooms", y="Price", palette="Blues", showmeans=True)
    plt.title("Price Distribution Across Bedroom Counts", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Price (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_price_vs_bedrooms.png", dpi=300)
    plt.close()

    # 5. Price vs Bathrooms
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df_feat, x="Bathrooms", y="Price", palette="Greens", showmeans=True)
    plt.title("Price Distribution Across Bathroom Counts", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Bathrooms")
    plt.ylabel("Price (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_price_vs_bathrooms.png", dpi=300)
    plt.close()

    # 6. Price by Location
    plt.figure(figsize=(10, 5))
    loc_order = df_feat.groupby("Location")["Price"].median().sort_values(ascending=False).index
    sns.barplot(data=df_feat, x="Location", y="Price", order=loc_order, palette="viridis", errorbar=None)
    plt.title("Median Property Valuation by Prime Location", fontsize=13, fontweight="bold")
    plt.xlabel("Location")
    plt.ylabel("Average Valuation (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_price_by_location.png", dpi=300)
    plt.close()

    # 7. Price by Condition
    plt.figure(figsize=(8, 5))
    cond_order = ["Excellent", "Good", "Fair", "Poor"]
    sns.violinplot(data=df_feat, x="Condition", y="Price", order=cond_order, palette="Set2", inner="quartile")
    plt.title("Price Distribution Across Property Condition Tiers", fontsize=13, fontweight="bold")
    plt.xlabel("Condition")
    plt.ylabel("Price (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_price_by_furnishing.png", dpi=300)
    plt.close()

    # 8. Price by Garage
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df_feat, x="Garage", y="Price", palette="magma", errorbar=None)
    plt.title("Average Market Price: Garage vs. No Garage", fontsize=13, fontweight="bold")
    plt.xlabel("Garage Available")
    plt.ylabel("Average Price (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_price_by_property_type.png", dpi=300)
    plt.close()

    # 9. Outlier Analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.boxenplot(y=df_feat["Area"], ax=axes[0], color="#3b82f6")
    axes[0].set_title("Living Area Distribution & Outliers", fontweight="bold")
    axes[0].set_ylabel("Area (sq ft)")
    
    sns.boxenplot(y=df_feat["Price"], ax=axes[1], color="#ef4444")
    axes[1].set_title("Price Distribution & Upper Whiskers", fontweight="bold")
    axes[1].yaxis.set_major_formatter(currency_formatter)
    axes[1].set_ylabel("Price (USD)")
    
    sns.boxenplot(y=df_feat["Price_per_sqft"], ax=axes[2], color="#8b5cf6")
    axes[2].set_title("Price/SqFt Dispersion", fontweight="bold")
    axes[2].set_ylabel("USD per sq ft")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_outlier_analysis.png", dpi=300)
    plt.close()

    # 10. Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    cols_to_plot = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt", "Total_Rooms"]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"]
    for i, col in enumerate(cols_to_plot):
        ax = axes[i // 3, i % 3]
        sns.histplot(df_feat[col], kde=True, ax=ax, color=colors[i], bins=20)
        ax.set_title(f"Distribution of {col}", fontweight="bold", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_feature_distributions.png", dpi=300)
    plt.close()

    # 11. Pairwise Relationships
    pair_cols = ["Price", "Area", "Total_Rooms", "YearBuilt"]
    g = sns.pairplot(df_feat[pair_cols], diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15, "color": "#1d4ed8"})
    g.fig.suptitle("Pairwise Relationships Among Core Drivers", y=1.02, fontweight="bold", fontsize=13)
    plt.savefig(FIGURES_DIR / "11_pairwise_relationships.png", dpi=300)
    plt.close()

    print("Generated all 11 EDA charts in reports/figures/")


if __name__ == "__main__":
    run_eda()
