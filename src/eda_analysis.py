"""
Exploratory Data Analysis (EDA) Module for House Price Prediction System.
Generates 11 professional publication-quality figures with business interpretations.
Saves all figures to reports/figures/ for reporting, notebooks, and Streamlit app.
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

# Set high-end visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8


def format_currency(x, pos):
    """Format tick numbers into clean USD notations ($K or $M)."""
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
    sns.histplot(df_feat["Price"], kde=True, ax=axes[0], color="#2563eb", bins=40)
    axes[0].set_title("Property Price Distribution (Right-Skewed)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Property Price (USD)")
    axes[0].xaxis.set_major_formatter(currency_formatter)
    axes[0].set_ylabel("Frequency")
    
    log_price = np.log1p(df_feat["Price"])
    sns.histplot(log_price, kde=True, ax=axes[1], color="#10b981", bins=40)
    axes[1].set_title("Log-Transformed Price Distribution (Near Normal)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Log(Price)")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "01_price_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = [
        "Price", "Area_sqft", "Bedrooms", "Bathrooms", "Parking_Spaces",
        "Property_Age", "Amenities_Count", "Total_Rooms", "Luxury_Score"
    ]
    corr = df_feat[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-0.2, vmax=1.0, cbar_kws={'label': 'Pearson Correlation'}
    )
    plt.title("Numerical Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "02_correlation_heatmap.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # 3. Price vs Area by Property Type
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("deep", n_colors=df_feat["Property_Type"].nunique())
    sns.scatterplot(
        data=df_feat, x="Area_sqft", y="Price", hue="Property_Type",
        palette=palette, alpha=0.65, s=40
    )
    plt.title("Property Price vs. Area (sq ft) Segmented by Property Type", fontsize=13, fontweight="bold")
    plt.xlabel("Living Area (Square Feet)")
    plt.ylabel("Price (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.legend(title="Property Type", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "03_price_vs_area.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # 4. Price vs Bedrooms
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df_feat, x="Bedrooms", y="Price", palette="Blues", showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red"})
    plt.title("Price Distribution Across Bedroom Counts", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Price (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "04_price_vs_bedrooms.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # 5. Price vs Bathrooms
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df_feat, x="Bathrooms", y="Price", palette="Greens", showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"blue", "markeredgecolor":"blue"})
    plt.title("Price Distribution Across Bathroom Counts", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Bathrooms")
    plt.ylabel("Price (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.tight_layout()
    fig5_path = FIGURES_DIR / "05_price_vs_bathrooms.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    # 6. Price by Location
    plt.figure(figsize=(12, 6))
    loc_order = df_feat.groupby("Location")["Price"].median().sort_values(ascending=False).index
    sns.barplot(data=df_feat, x="Location", y="Price", order=loc_order, palette="viridis", errorbar=None)
    plt.title("Median Property Valuation by Prime Geographic Location", fontsize=13, fontweight="bold")
    plt.xlabel("Location")
    plt.ylabel("Average Valuation (USD)")
    plt.xticks(rotation=35, ha='right')
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.tight_layout()
    fig6_path = FIGURES_DIR / "06_price_by_location.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()

    # 7. Price by Furnishing Status
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df_feat, x="Furnishing_Status", y="Price", palette="Set2", inner="quartile")
    plt.title("Price Distribution Across Furnishing Status Tiers", fontsize=13, fontweight="bold")
    plt.xlabel("Furnishing Status")
    plt.ylabel("Price (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.tight_layout()
    fig7_path = FIGURES_DIR / "07_price_by_furnishing.png"
    plt.savefig(fig7_path, dpi=300)
    plt.close()

    # 8. Price by Property Type
    plt.figure(figsize=(10, 5))
    type_order = df_feat.groupby("Property_Type")["Price"].median().sort_values(ascending=False).index
    sns.barplot(data=df_feat, x="Property_Type", y="Price", order=type_order, palette="magma", errorbar=None)
    plt.title("Average Market Price Across Asset Types", fontsize=13, fontweight="bold")
    plt.xlabel("Asset Class")
    plt.ylabel("Average Price (USD)")
    plt.gca().yaxis.set_major_formatter(currency_formatter)
    plt.tight_layout()
    fig8_path = FIGURES_DIR / "08_price_by_property_type.png"
    plt.savefig(fig8_path, dpi=300)
    plt.close()

    # 9. Outlier Analysis (Boxenplots of Key Numerical Features)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.boxenplot(y=df_feat["Area_sqft"], ax=axes[0], color="#3b82f6")
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
    fig9_path = FIGURES_DIR / "09_outlier_analysis.png"
    plt.savefig(fig9_path, dpi=300)
    plt.close()

    # 10. Feature Distributions (Multivariate Grid)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    cols_to_plot = ["Area_sqft", "Property_Age", "Parking_Spaces", "Amenities_Count", "Total_Rooms", "Luxury_Score"]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"]
    for i, col in enumerate(cols_to_plot):
        ax = axes[i // 3, i % 3]
        sns.histplot(df_feat[col], kde=True, ax=ax, color=colors[i], bins=25)
        ax.set_title(f"Distribution of {col}", fontweight="bold", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    plt.tight_layout()
    fig10_path = FIGURES_DIR / "10_feature_distributions.png"
    plt.savefig(fig10_path, dpi=300)
    plt.close()

    # 11. Pairwise Relationships
    pair_cols = ["Price", "Area_sqft", "Total_Rooms", "Luxury_Score"]
    g = sns.pairplot(df_feat[pair_cols], diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15, "color": "#1d4ed8"})
    g.fig.suptitle("Pairwise Relationships Among Core Valuation Drivers", y=1.02, fontweight="bold", fontsize=13)
    fig11_path = FIGURES_DIR / "11_pairwise_relationships.png"
    g.savefig(fig11_path, dpi=300)
    plt.close()

    print("Successfully generated all 11 professional EDA visualizations in reports/figures/:")
    for p in sorted(FIGURES_DIR.glob("*.png")):
        print(f" - {p.name}")

if __name__ == "__main__":
    run_eda()
