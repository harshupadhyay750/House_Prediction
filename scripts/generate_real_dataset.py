"""
US-Baseline Housing Dataset Generator for PropIntel.
Generates 5,000 property records calibrated to the US National Average market.

The model predicts prices in USD at US-national-average levels.
Global market localization is handled post-prediction via regional multipliers
in the global_market.py module.

US National Average Reference (2024-2026 calibrated):
  Downtown/Central:   $200-450/sqft  (think Chicago, Boston, Denver CBD)
  Urban/Inner Ring:   $130-280/sqft  (Atlanta, Dallas, Portland inner ring)
  Suburban/Commuter:  $90-200/sqft   (typical US suburb)
  Rural/Countryside:  $50-120/sqft   (countryside / small towns)

Premium cities (NYC $900+/sqft, SF $800+/sqft, Mumbai ₹35K+/sqft)
are handled by the global multiplier system post-prediction.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_PATH

np.random.seed(42)

# ─── Location Tier Definitions (US National Average) ──────────────────────
LOCATION_TIERS = {
    "Downtown": {
        "ppsf_range": (200, 450),
        "area_range": (500, 5000),
        "year_range": (1900, 2026),
        "floor_probs": [0.10, 0.45, 0.30, 0.10, 0.05],
        "garage_prob": 0.35,
        "condition_probs": [0.28, 0.45, 0.18, 0.09],
        "weight": 0.18,
    },
    "Urban": {
        "ppsf_range": (130, 280),
        "area_range": (700, 4500),
        "year_range": (1935, 2026),
        "floor_probs": [0.20, 0.50, 0.20, 0.07, 0.03],
        "garage_prob": 0.50,
        "condition_probs": [0.22, 0.48, 0.22, 0.08],
        "weight": 0.30,
    },
    "Suburban": {
        "ppsf_range": (90, 200),
        "area_range": (1000, 6000),
        "year_range": (1950, 2026),
        "floor_probs": [0.35, 0.48, 0.13, 0.03, 0.01],
        "garage_prob": 0.75,
        "condition_probs": [0.18, 0.45, 0.27, 0.10],
        "weight": 0.35,
    },
    "Rural": {
        "ppsf_range": (50, 120),
        "area_range": (1200, 8000),
        "year_range": (1940, 2026),
        "floor_probs": [0.55, 0.35, 0.08, 0.01, 0.01],
        "garage_prob": 0.80,
        "condition_probs": [0.12, 0.38, 0.33, 0.17],
        "weight": 0.17,
    },
}

# ─── Price Multipliers ────────────────────────────────────────────────────
CONDITION_MULT = {"Excellent": 1.20, "Good": 1.00, "Fair": 0.82, "Poor": 0.65}
GARAGE_MULT = {"Yes": 1.06, "No": 1.00}


def bedrooms_from_area(area_sqft: float) -> int:
    """Derive realistic bedroom count from living area."""
    if area_sqft < 700:
        return int(np.random.choice([1, 2], p=[0.60, 0.40]))
    elif area_sqft < 1200:
        return int(np.random.choice([2, 3], p=[0.55, 0.45]))
    elif area_sqft < 2000:
        return int(np.random.choice([2, 3, 4], p=[0.15, 0.55, 0.30]))
    elif area_sqft < 3000:
        return int(np.random.choice([3, 4, 5], p=[0.35, 0.45, 0.20]))
    elif area_sqft < 4500:
        return int(np.random.choice([4, 5, 6], p=[0.40, 0.40, 0.20]))
    else:
        return int(np.random.choice([4, 5, 6, 7], p=[0.20, 0.40, 0.30, 0.10]))


def bathrooms_from_bedrooms(beds: int) -> float:
    """Derive realistic bathroom count from bedrooms."""
    base = max(1.0, beds - 0.5)
    noise = np.random.choice([0, 0.5, 1.0], p=[0.40, 0.35, 0.25])
    return round(min(base + noise, beds + 1.0), 1)


def generate_price(
    area: float,
    ppsf_range: tuple,
    condition: str,
    garage: str,
    year_built: int,
    floors: int,
    bedrooms: int,
    bathrooms: float,
) -> float:
    """
    Generate realistic hedonic price using econometric model.

    Price = Area × BasePPSF × AreaFactor × AgeFactor × FloorPremium
            × ConditionMult × GarageMult × RoomPremium × (1 + noise)
    """
    ppsf_lo, ppsf_hi = ppsf_range

    # Triangular distribution: slight skew toward mid-range
    base_ppsf = np.random.triangular(ppsf_lo, (ppsf_lo + ppsf_hi) * 0.48, ppsf_hi)

    # Diminishing returns for very large properties (>4000 sqft)
    if area > 4000:
        area_factor = 1.0 - 0.000012 * (area - 4000)
    else:
        area_factor = 1.0

    # Age depreciation: ~0.4% per year, minimum 60%
    age = max(0, 2026 - year_built)
    age_factor = max(0.60, 1.0 - age * 0.004)

    # Floor premium: multi-story homes command slight premium
    floor_premium = 1.0 + 0.015 * (floors - 1)

    # Room premium: extra bathrooms beyond base add value
    extra_baths = max(0, bathrooms - bedrooms)
    room_premium = 1.0 + 0.012 * extra_baths

    base_price = area * base_ppsf * area_factor * age_factor * floor_premium * room_premium
    base_price *= CONDITION_MULT[condition]
    base_price *= GARAGE_MULT[garage]

    # Heteroskedastic noise: larger homes have slightly more variance
    noise_pct = np.random.normal(0, 0.04 + 0.000020 * area)
    price = base_price * (1 + noise_pct)

    return max(35000.0, round(price, -2))


def generate_dataset(total_records: int = 5000) -> pd.DataFrame:
    """Generate US-baseline property dataset."""
    locations = list(LOCATION_TIERS.keys())
    weights = np.array([LOCATION_TIERS[loc]["weight"] for loc in locations])
    weights /= weights.sum()

    loc_counts = np.round(weights * total_records).astype(int)
    diff = total_records - loc_counts.sum()
    loc_counts[np.argmax(weights)] += diff

    records = []
    row_id = 1

    for loc, count in zip(locations, loc_counts):
        tier = LOCATION_TIERS[loc]

        for _ in range(count):
            # Sample core attributes
            area = float(np.round(np.random.uniform(*tier["area_range"]), -1))
            beds = bedrooms_from_area(area)
            baths = bathrooms_from_bedrooms(beds)
            yr = int(np.random.randint(tier["year_range"][0], tier["year_range"][1] + 1))
            fl = int(np.random.choice([1, 2, 3, 4, 5], p=tier["floor_probs"]))
            cond = np.random.choice(
                ["Excellent", "Good", "Fair", "Poor"],
                p=tier["condition_probs"],
            )
            gar = "Yes" if np.random.random() < tier["garage_prob"] else "No"

            price = generate_price(area, tier["ppsf_range"], cond, gar, yr, fl, beds, baths)

            records.append({
                "Id": row_id,
                "Area": area,
                "Bedrooms": beds,
                "Bathrooms": baths,
                "Floors": fl,
                "YearBuilt": yr,
                "Location": loc,
                "Condition": cond,
                "Garage": gar,
                "Price": price,
            })
            row_id += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["Id"] = range(1, len(df) + 1)
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("PropIntel US-Baseline Dataset Generator")
    print("=" * 60)

    df = generate_dataset(5000)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nPrice statistics (USD):")
    print(df["Price"].describe().to_string())

    print(f"\nPrice by Location Tier (median USD):")
    print(df.groupby("Location")["Price"].median().sort_values(ascending=False).to_string())

    print(f"\nPrice per sqft by Location:")
    df["_ppsf"] = df["Price"] / df["Area"]
    print(df.groupby("Location")["_ppsf"].median().sort_values(ascending=False).to_string())

    print(f"\nLocation distribution:")
    print(df["Location"].value_counts().to_string())

    print(f"\nCondition distribution:")
    print(df["Condition"].value_counts().to_string())

    print(f"\nGarage distribution:")
    print(df["Garage"].value_counts().to_string())

    print(f"\nArea stats (sq ft):")
    print(df["Area"].describe().to_string())

    print(f"\nTop 5 most expensive:")
    print(df.nlargest(5, "Price")[["Id", "Area", "Location", "Condition", "Price"]].to_string())

    print(f"\nTop 5 least expensive:")
    print(df.nsmallest(5, "Price")[["Id", "Area", "Location", "Condition", "Price"]].to_string())

    # Save
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_save = df.drop(columns=["_ppsf"], errors="ignore")
    df_save.to_csv(RAW_DATA_PATH, index=False)
    print(f"\nSaved {len(df_save)} records to {RAW_DATA_PATH}")
