"""
Dataset Generation Script for House Price Prediction & Property Analytics System.
Generates realistic raw property transaction data with real-world complexities:
- Realistic correlation between area, rooms, location premiums, and pricing
- Injected missing values (NaN) in numerical and categorical columns
- Injected duplicates and inconsistent string casing
- Injected outlier and invalid edge cases for cleaning demonstration
"""

import os
import numpy as np
import pandas as pd

def generate_property_data(n_samples: int = 6000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    locations = [
        'Downtown Central', 'Silicon Hills', 'Greenwood Heights',
        'Riverside District', 'Lakeside Estates', 'Harbor Point',
        'Midtown Corridor', 'Oakridge Valley', 'Sunset Park', 'Westend Terrace'
    ]
    
    # Location base price per sqft ($/sqft)
    location_base_rates = {
        'Downtown Central': 480,
        'Silicon Hills': 520,
        'Greenwood Heights': 360,
        'Riverside District': 410,
        'Lakeside Estates': 450,
        'Harbor Point': 490,
        'Midtown Corridor': 430,
        'Oakridge Valley': 320,
        'Sunset Park': 340,
        'Westend Terrace': 380
    }
    
    property_types = ['Apartment', 'Independent House', 'Luxury Villa', 'Penthouse', 'Studio Apartment']
    type_weights = [0.45, 0.25, 0.12, 0.08, 0.10]
    
    type_multiplier = {
        'Apartment': 1.0,
        'Independent House': 1.15,
        'Luxury Villa': 1.40,
        'Penthouse': 1.50,
        'Studio Apartment': 0.95
    }

    # Sampling locations and types
    loc_sample = np.random.choice(locations, size=n_samples)
    type_sample = np.random.choice(property_types, size=n_samples, p=type_weights)
    
    # Area based on property type
    area = []
    bedrooms = []
    bathrooms = []
    floors = []
    
    for p_type in type_sample:
        if p_type == 'Studio Apartment':
            a = int(np.random.normal(550, 80))
            a = max(350, min(800, a))
            b_bed = 1
            b_bath = 1.0
            fl = np.random.randint(1, 20)
        elif p_type == 'Apartment':
            beds = np.random.choice([1, 2, 3, 4], p=[0.15, 0.50, 0.30, 0.05])
            a = int(np.random.normal(600 + beds * 350, 180))
            a = max(500, a)
            b_bed = beds
            b_bath = float(np.random.choice([beds, max(1, beds - 1), beds + 1], p=[0.6, 0.3, 0.1]))
            fl = np.random.randint(1, 25)
        elif p_type == 'Independent House':
            beds = np.random.choice([2, 3, 4, 5], p=[0.1, 0.45, 0.35, 0.1])
            a = int(np.random.normal(1200 + beds * 400, 300))
            a = max(1000, a)
            b_bed = beds
            b_bath = float(np.random.choice([beds, beds - 1, beds + 1], p=[0.5, 0.3, 0.2]))
            fl = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        elif p_type == 'Luxury Villa':
            beds = np.random.choice([3, 4, 5, 6], p=[0.15, 0.45, 0.30, 0.10])
            a = int(np.random.normal(2500 + beds * 500, 450))
            a = max(2200, a)
            b_bed = beds
            b_bath = float(beds + np.random.choice([0, 1], p=[0.4, 0.6]))
            fl = np.random.choice([2, 3, 4], p=[0.4, 0.45, 0.15])
        else: # Penthouse
            beds = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
            a = int(np.random.normal(2800 + beds * 450, 500))
            a = max(2400, a)
            b_bed = beds
            b_bath = float(beds + np.random.choice([1, 2], p=[0.7, 0.3]))
            fl = np.random.randint(18, 35)
            
        area.append(a)
        bedrooms.append(b_bed)
        bathrooms.append(b_bath)
        floors.append(fl)
        
    area = np.array(area)
    bedrooms = np.array(bedrooms)
    bathrooms = np.array(bathrooms)
    floors = np.array(floors)
    
    # Furnishing
    furnishing_choices = ['Furnished', 'Semi-Furnished', 'Unfurnished']
    furnishing = np.random.choice(furnishing_choices, size=n_samples, p=[0.35, 0.45, 0.20])
    furnishing_add = {'Furnished': 35000, 'Semi-Furnished': 15000, 'Unfurnished': 0}
    
    # Parking spaces
    parking = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.15, 0.50, 0.28, 0.07])
    
    # Property age (years)
    age = np.clip(np.random.exponential(scale=8, size=n_samples).astype(int), 0, 40)
    
    # Balconies
    balconies = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.12, 0.48, 0.32, 0.08])
    
    # Amenities count (0 to 10)
    amenities = np.random.poisson(lam=5.5, size=n_samples)
    amenities = np.clip(amenities, 1, 10)
    
    # Availability
    availability = np.random.choice(['Ready to Move', 'Under Construction'], size=n_samples, p=[0.78, 0.22])
    avail_discount = {'Ready to Move': 1.0, 'Under Construction': 0.94}
    
    # Nearby schools count (1 to 5)
    schools = np.random.randint(1, 6, size=n_samples)
    
    # Target Price Calculation (Realistic hedonic valuation function)
    price = []
    for i in range(n_samples):
        base_rate = location_base_rates[loc_sample[i]]
        t_mult = type_multiplier[type_sample[i]]
        
        # Base valuation from square footage
        val = area[i] * base_rate * t_mult
        
        # Room premium
        val += bedrooms[i] * 18000 + bathrooms[i] * 14000
        
        # Parking, furnishing, amenities
        val += parking[i] * 12500
        val += furnishing_add[furnishing[i]]
        val += amenities[i] * 6500
        val += balconies[i] * 4000
        val += schools[i] * 3500
        
        # Age depreciation: -0.7% per year
        age_factor = max(0.68, 1.0 - (age[i] * 0.007))
        val *= age_factor
        
        # Availability adjustment
        val *= avail_discount[availability[i]]
        
        # Add realistic market noise (~5% relative noise)
        noise = np.random.normal(0, val * 0.05)
        final_price = round(val + noise, -2)
        price.append(max(50000.0, final_price))
        
    df = pd.DataFrame({
        'Property_ID': [f'PROP_{10001 + i}' for i in range(n_samples)],
        'Location': loc_sample,
        'Property_Type': type_sample,
        'Area_sqft': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Furnishing_Status': furnishing,
        'Parking_Spaces': parking,
        'Floors': floors,
        'Property_Age': age,
        'Balconies': balconies,
        'Amenities_Count': amenities,
        'Availability': availability,
        'Nearby_Schools': schools,
        'Price': price
    })
    
    # --- Inject Real-World Flaws for Data Cleaning Demonstration ---
    
    # 1. Dirty categorical casing in Furnishing_Status
    dirty_indices = np.random.choice(df.index, size=180, replace=False)
    for idx in dirty_indices:
        curr = df.loc[idx, 'Furnishing_Status']
        if curr == 'Furnished':
            df.loc[idx, 'Furnishing_Status'] = np.random.choice(['furnished', 'FURNISHED'])
        elif curr == 'Semi-Furnished':
            df.loc[idx, 'Furnishing_Status'] = np.random.choice(['semi-furnished', 'SEMI-FURNISHED', 'Semi-furnished'])
        elif curr == 'Unfurnished':
            df.loc[idx, 'Furnishing_Status'] = np.random.choice(['unfurnished', 'UNFURNISHED'])

    # 2. Missing values (NaN) in Property_Age, Bathrooms, Furnishing_Status, Nearby_Schools
    nan_age_idx = np.random.choice(df.index, size=150, replace=False)
    df.loc[nan_age_idx, 'Property_Age'] = np.nan
    
    nan_bath_idx = np.random.choice(df.index, size=85, replace=False)
    df.loc[nan_bath_idx, 'Bathrooms'] = np.nan
    
    nan_furn_idx = np.random.choice(df.index, size=70, replace=False)
    df.loc[nan_furn_idx, 'Furnishing_Status'] = np.nan
    
    nan_schools_idx = np.random.choice(df.index, size=60, replace=False)
    df.loc[nan_schools_idx, 'Nearby_Schools'] = np.nan

    # 3. Impossible values (e.g. Negative area or zero bedrooms)
    err_idx = np.random.choice(df.index, size=8, replace=False)
    df.loc[err_idx[:4], 'Area_sqft'] = -1 * df.loc[err_idx[:4], 'Area_sqft']
    df.loc[err_idx[4:], 'Bedrooms'] = 0

    # 4. Outliers (Extreme unrealistic prices / luxury outliers)
    outlier_idx = np.random.choice(df.index, size=6, replace=False)
    df.loc[outlier_idx[:3], 'Price'] = df.loc[outlier_idx[:3], 'Price'] * 6.5
    df.loc[outlier_idx[3:], 'Area_sqft'] = 19500

    # 5. Duplicate rows
    dup_rows = df.sample(n=25, random_state=random_seed)
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    os.makedirs('src', exist_ok=True)
    os.makedirs('app/templates', exist_ok=True)
    os.makedirs('tests', exist_ok=True)
    
    data = generate_property_data(n_samples=6000, random_seed=42)
    output_path = 'data/raw/housing_raw.csv'
    data.to_csv(output_path, index=False)
    print(f"Generated raw dataset with {len(data)} rows and {len(data.columns)} columns at {output_path}")
    print(data.head())
    print("\nMissing values:")
    print(data.isnull().sum())
