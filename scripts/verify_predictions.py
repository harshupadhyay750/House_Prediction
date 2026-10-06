import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_house_price

cases = [
    {
        "name": "Mumbai Downtown 2BHK (1200 sqft, Good)",
        "p": {"Area": 1200, "Bedrooms": 2, "Bathrooms": 2.0, "Floors": 1, "YearBuilt": 2018, "Location": "Downtown", "Condition": "Good", "Garage": "Yes", "Country": "India", "City": "Mumbai (MMR)", "Currency": "INR"}
    },
    {
        "name": "Mumbai Suburban 3BHK (1500 sqft, Good)",
        "p": {"Area": 1500, "Bedrooms": 3, "Bathrooms": 2.5, "Floors": 1, "YearBuilt": 2019, "Location": "Suburban", "Condition": "Good", "Garage": "Yes", "Country": "India", "City": "Mumbai (MMR)", "Currency": "INR"}
    },
    {
        "name": "Bengaluru Urban 3BHK (1800 sqft, Excellent)",
        "p": {"Area": 1800, "Bedrooms": 3, "Bathrooms": 3.0, "Floors": 2, "YearBuilt": 2022, "Location": "Urban", "Condition": "Excellent", "Garage": "Yes", "Country": "India", "City": "Bengaluru (Silicon Valley of India)", "Currency": "INR"}
    },
    {
        "name": "Delhi NCR Suburban 4BHK (2500 sqft, Good)",
        "p": {"Area": 2500, "Bedrooms": 4, "Bathrooms": 3.5, "Floors": 2, "YearBuilt": 2015, "Location": "Suburban", "Condition": "Good", "Garage": "Yes", "Country": "India", "City": "Delhi NCR (Gurugram/Noida)", "Currency": "INR"}
    },
    {
        "name": "NYC Downtown Loft (1400 sqft, Good)",
        "p": {"Area": 1400, "Bedrooms": 2, "Bathrooms": 2.0, "Floors": 1, "YearBuilt": 2010, "Location": "Downtown", "Condition": "Good", "Garage": "No", "Country": "United States", "City": "New York City", "Currency": "USD"}
    },
    {
        "name": "Dallas Suburb Home (3200 sqft, Good)",
        "p": {"Area": 3200, "Bedrooms": 4, "Bathrooms": 3.0, "Floors": 2, "YearBuilt": 2012, "Location": "Suburban", "Condition": "Good", "Garage": "Yes", "Country": "United States", "City": "Dallas", "Currency": "USD"}
    },
    {
        "name": "London Central Flat (1200 sqft, Good)",
        "p": {"Area": 1200, "Bedrooms": 2, "Bathrooms": 2.0, "Floors": 1, "YearBuilt": 2005, "Location": "Downtown", "Condition": "Good", "Garage": "No", "Country": "United Kingdom", "City": "Central London", "Currency": "GBP"}
    }
]

print("=" * 70)
print("REAL ESTATE VALUATION VERIFICATION")
print("=" * 70)

for c in cases:
    res = predict_house_price(c["p"])
    print(f"\n{c['name']}:")
    print(f"  Base US-National USD: ${res['base_usd_price']:,.2f}")
    print(f"  Regional Multiplier : {res['regional_multiplier']}x")
    print(f"  Final Valuation     : {res['price_formatted']}")
    print(f"  Unit Rate           : {res['price_per_sqft_formatted']} / sq ft")
    print(f"  95% Interval        : {res['prediction_interval_95']['lower_formatted']} - {res['prediction_interval_95']['upper_formatted']}")
