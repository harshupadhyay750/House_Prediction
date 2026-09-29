"""
Global Real Estate Market & Multi-Currency Engine for PropIntel.
Supports worldwide property valuation across any region on Earth:
- 20+ Global Countries and 100+ Metro Hubs + Custom Any-Location mode
- Multi-currency conversion: USD, INR (with Lakh/Crore formatting), EUR, GBP, AED, CAD, AUD, JPY, SGD, CHF, SAR, CNY, BRL, ZAR
- Area unit conversion: Square Feet (sq ft) and Square Meters (sq m / m²)
- Hedonic settlement tier mapping (Downtown, Urban, Suburban, Rural)
"""

from typing import Dict, Any, Tuple, Optional, List

# Exchange rates relative to 1.00 USD (Production Market Averages)
EXCHANGE_RATES: Dict[str, Dict[str, Any]] = {
    "USD": {
        "symbol": "$",
        "name": "US Dollar (USD)",
        "rate": 1.00,
        "flag": "🇺🇸",
        "format": "${:,.2f}"
    },
    "INR": {
        "symbol": "₹",
        "name": "Indian Rupee (INR)",
        "rate": 86.50,
        "flag": "🇮🇳",
        "format": "₹{:,.0f}"
    },
    "EUR": {
        "symbol": "€",
        "name": "Euro (EUR)",
        "rate": 0.92,
        "flag": "🇪🇺",
        "format": "€{:,.2f}"
    },
    "GBP": {
        "symbol": "£",
        "name": "British Pound (GBP)",
        "rate": 0.79,
        "flag": "🇬🇧",
        "format": "£{:,.2f}"
    },
    "AED": {
        "symbol": "AED ",
        "name": "UAE Dirham (AED)",
        "rate": 3.67,
        "flag": "🇦🇪",
        "format": "AED {:,.2f}"
    },
    "CAD": {
        "symbol": "CA$",
        "name": "Canadian Dollar (CAD)",
        "rate": 1.38,
        "flag": "🇨🇦",
        "format": "CA${:,.2f}"
    },
    "AUD": {
        "symbol": "A$",
        "name": "Australian Dollar (AUD)",
        "rate": 1.54,
        "flag": "🇦🇺",
        "format": "A${:,.2f}"
    },
    "JPY": {
        "symbol": "¥",
        "name": "Japanese Yen (JPY)",
        "rate": 152.00,
        "flag": "🇯🇵",
        "format": "¥{:,.0f}"
    },
    "SGD": {
        "symbol": "SG$",
        "name": "Singapore Dollar (SGD)",
        "rate": 1.34,
        "flag": "🇸🇬",
        "format": "SG${:,.2f}"
    },
    "CHF": {
        "symbol": "CHF ",
        "name": "Swiss Franc (CHF)",
        "rate": 0.88,
        "flag": "🇨🇭",
        "format": "CHF {:,.2f}"
    },
    "SAR": {
        "symbol": "SAR ",
        "name": "Saudi Riyal (SAR)",
        "rate": 3.75,
        "flag": "🇸🇦",
        "format": "SAR {:,.2f}"
    },
    "CNY": {
        "symbol": "CN¥",
        "name": "Chinese Yuan (CNY)",
        "rate": 7.23,
        "flag": "🇨🇳",
        "format": "CN¥{:,.2f}"
    },
    "BRL": {
        "symbol": "R$",
        "name": "Brazilian Real (BRL)",
        "rate": 5.60,
        "flag": "🇧🇷",
        "format": "R${:,.2f}"
    },
    "ZAR": {
        "symbol": "R ",
        "name": "South African Rand (ZAR)",
        "rate": 18.20,
        "flag": "🇿🇦",
        "format": "R {:,.2f}"
    }
}

# Real estate price multipliers calibrated to global metropolitan real estate markets
# Each country has default currency and cities with property indices relative to baseline USD market
GLOBAL_COUNTRIES: Dict[str, Dict[str, Any]] = {
    "United States": {
        "flag": "🇺🇸",
        "default_currency": "USD",
        "base_multiplier": 1.00,
        "cities": {
            "National Benchmark": 1.00,
            "New York City": 1.95,
            "San Francisco Bay Area": 1.85,
            "Los Angeles": 1.65,
            "Seattle": 1.45,
            "Boston": 1.40,
            "Miami": 1.35,
            "Austin": 1.20,
            "Chicago": 1.10,
            "Phoenix": 1.00,
            "Houston": 0.95
        }
    },
    "India": {
        "flag": "🇮🇳",
        "default_currency": "INR",
        "base_multiplier": 0.85,
        "cities": {
            "National Benchmark": 0.90,
            "Mumbai (MMR)": 1.60,
            "Delhi NCR (Gurugram/Noida)": 1.35,
            "Bengaluru (Silicon Valley of India)": 1.25,
            "Hyderabad": 1.10,
            "Pune": 1.00,
            "Chennai": 0.95,
            "Goa (Coastal Luxury)": 1.20,
            "Kolkata": 0.80,
            "Ahmedabad": 0.80,
            "Chandigarh": 0.95,
            "Jaipur": 0.75,
            "Kochi": 0.80
        }
    },
    "United Kingdom": {
        "flag": "🇬🇧",
        "default_currency": "GBP",
        "base_multiplier": 1.20,
        "cities": {
            "National Benchmark": 1.10,
            "Central London": 1.95,
            "Greater London": 1.65,
            "Cambridge": 1.40,
            "Oxford": 1.35,
            "Edinburgh": 1.15,
            "Bristol": 1.15,
            "Manchester": 1.00,
            "Birmingham": 0.95,
            "Leeds": 0.90
        }
    },
    "United Arab Emirates": {
        "flag": "🇦🇪",
        "default_currency": "AED",
        "base_multiplier": 1.30,
        "cities": {
            "National Benchmark": 1.15,
            "Dubai (Downtown/Marina/Palm)": 1.55,
            "Dubai (Suburban/Hills)": 1.35,
            "Abu Dhabi (Al Reem/Saadiyat)": 1.30,
            "Sharjah": 0.85,
            "Ras Al Khaimah": 0.80,
            "Ajman": 0.70
        }
    },
    "Canada": {
        "flag": "🇨🇦",
        "default_currency": "CAD",
        "base_multiplier": 1.15,
        "cities": {
            "National Benchmark": 1.10,
            "Vancouver (Metro)": 1.65,
            "Toronto (GTA)": 1.55,
            "Victoria": 1.30,
            "Montreal": 1.10,
            "Ottawa": 1.05,
            "Calgary": 0.95,
            "Edmonton": 0.85
        }
    },
    "Australia": {
        "flag": "🇦🇺",
        "default_currency": "AUD",
        "base_multiplier": 1.25,
        "cities": {
            "National Benchmark": 1.20,
            "Sydney (Eastern Suburbs/CBD)": 1.75,
            "Melbourne": 1.40,
            "Brisbane": 1.15,
            "Gold Coast": 1.20,
            "Perth": 1.00,
            "Canberra": 1.25,
            "Adelaide": 0.95
        }
    },
    "Germany": {
        "flag": "🇩🇪",
        "default_currency": "EUR",
        "base_multiplier": 1.20,
        "cities": {
            "National Benchmark": 1.15,
            "Munich": 1.65,
            "Frankfurt": 1.40,
            "Berlin": 1.30,
            "Hamburg": 1.30,
            "Stuttgart": 1.25,
            "Cologne": 1.15,
            "Leipzig": 0.90
        }
    },
    "France": {
        "flag": "🇫🇷",
        "default_currency": "EUR",
        "base_multiplier": 1.20,
        "cities": {
            "National Benchmark": 1.15,
            "Paris Central": 1.90,
            "Nice & French Riviera": 1.40,
            "Lyon": 1.20,
            "Bordeaux": 1.20,
            "Marseille": 1.00,
            "Toulouse": 1.00
        }
    },
    "Singapore": {
        "flag": "🇸🇬",
        "default_currency": "SGD",
        "base_multiplier": 2.15,
        "cities": {
            "Central Core District (CCR)": 2.40,
            "Rest of Central Region (RCR)": 2.10,
            "Outside Central Region (OCR)": 1.85,
            "Island-Wide Average": 2.15
        }
    },
    "Japan": {
        "flag": "🇯🇵",
        "default_currency": "JPY",
        "base_multiplier": 1.10,
        "cities": {
            "National Benchmark": 1.05,
            "Tokyo (23 Special Wards)": 1.65,
            "Tokyo Metro / Yokohama": 1.30,
            "Kyoto": 1.15,
            "Osaka": 1.10,
            "Fukuoka": 0.95,
            "Sapporo": 0.85
        }
    },
    "Switzerland": {
        "flag": "🇨🇭",
        "default_currency": "CHF",
        "base_multiplier": 2.10,
        "cities": {
            "National Benchmark": 1.95,
            "Zurich": 2.35,
            "Geneva": 2.30,
            "Lausanne": 1.90,
            "Basel": 1.85,
            "Bern": 1.70
        }
    },
    "Spain": {
        "flag": "🇪🇸",
        "default_currency": "EUR",
        "base_multiplier": 0.95,
        "cities": {
            "National Benchmark": 0.95,
            "Madrid": 1.30,
            "Barcelona": 1.35,
            "Mallorca & Balearics": 1.40,
            "Malaga & Costa del Sol": 1.15,
            "Valencia": 0.95,
            "Seville": 0.85
        }
    },
    "Italy": {
        "flag": "🇮🇹",
        "default_currency": "EUR",
        "base_multiplier": 1.00,
        "cities": {
            "National Benchmark": 1.00,
            "Milan": 1.50,
            "Rome": 1.30,
            "Florence": 1.25,
            "Bologna": 1.10,
            "Turin": 0.90,
            "Naples": 0.85
        }
    },
    "Netherlands": {
        "flag": "🇳🇱",
        "default_currency": "EUR",
        "base_multiplier": 1.30,
        "cities": {
            "National Benchmark": 1.25,
            "Amsterdam": 1.80,
            "Utrecht": 1.40,
            "The Hague": 1.25,
            "Rotterdam": 1.20,
            "Eindhoven": 1.15
        }
    },
    "Saudi Arabia": {
        "flag": "🇸🇦",
        "default_currency": "SAR",
        "base_multiplier": 1.10,
        "cities": {
            "National Benchmark": 1.05,
            "Riyadh": 1.30,
            "Jeddah": 1.15,
            "Khobar / Dammam": 1.05,
            "Mecca / Medina": 1.35
        }
    },
    "China": {
        "flag": "🇨🇳",
        "default_currency": "CNY",
        "base_multiplier": 1.20,
        "cities": {
            "National Benchmark": 1.10,
            "Shanghai": 1.75,
            "Beijing": 1.70,
            "Shenzhen": 1.65,
            "Guangzhou": 1.35,
            "Hangzhou": 1.25,
            "Chengdu": 1.00
        }
    },
    "Brazil": {
        "flag": "🇧🇷",
        "default_currency": "BRL",
        "base_multiplier": 0.75,
        "cities": {
            "National Benchmark": 0.85,
            "São Paulo": 1.20,
            "Rio de Janeiro": 1.15,
            "Brasília": 1.10,
            "Florianópolis": 1.05,
            "Salvador": 0.80
        }
    },
    "South Africa": {
        "flag": "🇿🇦",
        "default_currency": "ZAR",
        "base_multiplier": 0.70,
        "cities": {
            "National Benchmark": 0.85,
            "Cape Town (Atlantic Seaboard)": 1.45,
            "Cape Town (Suburbs)": 1.20,
            "Johannesburg (Sandton)": 1.05,
            "Pretoria": 0.95,
            "Durban": 0.90
        }
    },
    "New Zealand": {
        "flag": "🇳🇿",
        "default_currency": "AUD",
        "base_multiplier": 1.20,
        "cities": {
            "National Benchmark": 1.15,
            "Auckland": 1.50,
            "Queenstown": 1.55,
            "Wellington": 1.25,
            "Christchurch": 1.00
        }
    },
    "Global Custom Location": {
        "flag": "🌍",
        "default_currency": "USD",
        "base_multiplier": 1.00,
        "cities": {
            "Global Average Benchmark": 1.00,
            "High-Cost Metro Tier": 1.50,
            "Prime Tier": 1.25,
            "Standard Market Tier": 1.00,
            "Developing / Emerging Market Tier": 0.75,
            "Affordable / Rural Market Tier": 0.50
        }
    }
}

# Settlement density tiers mapping to base model Location feature
SETTLEMENT_TIERS = {
    "Downtown / Central Core": "Downtown",
    "Urban / Inner Ring": "Urban",
    "Suburban / Metro Outer": "Suburban",
    "Rural / Countryside": "Rural"
}

# Unit conversion constants
SQM_TO_SQFT = 10.7639104
SQFT_TO_SQM = 0.09290304


def format_indian_currency(amount: float) -> Tuple[str, str]:
    """
    Format numerical amounts using Indian numbering system (Crores & Lakhs).
    Returns (compact_formatted, full_formatted)
    Example: 15200000 -> ('₹1.52 Cr', '₹1,52,00,000')
    """
    if amount >= 10000000:
        cr_val = amount / 10000000.0
        compact = f"₹{cr_val:,.2f} Cr"
    elif amount >= 100000:
        lakh_val = amount / 100000.0
        compact = f"₹{lakh_val:,.2f} Lakh"
    elif amount >= 1000:
        k_val = amount / 1000.0
        compact = f"₹{k_val:,.1f} K"
    else:
        compact = f"₹{amount:,.0f}"

    # Generate Indian comma separated notation (e.g. 1,23,45,678)
    int_part = int(round(amount))
    s = str(int_part)
    if len(s) <= 3:
        indian_str = s
    else:
        last3 = s[-3:]
        remaining = s[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        groups.append(last3)
        indian_str = ",".join(groups)

    full_formatted = f"₹{indian_str}"
    return compact, full_formatted


def format_currency_value(amount: float, currency_code: str = "USD") -> Dict[str, str]:
    """
    Format monetary valuation according to local currency conventions.
    Returns dictionary with standard, compact, symbol, and currency_code.
    """
    curr = EXCHANGE_RATES.get(currency_code.upper(), EXCHANGE_RATES["USD"])
    symbol = curr["symbol"]

    if currency_code.upper() == "INR":
        compact, full = format_indian_currency(amount)
        return {
            "formatted": full,
            "compact": compact,
            "symbol": symbol,
            "currency": "INR",
            "display": f"{compact} ({full})"
        }
    elif amount >= 1000000:
        m_val = amount / 1000000.0
        compact = f"{symbol}{m_val:,.2f}M"
        full = curr["format"].format(amount)
        return {
            "formatted": full,
            "compact": compact,
            "symbol": symbol,
            "currency": currency_code.upper(),
            "display": full
        }
    elif amount >= 1000:
        k_val = amount / 1000.0
        compact = f"{symbol}{k_val:,.1f}K"
        full = curr["format"].format(amount)
        return {
            "formatted": full,
            "compact": compact,
            "symbol": symbol,
            "currency": currency_code.upper(),
            "display": full
        }
    else:
        full = curr["format"].format(amount)
        return {
            "formatted": full,
            "compact": full,
            "symbol": symbol,
            "currency": currency_code.upper(),
            "display": full
        }


def convert_and_localize_price(
    usd_price: float,
    currency_code: str = "USD",
    country_name: str = "United States",
    city_name: Optional[str] = None,
    custom_city: Optional[str] = None,
    custom_multiplier: Optional[float] = None
) -> Dict[str, Any]:
    """
    Scales the base hedonic USD valuation by the regional real estate multiplier
    and converts it into the requested global currency.
    """
    # Determine Regional Real Estate Multiplier
    multiplier = 1.00
    if custom_multiplier is not None and custom_multiplier > 0:
        multiplier = float(custom_multiplier)
    elif country_name in GLOBAL_COUNTRIES:
        c_info = GLOBAL_COUNTRIES[country_name]
        cities = c_info.get("cities", {})
        if city_name and city_name in cities:
            multiplier = float(cities[city_name])
        else:
            multiplier = float(c_info.get("base_multiplier", 1.00))

    # Apply Regional Property Scale
    regional_usd_price = usd_price * multiplier

    # Convert to Selected Currency
    curr = EXCHANGE_RATES.get(currency_code.upper(), EXCHANGE_RATES["USD"])
    exchange_rate = curr["rate"]
    localized_amount = round(regional_usd_price * exchange_rate, 2)

    fmt = format_currency_value(localized_amount, currency_code.upper())

    return {
        "raw_usd_price": round(usd_price, 2),
        "regional_multiplier": multiplier,
        "regional_usd_price": round(regional_usd_price, 2),
        "currency": currency_code.upper(),
        "exchange_rate": exchange_rate,
        "localized_price": localized_amount,
        "formatted_price": fmt["formatted"],
        "compact_price": fmt["compact"],
        "display_price": fmt["display"],
        "symbol": fmt["symbol"],
        "country": country_name,
        "city": custom_city or city_name or "Standard Benchmark"
    }


def convert_area_to_sqft(area_val: float, unit: str = "sq ft") -> float:
    """Standardizes area into sq ft for model prediction."""
    unit_clean = unit.lower().replace(" ", "").replace("_", "")
    if "sqm" in unit_clean or "m2" in unit_clean or "meter" in unit_clean:
        return area_val * SQM_TO_SQFT
    return area_val


def convert_sqft_to_sqm(sqft_val: float) -> float:
    """Converts square feet to square meters."""
    return sqft_val * SQFT_TO_SQM
