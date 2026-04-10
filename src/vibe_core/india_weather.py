# india_weather.py
# Gap fix: FarmVibes.AI has no India weather support.
# This connector fetches IMD-compatible weather data for Indian coordinates
# using Open-Meteo (free, no API key needed).

import requests
from datetime import datetime, timedelta
from typing import Dict, Tuple

INDIA_CROPS = {
    "wheat":     {"sow": "Nov", "harvest": "Apr", "states": ["Punjab", "Haryana", "UP"]},
    "rice":      {"sow": "Jun", "harvest": "Nov", "states": ["WB", "Punjab", "AP"]},
    "mustard":   {"sow": "Oct", "harvest": "Feb", "states": ["Rajasthan", "UP"]},
    "sugarcane": {"sow": "Feb", "harvest": "Dec", "states": ["UP", "Maharashtra"]},
}

INDIA_FARM_LOCATIONS = {
    "punjab_wheat":    (30.9010, 75.8573),
    "up_rice":         (26.8467, 80.9462),
    "maharashtra_cane":(19.7515, 75.7139),
    "rajasthan_mustard":(27.0238, 74.2179),
}

def fetch_india_weather(
    lat: float,
    lon: float,
    days_back: int = 30
) -> Dict:
    """
    Fetch historical weather for Indian farm coordinates.
    Uses Open-Meteo — free, no API key, global coverage.
    
    Gap this fixes: FarmVibes.AI only supports NOAA (US) weather data.
    This adds Indian coordinate support with agri-relevant variables.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days_back)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":              lat,
        "longitude":             lon,
        "start_date":            start_date.strftime("%Y-%m-%d"),
        "end_date":              end_date.strftime("%Y-%m-%d"),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "et0_fao_evapotranspiration",  # critical for Indian irrigation
            "windspeed_10m_max",
            "shortwave_radiation_sum",     # for crop growth models
        ],
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    # Compute agronomic summary
    daily = data["daily"]
    temps_max = daily["temperature_2m_max"]
    temps_min = daily["temperature_2m_min"]
    rainfall   = daily["precipitation_sum"]

    summary = {
        "location":        {"lat": lat, "lon": lon},
        "period_days":     days_back,
        "avg_temp_max_c":  round(sum(t for t in temps_max if t) / len(temps_max), 1),
        "avg_temp_min_c":  round(sum(t for t in temps_min if t) / len(temps_min), 1),
        "total_rainfall_mm": round(sum(r for r in rainfall if r), 1),
        "dry_days":        sum(1 for r in rainfall if r is not None and r < 1),
        "raw":             daily,
    }
    return summary


def get_crop_weather_risk(weather: Dict, crop: str) -> Dict:
    """
    Rule-based risk assessment combining weather + Indian crop knowledge.
    This is the innovation: linking weather context to crop-specific risk.
    """
    risk = {"crop": crop, "alerts": [], "risk_level": "LOW"}

    avg_max  = weather["avg_temp_max_c"]
    rainfall = weather["total_rainfall_mm"]
    dry_days = weather["dry_days"]

    if crop == "wheat":
        if avg_max > 35:
            risk["alerts"].append("Heat stress: temps above 35°C damage wheat grain fill")
            risk["risk_level"] = "HIGH"
        if dry_days > 20:
            risk["alerts"].append("Drought stress: irrigation needed in Punjab/Haryana belt")
            risk["risk_level"] = "HIGH"

    elif crop == "rice":
        if rainfall < 50:
            risk["alerts"].append("Low rainfall: supplemental irrigation likely needed")
            risk["risk_level"] = "MEDIUM"
        if avg_max > 38:
            risk["alerts"].append("Spikelet sterility risk above 38°C for rice")
            risk["risk_level"] = "HIGH"

    elif crop == "mustard":
        if avg_max > 30:
            risk["alerts"].append("Premature flowering risk: mustard sensitive to heat")
            risk["risk_level"] = "MEDIUM"

    if not risk["alerts"]:
        risk["alerts"].append("Conditions normal for selected crop")

    return risk


if __name__ == "__main__":
    print("Testing India Weather Connector for FarmVibes.AI\n")
    print("=" * 50)

    for location_name, (lat, lon) in INDIA_FARM_LOCATIONS.items():
        crop = location_name.split("_")[1]
        print(f"\nLocation: {location_name}  ({lat}, {lon})")
        weather = fetch_india_weather(lat, lon, days_back=30)
        print(f"  Avg Max Temp : {weather['avg_temp_max_c']} °C")
        print(f"  Total Rainfall: {weather['total_rainfall_mm']} mm")
        print(f"  Dry Days      : {weather['dry_days']} / 30")

        if crop in INDIA_CROPS:
            risk = get_crop_weather_risk(weather, crop)
            print(f"  Risk Level    : {risk['risk_level']}")
            for alert in risk["alerts"]:
                print(f"  ⚠  {alert}")