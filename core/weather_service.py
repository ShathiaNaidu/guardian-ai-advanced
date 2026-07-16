from __future__ import annotations

from typing import Any

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Heavy rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Severe thunderstorm with hail",
}


def geocode_city(city: str) -> dict[str, Any] | None:
    response = requests.get(
        GEOCODE_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    return results[0] if results else None


def get_weather(city: str) -> dict[str, Any]:
    place = geocode_city(city)
    if not place:
        raise ValueError("Location not found.")
    params = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "timezone": "auto",
        "forecast_days": 7,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,weather_code,wind_speed_10m",
        "hourly": "temperature_2m,precipitation_probability,weather_code",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset",
    }
    response = requests.get(FORECAST_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    data["place"] = place
    current_code = data.get("current", {}).get("weather_code")
    data["current_description"] = WEATHER_CODES.get(current_code, "Unknown conditions")
    daily_codes = data.get("daily", {}).get("weather_code", [])
    data["daily_descriptions"] = [WEATHER_CODES.get(code, "Unknown") for code in daily_codes]
    return data
