"""Fetch current weather conditions from Open-Meteo API (free, no API key).

Used by the rate predictor to adjust freight rates based on weather severity
at origin and destination locations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import aiohttp
from loguru import logger

# Cache: (rounded_lat, rounded_lng) -> (WeatherCondition, timestamp)
_weather_cache: Dict[Tuple[float, float], Tuple["WeatherCondition", float]] = {}
CACHE_TTL = 1800  # 30 minutes


# WMO weather codes → severity category
# https://open-meteo.com/en/docs#weathervariables
_WMO_SEVERE = {
    # Thunderstorm
    95, 96, 99,
    # Freezing rain / drizzle
    56, 57, 66, 67,
    # Heavy snow / snow showers
    73, 75, 77, 85, 86,
    # Blizzard-like (heavy snow + wind handled via wind_speed)
}
_WMO_MODERATE = {
    # Moderate rain / drizzle
    55, 63, 65,
    # Light freezing
    56, 66,
    # Light-moderate snow
    71, 73, 85,
    # Fog
    45, 48,
}
_WMO_LIGHT = {
    # Light rain / drizzle
    51, 53, 61,
    # Slight snow
    71,
    # Overcast
    3,
}


@dataclass(frozen=True)
class WeatherCondition:
    """Current weather at a location."""

    temperature_f: float          # Fahrenheit
    wind_speed_mph: float         # Miles per hour
    precipitation_inch: float     # Inches (rain equivalent)
    snowfall_inch: float          # Inches of snow
    weather_code: int             # WMO weather interpretation code
    is_day: bool

    @property
    def severity_score(self) -> float:
        """0.0 = clear, 1.0 = extreme. Used as rate multiplier input."""
        score = 0.0

        # Snow severity
        if self.snowfall_inch > 4:
            score += 0.4
        elif self.snowfall_inch > 1:
            score += 0.25
        elif self.snowfall_inch > 0:
            score += 0.1

        # Precipitation (rain)
        if self.precipitation_inch > 1.0:
            score += 0.2
        elif self.precipitation_inch > 0.3:
            score += 0.1

        # Wind
        if self.wind_speed_mph > 45:
            score += 0.3
        elif self.wind_speed_mph > 30:
            score += 0.15
        elif self.wind_speed_mph > 20:
            score += 0.05

        # Extreme cold (diesel gelling, black ice)
        if self.temperature_f < 0:
            score += 0.3
        elif self.temperature_f < 15:
            score += 0.2
        elif self.temperature_f < 32:
            score += 0.1

        # Extreme heat (tire blowouts, reefer strain)
        if self.temperature_f > 110:
            score += 0.2
        elif self.temperature_f > 100:
            score += 0.1

        # WMO code severity
        if self.weather_code in _WMO_SEVERE:
            score += 0.2
        elif self.weather_code in _WMO_MODERATE:
            score += 0.1

        return min(score, 1.0)

    @property
    def condition_label(self) -> str:
        """Human-readable short label for the weather."""
        if self.snowfall_inch > 1:
            return "snow"
        if self.weather_code in _WMO_SEVERE:
            return "storm"
        if self.temperature_f < 15:
            return "freezing"
        if self.precipitation_inch > 0.3:
            return "rain"
        if self.wind_speed_mph > 30:
            return "wind"
        if self.temperature_f > 100:
            return "extreme heat"
        if self.weather_code in _WMO_MODERATE:
            return "fog" if self.weather_code in {45, 48} else "precip"
        return "clear"


# Fallback when weather API fails — neutral conditions
NEUTRAL_WEATHER = WeatherCondition(
    temperature_f=65.0,
    wind_speed_mph=8.0,
    precipitation_inch=0.0,
    snowfall_inch=0.0,
    weather_code=0,
    is_day=True,
)


def _cache_key(lat: float, lng: float) -> Tuple[float, float]:
    """Round to ~11km grid to reduce API calls."""
    return (round(lat, 1), round(lng, 1))


async def get_weather(
    lat: float, lng: float, session: aiohttp.ClientSession
) -> WeatherCondition:
    """Fetch current weather for a location. Returns NEUTRAL_WEATHER on failure."""
    key = _cache_key(lat, lng)
    now = time.time()

    # Check cache
    if key in _weather_cache:
        cached, ts = _weather_cache[key]
        if now - ts < CACHE_TTL:
            return cached

    try:
        params = {
            "latitude": key[0],
            "longitude": key[1],
            "current": "temperature_2m,wind_speed_10m,precipitation,snowfall,weather_code,is_day",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "auto",
        }
        async with session.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                logger.debug(f"Open-Meteo HTTP {resp.status} for ({key[0]}, {key[1]})")
                return NEUTRAL_WEATHER

            data = await resp.json()
            current = data.get("current", {})

            condition = WeatherCondition(
                temperature_f=float(current.get("temperature_2m", 65)),
                wind_speed_mph=float(current.get("wind_speed_10m", 8)),
                precipitation_inch=float(current.get("precipitation", 0)),
                snowfall_inch=float(current.get("snowfall", 0)),
                weather_code=int(current.get("weather_code", 0)),
                is_day=bool(current.get("is_day", 1)),
            )

            _weather_cache[key] = (condition, now)
            logger.debug(
                f"Weather ({key[0]}, {key[1]}): {condition.temperature_f:.0f}°F, "
                f"wind {condition.wind_speed_mph:.0f}mph, "
                f"snow {condition.snowfall_inch:.1f}in, "
                f"severity={condition.severity_score:.2f} [{condition.condition_label}]"
            )
            return condition

    except Exception as e:
        logger.debug(f"Weather fetch failed for ({key[0]}, {key[1]}): {e}")
        return NEUTRAL_WEATHER
