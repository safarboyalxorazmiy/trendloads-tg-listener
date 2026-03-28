"""Estimate driving distance between US cities using Nominatim geocoding + Haversine."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import aiohttp
from loguru import logger

# In-memory geocode cache: "City, ST" -> (lat, lng) or None
_geo_cache: Dict[str, Optional[Tuple[float, float]]] = {}

ROAD_FACTOR = 1.3  # straight-line to driving distance multiplier


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in miles."""
    R = 3959  # Earth radius in miles
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)
    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def _geocode(city: str, state: str, session: aiohttp.ClientSession) -> Optional[Tuple[float, float]]:
    """Geocode a US city via Nominatim. Returns (lat, lng) or None."""
    key = f"{city}, {state}"
    if key in _geo_cache:
        return _geo_cache[key]

    try:
        params = {"q": f"{city}, {state}, USA", "format": "json", "limit": "1"}
        headers = {"User-Agent": "TrendloadsBot/1.0 (trendloads.com)"}
        async with session.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data:
                    lat, lng = float(data[0]["lat"]), float(data[0]["lon"])
                    _geo_cache[key] = (lat, lng)
                    return (lat, lng)
    except Exception as e:
        logger.debug(f"Geocode failed for '{key}': {e}")

    _geo_cache[key] = None
    return None


async def estimate_load_distances(loads: List) -> None:
    """Estimate distance + payout for loads missing them. Mutates loads in-place.

    - If rate exists from GPT extraction -> use it (payoutPredicted=False)
    - If rate missing -> predict from distance * RATE_PER_MILE (payoutPredicted=True)
    - Distance always predicted via geocoding (distancePredicted=True) unless already set
    """
    async with aiohttp.ClientSession() as session:
        for load in loads:
            origin_city = load.originCity
            origin_state = load.originState
            dest_city = load.destCity
            dest_state = load.destState

            # Skip if no origin/dest
            if not (origin_city and origin_state and dest_city and dest_state):
                continue

            # Geocode origin and destination
            o_coord = await _geocode(origin_city, origin_state, session)
            d_coord = await _geocode(dest_city, dest_state, session)

            if not o_coord or not d_coord:
                continue

            # Estimate distance
            straight = _haversine(o_coord[0], o_coord[1], d_coord[0], d_coord[1])
            est_distance = round(straight * ROAD_FACTOR, 1)
            load.estimatedDistance = est_distance
            load.distancePredicted = True

            # Payout: use real rate if available, otherwise weather-aware prediction
            if load.rate and load.rate > 0:
                load.estimatedPayout = load.rate
                load.payoutPredicted = False
            elif hasattr(load, 'estimatedRate') and load.estimatedRate and load.estimatedRate > 0:
                # GPT predicted a market rate with weather context — use it
                load.estimatedPayout = load.estimatedRate
                load.payoutPredicted = True
            else:
                # Fallback only if GPT somehow didn't predict (shouldn't happen now)
                load.estimatedPayout = round(est_distance * 2.50)
                load.payoutPredicted = True

            # ratePerMile: use real rate, GPT prediction, or calculate from payout
            if load.ratePerMile and load.ratePerMile > 0:
                pass  # already set from message
            elif hasattr(load, 'estimatedRatePerMile') and load.estimatedRatePerMile and load.estimatedRatePerMile > 0:
                # GPT predicted a per-mile rate with weather context
                load.ratePerMile = load.estimatedRatePerMile
            elif load.estimatedPayout and est_distance > 0:
                # Calculate from payout
                load.ratePerMile = round(load.estimatedPayout / est_distance, 2)
