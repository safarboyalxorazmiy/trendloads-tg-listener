"""State centroid coordinates for pre-fetching weather before GPT extraction.

detect_states() scans raw message text for US state abbreviations,
returning coords so weather can be fetched and included in the extraction prompt.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# State abbreviation → approximate centroid (lat, lng)
STATE_COORDS: Dict[str, Tuple[float, float]] = {
    "AL": (32.8, -86.8), "AK": (64.2, -152.5), "AZ": (34.3, -111.7),
    "AR": (34.8, -92.2), "CA": (36.8, -119.4), "CO": (39.0, -105.5),
    "CT": (41.6, -72.7), "DE": (39.0, -75.5), "FL": (28.6, -82.4),
    "GA": (32.7, -83.5), "HI": (20.5, -157.5), "ID": (44.1, -114.7),
    "IL": (40.0, -89.0), "IN": (39.8, -86.3), "IA": (42.0, -93.5),
    "KS": (38.5, -98.3), "KY": (37.8, -85.3), "LA": (31.0, -92.0),
    "ME": (45.3, -69.2), "MD": (39.0, -76.8), "MA": (42.2, -71.5),
    "MI": (44.3, -84.5), "MN": (46.3, -94.3), "MS": (32.7, -89.7),
    "MO": (38.4, -92.5), "MT": (47.0, -109.6), "NE": (41.5, -99.8),
    "NV": (39.5, -116.9), "NH": (43.7, -71.6), "NJ": (40.1, -74.7),
    "NM": (34.4, -106.1), "NY": (42.9, -75.5), "NC": (35.6, -79.8),
    "ND": (47.4, -100.5), "OH": (40.4, -82.8), "OK": (35.6, -97.5),
    "OR": (43.9, -120.6), "PA": (41.0, -77.6), "RI": (41.6, -71.5),
    "SC": (33.9, -80.9), "SD": (44.4, -100.2), "TN": (35.9, -86.4),
    "TX": (31.5, -99.4), "UT": (39.3, -111.7), "VT": (44.0, -72.7),
    "VA": (37.5, -78.9), "WA": (47.4, -120.7), "WV": (38.6, -80.6),
    "WI": (44.6, -89.8), "WY": (43.0, -107.6), "DC": (38.9, -77.0),
}

_STATE_RE = re.compile(
    r'\b(' + '|'.join(STATE_COORDS.keys()) + r')\b'
)


def detect_states(text: str) -> List[str]:
    """Extract unique US state abbreviations from message text."""
    return list(dict.fromkeys(_STATE_RE.findall(text.upper())))


def get_state_coords(state: str) -> Tuple[float, float] | None:
    """Return (lat, lng) for a state abbreviation."""
    return STATE_COORDS.get(state.upper())
