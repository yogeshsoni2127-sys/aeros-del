"""
Indian National Air Quality Index (NAQI) Calculator

Pure Python implementation of CPCB-compliant NAQI scoring with:
- Breakpoints for all 6 criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3)
- Sub-index calculation with linear interpolation
- Overall AQI = max(sub-indices)
- 7 categories from Good (0-50) to Severe+ (>500)
- Dominant pollutant identification
- Health advisory per category
- Trend classification
- Batch DataFrame processing
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ── CPCB Breakpoint Tables ──────────────────────────────────────────
# Format: [(C_Lo, C_Hi, AQI_Lo, AQI_Hi), ...]
# Source: CPCB National Air Quality Index document

BREAKPOINTS = {
    "pm25": [
        # 24-hour average (µg/m³)
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    "pm10": [
        # 24-hour average (µg/m³)
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    "no2": [
        # 24-hour average (µg/m³)
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 800, 401, 500),
    ],
    "so2": [
        # 24-hour average (µg/m³)
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2400, 401, 500),
    ],
    "co": [
        # 8-hour average (mg/m³)
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 50.0, 401, 500),
    ],
    "o3": [
        # 8-hour average (µg/m³)
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
}

# ── AQI Categories ──────────────────────────────────────────────────
AQI_CATEGORIES = [
    {
        "label": "Good",
        "min": 0,
        "max": 50,
        "color": "#00e400",
        "health_impact": "Minimal impact",
        "advisory": "Enjoy outdoor activities",
    },
    {
        "label": "Satisfactory",
        "min": 51,
        "max": 100,
        "color": "#9cff9c",
        "health_impact": "Minor breathing discomfort to sensitive people",
        "advisory": "Sensitive individuals should reduce prolonged outdoor exertion",
    },
    {
        "label": "Moderate",
        "min": 101,
        "max": 200,
        "color": "#ffff00",
        "health_impact": "Breathing discomfort to people with lungs, asthma, and heart diseases",
        "advisory": "People with heart or lung disease, older adults, and children should reduce prolonged outdoor exertion",
    },
    {
        "label": "Poor",
        "min": 201,
        "max": 300,
        "color": "#ff7e00",
        "health_impact": "Breathing discomfort on prolonged exposure",
        "advisory": "Everyone should reduce prolonged outdoor exertion. People with heart or lung disease should avoid outdoor activity",
    },
    {
        "label": "Very Poor",
        "min": 301,
        "max": 400,
        "color": "#ff0000",
        "health_impact": "Respiratory illness on prolonged exposure",
        "advisory": "Everyone should avoid outdoor exertion. People with heart or lung disease, older adults, and children should remain indoors",
    },
    {
        "label": "Severe",
        "min": 401,
        "max": 500,
        "color": "#99004c",
        "health_impact": "Affects healthy people and seriously impacts those with existing diseases",
        "advisory": "Everyone should avoid all outdoor exertion. People with heart or lung disease should remain indoors and keep activity levels low",
    },
    {
        "label": "Severe+",
        "min": 501,
        "max": 999,
        "color": "#7e0023",
        "health_impact": "Health emergency — everyone may experience serious health effects",
        "advisory": "Stay indoors. Keep all windows and doors closed. Use air purifier if available. Seek medical help if experiencing breathing difficulty",
    },
]


@dataclass
class NAQIResult:
    """Result of NAQI calculation for a single station/time."""

    overall_aqi: float
    category: str
    color: str
    dominant_pollutant: str
    sub_indices: Dict[str, float]
    health_impact: str
    advisory: str


class NAQICalculator:
    """
    Indian National Air Quality Index (NAQI) Calculator.

    Implements CPCB breakpoint-based sub-index calculation with
    linear interpolation. No external dependencies — pure Python.
    """

    def __init__(self):
        self.breakpoints = BREAKPOINTS
        self.categories = AQI_CATEGORIES

    def calculate_sub_index(
        self,
        pollutant: str,
        concentration: float,
    ) -> Optional[float]:
        """
        Calculate AQI sub-index for a single pollutant.

        AQI_p = ((AQI_Hi - AQI_Lo) / (C_Hi - C_Lo)) * (C_p - C_Lo) + AQI_Lo

        Args:
            pollutant: Pollutant name (pm25, pm10, no2, so2, co, o3)
            concentration: Measured concentration value

        Returns:
            AQI sub-index value, or None if pollutant not recognized
        """
        breakpoint_table = self.breakpoints.get(pollutant)
        if breakpoint_table is None:
            logger.warning(f"Unknown pollutant: {pollutant}")
            return None

        if concentration < 0:
            return None

        # Find the breakpoint range. CPCB tables use integer seams
        # (e.g. pm25 30 | 31): decimal model outputs land in the seam, so
        # treat breakpoints as CONTINUOUS (previous bracket's high edge).
        # 60.1 -> ~100, never a skipped pollutant or AQI 0 "Unknown".
        table = breakpoint_table
        for i, (c_lo, c_hi, aqi_lo, aqi_hi) in enumerate(table):
            eff_lo_c = c_lo if i == 0 else table[i - 1][1]
            eff_lo_a = aqi_lo if i == 0 else table[i - 1][3]
            if eff_lo_c <= concentration <= c_hi:
                denom = max(c_hi - eff_lo_c, 0.001)
                sub_index = eff_lo_a + ((aqi_hi - eff_lo_a) / denom) * (concentration - eff_lo_c)
                return round(sub_index, 1)

        # Beyond last breakpoint — extrapolate for Severe+ (>500)
        if concentration > breakpoint_table[-1][1]:
            c_lo, c_hi, aqi_lo, aqi_hi = breakpoint_table[-1]
            slope = (aqi_hi - aqi_lo) / max(c_hi - c_lo, 0.001)
            sub_index = aqi_hi + slope * (concentration - c_hi)
            return round(sub_index, 1)

        return None

    def calculate_naqi(
        self,
        pollutants: Dict[str, Optional[float]],
    ) -> NAQIResult:
        """
        Calculate overall NAQI from pollutant concentrations.

        Overall AQI = max(sub-index for each pollutant)

        Args:
            pollutants: Dict mapping pollutant names to concentrations.
                        None values are skipped.

        Returns:
            NAQIResult with overall AQI, category, dominant pollutant, etc.
        """
        sub_indices = {}

        for pollutant, concentration in pollutants.items():
            if concentration is None:
                continue
            sub_index = self.calculate_sub_index(pollutant, concentration)
            if sub_index is not None:
                sub_indices[pollutant] = sub_index

        if not sub_indices:
            return NAQIResult(
                overall_aqi=0,
                category="Unknown",
                color="#808080",
                dominant_pollutant="N/A",
                sub_indices={},
                health_impact="Insufficient data",
                advisory="No data available for AQI calculation",
            )

        # Overall AQI = max of all sub-indices
        overall_aqi = max(sub_indices.values())
        dominant_pollutant = max(sub_indices, key=sub_indices.get)

        # Find category
        category_info = self._get_category(overall_aqi)

        return NAQIResult(
            overall_aqi=round(overall_aqi, 1),
            category=category_info["label"],
            color=category_info["color"],
            dominant_pollutant=dominant_pollutant,
            sub_indices=sub_indices,
            health_impact=category_info["health_impact"],
            advisory=category_info["advisory"],
        )

    def _get_category(self, aqi: float) -> Dict:
        """Get AQI category info for a given AQI value.

        Continuous lookup (first category whose max covers the value) so
        decimal AQIs like 50.5 or 100.5 land correctly instead of falling
        through integer seams to Severe+.
        """
        for cat in self.categories:
            if aqi <= cat["max"]:
                return cat
        # Beyond Severe+
        return self.categories[-1]

    def classify_trend(
        self,
        current_aqi: float,
        previous_aqi: float,
        hours_apart: float = 1.0,
    ) -> Tuple[str, str]:
        """
        Classify AQI trend based on current vs previous value.

        Args:
            current_aqi: Current AQI value
            previous_aqi: Previous AQI value
            hours_apart: Time difference in hours

        Returns:
            Tuple of (trend_label, trend_icon)
        """
        if hours_apart <= 0:
            return "Stable", "→"

        rate = (current_aqi - previous_aqi) / hours_apart

        if rate < -20:
            return "Rapidly Improving", "⬇⬇"
        elif rate < -5:
            return "Improving", "⬇"
        elif rate <= 5:
            return "Stable", "→"
        elif rate <= 20:
            return "Deteriorating", "⬆"
        else:
            return "Rapidly Deteriorating", "⬆⬆"

    def batch_calculate(
        self,
        records: List[Dict[str, Optional[float]]],
    ) -> List[NAQIResult]:
        """
        Calculate NAQI for a batch of records (e.g., DataFrame rows).

        Args:
            records: List of dicts, each mapping pollutant → concentration

        Returns:
            List of NAQIResult objects
        """
        return [self.calculate_naqi(r) for r in records]

    def to_dict(self, result: NAQIResult) -> Dict:
        """Convert NAQIResult to a JSON-serializable dict."""
        return {
            "overall_aqi": result.overall_aqi,
            "category": result.category,
            "color": result.color,
            "dominant_pollutant": result.dominant_pollutant,
            "sub_indices": result.sub_indices,
            "health_impact": result.health_impact,
            "advisory": result.advisory,
        }
