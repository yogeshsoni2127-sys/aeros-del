"""
Environment & API Configuration for SIH AQI Forecasting System.

Centralizes all configuration, environment variable loading, and constants.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = BACKEND_ROOT / "models"


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # ── API Keys ──────────────────────────────────────────────────────
    openaq_api_key: Optional[str] = None
    waqi_api_key: Optional[str] = None
    openweather_api_key: Optional[str] = None
    nasa_firms_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    maptiler_key: Optional[str] = None
    carto_key: Optional[str] = None

    # ── Server ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])

    # ── Data Pipeline ─────────────────────────────────────────────────
    data_refresh_interval: int = 300  # seconds
    database_path: str = "data/aqi_data.db"
    demo_mode: bool = False
    # When True, policy alerts are optionally rephrased by Gemini.
    # Defaults to OFF: the deterministic template engine is always used,
    # forecasting never touches an external LLM.
    llm_enhanced: bool = False

    # ── Delhi NCR Bounding Box ────────────────────────────────────────
    delhi_lat_min: float = 28.30
    delhi_lat_max: float = 28.90
    delhi_lon_min: float = 76.80
    delhi_lon_max: float = 77.50
    delhi_center_lat: float = 28.6139
    delhi_center_lon: float = 77.2090
    delhi_radius_km: float = 80.0

    # ── OpenAQ Configuration ─────────────────────────────────────────
    openaq_base_url: str = "https://api.openaq.org/v3"
    openaq_rate_limit: float = 0.1  # seconds between requests
    openaq_max_concurrency: int = 4  # parallel /latest fetches

    # ── Model Configuration ──────────────────────────────────────────
    forecast_horizon_hours: int = 72
    history_window_hours: int = 72
    tft_weight: float = 0.5
    xgb_weight: float = 0.3
    lgbm_weight: float = 0.2

    # ── AISI Calibration (Delhi Winter) ──────────────────────────────
    aisi_alpha: float = 2.5   # Temperature gradient weight
    aisi_beta: float = 150.0  # Inverse PBL height weight
    aisi_gamma: float = 3.0   # Richardson number weight

    # ── Pollutants Tracked ───────────────────────────────────────────
    pollutants: list = field(default_factory=lambda: [
        "pm25", "pm10", "no2", "so2", "o3", "co"
    ])
    pollutant_display_names: dict = field(default_factory=lambda: {
        "pm25": "PM₂.₅",
        "pm10": "PM₁₀",
        "no2": "NO₂",
        "so2": "SO₂",
        "o3": "O₃",
        "co": "CO",
    })

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables (or .env file)."""
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass

        return cls(
            tft_weight=float(os.getenv("TFT_WEIGHT", "0.5")),
            xgb_weight=float(os.getenv("XGB_WEIGHT", "0.3")),
            lgbm_weight=float(os.getenv("LGBM_WEIGHT", "0.2")),
            openaq_api_key=os.getenv("OPENAQ_API_KEY"),
            waqi_api_key=os.getenv("WAQI_API_KEY"),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
            nasa_firms_api_key=os.getenv("NASA_FIRMS_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            maptiler_key=os.getenv("MAPTILER_KEY"),
            carto_key=os.getenv("CARTO_KEY") or os.getenv("CARTO_API_KEY"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            data_refresh_interval=int(os.getenv("DATA_REFRESH_INTERVAL", "300")),
            database_path=os.getenv("DATABASE_PATH", "data/aqi_data.db"),
            demo_mode=os.getenv("DEMO_MODE", "false").lower() == "true",
            llm_enhanced=os.getenv("LLM_ENHANCED", "false").lower() == "true",
        )


# ──────────────────────────────────────────────────────────────────────
# AQI Color Scale (CPCB Standard)
# ──────────────────────────────────────────────────────────────────────
AQI_COLORS = {
    "Good":            "#00e400",
    "Satisfactory":    "#9cff9c",
    "Moderate":        "#ffff00",
    "Poor":            "#ff7e00",
    "Very Poor":       "#ff0000",
    "Severe":          "#99004c",
    "Severe+":         "#7e0023",
}

AQI_CATEGORIES = [
    {"label": "Good",         "min": 0,   "max": 50,  "color": "#00e400"},
    {"label": "Satisfactory", "min": 51,  "max": 100, "color": "#9cff9c"},
    {"label": "Moderate",     "min": 101, "max": 200, "color": "#ffff00"},
    {"label": "Poor",         "min": 201, "max": 300, "color": "#ff7e00"},
    {"label": "Very Poor",    "min": 301, "max": 400, "color": "#ff0000"},
    {"label": "Severe",       "min": 401, "max": 500, "color": "#99004c"},
    {"label": "Severe+",      "min": 501, "max": 999, "color": "#7e0023"},
]


# ──────────────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────────────
settings = Settings.from_env()
