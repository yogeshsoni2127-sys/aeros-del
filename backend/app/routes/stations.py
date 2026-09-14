"""
Station Data Routes — Station list, current readings, history
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/stations", tags=["stations"])


def _get_service(request: Request):
    return request.app.state.service


@router.get("")
async def list_stations(request: Request, category: Optional[str] = None):
    """List all Delhi NCR stations with current readings and AQI."""
    service = _get_service(request)
    stations = service.get_stations()
    if category:
        stations = [
            s for s in stations
            if s.get("current") and s["current"].get("category") == category
        ]
    return {"count": len(stations), "stations": stations}


@router.get("/{station_id}")
async def station_detail(request: Request, station_id: str):
    """Get full detail for a single station."""
    service = _get_service(request)
    station = service.get_station(station_id)
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")
    station["forecast"] = service.get_forecast(station_id)
    return station


@router.get("/{station_id}/current")
async def station_current(request: Request, station_id: str):
    """Current pollutant levels + NAQI for a station."""
    service = _get_service(request)
    station = service.get_station(station_id)
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")
    if not station.get("current"):
        raise HTTPException(
            status_code=503,
            detail="No current reading available for this station",
        )
    return station["current"]


@router.get("/{station_id}/forecast")
async def station_forecast(
    request: Request,
    station_id: str,
    hours: int = 72,
):
    """72-hour forecast timeseries for a station."""
    service = _get_service(request)
    forecast = service.get_forecast(station_id)
    if not forecast:
        raise HTTPException(
            status_code=404,
            detail="No forecast available for this station",
        )
    data = dict(forecast)
    if hours and 0 < hours < len(data["timestamps"]):
        for key in ("timestamps", "pm25", "pm10", "no2", "o3", "aqi",
                    "category", "colors", "lower", "upper",
                    "pm10_lower", "pm10_upper",
                    "no2_lower", "no2_upper", "o3_lower", "o3_upper"):
            if key in data:
                data[key] = data[key][:hours]
    return data


@router.get("/{station_id}/history")
async def station_history(
    request: Request,
    station_id: str,
    hours: int = 72,
):
    """Historical readings for a station (from persisted SQLite)."""
    service = _get_service(request)
    station = service.get_station(station_id)
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")
    history = (station.get("history") or [])[-hours:]
    return {"station_id": station_id, "count": len(history), "history": history}