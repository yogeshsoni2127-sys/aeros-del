"""
CPCB data.gov.in Client — PRIMARY Ground-Truth Source.

Fetches hourly CPCB CAAQMS readings for Delhi NCR via the official
data.gov.in resource "Real time Air Quality Index from various locations"
(3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69). Mirrors
OpenAQClient.get_latest_measurements() and returns StationReading lists
so the service chain can treat it as a drop-in primary.

Needs DATAGOV_API_KEY in .env (free key from data.gov.in; the
CPCB_DATAGOV_KEY alias is also accepted). Without a key every method
returns [] (no crash — the chain falls through to WAQI/OpenAQ/demo).

Response notes (verified against live records, Sept 2026):
- Value columns are `avg_value` / `max_value` / `min_value` (older docs
  say `pollutant_avg`, both spellings are accepted).
- `latitude` / `longitude` ship as strings; "0.0" fallback when absent.
- `last_update` looks like "14-09-2026 16:00:00" (IST, naive).
- Missing values arrive as "X"/"NA"/"" and are skipped.

Sub-index caveat: per CPCB docs the Min/Max/Avg columns are AQI
SUB-INDEX values, not µg/m³ (e.g. CO avg "22" is an index — 22 mg/m³
ambient CO would be absurd). Downstream expects concentrations, so the
default `value_mode="subindex"` inverts each Avg back to an estimated
concentration by inverse-interpolating naqi_calculator.BREAKPOINTS. The
round trip is exact (piecewise-linear both ways), so recomputed NAQI
matches the published sub-index. If CPCB ever ships raw concentrations,
set value_mode="concentration" (or CPCB_DATAGOV_VALUE_MODE env).
"""

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .openaq_client import StationReading, POLLUTANT_BOUNDS
from .naqi_calculator import BREAKPOINTS

RESOURCE_UUID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_UUID}"

# Delhi NCR cities as named by CPCB in this resource.
CPCB_CITIES = ["Delhi", "New Delhi", "Noida", "Ghaziabad", "Gurugram",
               "Faridabad"]

# data.gov.in pollutant_id -> our internal pollutant names.
# NH3/metals/VOCs have no NAQI mirror and are skipped.
POLLUTANT_ID_MAP = {
    "PM2.5": "pm25",
    "PM25": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "OZONE": "o3",
    "O3": "o3",
}

MISSING_TOKENS = {"", "NA", "N/A", "X", "NONE", "NULL", "-", "--"}

_TS_FORMATS = (
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
)


def _parse_float(value: Any) -> Optional[float]:
    """Parse a data.gov.in numeric cell; X/NA/'' -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().strip(",")
    if text.upper() in MISSING_TOKENS:
        return None
    try:
        return float(text.replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_last_update(raw: Any) -> Optional[str]:
    """
    Parse a CPCB `last_update` cell to a UTC ISO timestamp.

    Naive values are IST (CPCB station time). Returns None when
    unparseable — callers treat that as stale, never as fresh.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for fmt in _TS_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    else:
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        try:
            from zoneinfo import ZoneInfo
            dt = dt.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        except Exception:
            from datetime import timedelta
            dt = dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
    return dt.astimezone(timezone.utc).isoformat()


def subindex_to_concentration(pollutant: str,
                              subindex: float) -> Optional[float]:
    """
    Invert a CPCB sub-index to an estimated concentration (µg/m³,
    mg/m³ for co) via inverse-linear interpolation of the NAQI
    breakpoint table. Beyond the top row, extrapolate the last
    segment's slope; out-of-range results are rejected by the
    caller's POLLUTANT_BOUNDS check.
    """
    rows = BREAKPOINTS.get(pollutant)
    if not rows or subindex < 0:
        return None
    row = None
    for r in rows:
        if r[2] <= subindex <= r[3]:
            row = r
            break
    if row is None:
        if subindex < rows[0][2]:
            return None
        row = rows[-1]  # extrapolate worst-row slope
    c_lo, c_hi, a_lo, a_hi = row
    if a_hi == a_lo:
        return None
    return c_lo + (subindex - a_lo) * (c_hi - c_lo) / (a_hi - a_lo)


class CPCBDataGovClient:
    """
    Async client for the CPCB data.gov.in resource.

    Usage mirrors OpenAQClient: `await get_latest_measurements()`
    returns one StationReading per CPCB station (coords attached when
    sane, else 0.0 — match by station name downstream, proximity as
    fallback).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cities: Optional[List[str]] = None,
        limit: int = 100,
        rate_limit: float = 0.5,
        timeout: float = 15.0,
        value_mode: Optional[str] = None,
    ):
        self.api_key = (api_key or "").strip()
        self.cities = cities or list(CPCB_CITIES)
        self.limit = limit
        self.rate_limit = rate_limit
        self.timeout = timeout
        mode = (value_mode or os.getenv("CPCB_DATAGOV_VALUE_MODE")
                or "subindex").lower()
        self.value_mode = mode if mode in ("subindex", "concentration") \
            else "subindex"
        self._last = 0.0
        self._http = None
        self._no_key_warned = False
        # Observability for the probe script / refresh logs.
        self.last_fetch: Dict[str, Any] = {}

    async def _get(self, params: Dict[str, Any]) -> Optional[Dict]:
        if not self.api_key:
            if not self._no_key_warned:
                logger.warning("DATAGOV_API_KEY missing — CPCB primary "
                               "disabled, chain falls through")
                self._no_key_warned = True
            return None
        elapsed = time.time() - self._last
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        params = {
            "api-key": self.api_key,
            "format": "json",
            **params,
        }
        try:
            import httpx
        except ImportError:
            httpx = None
        if httpx is None:
            # aiohttp fallback (fresh session per call, like OpenAQClient).
            try:
                import aiohttp
                self._last = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        BASE_URL, params=params,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status != 200:
                            logger.warning("data.gov.in -> HTTP %s",
                                           resp.status)
                            return None
                        return await resp.json()
            except Exception as e:
                logger.warning("data.gov.in request failed: %s: %s",
                               type(e).__name__, e)
                return None
        try:
            if self._http is None:
                self._http = httpx.AsyncClient(timeout=self.timeout)
            self._last = time.time()
            r = await self._http.get(BASE_URL, params=params)
            if r.status_code != 200:
                logger.warning("data.gov.in -> HTTP %s", r.status_code)
                return None
            body = r.json()
            if body.get("status") not in ("ok", None) and "records" not in body:
                logger.warning("data.gov.in: %s",
                               str(body.get("message"))[:120])
                return None
            return body
        except Exception as e:
            logger.warning("data.gov.in request failed: %s: %s",
                           type(e).__name__, e)
            return None

    async def fetch_city_records(self, city: str) -> List[Dict]:
        """All raw records for one city, following offset pagination."""
        records: List[Dict] = []
        offset = 0
        while True:
            body = await self._get({
                "limit": self.limit,
                "offset": offset,
                "filters[city]": city,
            })
            if not body:
                break
            batch = body.get("records") or []
            records.extend(batch)
            try:
                total = int(body.get("total") or 0)
            except (TypeError, ValueError):
                total = 0
            offset += len(batch)
            if not batch or len(batch) < self.limit:
                break
            if total and offset >= total:
                break
            if offset > 5000:  # sanity cap
                logger.warning("data.gov.in pagination cap hit for %s", city)
                break
        return records

    @staticmethod
    def _norm_key(key: Any) -> str:
        return re.sub(r"[\s\-]+", "_", str(key).strip().lower())

    def _cell(self, rec: Dict, *names: str) -> Any:
        normed = {self._norm_key(k): v for k, v in rec.items()}
        for n in names:
            if n in normed:
                return normed[n]
        return None

    def _record_to_value(self, rec: Dict) -> Optional[tuple]:
        """(internal_pollutant, concentration, timestamp) or None."""
        pid = str(self._cell(rec, "pollutant_id") or "").strip().upper()
        internal = POLLUTANT_ID_MAP.get(pid)
        if not internal:
            return None
        avg = _parse_float(self._cell(
            rec, "pollutant_avg", "avg_value", "avg", "average"))
        if avg is None:
            # Fall back to max, then min — a stale extreme beats nothing,
            # and the freshness gate still applies to the timestamp.
            avg = _parse_float(self._cell(rec, "pollutant_max", "max_value",
                                          "max"))
        if avg is None:
            avg = _parse_float(self._cell(rec, "pollutant_min", "min_value",
                                          "min"))
        if avg is None:
            return None
        if self.value_mode == "subindex":
            conc = subindex_to_concentration(internal, avg)
        else:
            conc = avg
        if conc is None:
            return None
        lo, hi = POLLUTANT_BOUNDS[internal]
        if not (lo <= conc <= hi):
            return None
        ts = parse_last_update(self._cell(rec, "last_update"))
        return internal, conc, ts

    async def get_latest_measurements(self) -> List[StationReading]:
        """
        One StationReading per CPCB station across the NCR cities.

        Groups per-pollutant records by station name; timestamp is the
        newest last_update seen for that station. Coordinates attached
        when sane — match by station name downstream, proximity fallback.
        """
        stats: Dict[str, Any] = {
            "per_city": {}, "records_raw": 0, "stations_grouped": 0,
            "dropped_no_pollutant": 0, "inverted": 0, "passthrough": 0,
            "unparsed_timestamps": 0, "value_mode": self.value_mode,
        }
        if not self.api_key:
            await self._get({})  # emits the one-time missing-key warning
            self.last_fetch = stats
            return []

        grouped: Dict[str, Dict[str, Any]] = {}
        # Cities in parallel (bounded): data.gov.in can be slow from some
        # networks, and serial cities x timeout stalled whole refreshes.
        # Worst case is now ~1 timeout, not N. Six concurrent polite
        # GETs per 5-min cycle is well within fair use.
        _city_sem = asyncio.Semaphore(3)

        async def _one(city: str):
            async with _city_sem:
                return city, await self.fetch_city_records(city)

        city_results = await asyncio.gather(
            *(_one(c) for c in self.cities))
        for city, recs in city_results:
            stats["per_city"][city] = len(recs)
            stats["records_raw"] += len(recs)
            for rec in recs:
                name = str(self._cell(rec, "station") or "").strip()
                if not name:
                    continue
                parsed = self._record_to_value(rec)
                if parsed is None:
                    stats["dropped_no_pollutant"] += 1
                    continue
                internal, conc, ts = parsed
                if self.value_mode == "subindex":
                    stats["inverted"] += 1
                else:
                    stats["passthrough"] += 1
                if ts is None:
                    stats["unparsed_timestamps"] += 1
                slot = grouped.setdefault(name, {
                    "city": str(self._cell(rec, "city") or city),
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "pollutants": {},
                    "timestamp": None,
                })
                lat = _parse_float(self._cell(rec, "latitude", "lat"))
                lon = _parse_float(self._cell(rec, "longitude", "lon", "lng"))
                # NCR-wide sanity box; garbage stays 0.0 (name-matched).
                if (lat is not None and lon is not None
                        and 27.5 <= lat <= 29.5 and 76.0 <= lon <= 78.5):
                    slot["latitude"] = lat
                    slot["longitude"] = lon
                # Keep the strongest reading per pollutant (max is the
                # protective choice; NAQI takes max sub-index anyway).
                prev = slot["pollutants"].get(internal)
                if prev is None or conc > prev:
                    slot["pollutants"][internal] = round(conc, 2)
                if ts and (slot["timestamp"] is None or ts > slot["timestamp"]):
                    slot["timestamp"] = ts

        readings: List[StationReading] = []
        for name, slot in grouped.items():
            if not slot["pollutants"]:
                continue
            readings.append(StationReading(
                station_id="cpcb-" + re.sub(r"[^a-z0-9]+", "-",
                                            name.lower()).strip("-"),
                station_name=name,
                latitude=slot["latitude"],
                longitude=slot["longitude"],
                timestamp=(slot["timestamp"]
                           or datetime.now(timezone.utc).isoformat()),
                pollutants=slot["pollutants"],
                source="cpcb",
            ))
        stats["stations_grouped"] = len(readings)
        self.last_fetch = stats
        logger.info("CPCB data.gov.in: %d stations from %d records "
                    "(%s)", len(readings), stats["records_raw"],
                    ", ".join(f"{c}={n}" for c, n in stats["per_city"].items()))
        return readings

    async def close(self):
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None
