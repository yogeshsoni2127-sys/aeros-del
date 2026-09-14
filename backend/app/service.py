"""
AQI Service — ORCHESTRATION LAYER

Wires together the full pipeline:
live data ingestion → physics diagnostics → ML ensemble forecast →
NLP advisory → in-memory state consumed by REST + WebSocket routes.

Runs a defensive refresh cycle so the system produces coherent output
even when upstream APIs are unreachable (demo-fallback generators).
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from backend.app.config import Settings, PROJECT_ROOT, DATA_DIR, MODELS_DIR

from backend.data.openaq_client import OpenAQClient
from backend.data.cpcb_datagov_client import CPCBDataGovClient
from backend.data.waqi_client import WAQIClient
from backend.data.weather_client import WeatherClient
from backend.data.fire_client import FireClient
from backend.data.naqi_calculator import NAQICalculator
from backend.data.preprocessor import DataPreprocessor

from backend.physics.aisi_calculator import AISICalculator
from backend.physics.radiation_feedback import RadiationFeedback
from backend.physics.plume_transport import PlumeTransportModel
from backend.physics.emission_processor import EmissionProcessor

from backend.ml.feature_engineering import FeatureEngineer
from backend.ml.ensemble import EnsembleForecaster, ForecastResult

from backend.nlp.alert_generator import AlertGenerator
from backend.nlp.severity_classifier import SeverityClassifier


class AQIService:
    """
    Central service that owns data clients, physics, ML and NLP engines,
    and exposes a consistent snapshot for the API layer.
    """

    # Readings older than this are rejected by the freshness gate and
    # their stations treated as dark (filled by the next chain source).
    FRESHNESS_MAX_AGE_H = 3.0
    # Minimum fresh stations for data_source="live"; fewer -> "degraded".
    MIN_FRESH_FOR_LIVE = 5

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = (settings if settings is not None
                         else Settings.from_env())

        # ── Data clients ────────────────────────────────────────────
        # Chain: CPCB data.gov.in (primary ground truth) -> WAQI fill ->
        # OpenAQ fill -> demo. See _collect_raw_data.
        self.cpcb_datagov = CPCBDataGovClient(
            api_key=self.settings.datagov_api_key,
        )
        self.openaq = OpenAQClient(
            api_key=self.settings.openaq_api_key,
            base_url=self.settings.openaq_base_url,
            rate_limit=self.settings.openaq_rate_limit,
            max_concurrency=self.settings.openaq_max_concurrency,
        )
        # Secondary source: fills stations the primary missed (bounded
        # per cycle so the free-token rate limit never stalls refreshes).
        self.waqi = WAQIClient(api_key=self.settings.waqi_api_key)
        self.waqi_fill_cap = 16
        self.weather = WeatherClient()
        self.fire = FireClient(api_key=self.settings.nasa_firms_api_key)
        self.naqi = NAQICalculator()
        db_path = str(PROJECT_ROOT / self.settings.database_path)
        self.preprocessor = DataPreprocessor(database_path=db_path)

        # ── Physics engines ─────────────────────────────────────────
        self.aisi = AISICalculator(
            alpha=self.settings.aisi_alpha,
            beta=self.settings.aisi_beta,
            gamma=self.settings.aisi_gamma,
        )
        self.radiation = RadiationFeedback(
            latitude=self.settings.delhi_center_lat,
            longitude=self.settings.delhi_center_lon,
        )
        self.plume = PlumeTransportModel()
        self.emission_processor = EmissionProcessor()

        # ── ML stack ────────────────────────────────────────────────
        self.feature_engineer = FeatureEngineer(
            history_window=self.settings.history_window_hours,
            horizon=self.settings.forecast_horizon_hours,
        )
        self.ensemble = EnsembleForecaster(
            tft_weight=self.settings.tft_weight,
            xgb_weight=self.settings.xgb_weight,
            lgbm_weight=self.settings.lgbm_weight,
            model_dir=str(MODELS_DIR),
        )

        # ── NLP engines ─────────────────────────────────────────────
        self.severity = SeverityClassifier()
        # NLP-first: try Gemini whenever a key exists (LLM_ENHANCED=true
        # forces it even with a weak key); otherwise the local NLG engine
        # generates the advisory. Forecasting never depends on an LLM.
        llm_key = self.settings.gemini_api_key or None
        self.alert_generator = AlertGenerator(
            gemini_api_key=llm_key,
            classifier=self.severity,
        )

        # ── Domain metadata ─────────────────────────────────────────
        self.stations = self._load_stations()

        # ── Shared lock + state snapshot ────────────────────────────
        self._lock = asyncio.Lock()
        self.state: Dict[str, Any] = {
            "initialized": False,
            "last_update": None,
            "data_source": "unknown",         # live | degraded | demo
            "stations": [],                   # enriched station snapshots
            "weather": {},
            "fires": [],
            "fire_stats": {},
            "plume": {},
            "aisi": {},
            "radiation": {},
            "emissions": {},
            "forecasts": {},                  # station_id → ForecastResult
            "spatial": {},
            "alerts": [],
            "domain_summary": {},
        }

    # ────────────────────────────────────────────────────────────────
    # Public lifecycle
    # ────────────────────────────────────────────────────────────────

    async def initialize(self):
        """One-time setup: DB init and initial data refresh."""
        await self.preprocessor.initialize_database()
        await self.refresh(force=True)
        self.state["initialized"] = True

    async def refresh(self, force: bool = False) -> Dict[str, Any]:
        """
        Execute one full pipeline refresh (data → physics → ML → NLP).

        Args:
            force: Bypass the polling interval gate.

        Returns:
            A summary dict describing what was refreshed.
        """
        if not force:
            last = self.state.get("last_update")
            if last is not None:
                delta = (datetime.now(timezone.utc) -
                         datetime.fromisoformat(last)).total_seconds()
                if delta < self.settings.data_refresh_interval:
                    return {"refreshed": False, "last_update": last}

        async with self._lock:
            try:
                await self._collect_raw_data()
                await self._compute_physics()
                await self._build_station_snapshots()
                await self._compute_forecasts()
                await self._generate_alerts()
                self._build_domain_summary()

                self.state["last_update"] = datetime.now(timezone.utc).isoformat()
                return {"refreshed": True, "last_update": self.state["last_update"]}

            except Exception as e:
                logger.exception("Refresh cycle failed: %s", e)
                return {"refreshed": False, "error": str(e)}

    # ────────────────────────────────────────────────────────────────
    # Pipeline steps
    # ────────────────────────────────────────────────────────────────

    async def _collect_raw_data(self):
        """Fetch readings, weather, fires, and persist to SQLite.

        Source chain: CPCB data.gov.in (primary, name-matched) -> WAQI
        (fills dark stations, capped + bounded-parallel) -> OpenAQ
        (fills still-dark stations) -> demo (only when NO source
        returned anything). A freshness gate then rejects readings
        older than FRESHNESS_MAX_AGE_H; data_source is "live" only
        with >= MIN_FRESH_FOR_LIVE fresh stations, else "degraded".
        """
        mapped: Dict[str, Dict] = {}

        # 1) CPCB primary — official ground truth.
        cpcb_readings = await self._safe(
            self.cpcb_datagov.get_latest_measurements()) or []
        for sid, entry in self._match_cpcb_readings(cpcb_readings).items():
            mapped[sid] = entry

        # 2) WAQI secondary fill for stations the primary missed.
        # Only fires when a key exists AND some stations are still dark.
        waqi_targets = [s for s in self.stations
                        if s["id"] not in mapped][:self.waqi_fill_cap]
        if waqi_targets and self.settings.waqi_api_key:
            _sem = asyncio.Semaphore(3)

            async def _fill(station):
                async with _sem:
                    feed = await self._safe(self.waqi.get_station_feed(
                        station["latitude"], station["longitude"]))
                if feed and feed.get("pollutants"):
                    return station["id"], {
                        "pollutants": feed["pollutants"],
                        "timestamp": feed["timestamp"],
                        "source": "waqi",
                    }
                return station["id"], None

            for sid, entry in await asyncio.gather(
                    *(_fill(s) for s in waqi_targets)):
                if entry is not None:
                    mapped[sid] = entry

        # 3) OpenAQ tertiary fill for still-dark stations.
        if len(mapped) < len(self.stations):
            openaq_readings = await self._safe(
                self.openaq.get_latest_measurements()) or []
            if openaq_readings:
                for sid, entry in self._match_readings(
                        openaq_readings, source="openaq").items():
                    mapped.setdefault(sid, entry)

        had_any_readings = bool(mapped)

        # 4) Freshness gate: reject readings older than 3h (30h lag must
        # never silently poison history/features/AISI again).
        freshness = self._apply_freshness_gate(mapped)
        self.state["freshness"] = freshness
        logger.info("ingest freshness: %d fresh / %d stale "
                    "(cpcb_fresh=%d waqi_filled=%d openaq_stale=%d)",
                    freshness["fresh"], freshness["stale"],
                    freshness["by_source"].get("cpcb_fresh", 0),
                    freshness["by_source"].get("waqi_fresh", 0),
                    freshness["by_source"].get("openaq_stale", 0))
        self.state["waqi_filled"] = (
            freshness["by_source"].get("waqi_fresh", 0))

        # Fall back to demo generation when no source returned anything.
        # Stale-but-present data -> "degraded" with dark stations (never
        # demo-masked, so lag stays visible instead of silent).
        if not had_any_readings:
            mapped = self._demo_readings()
            self.state["data_source"] = "demo"
        elif freshness["fresh"] >= self.MIN_FRESH_FOR_LIVE:
            self.state["data_source"] = "live"
        else:
            self.state["data_source"] = "degraded"

        # Normalize each station entry into persisted reading format
        records = []
        for station in self.stations:
            reading = mapped.get(station["id"])
            if reading is None:
                continue
            pollutants = reading.get("pollutants", {})
            result = self.naqi.calculate_naqi(pollutants)
            record = {
                "station_id": station["id"],
                "station_name": station["name"],
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "timestamp": reading.get("timestamp",
                                          datetime.now(timezone.utc).isoformat()),
                "age_hours": reading.get("age_hours"),
                "pollutants": pollutants,
                "aqi": result.overall_aqi,
                "category": result.category,
                "dominant_pollutant": result.dominant_pollutant,
                "source": reading.get("source", self.state["data_source"]),
            }
            records.append(record)

        await self._safe(self.preprocessor.store_readings(records))
        self.state["raw_records"] = records

        # Weather (Delhi center)
        weather = await self._safe(self.weather.get_current_weather(
            self.settings.delhi_center_lat,
            self.settings.delhi_center_lon,
        ))
        self.state["weather"] = weather or {}

        # Fires
        fires = await self._safe(self.fire.get_active_fires(days_back=2))
        self.state["fires"] = fires or []
        self.state["fire_stats"] = self.fire.aggregate_fire_stats(
            self.state["fires"]
        )

    async def _compute_physics(self):
        """Run AISI, radiation, plume and emission engines on current data."""
        weather = dict(self.state.get("weather", {}))

        # Inject the real ERA5 PBL height for the current hour BEFORE the
        # AISI/PBL computation — otherwise the physics falls back to a
        # coarse 350/1200 m day-night heuristic that swings AISI wildly.
        wind_forecast = await self._safe(self.weather.get_hourly_forecast(
            self.settings.delhi_center_lat,
            self.settings.delhi_center_lon,
            forecast_days=2,
        ))
        if wind_forecast:
            pbl_series = wind_forecast.get("pbl_height_m") or []
            if pbl_series:
                try:
                    from zoneinfo import ZoneInfo
                    from datetime import datetime, timezone as _tz
                    now_ist = datetime.now(_tz.utc).astimezone(
                        ZoneInfo("Asia/Kolkata"))
                    fc_times = wind_forecast.get("timestamps") or []
                    idx = 0
                    for i, ts in enumerate(fc_times):
                        if str(ts)[:13] <= now_ist.strftime("%Y-%m-%dT%H"):
                            idx = i
                    pbl_now = pbl_series[min(idx, len(pbl_series) - 1)]
                    if pbl_now:
                        weather["pbl_height_m"] = float(pbl_now)
                except Exception as e:
                    logger.debug("PBL injection failed: %s", e)

        # Mean PM2.5 across resolved stations (needed BEFORE AISI so
        # GRAP stage agrees with the AQI category, not just meteorology).
        # GRAP uses the WORST station (protective, matches the "Worst
        # Station" header) — mean would hide a Poor hotspot behind a
        # clean city average (e.g. mean AQI 72 + worst 285).
        records = self.state.get("raw_records", [])
        pm25s = [r.get("pollutants", {}).get("pm25") for r in records]
        pm25s = [p for p in pm25s if p is not None]
        mean_pm25 = sum(pm25s) / len(pm25s) if pm25s else 120.0
        worst_pm25 = max(pm25s) if pm25s else mean_pm25

        aisi_result = await self._safe(
            self.aisi.compute(weather, pm25=worst_pm25))
        if aisi_result:
            aisi_result["grap_basis"] = {
                "pm25_used": round(worst_pm25, 1),
                "mean_pm25": round(mean_pm25, 1),
                "note": "GRAP from worst station (protective); "
                        "AISI itself is meteorology-only.",
            }
        self.state["aisi"] = aisi_result or {}

        pblh = (aisi_result or {}).get("pbl", {}).get("pbl_height_m", 700.0)
        rad_result = await self._safe(self.radiation.compute(
            pm25=mean_pm25,
            cloud_cover_pct=weather.get("cloud_cover_pct"),
            baseline_pbl_m=pblh,
        ))
        self.state["radiation"] = rad_result or {}

        # Wind forecast for plume transport (reuses the hourly fetch above)
        plume_result = await self._safe(self.plume.compute_trajectories(
            self.state.get("fires", []),
            wind_forecast=wind_forecast,
        ))
        self.state["plume"] = plume_result or {}

        emissions = await self._safe(self.emission_processor.compute(
            fires=self.state.get("fires", []),
        ))
        self.state["emissions"] = emissions or {}

    async def _build_station_snapshots(self):
        """Attach current reading + AQI metadata to each station."""
        records = {
            r["station_id"]: r for r in self.state.get("raw_records", [])
        }
        snapshots = []

        for station in self.stations:
            record = records.get(station["id"])
            current = None
            if record:
                current = self._station_payload(record)

            # History for forecast feature building
            history = await self._safe(self.preprocessor.get_station_history(
                station["id"], hours=72
            ))

            snapshot = {
                "id": station["id"],
                "name": station["name"],
                "short_name": station["short_name"],
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "city": station["city"],
                "zone": station["zone"],
                "type": station["type"],
                "elevation_m": station["elevation_m"],
                "current": current,
                "history_count": len(history),
                "history": history[-72:],
            }
            snapshots.append(snapshot)

        self.state["stations"] = snapshots

    async def _compute_forecasts(self):
        """Generate 72-hour ensemble forecasts for every station."""
        aisi_state = self.state.get("aisi", {})
        plume = self.state.get("plume", {})
        pblh = (aisi_state.get("pbl") or {}).get("pbl_height_m", 700.0)

        # Fire contribution estimate (sum of plume contributions arriving)
        fire_contrib = sum(
            e.get("estimated_contribution_pm25", 0.0)
            for e in plume.get("arrival_estimates", [])
        )

        contexts = []
        for station in self.state.get("stations", []):
            current = station.get("current") or {}
            pollutants = current.get("pollutants", {})
            pm25 = pollutants.get("pm25")
            if pm25 is None:
                continue

            history = [r.get("pm25") for r in station.get("history", [])]
            history = [h for h in history if h is not None]

            # Per-station PM10/PM2.5 ratio from live observations (clamped),
            # else median of history, else Delhi-typical 1.35 fallback.
            pm10_ratio = self._station_pm10_ratio(
                station, pollutants.get("pm25"), pollutants.get("pm10"))

            features = self.feature_engineer.build_features(
                station_id=station["id"],
                readings=station.get("history", []),
                weather=self.state.get("weather", {}),
                fire_summary=self.state.get("fire_stats", {}),
                physics=aisi_state,
            )

            ctx = self.ensemble.build_context(
                station_id=station["id"],
                current_pm25=pm25,
                current_pm10=pollutants.get("pm10"),
                history_pm25=history,
                aisi=aisi_state.get("aisi", 2.0),
                pbl_height_m=pblh,
                inversion_strength=(aisi_state.get("pbl") or {})
                    .get("inversion_strength_k", 0.0),
                fire_contribution=fire_contrib,
                features=features,
                pm10_ratio=pm10_ratio,
            )
            contexts.append(ctx)

        forecasts = await self.ensemble.forecast_domain(contexts)
        self.state["forecasts"] = {
            f.station_id: f.to_dict() for f in forecasts
        }

        # REAL-model overlay: SIH-p2 exporter output (daily TFT+XGB+LGBM,
        # hourly-downscaled) wins over the baseline ensemble wherever it
        # is fresh (<60h, daily_refresh.sh runs every 24h). Falls back
        # to baseline when stale/missing; the UI vintage badge warns first.
        self._apply_real_overlay()

        # Spatial forecast grid overlay (from state so the real-model
        # overlay is reflected on the map, not just the live ensemble).
        self.state["spatial"] = self._build_spatial_geojson(
            self.state.get("forecasts", {}))

    def _load_real_overlay(self) -> Dict[str, Any]:
        """Read exporter output (data/sample_forecasts/) with freshness gate."""
        overlay: Dict[str, Any] = {}
        try:
            idx_path = DATA_DIR / "sample_forecasts" / "index.json"
            idx = json.loads(idx_path.read_text())
            gen = datetime.fromisoformat(str(idx.get("generated_at", "")))
            age_h = (datetime.now(timezone.utc) - gen).total_seconds() / 3600.0
            # 60h gate: daily_refresh.sh runs every 24h, so one missed day
            # degrades via the UI vintage badge instead of silently
            # swapping to baseline. Beyond 60h the feed is truly dead.
            if age_h > 60:
                logger.info("Real overlay stale (%.1fh) — using live ensemble", age_h)
                return {}
            for fname in idx.get("station_forecasts", []):
                p = DATA_DIR / "sample_forecasts" / fname
                try:
                    payload = json.loads(p.read_text())
                except Exception:
                    continue
                if (payload.get("source", "").startswith("sih-p2")
                        and len(payload.get("timestamps", [])) == 72):
                    overlay[payload["station_id"]] = payload
            logger.info("Real overlay: %d stations (age %.1fh)",
                        len(overlay), age_h)
        except Exception as e:
            logger.debug("Real overlay unavailable: %s", e)
        return overlay

    def _apply_real_overlay(self) -> None:
        overlay = self._load_real_overlay()
        self.state["real_overlay_n"] = len(overlay)
        if not overlay:
            return
        for sid, payload in overlay.items():
            try:
                self.state["forecasts"][sid] = {
                    **ForecastResult(
                        station_id=sid,
                        timestamps=payload["timestamps"],
                        pm25=[float(v) for v in payload["pm25"]],
                        pm10=[float(v) for v in payload["pm10"]],
                        aqi=[float(v) for v in payload["aqi"]],
                        category=list(payload["category"]),
                        colors=list(payload["colors"]),
                        lower=[float(v) for v in payload.get("lower", [])],
                        upper=[float(v) for v in payload.get("upper", [])],
                        pm10_lower=[float(v) for v in payload.get("pm10_lower", [])],
                        pm10_upper=[float(v) for v in payload.get("pm10_upper", [])],
                        dominant_pollutant=payload.get("daily", [{}])[0].get(
                            "dominant_pollutant", "pm25"),
                        models=dict(payload.get("models", {})),
                        generated_at=payload.get("generated_at", ""),
                    ).to_dict(),
                    # Extras pass through to station-detail + accuracy table.
                    "no2": payload.get("no2", []),
                    "o3": payload.get("o3", []),
                    "lower": payload.get("lower", []),
                    "upper": payload.get("upper", []),
                    "pm10_lower": payload.get("pm10_lower", []),
                    "pm10_upper": payload.get("pm10_upper", []),
                    "no2_lower": payload.get("no2_lower", []),
                    "no2_upper": payload.get("no2_upper", []),
                    "o3_lower": payload.get("o3_lower", []),
                    "o3_upper": payload.get("o3_upper", []),
                    "daily": payload.get("daily", []),
                    "blend_used": payload.get("blend_used", {}),
                    "provenance": "sih-p2 ensemble (daily model, hourly downscaled)",
                }
            except Exception as e:
                logger.debug("Overlay failed for %s: %s", sid, e)

    async def _generate_alerts(self):
        """Generate domain advisory from the worst station."""
        forecasts = self.state.get("forecasts", {})
        if not forecasts:
            return

        worst_station_id = max(
            forecasts,
            key=lambda sid: forecasts[sid]["aqi"][0] if forecasts[sid]["aqi"] else 0,
        )
        forecast = forecasts[worst_station_id]
        station = self._station_by_id(worst_station_id)

        aisi_state = self.state.get("aisi", {})
        fires = self.state.get("fires", [])
        plume = self.state.get("plume", {})

        trend_label, _ = self.severity.forecast_trend(forecast.get("aqi", []))
        dominants = ((station or {}).get("current") or {}).get("pollutants", {})
        if not dominants and self.state.get("raw_records"):
            dominants = self.state["raw_records"][0].get("pollutants", {})

        # Recompute GRAP with peak PM2.5 so stage agrees with the AQI
        # category (AISI-only GRAP caused Moderate+Stage-II mismatches).
        from backend.formulas.aisi_formulas import grap_activation_level
        peak_pm25 = max(forecast["pm25"]) if forecast.get("pm25") else 0.0
        peak_aqi = max(forecast["aqi"]) if forecast.get("aqi") else 0.0
        grap = grap_activation_level(
            aisi=aisi_state.get("aisi", 0.0), pm25=peak_pm25,
        )

        alert = await self.alert_generator.generate(
            category=forecast["category"][0] if forecast.get("category") else "Moderate",
            peak_pm25=peak_pm25,
            peak_aqi=peak_aqi,
            aisi=aisi_state.get("aisi", 0.0),
            dominants=dominants,
            corridor=(plume.get("corridor") or {}).get("name", "North-Westerly"),
            active_fires=len(fires),
            trend=trend_label,
            aq_series=forecast.get("aqi"),
            grap=grap,
            horizon_hours=len(forecast["timestamps"]),
            language="en",
        )
        alert["station_id"] = worst_station_id
        alert["station_name"] = (station or {}).get("name", "Delhi NCR")
        # Canonical feed is English-only; Hindi is rendered on demand and
        # never mixed into this list (avoids showing the same advice twice).
        # history() already includes this alert newest-first; don't duplicate.
        self.state["alerts"] = self.alert_generator.history(
            limit=10, language="en")

        # Keep the inputs so the same advisory can be re-rendered in
        # another language on demand (see generate_alert_in_language).
        self.state["alert_context"] = {
            "inputs": {
                "category": (forecast["category"][0]
                             if forecast.get("category") else "Moderate"),
                "peak_pm25": peak_pm25,
                "peak_aqi": peak_aqi,
                "aisi": aisi_state.get("aisi", 0.0),
                "dominants": dominants,
                "corridor": (plume.get("corridor") or {}).get(
                    "name", "North-Westerly"),
                "active_fires": len(fires),
                "trend": trend_label,
                "aq_series": forecast.get("aqi"),
                "grap": grap,
                "horizon_hours": len(forecast["timestamps"]),
            },
            "station_id": worst_station_id,
            "station_name": (station or {}).get("name", "Delhi NCR"),
        }

    # ── Auto-retrain: data accumulates → weights teach themselves ──
    # Checked cheaply on every refresh cycle; trains at most once/day and
    # only when meaningful new history arrived. Same code + guardrails as
    # scripts/train_models.py. No restart needed: weights hot-reload.

    AUTO_TRAIN_MIN_INTERVAL_H = 24
    AUTO_TRAIN_MIN_NEW_READINGS = 1500

    async def maybe_auto_train(self) -> Dict[str, Any]:
        """Trigger a background retrain if due. Cheap gate, never blocks."""
        if getattr(self, "_auto_train_running", False):
            return {"auto_train": "already-running"}
        try:
            state = self._auto_train_state()
            import sqlite3
            from datetime import datetime, timezone
            db_path = str(PROJECT_ROOT / self.settings.database_path)
            try:
                con = sqlite3.connect(db_path)
                current = con.execute(
                    "SELECT COUNT(*) FROM station_readings").fetchone()[0]
                con.close()
            except Exception:
                return {"auto_train": "no-db"}
            now = datetime.now(timezone.utc)
            last = state.get("last_attempt_iso")
            age_h = 9999.0
            if last:
                try:
                    age_h = ((now - datetime.fromisoformat(last))
                             .total_seconds() / 3600.0)
                except ValueError:
                    pass
            growth = current - int(state.get("readings_at_attempt", 0))
            if (age_h >= self.AUTO_TRAIN_MIN_INTERVAL_H
                    and growth >= self.AUTO_TRAIN_MIN_NEW_READINGS):
                self._auto_train_running = True
                asyncio.create_task(self._auto_train_job(current))
                return {"auto_train": "started",
                        "new_readings": growth}
            return {"auto_train": "not-due",
                    "age_h": round(age_h, 1), "new_readings": growth}
        except Exception as e:
            logger.debug("auto-train gate failed: %s", e)
            return {"auto_train": "gate-error", "error": str(e)}

    async def _auto_train_job(self, readings_now: int):
        """Run training in background, hot-reload weights on success."""
        import json
        from datetime import datetime, timezone
        try:
            logger.info("Auto-train started (%d readings)", readings_now)
            import scripts.train_models as tm
            report = await tm.run_training(
                db=self.settings.database_path, force_sklearn=True)
            (PROJECT_ROOT / "scripts" / "training_report.json").write_text(
                json.dumps(report, indent=2, default=str))
            saved = bool((report.get("lgbm") or {}).get("saved")
                         or (report.get("xgboost") or {}).get("saved"))
            if saved:
                self.ensemble._load_weights()  # hot-reload, no restart
                logger.info("Auto-train SAVED weights — ensemble reloaded: %s",
                            report.get("verdict"))
            else:
                logger.info("Auto-train refused: %s", report.get("verdict"))
            self._write_auto_train_state({
                "last_attempt_iso": datetime.now(timezone.utc).isoformat(),
                "readings_at_attempt": readings_now,
                "last_saved": saved,
            })
        except Exception as e:
            logger.warning("Auto-train job failed: %s", e)
        finally:
            self._auto_train_running = False

    def _auto_train_state(self) -> Dict[str, Any]:
        import json
        path = MODELS_DIR / ".auto_train.json"
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def _write_auto_train_state(self, state: Dict[str, Any]):
        import json
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            (MODELS_DIR / ".auto_train.json").write_text(json.dumps(state))
        except Exception as e:
            logger.debug("auto-train state write failed: %s", e)

    async def generate_alert_in_language(
        self, language: str = "en"
    ) -> Optional[Dict]:
        """Regenerate the current domain advisory in another language."""
        ctx = self.state.get("alert_context")
        if not ctx:
            return None
        alert = await self.alert_generator.generate(
            **ctx["inputs"], language=language, record_history=False
        )
        alert["station_id"] = ctx.get("station_id")
        alert["station_name"] = ctx.get("station_name")
        return alert

    def _build_domain_summary(self):
        """Aggregate overall domain metrics for the header/ticker."""
        stations = self.state.get("stations", [])
        currents = [s["current"] for s in stations if s.get("current")]

        if not currents:
            self.state["domain_summary"] = {
                "station_count": 0,
                "mean_aqi": None,
                "worst": None,
                "category_counts": {},
                "fire_count": len(self.state.get("fires", [])),
                "data_source": self.state.get("data_source"),
                "waqi_filled": self.state.get("waqi_filled", 0),
                "freshness": self.state.get("freshness", {}),
            }
            return

        worst = max(currents, key=lambda c: c.get("aqi", 0))
        mean_aqi = sum(c.get("aqi", 0) for c in currents) / len(currents)
        category_counts = {}
        for c in currents:
            cat = c.get("category", "Unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        self.state["domain_summary"] = {
            "station_count": len(currents),
            "mean_aqi": round(mean_aqi, 1),
            "worst": {
                "station_id": worst.get("station_id"),
                "station_name": worst.get("station_name"),
                "aqi": worst.get("aqi"),
                "category": worst.get("category"),
                "color": worst.get("color"),
            },
            "category_counts": category_counts,
            "fire_count": len(self.state.get("fires", [])),
            "data_source": self.state.get("data_source"),
            "waqi_filled": self.state.get("waqi_filled", 0),
            "freshness": self.state.get("freshness", {}),
        }

    # ────────────────────────────────────────────────────────────────
    # Query helpers exposed to routes
    # ────────────────────────────────────────────────────────────────

    async def get_snapshot(self) -> Dict[str, Any]:
        """Return the full current state snapshot (for WebSocket push)."""
        return {
            "type": "snapshot",
            "last_update": self.state.get("last_update"),
            "data_source": self.state.get("data_source"),
            "freshness": self.state.get("freshness", {}),
            "domain_summary": self.state.get("domain_summary", {}),
            "stations": self.state.get("stations", []),
            "aisi": self.state.get("aisi", {}),
            "fire_stats": self.state.get("fire_stats", {}),
            "plume": self.state.get("plume", {}),
            "alerts": self.state.get("alerts", []),
            "forecasts": self.state.get("forecasts", {}),
            "spatial": self.state.get("spatial", {}),
            "radiation": self.state.get("radiation", {}),
        }

    def get_stations(self) -> List[Dict]:
        return self.state.get("stations", [])

    def get_station(self, station_id: str) -> Optional[Dict]:
        return self._station_by_id(station_id)

    def get_forecast(self, station_id: str) -> Optional[Dict]:
        return self.state.get("forecasts", {}).get(station_id)

    def get_aisi(self) -> Dict:
        aisi = self.state.get("aisi", {})
        return {
            **aisi,
            "history": self.aisi.history(limit=48),
        }

    def get_fires(self) -> Dict:
        return {
            "fires": self.state.get("fires", []),
            "stats": self.state.get("fire_stats", {}),
            "plume": self.state.get("plume", {}),
        }

    def get_alerts(self) -> List[Dict]:
        return self.state.get("alerts", [])

    def get_spatial(self) -> Dict:
        return self.state.get("spatial", {})

    def get_emissions(self) -> Dict:
        return self.state.get("emissions", {})

    def get_radiation(self) -> Dict:
        return self.state.get("radiation", {})

    # ────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────

    def _station_payload(self, record: Dict) -> Dict:
        pollutants = record.get("pollutants", {})
        result = self.naqi.calculate_naqi(pollutants)
        age = record.get("age_hours")
        if age is None:
            age = self._age_hours(record.get("timestamp"))
        return {
            "station_id": record.get("station_id"),
            "station_name": record.get("station_name"),
            "timestamp": record.get("timestamp"),
            "age_hours": round(age, 2) if age is not None else None,
            "is_stale": age is None or age > self.FRESHNESS_MAX_AGE_H,
            "pollutants": pollutants,
            "aqi": result.overall_aqi,
            "category": result.category,
            "color": result.color,
            "dominant_pollutant": result.dominant_pollutant,
            "sub_indices": result.sub_indices,
            "health_impact": result.health_impact,
            "source": record.get("source", self.state.get("data_source", "unknown")),
        }

    def _build_spatial_geojson(self, forecasts) -> Dict:
        """Pixel grid of interpolated PM2.5 for map heatmap overlay.

        Accepts ForecastResult objects (live ensemble) or plain dicts
        (real-model overlay in state) — attribute/dict access handled.
        """
        stations = self.state.get("stations", [])

        def _get(fc, key, idx=0):
            if fc is None:
                return None
            v = fc.get(key) if isinstance(fc, dict) else getattr(fc, key, None)
            if isinstance(v, list):
                return v[idx] if len(v) > idx else None
            return v

        by_id = {}
        if forecasts and not isinstance(forecasts, dict):
            by_id = {f.station_id: f for f in forecasts}
        else:
            by_id = dict(forecasts or {})

        features = []
        for station in stations:
            forecast = by_id.get(station["id"])
            current = station.get("current")
            value = None
            if forecast:
                value = _get(forecast, "pm25")
            elif current:
                value = current.get("pollutants", {}).get("pm25")
            if value is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [station["longitude"], station["latitude"]],
                },
                "properties": {
                    "id": station["id"],
                    "name": station["short_name"],
                    "pm25": value,
                    "pm10": (_get(forecast, "pm10")
                             or (current or {}).get("pollutants", {}).get("pm10", 0)),
                    "no2": (_get(forecast, "no2")
                            or (current or {}).get("pollutants", {}).get("no2", 0)),
                    "o3": (_get(forecast, "o3")
                           or (current or {}).get("pollutants", {}).get("o3", 0)),
                    "aqi": (_get(forecast, "aqi")
                            or (current or {}).get("aqi", 0)),
                    "category": (
                        _get(forecast, "category")
                        or (current or {}).get("category", "Unknown")
                    ),
                    "color": (
                        _get(forecast, "colors")
                        or (current or {}).get("color", "#808080")
                    ),
                },
            })

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    @staticmethod
    def _station_name_keys(station: Dict) -> List[str]:
        """Normalized name tokens used to anchor OpenAQ matches by name."""
        import re
        keys = [station.get("short_name", ""), station.get("name", "")]
        out = []
        for k in keys:
            k = re.sub(r"[^a-z0-9 ]", " ", str(k).lower())
            k = re.sub(r"\b(delhi|new|sector|sec|phase|gram|nagar|marg|road|rd|station|dpcc|cpcb|uppcb|hspcb|imd|iitm|sai|teri)\b", " ", k)
            for tok in k.split():
                if len(tok) >= 4:
                    out.append(tok)
        return out

    def _match_readings(self, readings: List[Any],
                        source: str = "live") -> Dict[str, Dict]:
        """Map OpenAQ readings to station metadata.

        Name-anchored first (an OpenAQ location whose name shares a token
        with our station wins regardless of distance), nearest-proximity
        fallback within 25 km. Fixes identical readings copied across
        neighbouring stations (e.g. ITO vs Mandir Marg).
        """
        if not readings:
            return {}

        mapped = {}
        for station in self.stations:
            lat, lon = station["latitude"], station["longitude"]
            name_keys = self._station_name_keys(station)

            # OpenAQ returns one reading per pollutant; pick the nearest
            # reading for each pollutant individually so a mid-city monitor
            # does not masquerade for a far suburb station.
            best_pollutants: Dict[str, float] = {}
            best_pollutant_dist: Dict[str, float] = {}
            best = None
            best_eff = 25.0  # km radius on effective (anchor-weighted) distance
            best_dist = 25.0

            for reading in readings:
                rlat = getattr(reading, "latitude", None)
                rlon = getattr(reading, "longitude", None)
                if rlat is None or rlon is None:
                    continue
                d = self._haversine(lat, lon, rlat, rlon)
                # Name anchor: shared token halves effective distance so the
                # correctly-named monitor wins over a nearer wrong one.
                rname = str(getattr(reading, "station_name", "") or "").lower()
                anchored = any(k in rname for k in name_keys)
                eff_d = d * 0.5 if anchored else d
                if eff_d <= 25.0:  # km radius on effective distance
                    if best is None or eff_d < best_eff:
                        best_eff = eff_d
                        best_dist = d
                        best = reading
                    for k, v in (reading.pollutants or {}).items():
                        if v is None:
                            continue
                        if k not in best_pollutant_dist or d < best_pollutant_dist[k]:
                            best_pollutant_dist[k] = d
                            best_pollutants[k] = v

            if best is not None:
                mapped[station["id"]] = {
                    "pollutants": best_pollutants,
                    "timestamp": best.timestamp,
                    "source": source,
                    "matched_distance_km": round(best_dist, 1),
                }
        return mapped

    def _match_cpcb_readings(self, readings: List[Any]) -> Dict[str, Dict]:
        """Map CPCB data.gov.in readings to station metadata.

        Pass 1 — name match: each CPCB station name is scored against
        our stations by shared name-key tokens (same tokenizer as the
        OpenAQ anchor) plus a short-name substring bonus for terse
        names like JNU/ITO. Best score wins; ties keep station order.
        Pass 2 — proximity fallback for name-unmatched readings that
        carry sane coordinates (generic proximity+anchor matcher).
        """
        import re
        if not readings:
            return {}

        def _tokens(text: str) -> List[str]:
            t = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower())
            t = re.sub(r"\b(delhi|new|sector|sec|phase|gram|nagar|marg|road|rd|station|dpcc|cpcb|uppcb|hspcb|imd|iitm|sai|teri)\b", " ", t)
            return [tok for tok in t.split() if len(tok) >= 4]

        station_keys = [(s, set(self._station_name_keys(s)),
                         str(s.get("short_name", "")).lower())
                        for s in self.stations]
        mapped: Dict[str, Dict] = {}
        leftovers = []
        for reading in readings:
            rname = str(getattr(reading, "station_name", "") or "")
            rtokens = set(_tokens(rname))
            rnorm = re.sub(r"[^a-z0-9]", "", rname.lower())
            best_sid = None
            best_score = 0
            for station, keys, short in station_keys:
                if station["id"] in mapped:
                    continue
                score = len(rtokens & keys)
                if short and len(short) >= 2 and short.replace(" ", "") in rnorm:
                    score += 2
                if score > best_score:
                    best_score = score
                    best_sid = station["id"]
            if best_sid is not None and best_score > 0:
                mapped[best_sid] = {
                    "pollutants": dict(reading.pollutants or {}),
                    "timestamp": reading.timestamp,
                    "source": "cpcb",
                    "matched_station_name": rname,
                }
            else:
                leftovers.append(reading)

        # Pass 2: proximity fallback for the name-unmatched remainder.
        if leftovers:
            geo = [r for r in leftovers
                   if getattr(r, "latitude", 0.0)
                   and getattr(r, "longitude", 0.0)]
            if geo:
                for sid, entry in self._match_readings(
                        geo, source="cpcb").items():
                    if sid not in mapped:
                        mapped[sid] = entry
        return mapped

    @staticmethod
    def _age_hours(timestamp: Any) -> Optional[float]:
        """Age of an upstream timestamp in hours; None if unparseable."""
        if not timestamp:
            return None
        try:
            dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds()
                       / 3600.0)
        except (ValueError, TypeError):
            return None

    def _apply_freshness_gate(self, mapped: Dict[str, Dict]) -> Dict[str, Any]:
        """Drop entries older than FRESHNESS_MAX_AGE_H (in place).

        Returns counters: fresh/stale totals plus per-source
        {source}_fresh / {source}_stale tallies. Surviving entries
        gain an `age_hours` field carried through to records.
        """
        by_source: Dict[str, int] = {}
        fresh = 0
        for sid in list(mapped.keys()):
            entry = mapped[sid]
            src = str(entry.get("source") or "unknown")
            age = self._age_hours(entry.get("timestamp"))
            if age is None or age > self.FRESHNESS_MAX_AGE_H:
                by_source[f"{src}_stale"] = by_source.get(f"{src}_stale", 0) + 1
                del mapped[sid]
            else:
                entry["age_hours"] = round(age, 2)
                by_source[f"{src}_fresh"] = by_source.get(f"{src}_fresh", 0) + 1
                fresh += 1
        stale = sum(v for k, v in by_source.items() if k.endswith("_stale"))
        return {
            "fresh": fresh,
            "stale": stale,
            "max_age_h": self.FRESHNESS_MAX_AGE_H,
            "min_for_live": self.MIN_FRESH_FOR_LIVE,
            "by_source": by_source,
        }

    def _demo_readings(self) -> Dict[str, Dict]:
        """
        Generate realistic demo readings for the Delhi NCR domain.

        Mean-reverting random walk per station: values evolve smoothly
        (±3% per cycle) around a diurnal target, so consecutive snapshots
        look like a real atmosphere instead of white noise. (The old
        version multiplied the base by up to ~9x, spraying 15–400 µg/m³
        garbage that poisoned history, features and AISI.)
        """
        import random
        try:
            from zoneinfo import ZoneInfo
            hour = datetime.now(timezone.utc).astimezone(
                ZoneInfo("Asia/Kolkata")).hour
        except Exception:
            hour = datetime.now(timezone.utc).hour
        import math
        # Delhi diurnal: calm-night peak, afternoon minimum
        diurnal = 1.0 + 0.35 * math.cos((hour - 1) * math.pi / 12.0)
        month = datetime.now().month
        winter = month in (11, 12, 1, 2)
        base = 140 if winter else 55

        if not hasattr(self, "_demo_state"):
            self._demo_state = {}
        mapped = {}
        for station in self.stations:
            zone_factor = self._zone_factor(station.get("zone", ""), station["latitude"])
            target = base * zone_factor * diurnal
            prev = self._demo_state.get(station["id"], target)
            # Mean reversion (30% toward target) + small innovation
            pm25 = prev + 0.30 * (target - prev) + prev * random.uniform(-0.03, 0.03)
            pm25 = max(8.0, min(500.0, pm25))
            self._demo_state[station["id"]] = pm25
            ratio = self._demo_state.get(station["id"] + ":ratio",
                                         random.uniform(1.6, 2.2))
            self._demo_state[station["id"] + ":ratio"] = ratio
            pm10 = pm25 * ratio
            no2 = max(8.0, pm25 * 0.32)
            so2 = max(4.0, pm25 * 0.09)
            o3 = 25.0 + 30.0 * (1.0 - (diurnal - 0.65) / 0.7)
            co = max(0.3, pm25 * 0.028)

            pollutants = {"pm25": round(pm25, 1), "pm10": round(pm10, 1),
                          "no2": round(no2, 1), "so2": round(so2, 1),
                          "o3": round(max(o3, 5.0), 1), "co": round(co, 2)}
            mapped[station["id"]] = {
                "pollutants": pollutants,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "demo",
            }
        return mapped

    def _zone_factor(self, zone: str, lat: float) -> float:
        """Spatial weighting to reproduce the polluted NW corridor."""
        import re
        north = 1.0 if lat > 28.65 else 0.85
        corridors = ["North", "North West", "East"]
        if any(word in zone for word in corridors):
            return 1.12 * north
        return 0.92 * north

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        import math
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _station_pm10_ratio(station: Dict, pm25: Optional[float],
                            pm10: Optional[float]) -> float:
        """Observed PM10/PM2.5 ratio for a station, clamped to [1.0, 2.5]."""
        if pm25 and pm10 and pm25 > 5:
            return max(1.0, min(2.5, pm10 / pm25))
        ratios = []
        for r in station.get("history", []) or []:
            p25, p10 = r.get("pm25"), r.get("pm10")
            if p25 and p10 and p25 > 5:
                ratios.append(max(1.0, min(2.5, p10 / p25)))
        if ratios:
            ratios.sort()
            return ratios[len(ratios) // 2]
        return 1.35

    def _station_by_id(self, station_id: str) -> Optional[Dict]:
        for s in self.state.get("stations", []):
            if s["id"] == station_id:
                return s
        return None

    def _load_stations(self) -> List[Dict]:
        """Load Delhi NCR station metadata from data/stations.json."""
        path = DATA_DIR / "stations.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("stations", [])
        except Exception as e:
            logger.error("Failed to load stations.json: %s", e)
            return []

    async def _safe(self, coroutine, default=None):
        """Await a coroutine, returning default on any failure."""
        try:
            return await coroutine
        except Exception as e:
            logger.warning("Step failed (%s): %s",
                           coroutine.__qualname__ if hasattr(coroutine, "__qualname__") else "?",
                           e)
            return default

    async def close(self):
        """Close all HTTP clients."""
        await self._safe(self.cpcb_datagov.close())
        await self._safe(self.openaq.close())
        await self._safe(self.waqi.close())
        await self._safe(self.weather.close())
        await self._safe(self.fire.close())