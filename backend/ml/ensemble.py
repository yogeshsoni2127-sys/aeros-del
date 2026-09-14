"""
Ensemble Forecaster — TFT + XGBoost Combination

Stacks the Temporal Fusion Transformer and XGBoost predictions using
configurable weights, merges uncertainty bands, and produces the final
AQI-rated 72-hour forecast consumed by the API and dashboard.

Includes fall-back continuity: if one model is unavailable, the other
(plus the statistical baseline) maintains prediction continuity.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from backend.ml.xgboost_forecaster import XGBoostForecaster
from backend.ml.transformer_forecaster import TFTForecaster
from backend.ml.lightgbm_forecaster import LightGBMForecaster
from backend.data.naqi_calculator import NAQICalculator
from backend.ml.xgboost_forecaster import statistical_baseline


@dataclass
class ForecastResult:
    """Final ensemble 72-hour forecast for a station."""

    station_id: str
    timestamps: List[str]
    pm25: List[float]
    pm10: List[float]
    aqi: List[float] = field(default_factory=list)
    category: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    lower: List[float] = field(default_factory=list)
    upper: List[float] = field(default_factory=list)
    pm10_lower: List[float] = field(default_factory=list)
    pm10_upper: List[float] = field(default_factory=list)
    no2: List[float] = field(default_factory=list)
    o3: List[float] = field(default_factory=list)
    so2: List[float] = field(default_factory=list)
    co: List[float] = field(default_factory=list)
    no2_lower: List[float] = field(default_factory=list)
    no2_upper: List[float] = field(default_factory=list)
    o3_lower: List[float] = field(default_factory=list)
    o3_upper: List[float] = field(default_factory=list)
    dominant_pollutant: str = "pm25"
    models: Dict[str, bool] = field(default_factory=dict)
    generated_at: str = ""

    def to_dict(self, include_arrays: bool = True) -> Dict:
        """Serialize to a JSON-friendly dict for API responses."""
        return {
            "station_id": self.station_id,
            "generated_at": self.generated_at,
            "timestamps": self.timestamps,
            "pm25": self.pm25,
            "pm10": self.pm10,
            "aqi": self.aqi,
            "category": self.category,
            "colors": self.colors,
            "lower": self.lower,
            "upper": self.upper,
            "pm10_lower": self.pm10_lower,
            "pm10_upper": self.pm10_upper,
            "no2": self.no2,
            "o3": self.o3,
            "so2": self.so2,
            "co": self.co,
            "no2_lower": self.no2_lower,
            "no2_upper": self.no2_upper,
            "o3_lower": self.o3_lower,
            "o3_upper": self.o3_upper,
            "dominant_pollutant": self.dominant_pollutant,
            "models": self.models,
            "horizon_hours": len(self.timestamps),
        }


class EnsembleForecaster:
    """
    Multi-model ensemble combiner for the Delhi NCR domain grid.
    """

    def __init__(
        self,
        tft_weight: float = 0.5,
        xgb_weight: float = 0.3,
        lgbm_weight: float = 0.2,
        model_dir: Optional[str] = None,
    ):
        self.tft_weight = tft_weight
        self.xgb_weight = xgb_weight
        self.lgbm_weight = lgbm_weight
        self.naqi = NAQICalculator()

        self.tft = TFTForecaster(
            model_path=f"{model_dir}/tft_pm25.pt" if model_dir else None
        )
        self.xgb = XGBoostForecaster(
            model_path=f"{model_dir}/xgboost_pm25.joblib" if model_dir else None
        )
        self.lgbm = LightGBMForecaster(
            model_path=f"{model_dir}/lgbm_pm25.joblib" if model_dir else None
        )
        self.model_dir = model_dir
        self._load_weights()

    # ── Public API ───────────────────────────────────────────────────

    async def forecast(
        self,
        context: Dict[str, Any],
        horizon: int = 72,
    ) -> ForecastResult:
        """
        Generate a merged 72-hour forecast from all member models.

        Args:
            context: Station forecast context (see _build_context).
            horizon: Forecast length in hours.

        Returns:
            ForecastResult with point estimates and uncertainty bands.
        """
        station_id = context.get("station_id", "unknown")
        timestamps = self._make_timestamps(horizon)

        tft_pred, xgb_pred, lgbm_pred = await asyncio.gather(
            self.tft.predict(context, horizon=horizon),
            self.xgb.predict(context, features=context.get("features"),
                             horizon=horizon),
            self.lgbm.predict(context, features=context.get("features"),
                              horizon=horizon),
        )

        members = [
            (self.tft_weight, tft_pred),
            (self.xgb_weight, xgb_pred),
            (self.lgbm_weight, lgbm_pred),
        ]
        wsum = sum(w for w, _ in members) or 1.0
        # Honest labels: only claim a member when its fitted weights are
        # actually loaded. Otherwise the dashboard shows "baseline".
        models = {
            "tft": bool(getattr(self.tft, "is_trained", False)),
            "xgboost": bool(getattr(self.xgb, "is_trained", False)),
            "lightgbm": bool(getattr(self.lgbm, "is_trained", False)),
        }
        if not any(models.values()):
            models = {"baseline": True}

        def _blend(key: str) -> List[float]:
            out = []
            for i in range(horizon):
                v = sum(w * p[key][i] for w, p in members) / wsum
                out.append(round(v, 1))
            return out

        def _band(key: str, fn) -> List[float]:
            out = []
            for i in range(horizon):
                vals = [p.get(key, p["lower" if "lower" in key else "upper"])[i]
                        for _, p in members]
                out.append(round(fn(vals), 1))
            return out

        # Weighted blend of point values; union envelope for uncertainty
        pm25 = _blend("pm25")
        pm10 = _blend("pm10")
        lower = _band("lower", min)
        upper = _band("upper", max)
        pm10_lower = _band("pm10_lower", min)
        pm10_upper = _band("pm10_upper", max)
        # Gases: every member now serves independent gas curves (own
        # diurnal shapes), so blend them like PM instead of copying PM.
        no2 = _blend("no2")
        o3 = _blend("o3")
        so2 = _blend("so2")
        co = _blend("co")
        no2_lower = _band("no2_lower", min)
        no2_upper = _band("no2_upper", max)
        o3_lower = _band("o3_lower", min)
        o3_upper = _band("o3_upper", max)

        return self._finalize(
            station_id=station_id,
            timestamps=timestamps,
            pm25=pm25,
            pm10=pm10,
            lower=lower,
            upper=upper,
            pm10_lower=pm10_lower,
            pm10_upper=pm10_upper,
            no2=no2,
            o3=o3,
            so2=so2,
            co=co,
            no2_lower=no2_lower,
            no2_upper=no2_upper,
            o3_lower=o3_lower,
            o3_upper=o3_upper,
            models=models,
        )

    async def forecast_domain(
        self,
        contexts: List[Dict[str, Any]],
        horizon: int = 72,
    ) -> List[ForecastResult]:
        """Forecast for a list of station contexts in parallel batches."""
        results = []
        for context in contexts:
            results.append(await self.forecast(context, horizon=horizon))
        return results

    async def train_all(self, station_outputs: Dict[str, Dict]) -> Dict:
        """
        Train both member models from prepared station datasets.

        Args:
            station_outputs: Mapping station_id → {features, targets,
                sequences}.

        Returns:
            Training report for both models.
        """
        all_features = []
        all_targets = []
        sequences = []

        for payload in station_outputs.values():
            all_features.extend(payload.get("features", []))
            all_targets.extend(payload.get("targets", []))
            sequences.extend(payload.get("sequences", []))

        xgb_report = await self.xgb.train(all_features, all_targets)
        tft_report = await self.tft.train(sequences)
        lgbm_report = await self.lgbm.train(all_features, all_targets)

        return {"xgboost": xgb_report, "tft": tft_report,
                "lightgbm": lgbm_report}

    # ── Helpers ──────────────────────────────────────────────────────

    def _finalize(
        self,
        station_id: str,
        timestamps: List[str],
        pm25: List[float],
        pm10: List[float],
        lower: List[float],
        upper: List[float],
        pm10_lower: List[float],
        pm10_upper: Optional[List[float]] = None,
        no2: Optional[List[float]] = None,
        o3: Optional[List[float]] = None,
        so2: Optional[List[float]] = None,
        co: Optional[List[float]] = None,
        no2_lower: Optional[List[float]] = None,
        no2_upper: Optional[List[float]] = None,
        o3_lower: Optional[List[float]] = None,
        o3_upper: Optional[List[float]] = None,
        models: Optional[Dict[str, bool]] = None,
    ) -> ForecastResult:
        aqi, category, colors = self._aqi_series(pm25, pm10, no2, o3)

        return ForecastResult(
            station_id=station_id,
            timestamps=timestamps,
            pm25=pm25,
            pm10=pm10,
            aqi=aqi,
            category=category,
            colors=colors,
            lower=lower,
            upper=upper,
            pm10_lower=pm10_lower,
            pm10_upper=pm10_upper or [],
            no2=no2 or [],
            o3=o3 or [],
            so2=so2 or [],
            co=co or [],
            no2_lower=no2_lower or [],
            no2_upper=no2_upper or [],
            o3_lower=o3_lower or [],
            o3_upper=o3_upper or [],
            models=models or {},
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _aqi_series(
        self,
        pm25: List[float],
        pm10: List[float],
        no2: Optional[List[float]] = None,
        o3: Optional[List[float]] = None,
    ) -> tuple:
        """Map hourly pollutant arrays to 4-pollutant NAQI series.

        Matches the research-daily path (pm25/pm10/no2/o3); the old
        2-pollutant version could understate AQI on NO2/O3-driven hours.
        """
        aqi, category, colors = [], [], []
        n = len(pm25)
        for i in range(n):
            payload = {
                "pm25": pm25[i],
                "pm10": pm10[i] if i < len(pm10) else None,
            }
            if no2 and i < len(no2) and no2[i]:
                payload["no2"] = no2[i]
            if o3 and i < len(o3) and o3[i]:
                payload["o3"] = o3[i]
            result = self.naqi.calculate_naqi(payload)
            aqi.append(result.overall_aqi)
            category.append(result.category)
            colors.append(result.color)
        return aqi, category, colors

    def _make_timestamps(self, horizon: int) -> List[str]:
        """Hourly UTC timestamps starting from the next full hour."""
        start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) \
            + timedelta(hours=1)
        return [
            (start + timedelta(hours=h)).isoformat() for h in range(horizon)
        ]

    def _load_weights(self):
        """Attempt to load persisted model weights (silent on failure)."""
        tft_loaded = self.tft.load()
        xgb_loaded = self.xgb.load()
        lgbm_loaded = self.lgbm.load()
        if tft_loaded or xgb_loaded or lgbm_loaded:
            logger.info(
                "Loaded model weights — TFT: %s, XGBoost: %s, LightGBM: %s",
                tft_loaded, xgb_loaded, lgbm_loaded,
            )
        else:
            logger.info(
                "No trained weights found — using statistical baseline mode"
            )

    @staticmethod
    def build_context(
        station_id: str,
        current_pm25: float,
        current_pm10: Optional[float] = None,
        history_pm25: Optional[List[float]] = None,
        aisi: float = 2.0,
        pbl_height_m: float = 700.0,
        inversion_strength: float = 0.0,
        fire_contribution: float = 0.0,
        features: Optional[Dict] = None,
        pm10_ratio: float = 1.35,
        current_no2: Optional[float] = None,
        current_o3: Optional[float] = None,
        current_so2: Optional[float] = None,
        current_co: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Construct a canonical forecast context dict for a station."""
        return {
            "station_id": station_id,
            "current_pm25": current_pm25,
            "current_pm10": current_pm10,
            "history_pm25": history_pm25 or [],
            "aisi": aisi,
            "pbl_height_m": pbl_height_m,
            "inversion_strength": inversion_strength,
            "fire_contribution": fire_contribution,
            "features": features,
            "pm10_ratio": pm10_ratio,
            "current_no2": current_no2,
            "current_o3": current_o3,
            "current_so2": current_so2,
            "current_co": current_co,
        }