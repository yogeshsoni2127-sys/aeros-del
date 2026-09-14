"""
XGBoost Gradient Boosting Forecaster — PM2.5/PM10 72-Hour Prediction

Gradient-boosted trees post-processor for pollutant forecasting.

- Trains on available station history (features → next-hour target)
- Learns systematic biases conditioned on time-of-day, season, and
  meteorological/physics regime
- Produces point forecast + uncertainty bands (quantile style)
- Falls back to a physics-aware statistical baseline when no model
  weights or training history are available (ensures demo continuity)
"""

import asyncio
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _ensure_numpy():
    if np is None:
        raise ImportError("numpy is required for XGBoostForecaster")
    return np


def statistical_baseline(
    current_pm25: float,
    history_pm25: List[Optional[float]],
    current_pm10: Optional[float] = None,
    aisi: float = 2.0,
    pbl_height_m: float = 700.0,
    fire_contribution: float = 0.0,
    horizon: int = 72,
    pm10_ratio: float = 1.35,
    current_no2: Optional[float] = None,
    current_o3: Optional[float] = None,
    current_so2: Optional[float] = None,
    current_co: Optional[float] = None,
) -> Dict[str, List[float]]:
    """
    Physics-aware statistical baseline forecast (no ML weights needed).

    Combines:
    - Persistence of current concentration
    - Per-pollutant diurnal modulation curves (Delhi PM2.5: min ~16h,
      peak ~09h/23h; NO2 twin traffic peaks; O3 afternoon photochemical
      peak; SO2/CO flatter)
    - Regime decay modulated by AISI (trapping) and PBL height
    - A growing uncertainty envelope with forecast horizon

    Returns:
        Dict with pm25, pm10, lower, upper 72-element lists plus
        independent no2/o3/so2/co series with their own bands.
    """
    np_ = _ensure_numpy()
    history = [h for h in history_pm25 if h is not None]
    recent = history[-12:] if len(history) >= 12 else history
    recent_mean = float(np_.mean(recent)) if recent else current_pm25

    # If history is lower than current (spike), blend to a smoothing factor
    current = max(current_pm25, 5.0)
    target = max(recent_mean, 5.0) if recent else current

    # AISI trapping factor: higher AISI → less ventilation → slower decay
    aisi_gate = max(0.3, min(1.0, aisi / 8.0))
    pbl_gate = max(0.3, min(1.0, pbl_height_m / 1000.0))
    decay = 1.0 - 0.10 * (1.0 - aisi_gate) * (0.5 + 0.5 * pbl_gate)

    # Diurnal modulation (1 + sine-shaped, normalized around 1.0)
    pm25 = []
    pm10 = []
    lower = []
    upper = []

    for h in range(horizon):
        hour_of_day = (datetime.now().hour + h) % 24
        diurnal = _delhi_diurnal_factor(hour_of_day)

        base = current * (decay ** (h / 24.0)) * diurnal
        nominal = base + fire_contribution * (1.0 - h / (2.0 * horizon))
        nominal = max(nominal, 5.0)

        # Uncertainty envelope grows non-linearly with horizon
        sigma = 0.10 * current + 0.05 * current * math.sqrt(h + 1)
        pm25.append(round(nominal, 1))
        pm10.append(round(nominal * pm10_ratio, 1))
        lower.append(round(max(nominal - 1.28 * sigma, 2.0), 1))
        upper.append(round(nominal + 1.28 * sigma, 1))

    pm10_lower = [round(max(v * pm10_ratio * 0.72, 2.0), 1) for v in lower]
    pm10_upper = [round(v * pm10_ratio * 1.35, 1) for v in upper]

    # Independent gas projections: each gas persists around its OWN
    # current observation with its OWN diurnal shape (NOT the PM curve),
    # so the NO2/O3 chart tabs stop mirroring PM2.5.
    gas_out = project_gases(
        current, decay, horizon, current_no2, current_o3,
        current_so2, current_co)

    return {"pm25": pm25, "pm10": pm10, "lower": lower, "upper": upper,
            "pm10_lower": pm10_lower, "pm10_upper": pm10_upper, **gas_out}


def _delhi_diurnal_factor(hour: int) -> float:
    """Empirical Delhi PM2.5 diurnal pattern (min ~15-16h, peaks ~9h/23h)."""
    curve = [
        1.18, 1.22, 1.18, 1.12, 1.06, 1.02,  # 00-05
        0.98, 1.02, 1.12, 1.24, 1.28, 1.22,  # 06-11
        1.10, 0.98, 0.88, 0.82, 0.80, 0.84,  # 12-17
        0.92, 1.04, 1.14, 1.20, 1.22, 1.20,  # 18-23
    ]
    return curve[hour % 24]


# Per-pollutant diurnal shapes (raw, normalized to mean 1.0 at load).
# PM follows the observed Delhi haze cycle; NO2 follows traffic (twin
# rush-hour peaks); O3 is photochemical (afternoon peak, dawn minimum);
# SO2/CO are flatter industrial/domestic signals. Previously every gas
# reused the PM curve, so all 72h chart tabs looked identical.
_GAS_DIURNAL_RAW = {
    "no2": [
        1.05, 1.02, 0.98, 0.94, 0.90, 0.88,  # 00-05
        0.92, 1.05, 1.20, 1.28, 1.18, 1.05,  # 06-11 (morning rush)
        0.95, 0.88, 0.82, 0.80, 0.82, 0.90,  # 12-17 (midday dip)
        1.05, 1.18, 1.25, 1.22, 1.15, 1.10,  # 18-23 (evening rush)
    ],
    "o3": [
        0.55, 0.50, 0.48, 0.47, 0.48, 0.52,  # 00-05 (night titration)
        0.60, 0.75, 0.95, 1.15, 1.32, 1.45,  # 06-11 (morning build)
        1.55, 1.62, 1.65, 1.60, 1.48, 1.30,  # 12-17 (afternoon peak)
        1.10, 0.92, 0.78, 0.68, 0.61, 0.57,  # 18-23 (evening decay)
    ],
    "so2": [
        1.02, 1.00, 0.98, 0.97, 0.97, 0.98,  # 00-05
        1.00, 1.04, 1.08, 1.10, 1.08, 1.05,  # 06-11
        1.02, 0.99, 0.96, 0.94, 0.94, 0.96,  # 12-17
        1.00, 1.04, 1.06, 1.05, 1.04, 1.03,  # 18-23
    ],
    "co": [
        1.08, 1.05, 1.02, 0.99, 0.96, 0.94,  # 00-05
        0.96, 1.04, 1.14, 1.18, 1.12, 1.04,  # 06-11
        0.97, 0.91, 0.87, 0.86, 0.88, 0.94,  # 12-17
        1.03, 1.11, 1.15, 1.14, 1.12, 1.10,  # 18-23
    ],
}
_GAS_DIURNAL = {}
for _gas, _shape in _GAS_DIURNAL_RAW.items():
    _mean = sum(_shape) / len(_shape)
    _GAS_DIURNAL[_gas] = [round(v / _mean, 4) for v in _shape]


def gas_diurnal_factor(target: str, hour: int) -> float:
    """Diurnal multiplier for one pollutant at one hour-of-day.

    pm25/pm10 keep the legacy Delhi haze curve (validated behaviour,
    untouched); gases use their own normalized shapes so the 72h tabs
    no longer mirror PM.
    """
    t = (target or "").lower()
    if t in ("pm25", "pm10"):
        return _delhi_diurnal_factor(hour)
    shape = _GAS_DIURNAL.get(t)
    if shape:
        return shape[int(hour) % 24]
    return _delhi_diurnal_factor(hour)


def project_gases(
    current_pm25: float,
    decay: float,
    horizon: int,
    current_no2: Optional[float] = None,
    current_o3: Optional[float] = None,
    current_so2: Optional[float] = None,
    current_co: Optional[float] = None,
) -> Dict[str, List[float]]:
    """Independent per-gas hourly projections with own diurnal shapes.

    Shared by the statistical baseline and the trained-model predict()
    paths so every mode serves distinct NO2/O3/SO2/CO curves (never a
    PM copy). See statistical_baseline for anchor/fallback semantics.
    """
    gas_anchors = {
        "no2": (current_no2, 5.0, 800.0),
        "o3": (current_o3, 5.0, 600.0),
        "so2": (current_so2, 4.0, 1000.0),
        "co": (current_co, 0.3, 50.0),
    }
    gas_out: Dict[str, List[float]] = {}
    for gas, (obs, lo, hi) in gas_anchors.items():
        if obs is not None and obs > 0:
            anchor = float(obs)
            estimated = False
        else:
            # Delhi-typical fallback ratios off PM2.5 (documented estimate).
            fallback_ratio = {"no2": 0.32, "o3": None, "so2": 0.09,
                              "co": 0.028}[gas]
            anchor = 45.0 if gas == "o3" else max(current_pm25 * fallback_ratio,
                                                  lo)
            estimated = True
        series, glo, ghi = [], [], []
        for h in range(horizon):
            hour_of_day = (datetime.now().hour + h) % 24
            shape = gas_diurnal_factor(gas, hour_of_day)
            nominal = max(min(anchor * (decay ** (h / 24.0)) * shape,
                              hi), lo)
            width = 1.5 if estimated else 1.0
            sigma = width * (0.10 * anchor + 0.05 * anchor * math.sqrt(h + 1))
            series.append(round(nominal, 2 if gas == "co" else 1))
            glo.append(round(max(nominal - 1.28 * sigma, lo * 0.5),
                             2 if gas == "co" else 1))
            ghi.append(round(min(nominal + 1.28 * sigma, hi),
                             2 if gas == "co" else 1))
        gas_out[gas] = series
        gas_out[f"{gas}_lower"] = glo
        gas_out[f"{gas}_upper"] = ghi
    return gas_out


class XGBoostForecaster:
    """
    Gradient-boosted ensemble forecaster with statistical fallback.

    Model weights are persisted to backend/models when trained so that
    inference runs without re-training on every boot.
    """

    MODEL_PATH_KEY = "xgboost_pm25"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._feature_names = None

    @property
    def is_trained(self) -> bool:
        """True only when real fitted weights are loaded (not baseline)."""
        return self._model is not None and bool(self._feature_names)

    async def train(
        self,
        features: List[Dict[str, Any]],
        targets: List[float],
    ) -> Dict:
        """
        Train the booster on feature vectors → next-hour PM2.5 targets.

        Args:
            features: List of feature dicts (from FeatureEngineer).
            targets: List of corresponding next-hour pm25 values.

        Returns:
            Training summary dict.
        """
        np_ = _ensure_numpy()
        try:
            import xgboost as xgb
        except ImportError as e:
            logger.warning(f"xgboost unavailable ({e}); using baseline fallback")
            return {"trained": False, "reason": "xgboost not installed",
                    "samples": 0}

        if not features or len(features) != len(targets):
            return {"trained": False, "reason": "insufficient data", "samples": 0}

        df = self._frame(features)
        X = df.drop(columns=["station_id"])
        y = np_.array([float(t) for t in targets])

        self._feature_names = list(X.columns)

        try:
            model = xgb.XGBRegressor(
                n_estimators=180,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
            )
            model.fit(X, y)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return {"trained": False, "reason": str(e), "samples": len(X)}

        self._model = model
        self._persist_if_possible()
        logger.info("XGBoost forecaster trained on %d samples", len(X))

        return {"trained": True, "samples": len(X), "features": len(X.columns)}

    async def predict(
        self,
        context: Dict[str, Any],
        features: Optional[Dict[str, Any]] = None,
        horizon: int = 72,
    ) -> Dict[str, List[float]]:
        """
        Predict the next `horizon` hours of PM2.5/PM10.

        Args:
            context: Dict with current_pm25, current_pm10, history_pm25,
                aisi, pbl_height_m, fire_contribution.
            features: Optional FeatureEngineer vector (used when model
                is trained and can iterate).
            horizon: Number of forecast hours.

        Returns:
            Dict with pm25, pm10, lower, upper arrays.
        """
        current = float(context.get("current_pm25", 120.0) or 120.0)
        pm10 = float(context.get("current_pm10", 0.0) or 0.0)
        ratio = float(context.get("pm10_ratio", 0.0) or 0.0)
        if ratio <= 0:
            ratio = 1.35
        if pm10 <= 0:
            pm10 = current * ratio

        history = context.get("history_pm25") or []
        aisi = float(context.get("aisi", 2.0) or 2.0)
        pblh = float(context.get("pbl_height_m", 700.0) or 700.0)
        fire = float(context.get("fire_contribution", 0.0) or 0.0)

        if self._model is None or features is None:
            return statistical_baseline(
                current_pm25=current,
                history_pm25=history,
                current_pm10=pm10,
                aisi=aisi,
                pbl_height_m=pblh,
                fire_contribution=fire,
                horizon=horizon,
                pm10_ratio=ratio,
                current_no2=context.get("current_no2"),
                current_o3=context.get("current_o3"),
                current_so2=context.get("current_so2"),
                current_co=context.get("current_co"),
            )

        # Iterative multi-step forecast using the trained model
        preds = self._model_predict_iterative(features, history, horizon)
        base = preds["pm25"]
        pm25 = base
        pm10 = [round(max(v * ratio, 5.0), 1) for v in base]
        lower = [round(max(v * 0.75, 2.0), 1) for v in base]
        upper = [round(v * 1.35, 1) for v in base]
        pm10_lower = [round(max(v * 0.72, 2.0), 1) for v in lower]
        pm10_upper = [round(v * 1.35, 1) for v in upper]
        aisi_gate = max(0.3, min(1.0, aisi / 8.0))
        pbl_gate = max(0.3, min(1.0, pblh / 1000.0))
        decay = 1.0 - 0.10 * (1.0 - aisi_gate) * (0.5 + 0.5 * pbl_gate)
        gases = project_gases(
            current, decay, horizon, context.get("current_no2"),
            context.get("current_o3"), context.get("current_so2"),
            context.get("current_co"))

        return {"pm25": pm25, "pm10": pm10, "lower": lower, "upper": upper,
                "pm10_lower": pm10_lower, "pm10_upper": pm10_upper, **gases}

    # ── Internals ────────────────────────────────────────────────────

    def _model_predict_iterative(
        self,
        features: Dict,
        history: List[float],
        horizon: int,
    ) -> Dict[str, List[float]]:
        """Roll the trained model forward hour-by-hour."""
        np_ = _ensure_numpy()
        row = {k: v for k, v in features.items() if k != "station_id"}
        out = []

        for h in range(horizon):
            X = np_.array([[float(row.get(c, 0.0)) for c in self._feature_names]])
            pred = float(self._model.predict(X)[0])
            pred = max(pred, 5.0)
            out.append(round(pred, 1))

            # Update feature window: shift lag features, set pm25_now
            row["pm25_now"] = pred
            for window in (24, 48, 72):
                mean_key = f"pm25_mean_{window}h"
                max_key = f"pm25_max_{window}h"
                if mean_key in row:
                    row[mean_key] = (row[mean_key] * window + pred) / max(window, 1)
                if max_key in row:
                    row[max_key] = max(row[max_key], pred)

        return {"pm25": out}

    def _frame(self, features: List[Dict]):
        """Build a simple table from feature dicts (pandas if available)."""
        try:
            import pandas as pd
            return pd.DataFrame(features)
        except ImportError:
            # Minimal dict-of-lists fallback
            keys = list(features[0].keys())
            table: Dict[str, list] = {k: [] for k in keys}
            for row in features:
                for k in keys:
                    table[k].append(row.get(k))
            table["_rows"] = table.pop("station_id")
            return type("Frame", (), {
                "columns": [k for k in table.keys() if k != "_rows"],
                "drop": lambda self, c: self,
                "__getitem__": lambda self, c: table.get(c),
                "get_data": lambda: table,
            })()

    def _persist_if_possible(self):
        """Save the trained model to disk for later inference."""
        if not self.model_path:
            return
        try:
            import joblib
            joblib.dump({"model": self._model, "features": self._feature_names},
                        self.model_path)
        except Exception as e:
            logger.warning(f"Could not persist XGBoost model: {e}")

    def load(self, path: Optional[str] = None) -> bool:
        """Load persisted model weights."""
        path = path or self.model_path
        if not path:
            return False
        try:
            import joblib
            data = joblib.load(path)
            self._model = data["model"]
            self._feature_names = data["features"]
            return True
        except Exception:
            return False