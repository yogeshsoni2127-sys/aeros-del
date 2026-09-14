"""
LightGBM Gradient Boosting Forecaster — PM2.5/PM10 72-Hour Prediction.

Third ensemble member alongside XGBoost and the TFT forecaster.
Leaf-wise tree growth trains faster and often scores better on the
wide, sparse feature vectors from FeatureEngineer.

- Trains on station history (features → next-hour target)
- Falls back to sklearn HistGradientBoostingRegressor when LightGBM
  is unavailable, and to the physics-aware statistical baseline when
  no model weights or training history exist.
"""

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
        raise ImportError("numpy is required for LightGBMForecaster")
    return np


class LightGBMForecaster:
    """
    Leaf-wise gradient boosting forecaster with statistical fallback.

    Weights persist to backend/models/lgbm_pm25.joblib when trained.
    """

    MODEL_PATH_KEY = "lgbm_pm25"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._feature_names = None
        self._backend = None  # "lightgbm" | "sklearn"

    @property
    def is_trained(self) -> bool:
        """True only when real fitted weights are loaded (not baseline)."""
        return self._model is not None and bool(self._feature_names)

    async def train(
        self,
        features: List[Dict[str, Any]],
        targets: List[float],
    ) -> Dict:
        """Train leaf-wise booster on feature vectors → next-hour PM2.5."""
        np_ = _ensure_numpy()
        if not features or len(features) != len(targets):
            return {"trained": False, "reason": "insufficient data", "samples": 0}

        df = self._frame(features)
        try:
            import pandas as pd  # noqa: F401
            X = df.drop(columns=["station_id"])
            cols = list(X.columns)
            Xv = X.values
        except ImportError:
            cols = [k for k in features[0].keys() if k != "station_id"]
            Xv = np_.array([[float(r.get(c, 0.0)) for c in cols]
                            for r in features])
        y = np_.array([float(t) for t in targets])
        self._feature_names = cols

        # Prefer LightGBM; fall back to sklearn HistGBM (same leaf-wise idea).
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=220,
                num_leaves=31,
                learning_rate=0.06,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
            model.fit(Xv, y)
            self._backend = "lightgbm"
        except ImportError:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                model = HistGradientBoostingRegressor(
                    max_iter=220,
                    max_leaf_nodes=31,
                    learning_rate=0.06,
                    l2_regularization=1.0,
                    random_state=42,
                )
                model.fit(Xv, y)
                self._backend = "sklearn"
            except Exception as e:
                logger.warning("no GBM backend available (%s)", e)
                return {"trained": False, "reason": "no backend", "samples": 0}
        except Exception as e:
            logger.error("LightGBM training failed: %s", e)
            return {"trained": False, "reason": str(e), "samples": len(Xv)}

        self._model = model
        self._persist_if_possible()
        logger.info("LightGBM forecaster trained on %d samples (%s)",
                    len(Xv), self._backend)
        return {"trained": True, "samples": len(Xv),
                "features": len(cols), "backend": self._backend}

    async def predict(
        self,
        context: Dict[str, Any],
        features: Optional[Dict[str, Any]] = None,
        horizon: int = 72,
    ) -> Dict[str, List[float]]:
        """Predict the next `horizon` hours of PM2.5/PM10 (+ gases)."""
        from backend.ml.xgboost_forecaster import statistical_baseline
        current = float(context.get("current_pm25", 120.0) or 120.0)
        ratio = float(context.get("pm10_ratio", 0.0) or 0.0) or 1.35
        history = context.get("history_pm25") or []
        aisi = float(context.get("aisi", 2.0) or 2.0)
        pblh = float(context.get("pbl_height_m", 700.0) or 700.0)
        fire = float(context.get("fire_contribution", 0.0) or 0.0)

        if self._model is None or features is None:
            return statistical_baseline(
                current_pm25=current, history_pm25=history,
                aisi=aisi, pbl_height_m=pblh, fire_contribution=fire,
                horizon=horizon, pm10_ratio=ratio,
                current_no2=context.get("current_no2"),
                current_o3=context.get("current_o3"),
                current_so2=context.get("current_so2"),
                current_co=context.get("current_co"),
            )

        np_ = _ensure_numpy()
        row = {k: v for k, v in features.items() if k != "station_id"}
        out = []
        for _ in range(horizon):
            X = np_.array([[float(row.get(c, 0.0))
                            for c in self._feature_names]])
            pred = max(float(self._model.predict(X)[0]), 5.0)
            out.append(round(pred, 1))
            row["pm25_now"] = pred

        pm25 = out
        pm10 = [round(max(v * ratio, 5.0), 1) for v in out]
        lower = [round(max(v * 0.76, 2.0), 1) for v in out]
        upper = [round(v * 1.33, 1) for v in out]
        aisi_gate = max(0.3, min(1.0, aisi / 8.0))
        pbl_gate = max(0.3, min(1.0, pblh / 1000.0))
        decay = 1.0 - 0.10 * (1.0 - aisi_gate) * (0.5 + 0.5 * pbl_gate)
        from backend.ml.xgboost_forecaster import project_gases
        gases = project_gases(
            current, decay, horizon, context.get("current_no2"),
            context.get("current_o3"), context.get("current_so2"),
            context.get("current_co"))
        return {"pm25": pm25, "pm10": pm10, "lower": lower, "upper": upper,
                "pm10_lower": [round(max(v * 0.73, 2.0), 1) for v in lower],
                "pm10_upper": [round(v * 1.34, 1) for v in upper], **gases}

    # ── Internals ────────────────────────────────────────────────────

    def _frame(self, features: List[Dict]):
        try:
            import pandas as pd
            return pd.DataFrame(features)
        except ImportError:
            return features

    def _persist_if_possible(self):
        if not self.model_path:
            return
        try:
            import joblib
            joblib.dump({"model": self._model,
                         "features": self._feature_names,
                         "backend": self._backend}, self.model_path)
        except Exception as e:
            logger.warning("Could not persist LightGBM model: %s", e)

    def load(self, path: Optional[str] = None) -> bool:
        path = path or self.model_path
        if not path:
            return False
        try:
            import joblib
            data = joblib.load(path)
            self._model = data["model"]
            self._feature_names = data["features"]
            self._backend = data.get("backend", "lightgbm")
            return True
        except Exception:
            return False
