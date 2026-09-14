"""
Research Daily Ensemble — live inference bridge.

Runs the SIH-p2 research tree models (XGBoost + LightGBM native weights
trained on data/processed/delhi_final_training_data.csv) INSIDE the live
service, seeded with live CPCB/WAQI values instead of a frozen exporter
file. This is what makes H+24, the 72h graph and the skill panel reflect
real ML predictions on Render (where data/sample_forecasts/ is frozen at
deploy time and goes stale after 60h).

Faithfulness rules (do not "improve"):
- Feature rows replicate scripts/forecasting/forecast_72h.py::_feature_row
  key-for-key (daily granularity, horizons = DAYS 1..3). Same None
  semantics; trees treat them as missing natively.
- Missing TFT (pytorch-forecasting is not installed) uses the research
  offline fallback exactly: tft_pred := xgb_pred ?? lgbm_pred, then the
  fitted horizon/day-1 blends apply unchanged. Provenance always says
  tree-only so nobody mistakes backtest numbers for 3-model skill.
- Station calibration (a*x+b) and train-fit clip bounds apply as fitted.

What is live vs vendored per refresh:
- LIVE: pollutant tails (seed + SQLite daily means + today partial),
  weather days (Open-Meteo state), "today" fire anchor.
- VENDORED (backend/models/research/ + data/research_seed_daily.csv):
  booster weights, blends, calibration, clip bounds, 45d seed tails,
  trailing fire history (seed ends 2026-09-11; staleness is logged).
"""

import csv
import logging
import math
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TARGETS = ["pm25", "pm10", "no2", "o3"]
ENCODER_DAYS = 14

_STOPWORDS = re.compile(
    r"\b(delhi|new|sector|sec|phase|gram|nagar|marg|road|rd|station|"
    r"dpcc|cpcb|uppcb|hspcb|imd|iitm|sai|teri|garden|area|crossing)\b")

# Day-dict key groups for TFT encoder history (service assembles these
# from seed rows + live SQLite daily means).
_WX_KEYS = ["temp_mean", "temp_max", "temp_min", "humidity_mean",
            "wind_speed_mean", "wind_direction_dominant", "precipitation",
            "pressure_mean", "wind_gusts_mean", "blh_max",
            "radiation_sum", "dewpoint_mean"]
_FIRE_KEYS = ["regional_fire_count", "regional_total_frp",
              "regional_max_frp", "local_fire_count", "local_total_frp",
              "local_max_frp"]
_POL_KEYS = ("pm25", "pm10", "no2", "o3", "no", "nox")


def _mean_last(vals: list, n: int) -> float:
    v = [x for x in vals if x is not None][-n:]
    return (sum(v) / len(v)) if v else 0.0


def downscale_daily_to_hourly(preds, rmses, naqi, start=None,
                                horizon_dates=None):
    """Pin Delhi-diurnal hourly shape to Day+1/2/3 means (index blocks).

    preds: {target: [d1, d2, d3]} calibrated daily means.
    rmses: {target: [r1, r2, r3]} horizon uncertainties.
    naqi: NAQICalculator (or compatible with calculate_naqi).
    start: next-full-hour datetime (UTC); defaults to now.
    Returns dict with timestamps/pm25/pm10/no2/o3, bands, aqi/category/
    colors and daily[] — the same arrays the dashboard chart verifies
    (24h index-block means == daily values, tolerance 0.15).
    """
    from datetime import datetime, timedelta, timezone
    from backend.ml.xgboost_forecaster import gas_diurnal_factor

    if start is None:
        now = datetime.now(timezone.utc)
        start = now.replace(minute=0, second=0, microsecond=0) \
            + timedelta(hours=1)
    timestamps = [(start + timedelta(hours=i)).isoformat()
                  for i in range(72)]
    hours = []
    for ts in timestamps:
        try:
            hours.append(datetime.fromisoformat(
                str(ts).replace("Z", "+00:00")).hour)
        except (ValueError, TypeError):
            hours.append(12)

    def _series(target):
        daily = preds.get(target) or []
        vals = []
        for i in range(72):
            h = i // 24 + 1
            if h > 3 or h - 1 >= len(daily) or daily[h - 1] is None:
                vals.append(None)
                continue
            # Per-pollutant shape: NO2 twin traffic peaks, O3 afternoon
            # photochemical peak — never the PM curve for gases.
            f = [gas_diurnal_factor(target, hours[j] % 24)
                 for j in range(72) if j // 24 + 1 == h]
            norm = (sum(f) / len(f)) if f else 1.0
            vals.append(round(max(
                daily[h - 1] * gas_diurnal_factor(target, hours[i] % 24)
                / (norm or 1.0), 1.0), 1))
        return vals

    def _band(target, vals):
        lo, hi = [], []
        for i, v in enumerate(vals):
            h = min(i // 24 + 1, 3)
            r = (rmses.get(target) or [0.0, 0.0, 0.0])[h - 1] or 0.0
            if r <= 0:
                r = max((v or 0) * 0.15, 5.0)
            lo.append(round(max((v or 0) - 1.28 * r, 2.0), 1))
            hi.append(round((v or 0) + 1.28 * r, 1))
        return lo, hi

    pm25, pm10 = _series("pm25"), _series("pm10")
    no2 = _series("no2")
    o3 = _series("o3")
    lower, upper = _band("pm25", pm25)
    pm10_lower, pm10_upper = _band("pm10", pm10)
    no2_lower, no2_upper = _band("no2", no2)
    o3_lower, o3_upper = _band("o3", o3)
    for arr in (no2, o3):
        for i, v in enumerate(arr):
            if v is None:
                arr[i] = 0.0
    aqi, category, colors = [], [], []
    for a, b, c, d in zip(pm25, pm10, no2, o3):
        r = naqi.calculate_naqi(
            {"pm25": a, "pm10": b, "no2": c, "o3": d})
        aqi.append(r.overall_aqi)
        category.append(r.category)
        colors.append(r.color)
    daily = []
    for h in (1, 2, 3):
        r = naqi.calculate_naqi({
            "pm25": (preds.get("pm25") or [None] * 3)[h - 1],
            "pm10": (preds.get("pm10") or [None] * 3)[h - 1],
            "no2": (preds.get("no2") or [None] * 3)[h - 1],
            "o3": (preds.get("o3") or [None] * 3)[h - 1]})
        entry = {"horizon_h": h,
                 "pm25": (preds.get("pm25") or [None] * 3)[h - 1],
                 "pm10": (preds.get("pm10") or [None] * 3)[h - 1],
                 "no2": (preds.get("no2") or [None] * 3)[h - 1],
                 "o3": (preds.get("o3") or [None] * 3)[h - 1],
                 "aqi": r.overall_aqi, "category": r.category,
                 "color": r.color,
                 "dominant_pollutant": r.dominant_pollutant}
        if horizon_dates and h - 1 < len(horizon_dates):
            try:
                entry["date"] = horizon_dates[h - 1].isoformat()
            except AttributeError:
                entry["date"] = str(horizon_dates[h - 1])
        daily.append(entry)
    return {"timestamps": timestamps, "pm25": pm25, "pm10": pm10,
            "no2": no2, "o3": o3, "lower": lower, "upper": upper,
            "pm10_lower": pm10_lower, "pm10_upper": pm10_upper,
            "no2_lower": no2_lower, "no2_upper": no2_upper,
            "o3_lower": o3_lower, "o3_upper": o3_upper,
            "aqi": aqi, "category": category, "colors": colors,
            "daily": daily}


def _tails_upto(hist: list, end) -> Dict[str, list]:
    tails = {t: [] for t in _POL_KEYS}
    for d in hist:
        if d.get("date") is not None and d.get("date") <= end:
            pol = d.get("pollutants") or {}
            for t in tails:
                tails[t].append(pol.get(t))
    return {t: [v for v in vals if v is not None][-14:]
            for t, vals in tails.items()}


def _fire_upto(hist: list, end) -> Dict[str, list]:
    out = {"regional_fire_count": [], "regional_total_frp": []}
    for d in hist:
        if d.get("date") is not None and d.get("date") <= end:
            fire = d.get("fire") or {}
            for k in out:
                out[k].append(fire.get(k))
    return {k: [v for v in vals if v is not None][-30:]
            for k, vals in out.items()}


def _tokens(text: str) -> set:
    t = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower())
    t = _STOPWORDS.sub(" ", t)
    return {tok for tok in t.split() if len(tok) >= 4}


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _past_window(hist: list, h: int, n: int) -> list:
    end = len(hist) - h + 1
    if end <= 0:
        return []
    return hist[max(0, end - n):end]


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


class ResearchDailyEnsemble:
    """Daily forecaster (Day+1/2/3 per pollutant).

    Trees (XGB+LGBM) always run when loaded. TFT is an optional third
    member attached via attach_tft(); without it the research offline
    fallback (tft := xgb ?? lgbm) applies and provenance says tree-only.
    """

    def __init__(self, model_dir: str, seed_csv: str):
        self.model_dir = model_dir
        self.seed_csv = seed_csv
        self.models: Dict[Tuple[str, str], Any] = {}
        self.feature_order: Dict[Tuple[str, str], List[str]] = {}
        self.blends_h: Dict[Tuple[str, int], tuple] = {}
        self.blends_day1: Dict[str, str] = {}
        self.cal: Dict[Tuple[str, str], tuple] = {}
        self.clip: Dict[str, tuple] = {}
        self.seed: Dict[str, List[Dict[str, Any]]] = {}
        self.loc_meta: Dict[str, Dict[str, float]] = {}
        self.tft = None
        self._loaded = False

    def attach_tft(self, tft) -> None:
        """Attach a loaded TFTDailyEnsemble (or None to detach)."""
        self.tft = tft if tft is not None and tft.available else None

    @property
    def tft_targets(self) -> List[str]:
        if self.tft is None:
            return []
        return [t for t in self.runnable_targets()
                if t in self.tft.runnable_targets()]

    # ── Loading ────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load weights + tables. True when at least one target is runnable."""
        import os
        xgb_mod = lgb_mod = None
        try:
            import xgboost as xgb_mod  # noqa: F811
        except ImportError:
            logger.warning("research-daily: xgboost missing (lgbm-only)")
        try:
            import lightgbm as lgb_mod  # noqa: F811
        except ImportError:
            logger.warning("research-daily: lightgbm missing (xgb-only)")

        d = self.model_dir
        for t in TARGETS:
            xf = os.path.join(d, f"xgb_{t}.json")
            lf = os.path.join(d, f"lgbm_{t}.txt")
            if xgb_mod is not None and os.path.exists(xf):
                try:
                    m = xgb_mod.XGBRegressor()
                    m.load_model(xf)
                    self.models[("xgb", t)] = m
                    try:
                        self.feature_order[("xgb", t)] = list(
                            m.feature_names_in_)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning("research-daily: load %s: %s", xf, e)
            if lgb_mod is not None and os.path.exists(lf):
                try:
                    b = lgb_mod.Booster(model_file=lf)
                    self.models[("lgbm", t)] = b
                    try:
                        self.feature_order[("lgbm", t)] = list(
                            b.feature_name())
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning("research-daily: load %s: %s", lf, e)

        for row in _read_csv_rows(os.path.join(d, "horizon_blend.csv")):
            try:
                self.blends_h[(row["target"], int(row["horizon_h"]))] = (
                    float(row["w_tft"]), float(row["w_xgb"]),
                    float(row["w_lgbm"]), float(row["bias"]),
                    float(row.get("val_rmse") or 0.0))
            except (KeyError, TypeError, ValueError):
                continue
        for row in _read_csv_rows(
                os.path.join(d, "best_blend_per_pollutant.csv")):
            if row.get("target") and row.get("chosen_blend"):
                self.blends_day1[row["target"]] = row["chosen_blend"]
        for row in _read_csv_rows(os.path.join(d, "station_calibration.csv")):
            try:
                self.cal[(row["location_name"], row["target"])] = (
                    float(row["a"]), float(row["b"]))
            except (KeyError, TypeError, ValueError):
                continue
        for row in _read_csv_rows(os.path.join(d, "feature_clip_bounds.csv")):
            try:
                self.clip[row["column"]] = (float(row["lo"]),
                                            float(row["hi"]))
            except (KeyError, TypeError, ValueError):
                continue

        for row in _read_csv_rows(self.seed_csv):
            loc = (row.get("location_name") or "").strip()
            if not loc:
                continue
            try:
                y, m, dd = [int(x) for x in str(row["date"])[:10].split("-")]
                day = date(y, m, dd)
            except (KeyError, TypeError, ValueError):
                continue
            rec = {"date": day}
            for k in ("no", "no2", "o3", "pm10", "pm25", "nox",
                      "regional_fire_count", "regional_total_frp",
                      "regional_max_frp", "local_fire_count",
                      "local_total_frp", "local_max_frp",
                      "temp_mean", "temp_max", "temp_min",
                      "humidity_mean", "wind_speed_mean",
                      "wind_direction_dominant", "precipitation",
                      "pressure_mean", "wind_gusts_mean", "blh_max",
                      "radiation_sum", "dewpoint_mean"):
                rec[k] = _num(row.get(k))
            self.seed.setdefault(loc, []).append(rec)
            self.loc_meta.setdefault(loc, {
                "latitude": _num(row.get("latitude")) or 0.0,
                "longitude": _num(row.get("longitude")) or 0.0,
                "station_id": int(float(row.get("station_id") or -1)),
            })
        for loc in self.seed:
            self.seed[loc].sort(key=lambda r: r["date"])

        runnable = [t for t in TARGETS
                    if ("xgb", t) in self.models or ("lgbm", t) in self.models]
        self._loaded = bool(runnable)
        if self._loaded:
            seed_max = max((r["date"] for rows in self.seed.values()
                            for r in rows), default=None)
            logger.info("research-daily loaded: %s (seed through %s)",
                        ",".join(runnable), seed_max)
        else:
            logger.warning("research-daily: no runnable targets")
        return self._loaded

    @property
    def available(self) -> bool:
        return self._loaded

    def runnable_targets(self) -> List[str]:
        return [t for t in TARGETS
                if ("xgb", t) in self.models or ("lgbm", t) in self.models]

    # ── Station mapping ────────────────────────────────────────────

    def map_station(self, short_name: str, name: str,
                    lat: float, lon: float) -> Optional[str]:
        """Runtime station -> research location_name (tokens + substring
        bonus, proximity tiebreak). Mirrors service name matching."""
        cands = list(self.seed.keys())
        if not cands:
            return None
        keys = _tokens(short_name) | _tokens(name)
        short = re.sub(r"[^a-z0-9]", "", str(short_name or "").lower())
        scored = []
        for loc in cands:
            lt = _tokens(loc)
            score = len(keys & lt)
            if len(short) >= 2 and short in re.sub(
                    r"[^a-z0-9]", "", loc.lower()):
                score += 2
            if score > 0:
                meta = self.loc_meta.get(loc, {})
                dist = _haversine(lat, lon, meta.get("latitude", 0.0),
                                  meta.get("longitude", 0.0))
                scored.append((score, -dist, loc))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][2]

    # ── Feature rows (port of forecast_72h._feature_row) ────────────

    @staticmethod
    def feature_row(base: dict, h: int, wx: dict, fire: dict,
                    recent: Dict[str, list],
                    fire_hist: Optional[Dict[str, list]] = None) -> dict:
        row = dict(base)
        row.update(wx or {})
        row.update(fire or {})
        for t in ("pm25", "pm10", "no2", "o3", "no", "nox"):
            hist = recent.get(t, []) or []
            row[f"{t}_lag1"] = hist[-h] if len(hist) >= h else None
            row[f"{t}_lag2"] = hist[-(h + 1)] if len(hist) >= h + 1 else None
            window = _past_window(hist, h, 3)
            row[f"{t}_roll3"] = (sum(window) / len(window)) if window else None
        fh = fire_hist or {}
        rfc = fh.get("regional_fire_count", []) or []
        rfrp = fh.get("regional_total_frp", []) or []
        row["regional_fire_count_lag1"] = rfc[-h] if len(rfc) >= h else None
        row["regional_fire_count_lag2"] = (
            rfc[-(h + 1)] if len(rfc) >= h + 1 else None)
        row["regional_total_frp_lag1"] = (
            rfrp[-h] if len(rfrp) >= h else None)
        row["regional_total_frp_lag3"] = (
            rfrp[-(h + 2)] if len(rfrp) >= h + 2 else None)
        for key, n in (("regional_fire_count_roll3", 3),
                       ("regional_fire_count_roll7", 7),
                       ("regional_total_frp_roll3", 3),
                       ("regional_total_frp_roll7", 7)):
            src = rfc if "count" in key else rfrp
            w = _past_window(src, h, n)
            row[key] = (sum(w) / len(w)) if w else None
        d = base["_date"]
        row["day_of_week"] = d.weekday()
        row["month"] = d.month
        row["is_stubble_season"] = int(d.month in (10, 11))
        # Present-but-None (== NaN): training used these columns, so the
        # keys must EXIST for the TFT dataset; trees see NaN either way,
        # identical to research inference which never set them.
        row["is_holiday"] = None
        row["is_diwali_week"] = None
        wd = (wx or {}).get("wind_direction_dominant")
        row["wind_toward_delhi"] = int(
            wd is not None and 270 <= wd <= 330)
        row["fire_wind_interaction"] = (
            (fire.get("regional_total_frp", 0) or 0)
            * row["wind_toward_delhi"]) if fire else 0
        # Derived interactions: keys must ALWAYS exist (None when
        # inputs absent) because the TFT variable-selection head has a
        # fixed input width from training; a missing column shifts every
        # index after it and crashes the forward pass (IndexError).
        row["ventilation"] = None
        row["o3_driver"] = None
        if "blh_max" in (wx or {}) and "wind_speed_mean" in (wx or {}):
            try:
                row["ventilation"] = (float(wx["blh_max"])
                                      * float(wx["wind_speed_mean"]))
            except (TypeError, ValueError):
                pass
        if "radiation_sum" in (wx or {}) and "temp_max" in (wx or {}):
            try:
                row["o3_driver"] = (float(wx["radiation_sum"])
                                    * float(wx["temp_max"]))
            except (TypeError, ValueError):
                pass
        return row

    def apply_clip(self, feat: dict) -> dict:
        for col, (lo, hi) in self.clip.items():
            if col in feat and feat[col] is not None:
                try:
                    feat[col] = min(hi, max(lo, float(feat[col])))
                except (TypeError, ValueError):
                    pass
        return feat

    # ── Blend + calibration (research offline semantics, tree-only) ─

    @staticmethod
    def _parse_day1_label(label: str) -> tuple:
        label = (label or "").strip()
        if label == "TFT only":
            return (1.0, 0.0, 0.0)
        if label == "XGB only":
            return (0.0, 1.0, 0.0)
        if label == "LGBM only":
            return (0.0, 0.0, 1.0)
        try:
            parts = dict(p.split("=") for p in label.split(", "))
            return (float(parts["tft"]), float(parts["xgb"]),
                    float(parts["lgbm"]))
        except (ValueError, KeyError, AttributeError):
            return (0.34, 0.33, 0.33)

    def blend(self, target: str, h: int, xp: Optional[float],
              lp: Optional[float], tp: Optional[float] = None
              ) -> Tuple[Optional[float], float, str]:
        """Returns (blended, band_rmse, blend_label).

        tp = live TFT prediction when the third member is wired;
        otherwise the research offline fallback (tft := xgb ?? lgbm)
        with fitted weights unchanged.
        """
        w_tft, w_xgb, w_lgbm, bias, rmse = self.blends_h.get(
            (target, h), (None, None, None, None, 0.0))
        label = None
        if w_tft is None:
            w_tft, w_xgb, w_lgbm = self._parse_day1_label(
                self.blends_day1.get(target, ""))
            bias = 0.0
            label = self.blends_day1.get(target, "equal")
        else:
            label = f"tft={w_tft}, xgb={w_xgb}, lgbm={w_lgbm}"
        if tp is None:
            tp = xp if xp is not None else lp
        if xp is None and lp is None and tp is None:
            return None, rmse, label
        val = (w_xgb * (xp if xp is not None else 0.0)
               + w_lgbm * (lp if lp is not None else 0.0)
               + w_tft * (tp if tp is not None else 0.0) + bias)
        return val, rmse, label

    def calibrate(self, location: str, target: str, value: float) -> float:
        a, b = self.cal.get((location, target), (1.0, 0.0))
        return value * a + b

    # ── Inference ──────────────────────────────────────────────────

    def _predict_one(self, feat: dict, target: str
                     ) -> Tuple[Optional[float], Optional[float]]:
        def _row(cols):
            row = []
            for c in cols:
                v = feat.get(c)
                try:
                    f = float(v)
                    row.append(f if math.isfinite(f) else float("nan"))
                except (TypeError, ValueError):
                    row.append(float("nan"))
            return [row]
        xp = lp = None
        xm = self.models.get(("xgb", target))
        if xm is not None:
            try:
                cols = (self.feature_order.get(("xgb", target))
                        or sorted(feat.keys()))
                xp = float(xm.predict(_row(cols))[0])
            except Exception as e:
                logger.debug("research-daily xgb %s: %s", target, e)
        lm = self.models.get(("lgbm", target))
        if lm is not None:
            try:
                cols = (self.feature_order.get(("lgbm", target))
                        or sorted(feat.keys()))
                lp = float(lm.predict(_row(cols))[0])
            except Exception as e:
                logger.debug("research-daily lgbm %s: %s", target, e)
        return xp, lp

    def predict_station(
        self,
        location: str,
        recent: Dict[str, list],
        fire_hist: Dict[str, list],
        fire_today: dict,
        wx_days: List[dict],
        horizon_dates: List[date],
        base_static: dict,
    ) -> Dict[str, Any]:
        """Recursive Day+1/2/3 predictions for one station.

        recent/fire_hist are mutated transiently (preds feed h+1 lags),
        mirroring forecast_72h.run_forecast — pass copies per station.
        """
        recent = {t: list(v or []) for t, v in recent.items()}
        fire_hist = {k: list(v or []) for k, v in fire_hist.items()}
        out: Dict[str, list] = {t: [] for t in self.runnable_targets()}
        rmses: Dict[str, list] = {t: [] for t in self.runnable_targets()}
        blends_used: Dict[str, str] = {}
        for h, fdate in zip((1, 2, 3), horizon_dates):
            wx = wx_days[h - 1] if h - 1 < len(wx_days) else {}
            base = dict(base_static)
            base["_date"] = fdate
            feat = self.feature_row(base, h, wx, fire_today, recent,
                                    fire_hist)
            feat = self.apply_clip(feat)
            feat_clean = {k: v for k, v in feat.items()
                          if not str(k).startswith("_")}
            for t in self.runnable_targets():
                xp, lp = self._predict_one(feat_clean, t)
                if xp is None and lp is None:
                    hist_vals = recent.get(t) or [None]
                    xp = lp = hist_vals[-1]  # persistence fallback
                if xp is None and lp is None:
                    continue
                blended, rmse, label = self.blend(t, h, xp, lp)
                if blended is None:
                    continue
                final = max(0.0, self.calibrate(location, t, blended))
                out[t].append(round(final, 1))
                rmses[t].append(rmse)
                blends_used[t] = label
                recent.setdefault(t, []).append(final)
            fire_hist.setdefault("regional_fire_count", []).append(
                fire_today.get("regional_fire_count", 0) or 0)
            fire_hist.setdefault("regional_total_frp", []).append(
                fire_today.get("regional_total_frp", 0) or 0)
        return {"preds": out, "rmses": rmses, "blends": blends_used}

    # ── Domain inference (batched TFT third member) ────────────────

    def predict_domain(self, station_inputs, horizon_dates):
        """Batched 3-member inference over many stations.

        station_inputs[sid] = dict(location, recent, fire_hist,
        fire_today, wx_days, base_static, hist_days) where hist_days =
        [{date, pollutants{}, wx{}, fire{}}] oldest->newest ending at
        the last history day (seed rows + live extension, built by the
        caller). TFT runs once per (target, horizon) over ALL seed
        locations (extra unmapped ones included with seed-only inputs
        so the location embedding matches training); per-station
        failures keep the tree-only fallback. Returns
        {sid: {preds, rmses, blends, tft_used}}.
        """
        sts = {}
        for sid, inp in (station_inputs or {}).items():
            sts[sid] = {
                "location": inp["location"],
                "recent": {t: list(v or [])
                           for t, v in (inp.get("recent") or {}).items()},
                "fire_hist": {k: list(v or [])
                              for k, v in (inp.get("fire_hist") or {}).items()},
                "fire_today": dict(inp.get("fire_today") or {}),
                "wx_days": inp.get("wx_days") or [],
                "base_static": inp.get("base_static") or {},
                "hist": [dict(d) for d in inp.get("hist_days", [])],
                "out": {t: [] for t in self.runnable_targets()},
                "rmses": {t: [] for t in self.runnable_targets()},
                "blends": {},
                "tft_used": {},
            }
        # Seed-only locations (no live station): TFT batch needs the
        # full training location set for stable categorical encoding.
        live_locs = {s["location"] for s in sts.values()}
        for loc, rows in (self.seed or {}).items():
            if loc in live_locs or len(rows) < 15:
                continue
            tail = rows[-30:]
            sts[f"__seed__{loc}"] = {
                "location": loc,
                "recent": {t: [r[t] for r in tail if r.get(t) is not None][-14:]
                           for t in ("pm25", "pm10", "no2", "o3",
                                     "no", "nox")},
                "fire_hist": {
                    "regional_fire_count": [
                        r["regional_fire_count"] for r in tail
                        if r.get("regional_fire_count") is not None],
                    "regional_total_frp": [
                        r["regional_total_frp"] for r in tail
                        if r.get("regional_total_frp") is not None]},
                "fire_today": {
                    "local_fire_count": 0.0, "local_total_frp": 0.0,
                    "local_max_frp": 0.0,
                    "regional_fire_count": _mean_last(
                        [r["regional_fire_count"] for r in tail
                         if r.get("regional_fire_count") is not None], 7),
                    "regional_total_frp": _mean_last(
                        [r["regional_total_frp"] for r in tail
                         if r.get("regional_total_frp") is not None], 7),
                    "regional_max_frp": 0.0},
                "wx_days": [{}, {}, {}],
                "base_static": {
                    "location_name": loc,
                    "latitude": (self.loc_meta.get(loc) or {}).get(
                        "latitude", 0.0),
                    "longitude": (self.loc_meta.get(loc) or {}).get(
                        "longitude", 0.0),
                    "station_id": (self.loc_meta.get(loc) or {}).get(
                        "station_id", -1)},
                "hist": [{"date": r["date"],
                          "pollutants": {t: r.get(t) for t in
                                         ("pm25", "pm10", "no2", "o3",
                                          "no", "nox")},
                          "wx": {k: r.get(k) for k in _WX_KEYS},
                          "fire": {k: r.get(k) for k in _FIRE_KEYS}}
                         for r in tail],
                "out": {}, "rmses": {}, "blends": {}, "tft_used": {},
                "seed_only": True,
            }
        for h, fdate in zip((1, 2, 3), horizon_dates):
            feats, treep = {}, {}
            for sid, st in sts.items():
                wx = st["wx_days"][h - 1] if h - 1 < len(st["wx_days"]) \
                    else {}
                base = dict(st["base_static"])
                base["_date"] = fdate
                feat = self.apply_clip(self.feature_row(
                    base, h, wx, st["fire_today"], st["recent"],
                    st["fire_hist"]))
                clean = {k: v for k, v in feat.items()
                         if not str(k).startswith("_")}
                feats[sid] = clean
                treep[sid] = {t: self._predict_one(clean, t)
                              for t in self.runnable_targets()}
            tftp = {}
            if self.tft is not None:
                frames = self._tft_frames(sts, feats, h, fdate)
                for t in self.tft_targets:
                    tftp[t] = self.tft.predict_horizon(t, frames)
            for sid, st in sts.items():
                if st.get("seed_only"):
                    continue
                loc = st["location"]
                for t in self.runnable_targets():
                    xp, lp = treep[sid][t]
                    tp = (tftp.get(t) or {}).get(loc)
                    if xp is None and lp is None and tp is None:
                        hist_vals = st["recent"].get(t) or [None]
                        xp = lp = hist_vals[-1]
                    if xp is None and lp is None and tp is None:
                        continue
                    blended, rmse, label = self.blend(t, h, xp, lp, tp)
                    if blended is None:
                        continue
                    final = max(0.0, self.calibrate(loc, t, blended))
                    st["out"][t].append(round(final, 1))
                    st["rmses"][t].append(rmse)
                    st["blends"][t] = label
                    if tp is not None:
                        st["tft_used"][t] = True
                    st["recent"].setdefault(t, []).append(final)
                self._extend_hist_day(sts[sid], fdate, h)
                st["fire_hist"].setdefault(
                    "regional_fire_count", []).append(
                    st["fire_today"].get("regional_fire_count", 0) or 0)
                st["fire_hist"].setdefault(
                    "regional_total_frp", []).append(
                    st["fire_today"].get("regional_total_frp", 0) or 0)
        return {sid: {"preds": st["out"], "rmses": st["rmses"],
                      "blends": st["blends"], "tft_used": st["tft_used"]}
                for sid, st in sts.items() if not st.get("seed_only")}

    def _extend_hist_day(self, st, fdate, h):
        """Append the just-predicted day to a station's date-anchored
        history so h+1 encoder windows slide forward on preds."""
        pol = {}
        for t in ("pm25", "pm10", "no2", "o3", "no", "nox"):
            arr = st["out"].get(t) or []
            pol[t] = arr[h - 1] if len(arr) >= h else None
        wx = st["wx_days"][h - 1] if h - 1 < len(st["wx_days"]) else {}
        st["hist"].append({"date": fdate, "pollutants": pol,
                           "wx": dict(wx),
                           "fire": dict(st["fire_today"])})

    def _tft_frames(self, sts, feats, h, fdate):
        """{location: [14 encoder dicts + 1 decoder dict]} for one
        horizon. Encoder rows rebuild lag features date-anchored via
        feature_row(h=1); decoder reuses the tree feat + target NaN."""
        from datetime import timedelta
        frames = {}
        for sid, st in sts.items():
            loc = st["location"]
            hist = st.get("hist") or []
            end = fdate - timedelta(days=1)
            cands = [d for d in hist if d.get("date") <= end][-14:]
            if len(cands) < 14:
                continue
            enc = []
            for d in cands:
                dd = d["date"]
                tails = _tails_upto(st["hist"], dd)
                fhist = _fire_upto(st["hist"], dd)
                base = dict(st["base_static"])
                base["_date"] = dd
                row = self.feature_row(
                    base, 1, d.get("wx") or {}, d.get("fire") or {},
                    tails, fhist)
                row = {k: v for k, v in row.items()
                       if not str(k).startswith("_")}
                for t in ("pm25", "pm10", "no2", "o3", "no", "nox"):
                    row[t + "_actual"] = (d.get("pollutants") or {}).get(t)
                row["location_name"] = loc
                row["date"] = dd.isoformat()
                enc.append(row)
            dec = dict(feats.get(sid) or {})
            dec["location_name"] = loc
            dec["date"] = fdate.isoformat()
            frames[loc] = enc + [dec]
        return frames
