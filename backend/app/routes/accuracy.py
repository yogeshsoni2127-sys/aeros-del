"""
Accuracy Routes — forecast skill, loss, and per-station AQI table.

- GET /api/v1/accuracy/summary
    Walk-forward backtest on persisted SQLite (same logic as
    scripts/evaluate_accuracy.py): baseline vs persistence MAE/RMSE/bias,
    Pearson r, AQI-category hit, plus:
      * skill_score = 1 - MAE_model / MAE_persistence  (>0 beats persistence)
      * loss_mse / loss_mae  (MSE is the training loss we optimise)
- GET /api/v1/accuracy/stations
    One row per station (all ~53): current PM2.5/PM10/AQI/category +
    H+24 forecast + history depth + last-step error. This is the "AQI col"
    table for the dashboard / judges.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/accuracy", tags=["accuracy"])


def _get_service(request: Request):
    return request.app.state.service


def _ensemble_72h_block():
    """Real SIH-p2 72h ensemble skill (no DB needed).
    Returns per-pollutant test RMSE, per-horizon backtest and AQI hit
    rates from scripts/ensemble_skill.json (written by the exporter),
    or None when the exporter has never run.
    """
    import json
    from backend.app.config import PROJECT_ROOT
    try:
        return json.loads((PROJECT_ROOT / "scripts" / "ensemble_skill.json").read_text())
    except Exception:
        return None


def _db_path(service) -> Path:
    from backend.app.config import PROJECT_ROOT
    return PROJECT_ROOT / service.settings.database_path


def _count_readings(db) -> int:
    import sqlite3
    try:
        con = sqlite3.connect(str(db))
        n = con.execute("SELECT COUNT(*) FROM station_readings").fetchone()[0]
        con.close()
        return int(n)
    except Exception:
        return 0


def _history_progress(db) -> dict:
    """Machine-readable warm-up state for the skill panel progress bar.

    pairs_needed = consecutive same-sensor pairs that make the live
    backtest meaningful. Until then the panel shows validated ensemble
    skill instead of a blank error.
    """
    import sqlite3
    readings, stations = 0, 0
    try:
        con = sqlite3.connect(str(db))
        readings = int(con.execute("SELECT COUNT(*) FROM station_readings").fetchone()[0])
        stations = int(con.execute(
            "SELECT COUNT(DISTINCT station_id) FROM station_readings").fetchone()[0])
        con.close()
    except Exception:
        pass
    need = 50
    # ~75 readings (25 stations x 3 consecutive cycles) yield the first
    # usable pairs; pct is progress toward a meaningful live backtest.
    pct = min(99, round(100 * readings / 75)) if readings else 0
    return {"readings": readings, "stations": stations,
            "pairs_available": 0, "pairs_needed": need, "pct": pct}


@router.get("/summary")
async def accuracy_summary(request: Request, include_gbm: bool = False):
    """Backtest skill of the live pipeline on SQLite history.

    Args:
        include_gbm: if true, also train/test a LightGBM/HistGBM next-step
            model (slower, ~seconds). Default false = fast baseline-only.
    """
    service = _get_service(request)
    try:
        import sys
        from backend.app.config import PROJECT_ROOT as _ROOT
        sys.path.insert(0, str(_ROOT))
        from scripts.evaluate_accuracy import (
            load_series, backtest, train_gbm,
        )
    except Exception as e:
        return {"available": False, "reason": f"import failed: {e}"}

    db = _db_path(service)
    if not db.exists():
        return {"available": False,
                "reason": f"No database at {db} — run server once first.",
                "ensemble_72h": _ensemble_72h_block()}
    try:
        series = load_series(str(db))
    except Exception as e:
        # Fresh disk: DB file exists but tables aren't created yet (first
        # refresh still running). Still serve the static ensemble skill so
        # the Forecast Skill panel is never blank.
        logger.debug("accuracy series unavailable: %s", e)
        return {"available": False,
                "reason": "History still warming up — ensemble skill below.",
                "history": _history_progress(db),
                "ensemble_72h": _ensemble_72h_block()}
    if not series:
        # Normal on free-tier ephemeral disk: every deploy/restart wipes
        # SQLite, so only the current cycle's readings exist (~25 = one
        # refresh) and there are no consecutive same-sensor pairs to score
        # yet. The static 72h ensemble skill below is the validated number.
        n_read = _count_readings(db)
        return {"available": False,
                "reason": (f"Live backtest needs consecutive history "
                           f"({n_read} readings stored so far — disk resets on "
                           f"each deploy). Validated 72h skill below."),
                "readings_stored": n_read,
                "history": _history_progress(db),
                "ensemble_72h": _ensemble_72h_block()}

    skill = backtest(series)

    # ── Live Day+1 skill for the served research-daily model ──────────
    # Persisted Day+1 predictions scored against realised means. Starts
    # empty after each deploy (ephemeral disk) and warms up within days;
    # until then the validated research backtest below is the headline.
    try:
        daily_live = await service.preprocessor.get_daily_skill(days=14)
    except Exception as e:
        logger.debug("daily skill unavailable: %s", e)
        daily_live = {}

    # ── Predictive score + loss (judge-friendly rollups) ──
    # skill_score per bucket: +20% means 20% lower MAE than persistence.
    scores = {}
    for bucket in ("<=1h", "1-6h", ">6h"):
        b = skill.get(bucket, {})
        m_mae = (b.get("baseline") or {}).get("mae")
        p_mae = (b.get("persistence") or {}).get("mae")
        if m_mae is not None and p_mae:
            scores[bucket] = round(1.0 - m_mae / p_mae, 3)
        else:
            scores[bucket] = 0.0
        # loss: MSE is what gradient boosters minimise; MAE is human-readable
        rmse = (b.get("baseline") or {}).get("rmse")
        if rmse is not None:
            b["baseline"]["loss_mse"] = round(rmse ** 2, 1)
            b["baseline"]["loss_mae"] = b["baseline"].get("mae")
    out = {
        "available": True,
        "db": str(db),
        "stations_with_history": len(series),
        "total_points": sum(len(v) for v in series.values()),
        "pairs": skill.get("pairs", 0),
        "buckets": {k: skill.get(k, {}) for k in ("<=1h", "1-6h", ">6h")},
        "skill_vs_persistence": scores,
        "pearson_r_baseline": skill.get("pearson_r_baseline", 0.0),
        "daily_live": daily_live,
        "ensemble_72h": _ensemble_72h_block(),
        "how_to_read": {
            "mae": "Mean absolute error µg/m³ (lower = better).",
            "rmse": "Root mean square error; penalises big misses.",
            "loss_mse": "MSE = RMSE² — the optimiser loss.",
            "skill_vs_persistence": ">0 beats naive carry-forward; "
                                   "+0.20 = 20% better.",
            "cat_acc": "Fraction where predicted AQI category == actual.",
            "pearson_r": "Correlation of predicted vs actual (1 = perfect).",
            "daily_live": "Served research-daily Day+1 predictions scored "
                          "against realised means (warms up after deploy).",
        },
    }
    if include_gbm:
        try:
            out["gbm_next_step"] = train_gbm(series)
        except Exception as e:
            out["gbm_next_step"] = {"trained": False, "reason": str(e)}
    return out


@router.get("/model-status")
async def model_status(request: Request):
    """What is actually forecasting right now — trained or baseline?

    Returns live ensemble member flags (True only with fitted weights on
    disk), last training/backfill reports if present, and DB depth.
    The dashboard Model panel reads this; judges can verify every claim.
    """
    import json
    import sqlite3
    from backend.app.config import PROJECT_ROOT

    service = _get_service(request)
    ens = getattr(service, "ensemble", None)

    def _read_json(name):
        try:
            return json.loads((PROJECT_ROOT / "scripts" / name).read_text())
        except Exception:
            return None

    db = _db_path(service)
    db_info = {"exists": db.exists(), "readings": 0, "stations": 0}
    if db.exists():
        try:
            con = sqlite3.connect(str(db))
            n, s = con.execute(
                "SELECT COUNT(*), COUNT(DISTINCT station_id) "
                "FROM station_readings").fetchone()
            con.close()
            db_info.update({"readings": n, "stations": s})
        except Exception as e:
            db_info["error"] = str(e)

    members = {}
    if ens is not None:
        for key, member in (("tft", getattr(ens, "tft", None)),
                            ("xgboost", getattr(ens, "xgb", None)),
                            ("lightgbm", getattr(ens, "lgbm", None))):
            members[key] = bool(getattr(member, "is_trained", False))
    members["baseline"] = not any(members.values())

    training = _read_json("training_report.json")
    backfill = _read_json("backfill_report.json")
    ensemble_72h = _read_json("ensemble_skill.json")
    members_72h = {}
    if ensemble_72h:
        for row in ensemble_72h.get("per_pollutant_test", []):
            members_72h[row["target"]] = row.get("blend")
    return {
        "members": members,
        "mode": ("baseline" if members.get("baseline")
                 else "+".join(k for k, v in members.items() if v)),
        "db": db_info,
        "training": training,
        "backfill": backfill,
        # Real SIH-p2 72h ensemble skill (daily TFT+XGB+LGBM, 27 stations).
        "ensemble_72h": ensemble_72h,
        "ensemble_72h_blends": members_72h,
        "real_overlay_n": service.state.get("real_overlay_n", 0),
    }


@router.get("/stations")
async def accuracy_stations(request: Request):
    """Per-station AQI table: current + H+24 forecast + last-step error.

    One row per configured station (~53). Columns:
    station_id, short_name, zone, data_source, history_n,
    current {pm25, pm10, aqi, category, color},
    forecast_h24 {pm25, pm10, aqi, category},
    last_step {actual, predicted, error, abs_error} or null.
    """
    service = _get_service(request)
    rows = []
    # Served-model Day+1 errors: yesterday's persisted research preds for
    # TODAY vs today's realised mean (falls back to the baseline nowcast
    # check per row when unavailable).
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    _ist = _tz(_td(hours=5, minutes=30))
    _today = _dt.now(_tz.utc).astimezone(_ist).date().isoformat()
    try:
        day1_pred_map = await service.preprocessor.get_daily_predictions_for(
            _today)
    except Exception:
        day1_pred_map = {}
    for s in service.get_stations():
        cur = s.get("current") or {}
        pol = cur.get("pollutants", {}) if cur else {}
        fc = service.get_forecast(s["id"]) or {}
        # H+24 = index 23 (0-based) when 72h horizon exists
        h24 = {}
        try:
            if fc.get("pm25") and len(fc["pm25"]) >= 24:
                h24 = {
                    "pm25": fc["pm25"][23],
                    "pm10": (fc.get("pm10") or [None])[23],
                    "aqi": (fc.get("aqi") or [None])[23],
                    "category": (fc.get("category") or [None])[23],
                }
        except Exception:
            h24 = {}
        hist = s.get("history") or []
        last_step = None
        try:
            dp = day1_pred_map.get(s["id"]) or {}
            if dp.get("pm25_pred") is not None:
                vals = []
                for hrow in hist:
                    try:
                        t = _dt.fromisoformat(
                            str(hrow.get("timestamp")).replace("Z", "+00:00"))
                        if t.tzinfo is None:
                            t = t.replace(tzinfo=_tz.utc)
                        if t.astimezone(_ist).date().isoformat() == _today \
                                and hrow.get("pm25") is not None:
                            vals.append(float(hrow["pm25"]))
                    except Exception:
                        continue
                if len(vals) >= 3:
                    actual = sum(vals) / len(vals)
                    pred = float(dp["pm25_pred"])
                    last_step = {
                        "actual": round(actual, 1),
                        "predicted": round(pred, 1),
                        "error": round(pred - actual, 1),
                        "abs_error": round(abs(pred - actual), 1),
                        "horizon_h": 24,
                        "model": "research-daily",
                    }
        except Exception as e:
            logger.debug("day1 error failed for %s: %s", s["id"], e)
        if last_step is None:
            try:
                if len(hist) >= 2 and hist[-1].get("pm25") is not None:
                    from backend.ml.xgboost_forecaster import statistical_baseline
                    actual = float(hist[-1]["pm25"])
                    prev = [h.get("pm25") for h in hist[:-1]
                            if h.get("pm25") is not None]
                    if prev:
                        h = 1
                        try:
                            from datetime import datetime as _dt
                            t1 = _dt.fromisoformat(str(hist[-1].get("timestamp")).replace("Z", "+00:00"))
                            t0 = _dt.fromisoformat(str(hist[-2].get("timestamp")).replace("Z", "+00:00"))
                            h = max(1, min(72, round((t1 - t0).total_seconds() / 3600)))
                        except Exception:
                            pass
                        base = statistical_baseline(
                            current_pm25=prev[-1], history_pm25=prev, horizon=h)
                        pred = float(base["pm25"][h - 1])
                        last_step = {
                            "actual": round(actual, 1),
                            "predicted": round(pred, 1),
                            "error": round(pred - actual, 1),
                            "abs_error": round(abs(pred - actual), 1),
                            "horizon_h": h,
                        }
            except Exception as e:
                logger.debug("last-step error failed for %s: %s", s["id"], e)
        rows.append({
            "station_id": s["id"],
            "short_name": s.get("short_name"),
            "zone": s.get("zone"),
            "city": s.get("city"),
            "latitude": s.get("latitude"),
            "longitude": s.get("longitude"),
            "data_source": (cur.get("source")
                            or service.state.get("data_source")),
            "history_n": s.get("history_count", len(hist)),
            "current": {
                "pm25": pol.get("pm25"),
                "pm10": pol.get("pm10"),
                "aqi": cur.get("aqi"),
                "category": cur.get("category"),
                "color": cur.get("color"),
                "timestamp": cur.get("timestamp"),
            } if cur else None,
            "forecast_h24": h24,
            "forecast_daily": fc.get("daily", []),
            "blend_used": fc.get("blend_used", {}),
            "provenance": fc.get("provenance", "live ensemble"),
            "last_step": last_step,
        })
    # Worst-AQI first so the table reads like a leaderboard
    rows.sort(key=lambda r: -((r.get("current") or {}).get("aqi") or -1))
    return {"count": len(rows), "stations": rows}
