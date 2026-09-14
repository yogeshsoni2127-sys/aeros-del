"""
Forecast Routes — Spatial grid, forecast trigger
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["forecast"])


def _get_service(request: Request):
    return request.app.state.service


@router.get("/forecast/spatial")
async def spatial_forecast(request: Request):
    """Gridded spatial forecast as a GeoJSON FeatureCollection."""
    service = _get_service(request)
    spatial = service.get_spatial()
    return {
        "type": "FeatureCollection",
        "features": spatial.get("features", []),
        "generated_at": service.state.get("last_update"),
    }


@router.get("/aisi/current")
async def current_aisi(request: Request):
    """Current AISI value, strength, trend and GRAP recommendation."""
    service = _get_service(request)
    return service.get_aisi()


@router.get("/aisi/verify")
async def verify_aisi(request: Request):
    """Cross-check the live inversion against an independent sounding.

    Returns model gradient/PBL/AISI plus the latest Wyoming VIDP
    (Delhi 42182) observed lowest-100m gradient when reachable
    (best-effort 8s fetch; sounding=null offline with manual links).
    Same tolerance logic as scripts/verify_aisi.py — run that script
    for the CLI verdict.
    """
    import sys
    service = _get_service(request)
    model = service.get_aisi()
    st = (model.get("sub_terms") or {})
    out = {
        "model": {
            "aisi": model.get("aisi"),
            "category": model.get("category"),
            "gradient_k_per_100m": st.get("temp_gradient_k_per_100m"),
            "pbl_height_m": (model.get("pbl") or {}).get("pbl_height_m"),
            "ri_bulk": st.get("ri_bulk"),
            "generated": service.state.get("last_update"),
        },
        "tolerance": {"gradient_k_per_100m": 1.5},
        "references": {
            "wyoming_sounding": ("https://weather.uwyo.edu/upperair/"
                                 "sounding.html (seasia, 42182 VIDP)"),
            "imd_radiosonde": ("https://ddgmui.imd.gov.in/ual2/"
                               "LastDataAscent.php"),
            "windy_ecmwf": "https://www.windy.com (surface vs 950/900hPa)",
            "script": "python scripts/verify_aisi.py --server <base>",
        },
        "sounding": None,
        "verdict": "unknown",
    }
    try:
        from backend.app.config import PROJECT_ROOT as _ROOT
        sys.path.insert(0, str(_ROOT))
        from scripts.verify_aisi import fetch_sounding, lowest_100m_gradient
        import asyncio as _aio
        # Short per-cycle timeout: first (latest) cycle usually hits;
        # the endpoint must stay fast even when Wyoming is slow.
        sonde = await _aio.wait_for(
            _aio.to_thread(fetch_sounding, 5), timeout=22)
        obs = lowest_100m_gradient(sonde["levels"])
        out["sounding"] = {"cycle": sonde.get("cycle"), **obs}
        mg = st.get("temp_gradient_k_per_100m")
        if mg is not None:
            d = abs(float(mg) - obs["gradient_k_per_100m"])
            sign_ok = (float(mg) > 0) == obs["inversion"]
            out["delta_gradient"] = round(d, 2)
            out["sign_agreement"] = sign_ok
            out["verdict"] = ("agree" if (d <= 1.5 and sign_ok)
                              else "disagree")
    except Exception as e:
        logger.debug("aisi verify sounding fetch failed: %s", e)
        out["sounding_error"] = "sounding unreachable — use manual links"
    return out


@router.get("/radiation/current")
async def current_radiation(request: Request):
    """Aerosol-radiation feedback diagnostics."""
    service = _get_service(request)
    return service.get_radiation()


@router.post("/forecast/trigger")
async def trigger_forecast(request: Request):
    """
    Manually trigger a full refresh cycle (data → physics → ML → NLP).

    Optionally passes `force=false` to respect the polling interval.
    """
    service = _get_service(request)
    force = (request.query_params.get("force", "true").lower() != "false")
    result = await service.refresh(force=force)
    if not result.get("refreshed"):
        raise HTTPException(status_code=409, detail=result)
    return result


@router.post("/forecasts/overlay")
async def ingest_overlay(request: Request):
    """Ingest a daily TFT-overlay bundle (CI/local runner).

    Body: {"generated_at": iso, "stations": {id: forecast_payload}}.
    Auth: X-Ingest-Key header must equal INGEST_KEY env (503 when the
    server has none configured). Each payload is validated exactly
    like frozen overlay files, persisted to data/sample_forecasts/
    (ingest_<id>.json + index.json refresh) and hot-applied to live
    state — no restart, no DB wipe. Fresh TFT overlays win over local
    tree-only research per the precedence rule; stale ones (>60h) are
    rejected at the door.
    """
    import hmac
    import json
    from datetime import datetime, timezone

    service = _get_service(request)
    want = (service.settings.ingest_key or "").strip()
    if not want:
        raise HTTPException(status_code=503,
                            detail="overlay ingest not configured")
    got = (request.headers.get("x-ingest-key") or "").strip()
    if not hmac.compare_digest(got, want):
        raise HTTPException(status_code=401, detail="bad ingest key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    generated_at = str(body.get("generated_at") or "")
    try:
        gen = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        if gen.tzinfo is None:
            gen = gen.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - gen).total_seconds() / 3600.0
    except (ValueError, TypeError):
        raise HTTPException(status_code=400,
                            detail="generated_at must be ISO8601")
    if not (0 <= age_h <= 60):
        raise HTTPException(status_code=422,
                            detail=f"bundle age {age_h:.1f}h outside 0-60h")
    stations = body.get("stations") or {}
    if not isinstance(stations, dict) or not stations:
        raise HTTPException(status_code=400,
                            detail="stations must be a non-empty object")

    accepted, rejected = [], []
    shaped = {}
    for sid, payload in stations.items():
        if not isinstance(payload, dict):
            rejected.append(sid)
            continue
        payload = {**payload, "station_id": sid,
                   "generated_at": generated_at}
        out = service._shape_overlay_forecast(payload)
        if out is None or not (out.get("models") or {}).get("tft"):
            rejected.append(sid)
            continue
        shaped[sid] = out
        accepted.append(sid)
    if not shaped:
        raise HTTPException(status_code=422,
                            detail={"accepted": [], "rejected": rejected})

    # Persist alongside frozen overlay files (same reader, same gate).
    try:
        from backend.app.config import DATA_DIR
        dest = DATA_DIR / "sample_forecasts"
        dest.mkdir(parents=True, exist_ok=True)
        names = []
        for sid, payload in shaped.items():
            fname = f"ingest_{sid}.json"
            (dest / fname).write_text(json.dumps(payload))
            names.append(fname)
        idx_path = dest / "index.json"
        try:
            idx = json.loads(idx_path.read_text())
        except Exception:
            idx = {}
        prior = [f for f in (idx.get("station_forecasts") or [])
                 if not str(f).startswith("ingest_")]
        idx["station_forecasts"] = prior + sorted(names)
        idx["generated_at"] = generated_at
        idx["description"] = (
            "TFT overlay bundles (ingested, win while fresh) + "
            "frozen exporter files")
        idx_path.write_text(json.dumps(idx, indent=1))
    except Exception as e:
        logger.warning("overlay persist failed (hot-apply continues): %s", e)

    applied = 0
    for sid, payload in shaped.items():
        service.state.setdefault("forecasts", {})[sid] = payload
        applied += 1
    service.state["real_overlay_n"] = len([
        f for f in service.state.get("forecasts", {}).values()
        if service._overlay_is_fresh_tft(f)])
    logger.info("tft-overlay ingested: %d stations (%d rejected)",
                applied, len(rejected))
    return {"ingested": applied, "rejected": rejected,
            "generated_at": generated_at}