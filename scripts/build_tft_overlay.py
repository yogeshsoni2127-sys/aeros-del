"""Build a TFT-overlay bundle WITHOUT the running server (CI runner).

Runs the full 3-member research inference (XGB+LGBM+TFT, CSV-trained)
seeded with live data, and emits an overlay-format bundle for
POST /api/v1/forecasts/overlay — or a local file with --out.

Live inputs (no Mac, no server needed):
- currents: CPCB data.gov.in snapshot (DATAGOV_API_KEY env)
- tails: OpenAQ per-sensor 14d history (WAQI_API_KEY not needed;
  OpenAQ key optional via OPENAQ_API_KEY env)
- weather: Open-Meteo hourly forecast (keyless)
- fires: seed trailing means (documented; same as the server path)

Usage:
    DATAGOV_API_KEY=... python scripts/build_tft_overlay.py --out bundle.json
    DATAGOV_API_KEY=... python scripts/build_tft_overlay.py \\
        --post https://aeros-del.onrender.com --key $INGEST_KEY

Self-gate (nothing is sent on failure): DAILY-exact per station
(block means == daily values, tol 0.15 — same as the chart verifier),
H+24 present, >= 20 stations applied with TFT active on >= 15.
Exit 0 sent/built-ok, 1 gate or fetch failure.
"""
import argparse
import asyncio
import json
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")
try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    pass

IST = timezone(timedelta(hours=5, minutes=30))
POLS = ("pm25", "pm10", "no2", "o3")


def log(msg):
    print(f"[overlay-build] {msg}", flush=True)


def _haversine(lat1, lon1, lat2, lon2):
    import math
    r = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def fetch_currents(svc):
    """Live CPCB currents mapped to runtime station ids."""
    from backend.data.cpcb_datagov_client import CPCBDataGovClient
    client = CPCBDataGovClient(api_key=os.getenv("DATAGOV_API_KEY"))
    try:
        readings = await client.get_latest_measurements()
    finally:
        await client.close()
    if not readings:
        return {}
    # Reuse the service matcher against stations.json metadata.
    svc.stations = svc._load_stations()
    mapped = svc._match_cpcb_readings(readings)
    out = {}
    for sid, entry in mapped.items():
        age = svc._age_hours(entry.get("timestamp"))
        if age is not None and age <= svc.FRESHNESS_MAX_AGE_H:
            out[sid] = entry["pollutants"]
    log(f"CPCB currents: {len(out)} fresh stations")
    return out


async def fetch_history_tails(svc, stations, days=14):
    """Trailing daily IST means per station via OpenAQ sensor history."""
    from backend.data.openaq_client import OpenAQClient
    client = OpenAQClient(api_key=os.getenv("OPENAQ_API_KEY") or None)
    try:
        locs = await client.get_locations_in_delhi()
    except Exception as e:
        log(f"OpenAQ locations failed: {e}")
        await client.close()
        return {}
    # Nearest PM-measuring location per runtime station.
    targets = {}
    for s in stations:
        best, best_d = None, 25.0
        for loc in locs:
            params = loc.get("parameters") or []
            if not any(p in ("pm25", "pm10") for p in params):
                continue
            if loc.get("latitude") is None:
                continue
            d = _haversine(s["latitude"], s["longitude"],
                           loc["latitude"], loc["longitude"])
            if d < best_d:
                best, best_d = loc, d
        if best is not None:
            targets[s["id"]] = best
    log(f"history: {len(targets)}/{len(stations)} stations near OpenAQ monitors")
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    tails = {}
    for sid, loc in targets.items():
        sens = {}
        for sensor in loc.get("sensors") or []:
            p = ((sensor.get("parameter") or {}).get("name", "") or "").lower()
            if p in POLS and sensor.get("id") is not None \
                    and p not in sens:
                sens[p] = sensor["id"]
        buckets = defaultdict(lambda: defaultdict(list))
        for pol, sensor_id in sens.items():
            try:
                rows = await client.get_sensor_measurements(
                    sensor_id, start, end, limit=1000, max_pages=10)
            except Exception as e:
                log(f"  {sid} {pol}: history failed ({e})")
                continue
            for m in rows:
                try:
                    dt = datetime.fromisoformat(
                        str(m["utc"]).replace("Z", "+00:00"))
                    day = dt.astimezone(IST).date().isoformat()
                    buckets[day][pol].append(float(m["value"]))
                except (ValueError, TypeError, KeyError):
                    continue
        tails[sid] = {
            day: {p: {"mean": round(sum(v) / len(v), 1), "n": len(v)}
                  for p, v in pols.items() if v}
            for day, pols in buckets.items()
        }
    try:
        await client.close()
    except Exception:
        pass
    return tails


async def fetch_weather(svc):
    from backend.data.weather_client import WeatherClient
    client = WeatherClient()
    try:
        current = await client.get_current_weather(
            svc.settings.delhi_center_lat, svc.settings.delhi_center_lon)
        hourly = await client.get_hourly_forecast(
            svc.settings.delhi_center_lat, svc.settings.delhi_center_lon,
            forecast_days=2)
    finally:
        try:
            await client.close()
        except Exception:
            pass
    return current or {}, hourly or {}


def gate_bundle(payloads):
    """DAILY-exact + H+24 + TFT coverage. Returns (ok, report)."""
    applied = len(payloads)
    tft_n = sum(1 for p in payloads.values()
                if (p.get("models") or {}).get("tft"))
    bad_daily, missing_h24 = [], []
    for sid, fc in payloads.items():
        try:
            for i, day in enumerate((fc.get("daily") or [])[:3]):
                seg = (fc.get("pm25") or [])[i * 24:(i + 1) * 24]
                want = day.get("pm25")
                if len(seg) != 24 or want is None:
                    bad_daily.append(sid)
                    break
                if abs(sum(seg) / 24 - want) > 0.15:
                    bad_daily.append(sid)
                    break
            if len(fc.get("pm25") or []) < 24:
                missing_h24.append(sid)
        except Exception:
            bad_daily.append(sid)
    ok = (applied >= 20 and tft_n >= 15
          and not bad_daily and not missing_h24)
    return ok, {"applied": applied, "tft": tft_n, "bad_daily": bad_daily,
                "missing_h24": missing_h24}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--bundle-in", default=None,
                    help="skip inference; POST this bundle file as-is")
    ap.add_argument("--post", default=None,
                    help="base URL, e.g. https://aeros-del.onrender.com")
    ap.add_argument("--key", default=None,
                    help="ingest key (or INGEST_KEY env)")
    args = ap.parse_args()

    if args.bundle_in and args.post and not args.out:
        # Send-only mode (CI step 2): no inference, just deliver.
        key = (args.key or os.getenv("INGEST_KEY") or "").strip()
        if not key:
            log("missing ingest key (--key or INGEST_KEY)")
            return 1
        try:
            with open(args.bundle_in, encoding="utf-8") as f:
                bundle = json.load(f)
            log(f"loaded bundle {args.bundle_in} "
                f"({len(bundle.get('stations', {}))} stations)")
        except Exception as e:
            log(f"cannot read bundle: {e}")
            return 1
        return _post_bundle(args.post, bundle, key)

    from backend.app.service import AQIService
    svc = AQIService()
    if not getattr(svc, "research_daily", None) \
            or not svc.research_daily.available:
        log("research bridge unavailable (models missing?)")
        return 1

    currents = await fetch_currents(svc)
    if len(currents) < 5:
        log(f"only {len(currents)} live stations — refusing to build")
        return 1
    stations = [s for s in svc._load_stations() if s["id"] in currents]
    for s in stations:
        s["current"] = {"pollutants": currents[s["id"]]}

    live_tails = await fetch_history_tails(svc, stations)
    _, hourly = await fetch_weather(svc)
    svc.state["weather_hourly"] = hourly
    today = datetime.now(timezone.utc).astimezone(IST).date()
    horizon_dates = [today + timedelta(days=h) for h in (1, 2, 3)]
    wx_days = svc._wx_days_for_research(horizon_dates)

    # Attach TFT locally (CI runners have RAM; same auto-gate applies).
    try:
        from backend.ml.tft_daily import TFTDailyEnsemble
        from backend.app.config import MODELS_DIR
        tft = TFTDailyEnsemble(str(MODELS_DIR / "research" / "tft"))
        svc.research_daily.attach_tft(tft if tft.load() else None)
    except Exception as e:
        log(f"TFT attach failed: {e}")

    inputs = svc._research_station_inputs(
        stations, live_tails, today, horizon_dates, wx_days)
    log(f"domain inputs: {len(inputs)} stations")
    try:
        results = svc.research_daily.predict_domain(inputs, horizon_dates)
    except Exception as e:
        log(f"domain inference failed: {e}")
        return 1

    generated_at = datetime.now(timezone.utc).isoformat()
    payloads = {}
    for sid, out in results.items():
        if not (out["preds"].get("pm25") or []) \
                or len(out["preds"]["pm25"]) < 3:
            continue
        wrapped = svc._wrap_research_payload(
            sid, out, horizon_dates, inputs[sid].get("tail_src", "seed"),
            provenance="local-tft 3-model (CI runner, live-seeded) "
                       "+ diurnal downscale")
        if wrapped is None:
            continue
        wrapped["station_id"] = sid
        wrapped["generated_at"] = generated_at
        payloads[sid] = wrapped

    ok, report = gate_bundle(payloads)
    log(f"gate: {report} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        return 1

    bundle = {"generated_at": generated_at, "stations": payloads}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(bundle, f)
        log(f"wrote {args.out} ({len(json.dumps(bundle)) // 1024} KB)")
    if args.post and not args.bundle_in:
        key = (args.key or os.getenv("INGEST_KEY") or "").strip()
        if not key:
            log("missing ingest key (--key or INGEST_KEY)")
            return 1
        return _post_bundle(args.post, bundle, key)
    try:
        await svc.close()
    except Exception:
        pass
    return 0


def _post_bundle(base_url, bundle, key):
    import urllib.request
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/v1/forecasts/overlay",
        data=json.dumps(bundle).encode(),
        headers={"Content-Type": "application/json",
                 "X-Ingest-Key": key},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            log(f"POST -> {resp.status}: {resp.read()[:200]}")
        return 0
    except Exception as e:
        log(f"POST failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
