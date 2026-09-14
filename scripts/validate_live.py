#!/usr/bin/env python
"""
Validate the live 72h forecast feed end-to-end (file checks + live site).

  python aeros-del/scripts/validate_live.py [--url https://aeros-del.onrender.com]

File checks (no server needed):
  1. Freshness — latest forecast_aqi_72h_*.csv + index.json generated_at
     vs the 60h overlay gate (fresh / aging>36h / STALE).
  2. Schema — 25 station files x 72 pts, bands (pm25/pm10/no2/o3),
     daily x3, models flags, provenance.
  3. Integrity — 24h means == daily model values (tolerance 0.15).
  4. Skill artifacts — ensemble_skill.json (18 horizons), backtest_FINAL.csv.
  5. Drift-if-overlap — ledger forecast dates vs observed truth dates:
     RMSE per pollutant when they overlap, else "pending" (honest, not fake).

Graph-confirmation checks (same language as the dashboard footer):
  TIME  — 72 hourly monotonic timestamps per station file.
  BAND  — median inside [lower, upper] at every hour, all 4 pollutants.
  FINITE — no NaN/neg/inf, all values inside physical bounds.
  COLOR — 72 colors/categories from the AQI palette.
  DAILY — 24h means == daily model values for pm25/pm10/no2/o3/aqi.

Retention check: live SQLite stays a rolling ~30d buffer (prune active).

Live checks (--url): model-status ensemble_72h + overlay_n==25,
one station forecast (72 pts + daily x3), summary ensemble block.

Exit 1 on any FAIL. Safe to run in CI / cron before deploys.
"""

import argparse
import csv
import glob
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_ROOT = REPO_ROOT / "aeros-del"
TARGETS = ["pm25", "pm10", "no2", "o3", "no", "nox"]
# Physical plausibility caps (mirror backend preprocessor bounds).
BOUNDS = {"pm25": (0, 1500), "pm10": (0, 2000), "no2": (0, 800),
          "o3": (0, 600), "no": (0, 800), "nox": (0, 1200),
          "aqi": (0, 999)}
AQI_COLORS = {"#00e400", "#9cff9c", "#ffff00", "#ff7e00",
              "#ff0000", "#99004c", "#7e0023"}
AQI_CATS = {"Good", "Satisfactory", "Moderate", "Poor",
            "Very Poor", "Severe", "Severe+"}
RETENTION_DAYS = 30
GATE_H = 60

failures: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def latest(pattern: str) -> Path | None:
    files = sorted((REPO_ROOT / "data" / "forecasts").glob(pattern))
    return files[-1] if files else None


def file_checks() -> dict:
    out: dict = {}
    fc = latest("forecast_aqi_72h_*.csv")
    check("forecast file exists", fc is not None)
    if fc is None:
        return out
    out["forecast_file"] = fc.name
    rows = list(csv.DictReader(fc.open()))
    stations = {r["location_name"] for r in rows}
    check("27 stations x 3 horizons", len(stations) == 27 and len(rows) == 81,
          f"{len(stations)} stations, {len(rows)} rows")
    dates = sorted({r["date"] for r in rows})
    out["dates"] = dates
    idx_p = SITE_ROOT / "data" / "sample_forecasts" / "index.json"
    age_h = float("inf")
    if idx_p.exists():
        gen = json.loads(idx_p.read_text())["generated_at"]
        age_h = ((datetime.now(timezone.utc) - datetime.fromisoformat(gen))
                 .total_seconds() / 3600.0)
    status = "fresh" if age_h <= 36 else ("aging" if age_h <= GATE_H else "STALE")
    check("overlay freshness (<60h gate)", age_h <= GATE_H, f"{age_h:.1f}h old → {status}")
    out["feed_age_h"] = round(age_h, 1)

    # Schema + integrity per station file.
    sdir = SITE_ROOT / "data" / "sample_forecasts"
    files = [p for p in sdir.glob("*.json") if p.name not in ("index.json", "spatial.json")]
    check("25 station files", len(files) == 25, f"{len(files)} found")
    bad_schema = bad_mean = 0
    need_bands = ["lower", "upper", "pm10_lower", "pm10_upper",
                  "no2_lower", "no2_upper", "o3_lower", "o3_upper"]
    for p in files:
        d = json.loads(p.read_text())
        if not (len(d.get("timestamps", [])) == 72 and len(d.get("pm25", [])) == 72
                and len(d.get("daily", [])) == 3
                and all(k in d and len(d[k]) == 72 for k in need_bands)
                and d.get("models", {}).get("baseline") is False):
            bad_schema += 1
            continue
        for i, day in enumerate(d["daily"]):
            seg = d["pm25"][i * 24:(i + 1) * 24]
            if abs(sum(seg) / 24 - day["pm25"]) > 0.15:
                bad_mean += 1
    check("schema (72pts, bands x4, daily x3, real models)", bad_schema == 0,
          f"{bad_schema} bad" if bad_schema else f"{len(files)} files")
    check("24h means == daily model values", bad_mean == 0,
          f"{bad_mean} drifted days" if bad_mean else "exact")

    # Skill artifacts.
    sk_p = SITE_ROOT / "scripts" / "ensemble_skill.json"
    sk = json.loads(sk_p.read_text()) if sk_p.exists() else {}
    check("ensemble_skill.json (18 horizons)", len(sk.get("per_horizon_backtest", [])) == 18)
    bt = REPO_ROOT / "data" / "forecasts" / "backtest_FINAL.csv"
    check("backtest_FINAL.csv present", bt.exists())
    out["skill"] = sk
    return out


def graph_checks() -> None:
    """Graph-confirmation audit: what the browser plots must be honest.

    Same language as the dashboard footer (GRAPH CHECK — TIME/BAND/DAILY).
    For every station JSON: 72 hourly monotonic timestamps, finite values
    inside physical bounds, uncertainty bands containing the median,
    72 colors/categories present and from the AQI palette, and 24h means
    == daily model values for pm25/pm10/no2/o3/aqi (tol 0.15).
    """
    sdir = SITE_ROOT / "data" / "sample_forecasts"
    files = [p for p in sdir.glob("*.json")
             if p.name not in ("index.json", "spatial.json")]
    if not files:
        check("graph payloads present", False, "no station files")
        return
    bad_time = bad_finite = bad_band = bad_color = bad_daily = 0
    band_pairs = {"pm25": ("lower", "upper"), "pm10": ("pm10_lower", "pm10_upper"),
                  "no2": ("no2_lower", "no2_upper"), "o3": ("o3_lower", "o3_upper")}
    day_keys = ["pm25", "pm10", "no2", "o3", "aqi"]
    for p in files:
        try:
            d = json.loads(p.read_text())
        except Exception:
            bad_finite += 1
            continue
        # TIME: 72 hourly monotonic timestamps.
        try:
            ts = [datetime.fromisoformat(str(t)) for t in d.get("timestamps", [])]
            hourly = (len(ts) == 72 and all(
                abs((ts[i] - ts[i - 1]).total_seconds() / 3600 - 1) < 0.05
                for i in range(1, len(ts))))
        except Exception:
            hourly = False
        if not hourly:
            bad_time += 1
        # FINITE + BOUNDS for every plotted pollutant + aqi.
        for k in list(band_pairs) + ["aqi"]:
            lo_b, hi_b = BOUNDS[k]
            for v in (d.get(k) or []):
                if (v is None or not isinstance(v, (int, float))
                        or not math.isfinite(v) or not (lo_b <= v <= hi_b)):
                    bad_finite += 1
                    break
        # BAND: median inside [lower, upper] at every hour.
        for k, (lo_k, hi_k) in band_pairs.items():
            base, lo, hi = d.get(k) or [], d.get(lo_k) or [], d.get(hi_k) or []
            if not (len(base) == len(lo) == len(hi) == 72):
                bad_band += 1
                break
            if any(not (l <= v <= h)
                   for v, l, h in zip(base, lo, hi)):
                bad_band += 1
                break
        # COLOR/CATEGORY: 72 entries from the AQI vocabulary.
        cols, cats = d.get("colors") or [], d.get("category") or []
        if not (len(cols) == len(cats) == 72
                and all(str(c).lower() in AQI_COLORS for c in cols)
                and all(c in AQI_CATS for c in cats)):
            bad_color += 1
        # DAILY: 24h means == daily model values (linear pollutants), and
        # daily AQI inside the day's hourly AQI envelope (AQI is nonlinear:
        # mean(hourly AQI) != AQI(daily-mean PM) by construction, so an
        # equality check there would be a validator bug, not a data bug).
        for i, day in enumerate((d.get("daily") or [])[:3]):
            for k in day_keys:
                seg = (d.get(k) or [])[i * 24:(i + 1) * 24]
                want = day.get(k)
                if len(seg) != 24 or want is None:
                    continue
                if k == "aqi":
                    if not (min(seg) - 1e-9 <= want <= max(seg) + 1e-9):
                        bad_daily += 1
                        break
                elif abs(sum(seg) / 24 - want) > 0.15:
                    bad_daily += 1
                    break
            else:
                continue
            break
    check("graph TIME (72 hourly monotonic)", bad_time == 0,
          f"{bad_time} bad" if bad_time else f"{len(files)} files")
    check("graph FINITE (bounds, no NaN/neg)", bad_finite == 0,
          f"{bad_finite} bad" if bad_finite else "all in bounds")
    check("graph BAND (median inside envelope)", bad_band == 0,
          f"{bad_band} breached" if bad_band else "72/72 hours inside")
    check("graph COLOR (AQI palette x72)", bad_color == 0,
          f"{bad_color} bad" if bad_color else "palette-clean")
    check("graph DAILY (24h means == daily, 5 series)", bad_daily == 0,
          f"{bad_daily} drifted" if bad_daily else "exact")


def retention_check() -> None:
    """Retention prune audit: live SQLite stays a rolling ~30d buffer."""
    import sqlite3
    db = SITE_ROOT / "data" / "aqi_data.db"
    if not db.exists() or db.stat().st_size == 0:
        print("  [ -- ] retention: live DB empty (fresh disk) — prune pending, not failed")
        return
    try:
        con = sqlite3.connect(str(db))
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "station_readings" not in tables:
            print("  [ -- ] retention: tables not created yet — prune pending, not failed")
            con.close()
            return
        n = con.execute("SELECT COUNT(*) FROM station_readings").fetchone()[0]
        row = con.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM station_readings").fetchone()
        con.close()
    except Exception as e:
        check("retention DB readable", False, str(e)[:100])
        return
    check("retention DB readable", True, f"{n} readings")
    try:
        span_d = (datetime.fromisoformat(str(row[1]).replace("Z", "+00:00"))
                  - datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
                  ).total_seconds() / 86400.0 if row[0] and row[1] else 0.0
    except Exception:
        span_d = 0.0
    check(f"retention span <={RETENTION_DAYS + 15}d (prune active)",
          span_d <= RETENTION_DAYS + 15, f"{span_d:.1f}d span")


def drift_check() -> None:
    """Score ledger rows against observed truth where dates overlap."""
    truth_p = REPO_ROOT / "data" / "processed" / "delhi_final_training_data.csv"
    led_p = REPO_ROOT / "data" / "forecasts" / "forecast_history.csv"
    if not (truth_p.exists() and led_p.exists()):
        print("  [ -- ] drift: ledger or truth missing, skipping")
        return
    import collections
    truth: dict = collections.defaultdict(dict)
    with truth_p.open() as fh:
        for r in csv.DictReader(fh):
            for t in TARGETS:
                try:
                    v = float(r[t])
                    if not math.isnan(v):
                        truth[(r["location_name"], r["date"])][t] = v
                except (ValueError, KeyError):
                    pass
    errs: dict = collections.defaultdict(lambda: [0.0, 0])
    n_overlap = 0
    with led_p.open() as fh:
        for r in csv.DictReader(fh):
            key = (r.get("location_name", ""), r.get("date", ""))
            obs = truth.get(key)
            if not obs:
                continue
            n_overlap += 1
            for t in TARGETS:
                try:
                    pv, av = float(r[t]), obs[t]
                    errs[t][0] += (pv - av) ** 2
                    errs[t][1] += 1
                except (ValueError, KeyError):
                    pass
    if not n_overlap:
        tmax = max(truth, key=lambda k: k[1])[1] if truth else "?"
        print(f"  [ -- ] drift: no ledger/truth date overlap yet "
              f"(truth ends {tmax}) — validation pending, not failed")
        return
    print(f"  [OK ] drift over {n_overlap} overlapping rows:")
    for t in TARGETS:
        s, n = errs[t]
        if n:
            print(f"         {t}: RMSE={math.sqrt(s / n):.2f} (n={n})")


def live_checks(url: str) -> None:
    import ssl
    import urllib.request
    # Some sandboxes lack CA bundles (curl works, python doesn't). Fall
    # back to explicitly-unverified TLS rather than failing the check.
    try:
        urllib.request.urlopen(url.rstrip("/") + "/api/v1/health", timeout=15).read()
        opener = urllib.request.build_opener()
    except Exception as e:
        if "CERTIFICATE" in str(e).upper() or "SSL" in str(e).upper():
            print("  [ -- ] system CA bundle missing; live checks use explicitly-unverified TLS")
            ctx = ssl._create_unverified_context()
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx))
        else:
            raise
    def get(path: str):
        with opener.open(url.rstrip("/") + path, timeout=30) as r:
            return json.loads(r.read().decode())
    try:
        ms = get("/api/v1/accuracy/model-status")
    except Exception as e:
        check("live model-status reachable", False, str(e)[:100])
        return
    check("live model-status reachable", True)
    check("live ensemble_72h served",
          bool((ms.get("ensemble_72h") or {}).get("per_horizon_backtest")))
    check("live overlay on 25 stations", ms.get("real_overlay_n") == 25,
          f"overlay_n={ms.get('real_overlay_n')}")
    try:
        stations = get("/api/v1/stations")
        sids = [s.get("id") for s in stations.get("stations", []) if s.get("id")]
        sid = "alipur" if "alipur" in sids else (sids[0] if sids else None)
        check("live stations list non-empty", bool(sids), f"{len(sids)} stations")
    except Exception as e:
        check("live stations list", False, str(e)[:100])
        sid = None
    try:
        if sid is None:
            raise RuntimeError("no station id to probe")
        fc = get(f"/api/v1/stations/{sid}/forecast?hours=72")
        check("live station forecast (72pts + daily)",
              len(fc.get("timestamps", [])) == 72 and len(fc.get("daily", [])) == 3,
              f"{sid}: {len(fc.get('timestamps', []))} pts")
    except Exception as e:
        check("live station forecast", False, str(e)[:100])
    try:
        sm = get("/api/v1/accuracy/summary")
        check("live summary carries ensemble skill",
              bool((sm.get("ensemble_72h") or {}).get("aqi_skill")))
    except Exception as e:
        check("live summary", False, str(e)[:100])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    args = ap.parse_args()
    print("== file checks ==")
    file_checks()
    print("== graph confirmation (what the browser plots) ==")
    graph_checks()
    print("== retention prune ==")
    retention_check()
    print("== drift (ledger vs truth) ==")
    drift_check()
    if args.url:
        print("== live checks ==")
        live_checks(args.url)
    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILING — {failures}")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
