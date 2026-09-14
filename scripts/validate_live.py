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
    import urllib.request
    def get(path: str):
        with urllib.request.urlopen(url.rstrip("/") + path, timeout=30) as r:
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
        fc = get("/api/v1/stations/alipur/forecast?hours=72")
        check("live station forecast (72pts + daily)",
              len(fc.get("timestamps", [])) == 72 and len(fc.get("daily", [])) == 3,
              f"{len(fc.get('timestamps', []))} pts")
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
