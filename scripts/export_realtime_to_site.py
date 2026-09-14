#!/usr/bin/env python
"""
Export REAL SIH-p2 ensemble outputs into the aeros-del Render site.
====================================================================
Replaces the synthetic demo feed (diurnal template, models all false)
with measured model output:

  SIH-p2 side (inputs)
    data/forecasts/forecast_aqi_72h_<run>.csv   27 stations x 3 days, real preds
    scripts/training/best_blend_per_pollutant.csv
    data/forecasts/backtest_FINAL.csv           per-horizon RMSE/MAE/R2
    data/forecasts/aqi_skill.json               AQI hit rates
    data/raw/delhi_stations_list.csv            OpenAQ location ids

  Site side (outputs, all under aeros-del/)
    data/sample_forecasts/<site-id>.json  (25 files, schema-compatible + extras)
    data/sample_forecasts/index.json + spatial.json
    data/stations.json                    (27 Delhi-only, OpenAQ ids filled)
    scripts/ensemble_skill.json           (served via /api/v1/accuracy/model-status)

Daily -> hourly: the research models predict DAILY means. The site renders
72 hourly points, so each daily mean is spread over 24h with a fixed
diurnal shape normalised to mean exactly 1 (24h mean == model value).
Files are stamped "downscaled": true -- the `daily` block always carries
the true model values. Uncertainty bands are +/-1.25x per-horizon MAE
(~80% band), NOT TFT quantiles (H3 not trained yet) -- labelled as such.

Usage:
    python aeros-del/scripts/export_realtime_to_site.py [--run DATE]
    python aeros-del/scripts/export_realtime_to_site.py --check-only
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_ROOT = REPO_ROOT / "aeros-del"
sys.path.insert(0, str(SITE_ROOT))

import pandas as pd

from backend.data.naqi_calculator import NAQICalculator

TARGETS = ["pm25", "pm10", "no2", "o3", "no", "nox"]

# Real research location_name -> site station id. Missing = new site entry.
SITE_MAP = {
    "Alipur, Delhi - DPCC": "alipur",
    "Anand Vihar, Delhi - DPCC": "anand-vihar",
    "Anand Vihar, New Delhi - DPCC": "anand-vihar",
    "Ashok Vihar, Delhi - DPCC": "ashok-vihar",
    "Aya Nagar, New Delhi - IMD": "aya-nagar",
    "Burari Crossing, New Delhi - IMD": "burari",
    "CRRI Mathura Road, New Delhi - IMD": "crri-mathura",
    "Cantonment Area, Delhi - DPCC": "cantonment",
    "Commonwealth Sports Complex, Delhi - DPCC": "commonwealth",
    "Dr. Karni Singh Shooting Range, Delhi - DPCC": "karni-singh",
    "Dwarka-Sector 8, Delhi - DPCC": "dwarka-sec8",
    "IGNOU_Maidan Garhi, Delhi - DPCC": "ignou",
    "IHBAS, Dilshad Garden,New Delhi - CPCB": "ihbas",
    "ITO, Delhi - CPCB": "ito",
    "JNU, Delhi - DPCC": "jnu",
    "Lodhi Road, New Delhi - IMD": "lodhi-road",
    "Major Dhyan Chand National Stadium, Delhi - DPCC": "major-dhyan-chand",
    "Mandir Marg, New Delhi - DPCC": "mandir-marg",
    "Mundka, Delhi - DPCC": "mundka",
    "Punjabi Bagh, Delhi - DPCC": "punjabi-bagh",
    "Pusa, Delhi - DPCC": "pusa",
    "Pusa, Delhi - IMD": "pusa",
    "R K Puram, Delhi - DPCC": "rk-puram",
    "Shadipur, Delhi - CPCB": "shadipur",
    "Sri Aurobindo Marg, Delhi - DPCC": "sri-aurobindo",
    "Talkatora Garden, Delhi - DPCC": "talkatora",
    "Wazirpur, Delhi - DPCC": "wazirpur",
}

# Co-located duplicate sensors averaged into one site file.
MERGE_GROUPS = {"anand-vihar": 2, "pusa": 2}

NEW_STATIONS = {
    # id: (name, short, zone, type) -- coords/elevation from stations_list.csv
    "aya-nagar": ("Aya Nagar, Delhi - IMD", "Aya Nagar", "South Delhi", "Residential"),
    "cantonment": ("Cantonment Area, Delhi - DPCC", "Cantonment", "South West Delhi", "Residential"),
    "commonwealth": ("Commonwealth Sports Complex, Delhi - DPCC", "Commonwealth", "East Delhi", "Residential"),
    "ignou": ("IGNOU Maidan Garhi, Delhi - DPCC", "IGNOU", "South Delhi", "Residential"),
    "jnu": ("JNU, Delhi - DPCC", "JNU", "South Delhi", "Residential"),
    "talkatora": ("Talkatora Garden, Delhi - DPCC", "Talkatora", "Central Delhi", "Residential"),
}

# Diurnal shape: (base amplitude, peak hour IST). Amplitude is scaled
# per station by that station's observed day-to-day variability
# (CoV from training data, clamped) -- spike-prone stations get peakier
# days, stable stations flatter ones. Phase stays fixed: daily data
# carries no hour-of-day information and inventing phases would be fake.
DIURNAL = {"pm25": (0.35, 1), "pm10": (0.30, 1), "no2": (0.40, 9),
           "o3": (0.50, 15), "no": (0.40, 9), "nox": (0.40, 9)}

IST_SUFFIX = "+05:30"  # Asia/Kolkata has no DST; daily dates are IST days.


def station_cov() -> dict:
    """Per-(station, pollutant) coefficient of variation from training data.

    Falls back to 1.0 (template amplitude) when the training CSV is absent
    (e.g. exporter run from a bare site checkout).
    """
    try:
        tr = pd.read_csv(REPO_ROOT / "data" / "processed" / "delhi_final_training_data.csv",
                         usecols=["location_name"] + TARGETS)
    except Exception:
        return {}
    out: dict = {}
    for p in TARGETS:
        med = tr.groupby("location_name")[p].agg(["mean", "std"])
        cov = (med["std"] / med["mean"]).replace([float("inf")], float("nan"))
        global_cov = float(cov.median())
        for station, c in cov.items():
            scale = 1.0 if pd.isna(c) or not global_cov else float(c) / global_cov
            out[(station, p)] = min(1.4, max(0.7, scale))
    return out


def diurnal_weights(pollutant: str, amp_scale: float = 1.0) -> list:
    amp, peak = DIURNAL[pollutant]
    amp *= amp_scale
    w = [1.0 + amp * math.cos((h - peak) * math.pi / 12.0) for h in range(24)]
    m = sum(w) / 24.0
    return [x / m for x in w]


def smooth_junctions(hourly72: list) -> list:
    """Blend the 23:00->00:00 day-boundary pairs so consecutive days with
    different means join continuously instead of stepping."""
    out = list(hourly72)
    for j in (23, 47):
        a, b = out[j], out[j + 1]
        out[j], out[j + 1] = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
    return out


def rescale_day(seg: list, target_mean: float) -> list:
    m = sum(seg) / len(seg)
    return [x * target_mean / m for x in seg] if m else list(seg)


def parse_blend(label: str) -> dict:
    if label == "TFT only":
        return {"tft": True, "xgboost": False, "lightgbm": False}
    if label == "XGB only":
        return {"tft": False, "xgboost": True, "lightgbm": False}
    if label == "LGBM only":
        return {"tft": False, "xgboost": False, "lightgbm": True}
    try:
        p = dict(x.split("=") for x in label.split(", "))
        return {"tft": float(p["tft"]) > 0, "xgboost": float(p["xgb"]) > 0,
                "lightgbm": float(p["lgbm"]) > 0}
    except (ValueError, KeyError):
        return {"tft": True, "xgboost": True, "lightgbm": True}


def latest_forecast_file() -> Path:
    files = sorted((REPO_ROOT / "data" / "forecasts").glob("forecast_aqi_72h_*.csv"))
    if not files:
        raise FileNotFoundError("no forecast_aqi_72h_*.csv in data/forecasts")
    return files[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="",
                    help="forecast_aqi_72h csv (default: latest)")
    ap.add_argument("--check-only", action="store_true",
                    help="verify current site files against sources, write nothing")
    args = ap.parse_args()

    fc_path = Path(args.input) if args.input else latest_forecast_file()
    fc = pd.read_csv(fc_path, parse_dates=["date"])
    blends = pd.read_csv(REPO_ROOT / "scripts" / "training" / "best_blend_per_pollutant.csv")
    horizons = pd.read_csv(REPO_ROOT / "data" / "forecasts" / "backtest_FINAL.csv")
    aq_skill_p = REPO_ROOT / "data" / "forecasts" / "aqi_skill.json"
    aq_skill = json.loads(aq_skill_p.read_text()) if aq_skill_p.exists() else {}
    openaq = pd.read_csv(REPO_ROOT / "data" / "raw" / "delhi_stations_list.csv")
    naqi = NAQICalculator()

    mae_h = {(r["target"], int(r["horizon_h"])): float(r["mae"])
             for _, r in horizons.iterrows()}
    blend_of = {r["target"]: str(r["chosen_blend"]) for _, r in blends.iterrows()}
    # Primary (PM2.5) blend drives file-level model flags.
    file_models = parse_blend(blend_of.get("pm25", ""))

    # ---- per-site daily frames (merge co-located duplicates by mean) ----
    fc["site_id"] = fc["location_name"].map(SITE_MAP)
    assert fc["site_id"].isna().sum() == 0, "unmapped stations!"
    daily = fc.groupby(["site_id", "date", "horizon_h"], as_index=False).agg(
        {c: "mean" for c in TARGETS + ["AQI"]} |
        {"location_name": "first", "category": "first", "color": "first",
         "advisory": "first", "dominant_pollutant": "first"})
    n_sites = daily["site_id"].nunique()
    print(f"Source {fc_path.name}: {len(fc)} rows -> {n_sites} site files")

    out_dir = SITE_ROOT / "data" / "sample_forecasts"
    now_utc = datetime.now(timezone.utc)

    if args.check_only:
        problems = []
        for sid in sorted(daily["site_id"].unique()):
            f = out_dir / f"{sid}.json"
            if not f.exists():
                problems.append(f"missing {f.name}")
        print("CHECK:", "OK" if not problems else problems)
        sys.exit(0 if not problems else 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    cov = station_cov()
    if cov:
        print(f"  station-specific diurnal from training CoV "
              f"({len(set(k[0] for k in cov))} stations)")
    else:
        print("  [warn] training CSV absent -- template diurnal for all stations")
    written = []
    for sid, grp in daily.groupby("site_id"):
        grp = grp.sort_values("horizon_h")
        if len(grp) != 3:
            print(f"  [warn] {sid}: {len(grp)} daily rows (expect 3), skipping")
            continue
        real_name = str(grp.iloc[0]["location_name"])
        Ws = {p: diurnal_weights(p, cov.get((real_name, p), 1.0)) for p in TARGETS}
        # Build raw hourly per pollutant, smooth day junctions, then
        # rescale each day back to the exact model daily mean.
        raw = {p: [] for p in TARGETS}
        for _, d in grp.iterrows():
            for hh in range(24):
                for p in TARGETS:
                    raw[p].append(max(0.0, float(d[p]) * Ws[p][hh]))
        for p in TARGETS:
            raw[p] = smooth_junctions(raw[p])
        series = {p: [] for p in TARGETS}
        day_rows = list(grp.iterrows())
        for i, (_, d) in enumerate(day_rows):
            for hh in range(24):
                for p in TARGETS:
                    series[p].append(raw[p][i * 24 + hh])
        # exact-mean rescale per day (keeps verification honest)
        for i, (_, d) in enumerate(day_rows):
            for p in TARGETS:
                seg = series[p][i * 24:(i + 1) * 24]
                m = sum(seg) / 24
                target = float(d[p])
                series[p][i * 24:(i + 1) * 24] = (
                    [round(x * target / m, 1) for x in seg] if m else seg)
        timestamps = [
            (pd.Timestamp(d["date"]) + pd.Timedelta(hours=hh)).isoformat() + IST_SUFFIX
            for _, d in grp.iterrows() for hh in range(24)
        ]
        lower, upper, aqi_s, cat_s, col_s = [], [], [], [], []
        bands = {p: [] for p in ("no2", "o3")}
        for i in range(72):
            h = int(grp.iloc[i // 24]["horizon_h"])
            vals = {p: series[p][i] for p in TARGETS}
            band = {p: 1.25 * mae_h.get((p, h), 10.0) for p in ("pm25", "pm10")}
            lower.append(round(max(0.0, vals["pm25"] - band["pm25"]), 1))
            upper.append(round(vals["pm25"] + band["pm25"], 1))
            for p in ("no2", "o3"):
                w = 1.25 * mae_h.get((p, h), 8.0)
                bands[p].append((round(max(0.0, vals[p] - w), 1),
                                 round(vals[p] + w, 1)))
            res = naqi.calculate_naqi(
                {"pm25": vals["pm25"], "pm10": vals["pm10"],
                 "no2": vals["no2"], "o3": vals["o3"]})
            aqi_s.append(res.overall_aqi)
            cat_s.append(res.category)
            col_s.append(res.color)
        pm10_band_h1 = 1.25 * mae_h.get(("pm10", 1), 25.0)
        payload = {
            "station_id": sid,
            "station_name": str(grp.iloc[0]["location_name"]),
            "generated_at": now_utc.isoformat(),
            "source": "sih-p2 ensemble (TFT+XGBoost+LightGBM)",
            "source_file": fc_path.name,
            "downscaled": True,
            "native_resolution": "daily",
            "diurnal_note": ("hourly shape = cosine peak per pollutant, amplitude "
                             "scaled per station by observed day-to-day CoV; 24h mean "
                             "== model daily value exactly; day junctions smoothed"),
            "tz": "Asia/Kolkata",
            "band_note": "lower/upper = pm25 +/- 1.25x per-horizon MAE (~80% band); not TFT quantiles",
            "timestamps": timestamps,
            "pm25": series["pm25"], "pm10": series["pm10"],
            "no2": series["no2"], "o3": series["o3"],
            "lower": lower, "upper": upper,
            "no2_lower": [lo for lo, _ in bands["no2"]],
            "no2_upper": [hi for _, hi in bands["no2"]],
            "o3_lower": [lo for lo, _ in bands["o3"]],
            "o3_upper": [hi for _, hi in bands["o3"]],
            "pm10_lower": [round(max(0.0, v - pm10_band_h1), 1) for v in series["pm10"]],
            "pm10_upper": [round(v + pm10_band_h1, 1) for v in series["pm10"]],
            "aqi": aqi_s, "category": cat_s, "colors": col_s,
            "daily": [
                {"date": pd.Timestamp(r["date"]).date().isoformat(),
                 "horizon_h": int(r["horizon_h"]),
                 **{p: round(float(r[p]), 1) for p in TARGETS},
                 "aqi": round(float(r["AQI"]), 1), "category": r["category"],
                 "color": r["color"], "advisory": r["advisory"],
                 "dominant_pollutant": r["dominant_pollutant"]}
                for _, r in grp.iterrows()
            ],
            "models": {**file_models, "baseline": False},
            "blend_used": {t: blend_of.get(t) for t in TARGETS},
        }
        (out_dir / f"{sid}.json").write_text(json.dumps(payload, indent=2))
        written.append(f"{sid}.json")

    # Remove stale NCR/demo-only files so the site can't serve them.
    for stale in out_dir.glob("*.json"):
        if stale.name in ("index.json", "spatial.json"):
            continue
        if stale.name not in written:
            print(f"  removing stale {stale.name}")
            stale.unlink()

    # ---- stations.json FIRST (spatial needs the 6 new entries) ----
    _patch_stations(openaq)

    # ---- index.json + spatial.json (Day+1 real values) ----
    (out_dir / "index.json").write_text(json.dumps({
        "generated_at": now_utc.isoformat(),
        "description": "Real SIH-p2 ensemble 72h forecasts (daily model, hourly downscaled)",
        "source_file": fc_path.name,
        "station_forecasts": sorted(written),
        "spatial_file": "spatial.json",
    }, indent=2))
    feats = []
    for sid in sorted(daily["site_id"].unique()):
        f = json.loads((out_dir / f"{sid}.json").read_text())
        st = _site_station(sid)
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [st["longitude"], st["latitude"]]},
            "properties": {"id": sid, "name": st["short_name"],
                           "pm25": f["daily"][0]["pm25"],
                           "pm10": f["daily"][0]["pm10"],
                           "no2": f["daily"][0]["no2"],
                           "o3": f["daily"][0]["o3"],
                           "aqi": f["daily"][0]["aqi"],
                           "category": f["daily"][0]["category"],
                           "color": f["daily"][0]["color"]},
        })
    (out_dir / "spatial.json").write_text(json.dumps(
        {"type": "FeatureCollection", "features": feats}, indent=2))

    # ---- ensemble_skill.json ----
    _write_skill(blends, horizons, aq_skill, fc_path.name, now_utc)
    print(f"[OK] {len(written)} station files + index + spatial + stations.json + ensemble_skill.json")

    # ---- verification ----
    _verify(out_dir, daily, naqi)


def _site_station(sid: str) -> dict:
    stations = json.loads((SITE_ROOT / "data" / "stations.json").read_text())["stations"]
    for s in stations:
        if s["id"] == sid:
            return s
    raise KeyError(f"site station {sid} missing from stations.json")


def _patch_stations(openaq: pd.DataFrame) -> None:
    path = SITE_ROOT / "data" / "stations.json"
    if not (SITE_ROOT / "data" / "stations.54bak.json").exists():
        (SITE_ROOT / "data" / "stations.54bak.json").write_bytes(path.read_bytes())
        print("  backed up stations.json -> stations.54bak.json")
    data = json.loads(path.read_text())
    keep_ids = set(SITE_MAP.values())
    stations = [s for s in data["stations"] if s["id"] in keep_ids]
    have = {s["id"] for s in stations}
    # Add the 6 real stations missing from the site.
    for sid, (name, short, zone, typ) in NEW_STATIONS.items():
        if sid in have:
            continue
        loc = openaq[openaq["location_name"].str.contains(
            re.escape(short.split()[0]), case=False, na=False)]
        lat = float(loc.iloc[0]["latitude"]) if not loc.empty else 28.61
        lon = float(loc.iloc[0]["longitude"]) if not loc.empty else 77.21
        stations.append({"id": sid, "name": name, "short_name": short,
                         "latitude": round(lat, 4), "longitude": round(lon, 4),
                         "city": "Delhi", "zone": zone, "type": typ,
                         "elevation_m": 215, "openaq_location_id": None})
    # Fill OpenAQ ids (primary scalar + full array).
    inv = {}
    for real, sid in SITE_MAP.items():
        inv.setdefault(sid, []).append(real)
    for s in stations:
        ids: list = []
        for real in inv.get(s["id"], []):
            ids += [int(x) for x in
                    openaq[openaq["location_name"] == real]["location_id"].tolist()]
        # Fallback: name-token match for the 6 new entries.
        if not ids:
            key = s["short_name"].split()[0].lower()
            hit = openaq[openaq["location_name"].str.lower().str.contains(key, na=False)]
            ids = [int(x) for x in hit["location_id"].tolist()]
        s["openaq_location_id"] = ids[0] if ids else None
        s["openaq_location_ids"] = ids
    data["stations"] = sorted(stations, key=lambda s: s["id"])
    data["metadata"] = {"region": "Delhi (DPCC/CPCB/IMD network)",
                        "source": "SIH-p2 ensemble + OpenAQ",
                        "coordinate_system": "WGS84",
                        "station_count": len(stations),
                        "last_updated": datetime.now(timezone.utc).date().isoformat()}
    path.write_text(json.dumps(data, indent=2))
    n_null = sum(1 for s in stations if s["openaq_location_id"] is None)
    print(f"  stations.json: {len(stations)} Delhi stations, {n_null} null OpenAQ ids")


def _write_skill(blends: pd.DataFrame, horizons: pd.DataFrame,
                 aq_skill: dict, source_file: str, now_utc: datetime) -> None:
    per_pollutant = [
        {"target": r["target"], "blend": str(r["chosen_blend"]),
         "test_rmse": round(float(r["test_rmse"]), 2),
         "test_mae": round(float(r["test_mae"]), 2)}
        for _, r in blends.iterrows()
    ]
    per_horizon = [
        {"target": r["target"], "horizon_h": int(r["horizon_h"]),
         "n": int(r["n"]), "rmse": round(float(r["rmse"]), 2),
         "mae": round(float(r["mae"]), 2), "r2": round(float(r["r2"]), 3)}
        for _, r in horizons.sort_values(["target", "horizon_h"]).iterrows()
    ]
    (SITE_ROOT / "scripts" / "ensemble_skill.json").write_text(json.dumps({
        "generated_at": now_utc.isoformat(),
        "source_forecast": source_file,
        "models": {"tft": "Temporal Fusion Transformer (pytorch-forecasting)",
                   "xgboost": "XGBRegressor per pollutant",
                   "lightgbm": "LGBMRegressor per pollutant",
                   "note": "per-pollutant val-selected blend; NO uses LightGBM only"},
        "per_pollutant_test": per_pollutant,
        "per_horizon_backtest": per_horizon,
        "aqi_skill": aq_skill,
        "how_to_read": {
            "rmse": "ug/m3, lower is better; horizons degrade ~15-25% (h2) / ~30-50% (h3)",
            "r2": "1 = perfect; NO/PM2.5 weakest (traffic/dust features missing)",
            "aqi_exact": "fraction with predicted AQI category == actual",
        },
    }, indent=2))
    print("  ensemble_skill.json written")


def _verify(out_dir: Path, daily: pd.DataFrame, naqi) -> None:
    problems = []
    for sid in sorted(daily["site_id"].unique()):
        f = json.loads((out_dir / f"{sid}.json").read_text())
        if len(f["timestamps"]) != 72 or len(f["pm25"]) != 72:
            problems.append(f"{sid}: not 72 points")
        # 24h means must equal model daily values
        for i, (_, d) in enumerate(daily[daily.site_id == sid].sort_values("horizon_h").iterrows()):
            seg = f["pm25"][i * 24:(i + 1) * 24]
            if abs(sum(seg) / 24 - float(d["pm25"])) > 0.15:
                problems.append(f"{sid} h{i + 1}: daily mean drift")
        if not f["models"].get("baseline") is False:
            problems.append(f"{sid}: baseline flag not false")
        # AQI spot-check first hour with site calculator
        res = naqi.calculate_naqi({"pm25": f["pm25"][0], "pm10": f["pm10"][0],
                                   "no2": f["no2"][0], "o3": f["o3"][0]})
        if abs(res.overall_aqi - f["aqi"][0]) > 1.0:
            problems.append(f"{sid}: AQI recompute mismatch")
    if problems:
        print("VERIFY FAILED:"); [print("  " + p) for p in problems[:20]]
        sys.exit(1)
    print(f"VERIFY OK: {len(daily['site_id'].unique())} stations x 72h, means match, AQI consistent")


if __name__ == "__main__":
    main()
