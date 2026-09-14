"""TFT gate: replay recorded TFT predictions through the live harness.

Compares harness TFT output vs predictions_tft.csv (research, offline)
on a sample of test-split (location, date) pairs using data strictly
before each target date (observed weather/fire from the training CSV).
PASS when harness MAE <= 1.15x recorded MAE per target; per-target
failures keep that target tree-only (see ResearchDailyEnsemble).

Run from the aeros-del dir (needs the sibling research tree for the
training CSV + recorded predictions):

    python scripts/validate_tft_gate.py [--pairs 8] [--out report.json]

Exit 0 all-pass, 1 any target fails, 2 missing inputs (skip, not fail).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

AEROS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AEROS))
RESEARCH = AEROS.parent

from backend.ml.research_daily import ResearchDailyEnsemble  # noqa: E402
from backend.ml.tft_daily import TFTDailyEnsemble  # noqa: E402

TARGETS = ["pm25", "pm10", "no2", "o3"]
TOLERANCE = 1.15

WX_KEYS = ["temp_mean", "temp_max", "temp_min", "humidity_mean",
           "wind_speed_mean", "wind_direction_dominant", "precipitation",
           "pressure_mean", "wind_gusts_mean", "blh_max",
           "radiation_sum", "dewpoint_mean"]
FIRE_KEYS = ["regional_fire_count", "regional_total_frp",
             "regional_max_frp", "local_fire_count", "local_total_frp",
             "local_max_frp"]
POLS = ("pm25", "pm10", "no2", "o3", "no", "nox")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=8,
                    help="test dates sampled per target")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train_csv = RESEARCH / "data" / "processed" / "delhi_final_training_data.csv"
    tft_csv = RESEARCH / "scripts" / "training" / "predictions_tft.csv"
    if not train_csv.exists() or not tft_csv.exists():
        print("SKIP: research tree (training CSV / predictions_tft.csv) "
              "not found next to aeros-del")
        return 2

    rec_rows = [r for r in load_csv(str(tft_csv)) if r.get("split") == "test"]
    by_target_date = defaultdict(list)
    for r in rec_rows:
        if r.get("target") in TARGETS:
            by_target_date[(r["target"], r["date"])].append(r)
    dates = sorted({d for (_, d) in by_target_date})
    sample_dates = dates[::max(1, len(dates) // args.pairs)][:args.pairs]
    print(f"replaying {len(sample_dates)} test dates x up to 25 stations")

    hist = load_csv(str(train_csv))
    by_loc = defaultdict(list)
    for r in hist:
        if r.get("location_name"):
            by_loc[r["location_name"]].append(r)
    for loc in by_loc:
        by_loc[loc].sort(key=lambda r: r.get("date", ""))

    def fnum(v):
        try:
            f = float(v)
            return f if f == f and abs(f) != float("inf") else None
        except (TypeError, ValueError):
            return None

    bridge = ResearchDailyEnsemble(
        str(AEROS / "backend" / "models" / "research"),
        str(AEROS / "data" / "research_seed_daily.csv"))
    bridge.load()
    tft = TFTDailyEnsemble(str(AEROS / "backend" / "models" / "research" / "tft"))
    if not tft.load():
        print("FAIL: TFT checkpoints did not load")
        return 1

    report = {}
    for target in TARGETS:
        errs_mine, errs_rec = [], []
        for dstr in sample_dates:
            pairs = [r for r in by_target_date.get((target, dstr), [])]
            if not pairs:
                continue
            frames = {}
            actual = {}
            for r in pairs:
                loc = r["location_name"]
                rows = [x for x in by_loc.get(loc, [])
                        if x.get("date", "") < dstr]
                if len(rows) < 14:
                    continue
                enc_days = rows[-14:]
                tails = {t: [fnum(x.get(t)) for x in rows
                             if fnum(x.get(t)) is not None][-14:]
                         for t in POLS}
                fh = {
                    "regional_fire_count": [
                        fnum(x.get("regional_fire_count")) for x in rows
                        if fnum(x.get("regional_fire_count")) is not None][-30:],
                    "regional_total_frp": [
                        fnum(x.get("regional_total_frp")) for x in rows
                        if fnum(x.get("regional_total_frp")) is not None][-30:],
                }
                enc = []
                for d in enc_days:
                    dd = datetime.strptime(d["date"][:10], "%Y-%m-%d").date()
                    upto = rows[:rows.index(d) + 1]
                    tails_d = {t: [fnum(x.get(t)) for x in upto
                                   if fnum(x.get(t)) is not None][-14:]
                               for t in POLS}
                    fh_d = {
                        "regional_fire_count": [
                            fnum(x.get("regional_fire_count")) for x in upto
                            if fnum(x.get("regional_fire_count"))
                            is not None][-30:],
                        "regional_total_frp": [
                            fnum(x.get("regional_total_frp")) for x in upto
                            if fnum(x.get("regional_total_frp"))
                            is not None][-30:],
                    }
                    base = {"location_name": loc,
                            "latitude": fnum(d.get("latitude")) or 0.0,
                            "longitude": fnum(d.get("longitude")) or 0.0,
                            "station_id": int(float(d.get("station_id") or -1)),
                            "_date": dd}
                    wx = {k: fnum(d.get(k)) for k in WX_KEYS}
                    fire = {k: fnum(d.get(k)) for k in FIRE_KEYS}
                    row = bridge.feature_row(base, 1, wx, fire, tails_d, fh_d)
                    row = {k: v for k, v in row.items()
                           if not str(k).startswith("_")}
                    for t in POLS:
                        row[t + "_actual"] = fnum(d.get(t))
                    row["location_name"] = loc
                    row["date"] = d["date"][:10]
                    enc.append(row)
                tails = {t: [fnum(x.get(t)) for x in rows
                             if fnum(x.get(t)) is not None][-14:]
                         for t in POLS}
                fh = {
                    "regional_fire_count": [
                        fnum(x.get("regional_fire_count")) for x in rows
                        if fnum(x.get("regional_fire_count")) is not None][-30:],
                    "regional_total_frp": [
                        fnum(x.get("regional_total_frp")) for x in rows
                        if fnum(x.get("regional_total_frp")) is not None][-30:],
                }
                # decoder = target date, observed covariates, NaN target
                drow = [x for x in by_loc.get(loc, [])
                        if x.get("date", "")[:10] == dstr]
                if not drow:
                    continue
                d0 = drow[0]
                dd = datetime.strptime(dstr, "%Y-%m-%d").date()
                base = {"location_name": loc,
                        "latitude": fnum(d0.get("latitude")) or 0.0,
                        "longitude": fnum(d0.get("longitude")) or 0.0,
                        "station_id": int(float(d0.get("station_id") or -1)),
                        "_date": dd}
                wx = {k: fnum(d0.get(k)) for k in WX_KEYS}
                fire = {k: fnum(d0.get(k)) for k in FIRE_KEYS}
                dec = bridge.feature_row(base, 1, wx, fire, tails, fh)
                dec = {k: v for k, v in dec.items()
                       if not str(k).startswith("_")}
                for t in POLS:
                    dec[t + "_actual"] = None
                dec["location_name"] = loc
                dec["date"] = dstr
                frames[loc] = enc + [dec]
                try:
                    actual[loc] = float(r["actual"])
                except (TypeError, ValueError):
                    continue
            if not frames:
                continue
            got = tft.predict_horizon(target, frames)
            for r in pairs:
                loc = r["location_name"]
                if got.get(loc) is None:
                    continue
                try:
                    a = float(r["actual"])
                except (TypeError, ValueError):
                    continue
                errs_mine.append(abs(got[loc] - a))
                try:
                    errs_rec.append(abs(float(r["tft_pred"]) - a))
                except (TypeError, ValueError):
                    pass
        import statistics
        m_mine = statistics.mean(errs_mine) if errs_mine else None
        m_rec = statistics.mean(errs_rec) if errs_rec else None
        ok = (m_mine is not None and m_rec
              and m_mine <= TOLERANCE * m_rec)
        report[target] = {"n": len(errs_mine), "harness_mae": m_mine,
                          "recorded_mae": m_rec, "pass": bool(ok)}
        print(f"{target}: harness MAE {m_mine} vs recorded {m_rec} "
              f"(n={len(errs_mine)}) -> {'PASS' if ok else 'FAIL'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0 if all(v["pass"] for v in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
