"""Probe the CPCB data.gov.in primary feed (no server needed).

Logs per-city station counts, client stats (bounds rejects, inversions),
an age histogram of upstream `last_update` values, and sample records —
so a 30h-lag regression is visible here before it ever reaches the map.
Optionally checks a live server's /api/v1/snapshot timestamps too:

    DATAGOV_API_KEY=... python scripts/probe_cpcb_datagov.py
    DATAGOV_API_KEY=... python scripts/probe_cpcb_datagov.py \
        --snapshot-url http://localhost:8000

Exit 1 when the feed yields zero readings (key/network/schema break).
"""
import argparse
import asyncio
import json
import os
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, ".")
try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    pass
from backend.data.cpcb_datagov_client import CPCBDataGovClient

BUCKETS = [(1, "<1h"), (3, "1-3h"), (6, "3-6h"), (24, "6-24h"),
           (float("inf"), ">24h")]


def age_hours(iso):
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds()
                   / 3600.0)
    except (ValueError, TypeError):
        return None


async def probe_feed(key):
    client = CPCBDataGovClient(api_key=key)
    try:
        readings = await client.get_latest_measurements()
    finally:
        await client.close()
    stats = client.last_fetch

    print("== per-city raw records ==")
    for city, n in (stats.get("per_city") or {}).items():
        print(f"  {city:12} {n} records")
    print(f"raw={stats.get('records_raw')} grouped={stats.get('stations_grouped')} "
          f"dropped_no_pollutant={stats.get('dropped_no_pollutant')} "
          f"inverted_subindex={stats.get('inverted')} "
          f"unparsed_ts={stats.get('unparsed_timestamps')} "
          f"(mode={stats.get('value_mode')})")

    hist = Counter()
    unparseable = 0
    for r in readings:
        a = age_hours(r.timestamp)
        if a is None:
            unparseable += 1
            continue
        for edge, label in BUCKETS:
            if a <= edge:
                hist[label] += 1
                break
    print("== upstream age histogram (all cpcb) ==")
    for _, label in BUCKETS:
        print(f"  {label:6} {hist.get(label, 0)}")
    if unparseable:
        print(f"  unparseable {unparseable}")

    print("== sample records ==")
    for r in readings[:3]:
        a = age_hours(r.timestamp)
        print(f"  {r.station_name} | age={a:.1f}h | {r.pollutants}")

    fresh = hist.get("<1h", 0) + hist.get("1-3h", 0)
    print(f"fresh(<3h)={fresh} / {len(readings)} stations")
    return readings


def check_snapshot(base):
    url = base.rstrip("/") + "/api/v1/snapshot"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            snap = json.load(resp)
    except Exception as e:
        print(f"snapshot check SKIPPED ({e})")
        return
    stations = snap.get("stations") or []
    ages = []
    for s in stations:
        cur = s.get("current") or {}
        a = age_hours(cur.get("timestamp"))
        if a is not None:
            ages.append(a)
    under2 = sum(1 for a in ages if a < 2)
    mx = max(ages) if ages else None
    print(f"snapshot: source={snap.get('data_source')} "
          f"freshness={json.dumps(snap.get('freshness') or {})}")
    print(f"snapshot: {under2}/{len(ages)} station timestamps <2h old"
          + (f", max age {mx:.1f}h" if mx is not None else ""))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=None)
    ap.add_argument("--snapshot-url", default=None)
    args = ap.parse_args()
    key = (args.key or os.getenv("DATAGOV_API_KEY")
           or os.getenv("CPCB_DATAGOV_KEY") or "").strip()
    if not key:
        print("NO KEY: set DATAGOV_API_KEY (free at data.gov.in) or --key")
        return 1
    readings = await probe_feed(key)
    if args.snapshot_url:
        check_snapshot(args.snapshot_url)
    return 0 if readings else 1


sys.exit(asyncio.run(main()))
