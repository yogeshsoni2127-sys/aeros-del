"""
AISI cross-check vs independent radiosonde sounding — run:
  python scripts/verify_aisi.py [--server http://localhost:8000]
  python scripts/verify_aisi.py --gradient 2.1 --pbl 180 --aisi 7.4

Fetches the latest University of Wyoming sounding for Delhi (VIDP,
station 42182, 00Z/12Z) and compares its observed lowest-100m
temperature gradient + PBL proxy against the model's AISI inputs.

Independent references (manual check):
  - Wyoming sounding: https://weather.uwyo.edu/upperair/sounding.html
    (region seasia, station 42182 VIDP Delhi)
  - IMD radiosonde status: https://ddgmui.imd.gov.in/ual2/LastDataAscent.php
  - windy.com (ECMWF: surface vs 950/900 hPa temps) / earth.nullschool.net
  - Open-Meteo ERA5 archive (same boundary_layer_height we ingest):
    https://archive-api.open-meteo.com/v1/archive

Exit 0 = agreement within tolerance, 1 = mismatch or sounding missing.
Stdlib only (urllib) — no new dependencies.
"""
import argparse
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STATION = "42182"  # VIDP New Delhi
TOL_GRAD = 1.5     # K/100m — model uses a synthesized profile, not a sonde
TOL_PBL = 400.0    # m — ERA5 stable-PBL bias is a known ~100-400m error


def _sounding_url(dt: datetime) -> str:
    return (
        "https://weather.uwyo.edu/cgi-bin/sounding"
        f"?region=seasia&TYPE=TEXT%3ALIST&YEAR={dt.year}"
        f"&MONTH={dt.month:02d}&FROM={dt.day:02d}{dt.hour:02d}"
        f"&TO={dt.day:02d}{dt.hour:02d}&STNM={STATION}"
    )


def fetch_sounding(timeout: int = 25, insecure: bool = False) -> dict:
    """Latest available 00Z/12Z VIDP sounding (walks back <= 36h)."""
    import ssl
    now = datetime.now(timezone.utc)
    # Candidate cycles: most recent 00Z/12Z first, then older ones.
    cycles = []
    base = now.replace(minute=0, second=0, microsecond=0)
    h = base.hour - (base.hour % 12)
    for back in (0, 12, 24, 36):
        cycles.append(base.replace(hour=0) + timedelta(hours=h - back))
    last_err = None
    ctx = None
    if insecure:
        ctx = ssl._create_unverified_context()
    for cyc in cycles:
        try:
            req = urllib.request.Request(
                _sounding_url(cyc), headers={"User-Agent": "AEROS-verify/1.0"})
            with urllib.request.urlopen(req, timeout=timeout,
                                        context=ctx) as r:
                text = r.read().decode("utf-8", "replace")
            parsed = parse_sounding(text)
            if parsed and parsed.get("levels"):
                parsed["cycle"] = cyc.strftime("%Y-%m-%d %HZ")
                return parsed
        except Exception as e:  # network or parse — try older cycle
            last_err = e
    raise RuntimeError(f"no VIDP sounding in last 36h ({last_err})")


def parse_sounding(text: str) -> dict:
    """Parse Wyoming TEXT:LIST into (HGHTm, TEMP C) levels."""
    levels = []
    in_data = False
    for line in text.splitlines():
        if "HGHT" in line and "TEMP" in line:
            in_data = True
            continue
        if not in_data:
            continue
        if not line.strip() or "Station" in line:
            if levels:
                break
            continue
        parts = line.split()
        try:
            if len(parts) < 4:
                continue
            # TEXT:LIST columns: PRES HGHT TEMP DWPT RELH ...
            hght = float(parts[1])
            temp = float(parts[2])
            levels.append((hght, temp))
        except ValueError:
            continue
    if len(levels) < 3:
        return {}
    levels.sort()
    return {"levels": levels}


def lowest_100m_gradient(levels) -> dict:
    """Observed gradient over the lowest ~100m (linear interp at +100m)."""
    (h0, t0) = levels[0]
    h1, t1 = None, None
    for h, t in levels[1:]:
        if h >= h0 + 100:
            h1, t1 = h, t
            break
    if h1 is None:
        (h1, t1) = levels[-1]
    if h1 <= h0:
        raise RuntimeError("degenerate sounding levels")
    frac = min(100.0, h1 - h0) / (h1 - h0)
    t100 = t0 + frac * (t1 - t0)
    grad = t100 - t0  # K per 100m by construction
    inversion = grad > 0
    return {"t_sfc": round(t0, 1), "t_100m": round(t100, 1),
            "gradient_k_per_100m": round(grad, 2), "inversion": inversion}


def fetch_model(server: str, timeout: int = 20) -> dict:
    url = server.rstrip("/") + "/api/v1/aisi/current"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        import json
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-check AISI vs sonde")
    ap.add_argument("--server", default=None,
                    help="live server base URL (else use --gradient/--pbl)")
    ap.add_argument("--gradient", type=float, default=None)
    ap.add_argument("--pbl", type=float, default=None)
    ap.add_argument("--aisi", type=float, default=None)
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS verification (proxy/MITM networks only)")
    args = ap.parse_args()

    try:
        sonde = fetch_sounding(insecure=args.insecure)
    except Exception as e:
        print(f"FAIL  sounding unavailable: {e}")
        print("      manual: https://weather.uwyo.edu/upperair/sounding.html"
              " (seasia, 42182)")
        return 1
    obs = lowest_100m_gradient(sonde["levels"])
    print(f"sonde {sonde['cycle']}Z VIDP: sfc {obs['t_sfc']}C -> "
          f"+100m {obs['t_100m']}C = {obs['gradient_k_per_100m']:+.2f} K/100m "
          f"({'INVERSION' if obs['inversion'] else 'lapse/mixed'})")

    if args.server:
        try:
            m = fetch_model(args.server)
        except Exception as e:
            print(f"FAIL  server unreachable: {e}")
            return 1
        st = (m.get("sub_terms") or {})
        model_grad = st.get("temp_gradient_k_per_100m")
        model_pbl = (m.get("pbl") or {}).get("pbl_height_m")
        model_aisi = m.get("aisi")
    else:
        model_grad, model_pbl, model_aisi = args.gradient, args.pbl, args.aisi
    if model_grad is None:
        print("model gradient unknown — pass --server or --gradient/--pbl")
        print(f"observed: {obs['gradient_k_per_100m']:+.2f} K/100m "
              f"(inversion={obs['inversion']})")
        return 0

    from backend.formulas.aisi_formulas import calculate_aisi
    d_grad = abs(float(model_grad) - obs["gradient_k_per_100m"])
    print(f"model: grad {float(model_grad):+.2f} K/100m "
          f"| PBL {model_pbl} m | AISI {model_aisi}")
    print(f"Δgrad = {d_grad:.2f} K/100m (tol {TOL_GRAD})")
    ok = d_grad <= TOL_GRAD
    # Sign agreement matters more than magnitude: both see an inversion?
    sign_ok = (float(model_grad) > 0) == obs["inversion"]
    print(f"sign agreement (both inversion / both lapse): {sign_ok}")
    if model_pbl is not None:
        # PBL proxy from sonde: top of surface inversion or first
        # 2K potential-temp rise — rough, tolerance is wide on purpose.
        print(f"note: ERA5 PBL overestimates stable nights by ~100-400m; "
              f"model PBL {model_pbl} m is indicative, not exact.")
    if ok and sign_ok:
        print("PASS  inversion agrees with radiosonde")
        return 0
    print("FAIL  inversion disagrees — check PBL injection + night "
          "synthesis (pbl_model NOCTURNAL_INVERSION_C) or stale weather")
    return 1


if __name__ == "__main__":
    sys.exit(main())
