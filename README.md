# AEROS · DEL — Delhi NCR AQI Forecasting System

> Coupled **Meteorology-Chemistry AQI Forecaster** for Delhi NCR — built for **Smart India Hackathon 2026**.
> Live PM2.5/PM10 + meteorology + stubble-burn fires → Physics Engine (PBL / AISI / Plumes) → ML Ensemble (72-hr forecast) → NLP Health Advisories → Live Mission-Control Dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688) ![MapLibre](https://img.shields.io/badge/MapLibre-GL_4.x-black) ![Docker](https://img.shields.io/badge/Docker-ready-2496ED) ![License](https://img.shields.io/badge/License-MIT-green)

**Live site (after you run it):**
- Dashboard → **http://localhost:8000**
- API Docs (Swagger) → **http://localhost:8000/docs**
- Health check → **http://localhost:8000/api/v1/health**

> There is no public internet URL until you deploy. Run locally (below) for instant access, or deploy to Render / Railway to get a shareable `https://your-app.onrender.com` link — see [Get a public link](#-get-a-public-shareable-link).

---

## What it does

1. **Ingests live data** — PM2.5/PM10 from OpenAQ (CPCB stations), meteorology from Open-Meteo, fire hotspots from NASA FIRMS VIIRS.
2. **Runs atmospheric physics** — PBL height + Bulk Richardson No., **AISI 0-10 Inversion Severity Index** with GRAP mapping, Lagrangian plume transport, radiation feedback, sector emissions.
3. **Forecasts 72 hours** — Physics-informed features → stacked **XGBoost + Transformer + statistical baseline** ensemble with confidence bands. Works with zero trained weights (graceful baseline fallback).
4. **Generates advisories** — Bilingual EN/HI templates + severity classifier + GRAP Stage I-IV actions, optionally rephrased by Gemini.
5. **Streams to dashboard** — MapLibre dark map (heatmap + fires + plumes + stations), Chart.js 72h curves, AISI gauge, alert ticker, WS latency monitor. Updates via `WS /ws/live` every 5 min.

```
Data (OpenAQ/Open-Meteo/FIRMS) → Physics (PBL/AISI/Plume/Radiation) → ML Ensemble (0-72h)
        → NLP Alerts (EN/HI + GRAP) → FastAPI (REST + WS) → Dashboard (MapLibre + Chart.js)
```

## Demo

Runs **fully offline in Demo Mode** if keys are missing — realistic sample data so judges see the full stack with no setup.

| Panel | What you see |
|-------|--------------|
| Map | PM2.5 heatmap, wind, fire markers, plume lines, station dots, 0-72h time slider |
| Right rail | AISI gauge + sparkline + GRAP badge, searchable station list, 72h forecast chart (PM2.5/PM10/NAQI + CSV export), health advisories (EN/हिं) |
| Header/ticker | Mean AQI, worst station, fire count, AISI, LIVE/DEMO badge, scrolling advisory |

## Quick Start (Windows)

```powershell
# 1. Clone / enter project
cd "C:\Users\Anant\Desktop\All Projects Btech\SIH"

# 2. Venv + install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt

# 3. Env (optional - demo works without keys)
copy .env.example .env

# 4. Run
python backend\run.py
# open http://localhost:8000
```

macOS/Linux:
```bash
python3 -m venv .venv; source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
python backend/run.py
```

Dev with reload:
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
python backend/run.py --demo   # force demo mode
```

Docker:
```bash
python scripts/generate_sample_data.py
docker compose up --build
# open http://localhost:8000
```

## Configuration (.env)

Missing keys = automatic demo fallback, no crash.

| Variable | Get it at | Purpose |
|----------|-----------|---------|
| `OPENAQ_API_KEY` | docs.openaq.org | Live PM2.5/PM10 |
| `OPENWEATHER_API_KEY` | openweathermap.org | (optional, Open-Meteo is default) |
| `NASA_FIRMS_API_KEY` | firms.modaps.eosdis.nasa.gov | Fire hotspots |
| `GEMINI_API_KEY` | aistudio.google.com | LLM advisory rephrase (optional) |
| `MAPTILER_KEY` | maptiler.com | Premium dark basemap (optional, OSM default otherwise) |
| `CARTO_KEY` | carto.com/basemaps/apikey (free) | CARTO raster basemap — key now required, OSM default without |
| `DEMO_MODE` | `true`/`false` | Force offline demo |
| `DATA_REFRESH_INTERVAL` | seconds (default 300) | Refresh cycle |

## API (all under `/api/v1`)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Status, last update, station count |
| `GET /snapshot` | Full state (stations + AISI + forecast + alerts + plume) |
| `GET /stations` | All stations `?category=Severe` |
| `GET /stations/{id}/current` | Current pollutants + NAQI |
| `GET /stations/{id}/forecast?hours=72` | 72h timeseries |
| `GET /stations/{id}/history` | SQLite history |
| `GET /forecast/spatial` | GeoJSON overlay |
| `GET /forecast/trigger?force=true` | Force refresh |
| `GET /aisi/current` | AISI + GRAP stage |
| `GET /alerts/current` | Latest advisories |
| `GET /fires/active` | Hotspots + plume ETA |
| `GET /radiation/current` | Radiation deficit term |
| `WS /ws/live` | Live push (send `ping`/`refresh`) |

Full interactive docs: `http://localhost:8000/docs`

## Project Structure

```
SIH/
├── backend/
│   ├── run.py, app/main.py, app/config.py, app/service.py, app/websocket_manager.py
│   ├── app/routes/      # stations, forecast, alerts, fire
│   ├── data/            # openaq, weather, firms, preprocessor + SQLite
│   ├── formulas/        # pure physics (AISI, PBL, plume…)
│   ├── physics/         # service wrappers
│   ├── ml/              # features, xgboost, transformer, ensemble
│   └── nlp/             # templates, severity_classifier, alert_generator
├── frontend/            # AEROS dashboard
│   ├── index.html
│   ├── css/main.css, dashboard.css, components.css, animations.css
│   └── js/app.js, map.js, charts.js, aisi.js, websocket.js, alerts.js, plume.js, utils.js
├── data/stations.json, data/aqi_data.db, data/sample_forecasts/
├── scripts/generate_sample_data.py
├── Dockerfile, docker-compose.yml, backend/requirements.txt
```

## How the pipeline works

`AQIService.refresh()` → pull readings/weather/fires → `PBLModel` + `AISICalculator` + `PlumeTransportModel` + `RadiationFeedback` + `EmissionProcessor` → `FeatureEngineer` → `EnsembleForecaster` (XGB + transformer + diurnal baseline → 72× PM2.5/PM10 + bounds) → `SeverityClassifier` + `AlertGenerator` (+Gemini) → REST + WebSocket → dashboard.

## Model Formulae (all of them)

Physics (`backend/formulas/`):

- Potential temperature: `θ = T·(P₀/P)^κ`, `κ = R_d/C_p ≈ 0.286`
- Virtual potential temperature: `θv = θ·(1 + 0.61·r)` (r = mixing ratio)
- Bulk Richardson number: `Ri_b(z) = (g/θv₀)·(θvz−θv₀)·(z−z₀) / [(uz−u₀)² + (vz−v₀)² + η·u*²]`, `η=100`, `u*=0.3`
- PBL height: lowest `z` where `Ri_b ≥ 0.25` (linear interpolation between levels); ERA5 `boundary_layer_height` preferred when available
- Inversion strength: `ΔT = T(100m) − T(surface)` (K/100m); 100m temperature from a smooth cosine diurnal blend (night inversion ↔ dry-adiabatic day)
- AISI: `AISI = min(10, α·(∂T/∂z) + β/max(PBLH,50) + γ·Ri_b)`, `α=2.5, β=150, γ=3.0`, EMA-smoothed (`α=0.65`, max step 2.5/cycle)
- AISI bands: 0–2 Well-Mixed, 2–5 Mild, 5–8 Moderate, 8–10 Severe
- GRAP: from AQI category (Poor→I, Very Poor→II, Severe→III, Severe+→IV); AISI may escalate by at most one stage (`reconcile_grap`)
- AOD: `τ = σ_ext·RH·PM2.5·H_eff/ρ_air`, `σ_ext=3.5e-6`, `H_eff=1000 m`
- Surface forcing: `ΔF = −(1−albedo)·S₀·cosθ·(1−e^(−τ·secθ))·SSA` (two-stream approx)
- BC warming: `H = (S₀·0.5·k_abs)/(ρ·Cp)` → °C/day (`k_abs` from 8% BC fraction)
- PBL suppression: `f = 0.5·(1−e^(−0.6·τ))`, `PBL_sup = PBL·(1−f)`
- Lagrangian step: `Δx=u·Δt, Δy=v·Δt` → degrees via Earth radius (hourly, 24h window, Delhi reached ≤60 km)
- Gaussian plume: `C = Q/(2πuσyσz)·exp(−y²/2σy²)·[exp(−(z−H)²/2σz²) + exp(−(z+H)²/2σz²)]`, receptor `z=2 m`, stack `H=50 m`, Pasquill–Gifford `σy=a·x^b, σz=c·x^d` (class D default)
- FRP→emissions (FINN-style): `biomass = FRP·3600·0.013 kg/h`, `PM2.5 = biomass·10.5/1000`, `PM10 = ·12/1000`, `CO = ·80/1000`
- Sector emissions: `E = base·diurnal(hour)·season + fire`; speciation (e.g. vehicular PM2.5 = 85% of PM10); 24-value diurnal profiles per sector

ML (`backend/ml/`):

- Features ≈80/station-hour: pollutant lags (24/48/72h mean/max/min/trend), temporal (hour, dow, season), meteorology, fire (count/FRP/nearest), physics (AISI, PBL, inversion, AOD, forcing)
- Statistical baseline: `C(h) = C₀·decay^(h/24)·diurnal(h) + fire·(1−h/144)`, `decay = 1−0.10·(1−AISI/8)·(0.5+0.5·PBL/1000)`, `σ(h)=0.10·C₀+0.05·C₀·√(h+1)`, 80% band `±1.28σ`
- PM10 from per-station observed `PM10/PM2.5` ratio (clamped 1.0–2.5, fallback 1.35) with its own band
- XGBoost: `XGBRegressor(n=180, depth=5, lr=0.08)` iterative 72-step
- LightGBM: `LGBMRegressor(n=220, leaves=31, lr=0.06)` (sklearn HistGBM fallback)
- TFT: `Linear→positional→MultiheadAttention(4, d=64)→LSTM→Linear(72h)`; numpy attention baseline when torch/weights absent
- Ensemble: `(0.5·TFT + 0.3·XGB + 0.2·LGBM)`, union uncertainty envelope

Air-quality index (`backend/data/naqi_calculator.py`):

- Sub-index per pollutant: `AQI_p = (AQI_Hi−AQI_Lo)/(C_Hi−C_Lo)·(C_p−C_Lo) + AQI_Lo` on CPCB breakpoints; overall `AQI = max(sub-indices)`; 7 categories Good→Severe+

Particle altitudes modelled: CPCB inlet / breathing zone `z ≈ 2 m`, plume injection `H = 50 m`, mixed through the PBL (typical ~300–1500 m day, ~100–400 m night; live value shown under the horizon slider).

## Accuracy (measured, not claimed)

**How it is calculated:** `python scripts/evaluate_accuracy.py` runs a walk-forward backtest on persisted SQLite observations — for every consecutive same-sensor pair it hindcasts with the live `statistical_baseline` at the true gap horizon and compares against naive persistence (carry-forward). Pairs are bucketed by lead (`≤1h / 1–6h / >6h`); metrics are MAE, RMSE, bias, Pearson r and AQI-category hit rate. Same numbers are served live at `GET /api/v1/accuracy/summary` and per station at `GET /api/v1/accuracy/stations`. Full report: `scripts/accuracy_report.json` (reproducible — rerun the script).

Measured 2026-09-12 (16,667 pairs, 41 stations):

| Lead | Baseline MAE / RMSE | Persistence MAE | Category hit | r |
|------|--------------------|-----------------|--------------|---|
| ≤1 h (n=16,196) | 11.9 / 21.1 µg/m³ | **8.0** | 53% (persist 77%) | 0.74 |
| 1–6 h (n=366) | 17.8 / 31.6 | 17.8 | 47% | — |
| >6 h (n=105) | **36.4** / 86.0 | 36.7 | 39% | — |

Reading it honestly: at ≤1 h, carry-forward is nearly unbeatable on smooth hourly air (true for every forecaster, including SAFAR's); the physics baseline pulls even by 1–6 h and edges ahead beyond 6 h. A 6-lag LightGBM probe on the same data already hits **MAE 7.2 vs 14.7 (mean-persistence), r = 0.83, n=4,921** — learning works; the full trainer (`scripts/train_models.py --force-sklearn`, ERA5 weather join, 100% coverage) holds releases to a stricter gate (must beat 1 h persistence, r > 0.2), which September-only data hasn't cleared yet (LGBM r=0.75, MAE +7%). When it clears, weights land in `backend/models/*.joblib`, the server hot-reloads them (daily auto-retrain, no restart), and the dashboard badge flips from `baseline` automatically.

**Where the 17k rows live:** in `data/aqi_data.db` on the machine that runs the server — backfilled from OpenAQ (CPCB stations, 15-min raw → hourly means) via `python scripts/backfill_history.py --days 30`, plus live 5-min refreshes. The DB is gitignored (too big/regenerating); the repo ships the code, the backfill + training scripts, and the JSON reports — anyone can reproduce the data with an `OPENAQ_API_KEY`. Reference apps reading the same CPCB CAAQMS sensors: **CPCB SAMEER**, **SAFAR-Air (IITM Pune)**, **DPCC website**, **IQAir**, **aqi.in**. *Current* AQI should agree near-identically (±calibration); published Delhi 24 h PM2.5 RMSEs for trained systems are ~30–50 µg/m³, r ≈ 0.7–0.85. Honest score today: **~6/10** — live end-to-end system with disclosed skill and a self-improving loop; path to ~8/10 is winter-regime data, not new code.

## 🌍 Get a public (shareable) link

Localhost works only on your PC. For a judge-friendly URL pick one:

**Option A — Render (free, recommended):**
1. Push to GitHub, go to render.com → New Web Service
2. Build: `pip install -r backend/requirements.txt`, Start: `python backend/run.py`
3. Add env vars from `.env.example` → Deploy → you get `https://aeros-del.onrender.com`

**Dependencies required for the global link (all free):**
- Python service: `backend/requirements.txt` — FastAPI, Uvicorn, httpx, numpy/pandas/scipy, aiosqlite, python-dotenv, scikit-learn; optional `xgboost`, `lightgbm`, `torch`, `google-genai` (app runs without them in baseline + local-NLP mode)
- Host port `8000` exposed (Render/Railway set `PORT` env — `run.py` already reads it)
- Persistent disk for `data/aqi_data.db` (else history resets each deploy — fine for demo)
- API keys (all optional — missing keys = demo fallback, nothing crashes): `OPENAQ_API_KEY` (live PM), `NASA_FIRMS_API_KEY` (fires), `GEMINI_API_KEY` (LLM advisories), `MAPTILER_KEY` (premium basemap; OSM default otherwise), `CARTO_KEY` (CARTO raster without watermark; OSM default otherwise)
- Open-Meteo needs no key. Outbound HTTPS to `api.openaq.org`, `api.open-meteo.com`, FIRMS must be allowed (default on Render/Railway)
- `DEMO_MODE=false`, `DATA_REFRESH_INTERVAL=300` as env vars

**Option B — Instant tunnel (demo in 30 sec):**
```powershell
ngrok http 8000
# share the https://xxxx.ngrok-free.app URL
```

**Option C — Railway / Fly.io:** same start command, expose port 8000.

## Tech Stack

Backend: FastAPI + Uvicorn + httpx/aiohttp + numpy/pandas/scipy + scikit-learn/XGBoost/torch (optional) + SQLite. Frontend: MapLibre GL 4.x + Chart.js 4.x + vanilla JS, Inter + JetBrains Mono. Data: OpenAQ/CPCB, Open-Meteo, NASA FIRMS, Gemini, CARTO/MapTiler.

## Roadmap

- [ ] Ship trained XGBoost/Transformer weights under `backend/models/`
- [ ] FINN emission ingestion + NRT fire feed
- [ ] Hindi toggle in UI + forecast skill scoring vs observed NAQI

## Credits

OpenAQ / CPCB, Open-Meteo, NASA FIRMS, MapLibre, Chart.js, CARTO. Built for SIH 2026 — Delhi NCR air-quality theme.

MIT — free for hackathon / research use.
