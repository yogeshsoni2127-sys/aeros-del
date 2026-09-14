"""
FastAPI Application — SIH Coupled Meteorology-Chemistry AQI System

- REST endpoints under /api/v1
- WebSocket /ws/live for live dashboard pushes
- Static frontend served at /
- Background refresh loop pushing state every data_refresh_interval seconds
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("sih.api")

from backend.app.config import settings, PROJECT_ROOT
from backend.app.service import AQIService
from backend.app.websocket_manager import WebSocketManager
from backend.app.routes import (
    stations_router,
    forecast_router,
    alerts_router,
    fire_router,
    accuracy_router,
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init service fast, initial refresh in background."""
    app.state.service = AQIService(settings)
    app.state.ws_manager = WebSocketManager()

    # ── Background initial refresh (non-blocking so Render port opens) ──
    async def _initial_refresh():
        try:
            logger.info("Performing initial pipeline refresh ...")
            await app.state.service.initialize()
        except Exception as e:
            logger.error("Initial refresh failed: %s", e)

    app.state.init_task = asyncio.create_task(_initial_refresh())
    await app.state.ws_manager.start_heartbeat(interval_seconds=30)

    # ── Background refresh + broadcast loop ────────────────────────
    async def _publish_loop():
        while True:
            await asyncio.sleep(settings.data_refresh_interval)
            try:
                result = await app.state.service.refresh()
                if result.get("refreshed"):
                    snapshot = await app.state.service.get_snapshot()
                    await app.state.ws_manager.broadcast({
                        **snapshot,
                        "type": "update",
                    })
                # Self-improving loop: retrains at most 1×/day once enough
                # new history accumulated; weights hot-reload, no restart.
                try:
                    await app.state.service.maybe_auto_train()
                except Exception as e:
                    logger.debug("auto-train check failed: %s", e)
            except Exception as e:
                logger.error("Background publish loop error: %s", e)

    app.state.publish_task = asyncio.create_task(_publish_loop())
    logger.info("SIH AQI system initialized")

    yield

    # ── Shutdown ───────────────────────────────────────────────────
    app.state.publish_task.cancel()
    try:
        app.state.init_task.cancel()
    except Exception:
        pass
    await app.state.ws_manager.stop_heartbeat()
    await app.state.service.close()
    logger.info("SIH AQI system shut down")


app = FastAPI(
    title="SIH — Delhi NCR AQI Forecasting System",
    description=(
        "Coupled meteorology-chemistry AQI forecasting for Delhi NCR: "
        "live data ingestion, physics-informed ML ensemble, and NLP alerts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST routers ───────────────────────────────────────────────────
app.include_router(stations_router)
app.include_router(forecast_router)
app.include_router(alerts_router)
app.include_router(fire_router)
app.include_router(accuracy_router)


# ── WebSocket live feed ────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    manager: WebSocketManager = websocket.app.state.ws_manager
    service: AQIService = websocket.app.state.service

    await manager.connect(websocket)
    try:
        # Push a full snapshot immediately on connection
        snapshot = await service.get_snapshot()
        await websocket.send_json(snapshot)

        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_json({"type": "pong"})
            elif message == "refresh":
                # Non-blocking: ack immediately, broadcast when done.
                await websocket.send_json({"type": "refresh_started"})

                async def _do_refresh():
                    try:
                        result = await service.refresh(force=True)
                        if result.get("refreshed"):
                            fresh = await service.get_snapshot()
                            await manager.broadcast({**fresh, "type": "update"})
                    except Exception as e:
                        logger.warning("Background refresh failed: %s", e)

                asyncio.create_task(_do_refresh())
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
        await manager.disconnect(websocket)


# ── Root health/status ─────────────────────────────────────────────
@app.get("/api/v1/health")
async def health():
    service = getattr(app.state, "service", None)
    status = "ready" if service is not None else "starting"
    return {
        "status": status,
        "initialized": (service.state.get("initialized", False)
                        if service else False),
        "last_update": (service.state.get("last_update")
                        if service else None),
        "data_source": (service.state.get("data_source")
                        if service else None),
        "station_count": len(service.get_stations()) if service else 0,
        "ws_connections": (app.state.ws_manager.connection_count
                           if hasattr(app.state, "ws_manager") else 0),
    }


@app.get("/api/v1/snapshot")
async def api_snapshot():
    """Full current state snapshot for clients that prefer REST polling."""
    service = getattr(app.state, "service", None)
    if service is None:
        return {"initialized": False}
    return await service.get_snapshot()


@app.get("/api/v1/config")
async def public_config():
    """Non-secret client configuration (map tiles, refresh cadence)."""
    return {
        "maptiler_key": settings.maptiler_key or None,
        "carto_key": settings.carto_key or None,
        "refresh_interval_s": settings.data_refresh_interval,
        "forecast_horizon_h": settings.forecast_horizon_hours,
    }


# ── Static frontend (must be last — it is a catch-all mount) ───────
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True),
          name="frontend")