"""
NETGUARD AI — FastAPI Backend Application.
Provides RESTful APIs for streaming anomaly detection, multi-signal feature analytics,
and adaptive threshold comparisons.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd

from backend.api.routes.dataset import router as dataset_router
from backend.api.routes.traffic import router as traffic_router
from backend.api.routes.detection import router as detection_router
from backend.api.routes.threshold import router as threshold_router
from backend.api.routes.models import router as models_router
from backend.api.routes.pipeline import router as pipeline_router
from backend.api.routes.live_monitor import router as live_router
from backend.core.data_store import data_store

app = FastAPI(
    title="NETGUARD AI API",
    description="Multi-Signal Self-Tuning Thresholds for Streaming Anomaly Detection",
    version="2.0.0",
)

# Enable CORS for React frontend (Vite default is 5173, production can be any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules under /api
app.include_router(dataset_router, prefix="/api")
app.include_router(traffic_router, prefix="/api")
app.include_router(detection_router, prefix="/api")
app.include_router(threshold_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")
app.include_router(live_router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Auto-load the real multi-signal dataset on startup so the dashboard is immediately populated."""
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "samples",
        "multi_signal_raw.csv",
    )
    if os.path.exists(sample_path) and data_store.df is None:
        try:
            print(f"[NETGUARD AI] Pre-loading benchmark dataset: {sample_path}")
            df = pd.read_csv(sample_path)
            data_store.set_dataset(df, name="multi_signal_raw.csv")
            print(f"[NETGUARD AI] Loaded {len(df):,} flows across {df.shape[1]} features.")
        except Exception as e:
            print(f"[NETGUARD AI] Warning: Failed to pre-load sample dataset: {e}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "project": "NETGUARD AI",
        "subtitle": "Multi-Signal Self-Tuning Thresholds for Streaming Anomaly Detection",
        "dataset_loaded": data_store.df is not None,
        "rows": len(data_store.df) if data_store.df is not None else 0,
        "features": len(data_store.df.columns) if data_store.df is not None else 0,
    }


# Production Single-Page Application (SPA) static serving
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import HTTPException

frontend_dist = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
)

if os.path.exists(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Allow API calls to pass through without being intercepted by SPA router
        if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            raise HTTPException(status_code=404, detail="API route not found")

        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)

        index_file = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend index.html not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
