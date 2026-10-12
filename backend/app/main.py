import logging
import os

# Ensure local Kubernetes API (127.0.0.1:6443) and microservices bypass HTTP proxies
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.rag.indexer import index_all_runbooks
from app.api.health import router as health_router
from app.api.alerts import router as alerts_router
from app.api.incidents import router as incidents_router
from app.api.rag import router as rag_router
from app.api.chaos import router as chaos_router
from app.api.benchmarks import router as benchmarks_router
from app.api.mesh import router as mesh_router
from app.api.ml import router as ml_router
from app.api.webhooks import router as webhooks_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up AIOps Platform Engine...")
    try:
        count = index_all_runbooks()
        logger.info(f"RAG Knowledge Base initialized with {count} runbooks.")
    except Exception as e:
        logger.error(f"Failed to auto-index runbooks at startup: {e}")

    try:
        from app.mesh.manager import get_mesh_manager
        mgr = get_mesh_manager()
        mgr.start_all()
        logger.info("Real Microservice Mesh active on ports 8081-8084 & 6380.")
    except Exception as e:
        logger.error(f"Failed to start microservice mesh: {e}")

    try:
        from app.ml.pipeline import get_rca_pipeline
        health = get_rca_pipeline().ensure_loaded().health()
        logger.info(f"ML RCA pipeline ready: {health}")
    except Exception as e:
        logger.error(f"Failed to initialize ML RCA pipeline: {e}")

    yield
    logger.info("Shutting down AIOps Platform Engine...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Chaos-Grounded Autonomous AIOps Platform for Multi-Tenant Incident Detection, Correlation & AI Root-Cause Investigation",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for the React Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers under /api/v1
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(alerts_router, prefix=settings.API_V1_PREFIX)
app.include_router(incidents_router, prefix=settings.API_V1_PREFIX)
app.include_router(rag_router, prefix=settings.API_V1_PREFIX)
app.include_router(chaos_router, prefix=settings.API_V1_PREFIX)
app.include_router(benchmarks_router, prefix=settings.API_V1_PREFIX)
app.include_router(mesh_router, prefix=settings.API_V1_PREFIX)
app.include_router(ml_router, prefix=settings.API_V1_PREFIX)
app.include_router(webhooks_router, prefix=settings.API_V1_PREFIX)

# Serve built React dashboard or embedded single-page app
DASHBOARD_HTML_FILE = Path(__file__).resolve().parent.parent / "static" / "index.html"
if not DASHBOARD_HTML_FILE.parent.exists():
    DASHBOARD_HTML_FILE.parent.mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    if DASHBOARD_HTML_FILE.exists():
        return DASHBOARD_HTML_FILE.read_text(encoding="utf-8")
    return "<h1>AIOps Platform Dashboard</h1><p>Dashboard HTML is initializing...</p>"

@app.get("/api")
def api_meta():
    return {
        "platform": settings.PROJECT_NAME,
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_PREFIX}/health"
    }
