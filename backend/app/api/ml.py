from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.pipeline import get_rca_pipeline
from app.agent.sre_agent import build_live_telemetry_rows

router = APIRouter(prefix="/ml", tags=["ML Root Cause Analysis"])


class TelemetryRow(BaseModel):
    timestamp: Optional[int] = None
    service_name: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    service_up: int = 1


class AnalyzeRequest(BaseModel):
    rows: List[TelemetryRow] = Field(
        default_factory=list,
        description="Long-format telemetry rows. Leave empty to analyze the live mesh snapshot.",
    )


@router.get("/status")
def ml_status():
    """RCA model health: artifact presence and training metrics summary."""
    pipeline = get_rca_pipeline().ensure_loaded()
    health = pipeline.health()
    try:
        from app.ml.train import METRICS_PATH
        import json
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    except Exception:
        metrics = {}
    return {
        "pipeline": health,
        "models": {
            "anomaly_detector": "sklearn.ensemble.IsolationForest",
            "root_cause_classifier": "lightgbm.LGBMClassifier (root_cause_service + fault_type)",
        },
        "metrics": metrics,
    }


@router.post("/analyze")
def analyze_telemetry(req: AnalyzeRequest):
    """Run Isolation Forest + LightGBM RCA over provided telemetry rows
    (or the live mesh snapshot when rows are empty)."""
    if req.rows:
        rows = [r.model_dump() for r in req.rows]
        source = "provided_rows"
    else:
        rows = build_live_telemetry_rows()
        source = "live_mesh"
        if not rows:
            raise HTTPException(status_code=503, detail="Live mesh telemetry unavailable")

    for row in rows:
        if row.get("timestamp") is None:
            import time
            row["timestamp"] = int(time.time())

    try:
        result = get_rca_pipeline().analyze(rows)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"source": source, "n_rows": len(rows), "analysis": result}
