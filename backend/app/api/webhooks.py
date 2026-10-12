"""Webhook API router — anomaly trigger endpoint.

Exposes two routes:

  POST /api/v1/webhooks/anomaly/trigger
      Accepts telemetry rows, runs the ML RCA pipeline, and — when an
      anomaly is detected — fires an outbound POST to the n8n webhook URL.

  POST /api/v1/webhooks/anomaly/notify
      Accepts a pre-formed AnomalyPayload and forwards it directly to n8n.
      Useful for testing or forwarding from other internal services.
"""
import time
import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.ml.pipeline import get_rca_pipeline
from app.webhooks.n8n_notifier import build_payload, notify_n8n

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks / n8n Integration"])


# ── Request / Response models ─────────────────────────────────────────────────

class TelemetryRow(BaseModel):
    timestamp: Optional[int] = None
    service_name: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    service_up: int = 1


class AnomalyTriggerRequest(BaseModel):
    rows: List[TelemetryRow] = Field(
        ...,
        min_length=1,
        description="Long-format telemetry rows to evaluate for anomalies.",
    )


class AnomalyPayload(BaseModel):
    """Pre-formed anomaly payload; forwarded as-is to n8n."""
    service_name: str = Field(..., description="Originating microservice name.")
    anomaly_type: str = Field(..., description="Human-readable fault classification (e.g. cpu_throttling).")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly probability [0, 1].")
    timestamp: Optional[float] = Field(None, description="Unix epoch; defaults to now.")
    status: str = Field("ANOMALY_DETECTED", description="Alert lifecycle status.")


class WebhookResponse(BaseModel):
    triggered: bool
    n8n_notified: bool
    payload: Optional[dict] = None
    analysis_summary: Optional[dict] = None
    message: str


# ── Helper ────────────────────────────────────────────────────────────────────

def _derive_anomaly_type(analysis: dict) -> str:
    """Map RCA pipeline output to a single anomaly_type string."""
    fault = analysis.get("root_cause_fault_type")
    service = analysis.get("root_cause_service")
    if fault and service:
        return f"{fault}@{service}"
    if fault:
        return fault
    if service:
        return f"anomaly@{service}"
    return "unknown_anomaly"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/anomaly/trigger",
    response_model=WebhookResponse,
    summary="Run ML anomaly detection and fire n8n webhook when anomaly found",
)
async def anomaly_trigger(req: AnomalyTriggerRequest, background_tasks: BackgroundTasks):
    """
    Accepts a telemetry snapshot, runs the Isolation Forest + LightGBM RCA
    pipeline, and — **only when an anomaly is confirmed** — fires a POST to
    the configured n8n webhook URL.

    The n8n notification is dispatched as a **background task** so the API
    response is returned immediately.

    ### Payload sent to n8n
    ```json
    {
      "service_name": "cart-service",
      "anomaly_type": "cpu_throttling@cart-service",
      "confidence_score": 0.87,
      "timestamp": 1718000000.0,
      "status": "ANOMALY_DETECTED",
      "anomaly_score": -0.1234,
      "root_cause_service": "cart-service",
      "root_cause_fault_type": "cpu_throttling"
    }
    ```
    """
    rows = [r.model_dump() for r in req.rows]
    for row in rows:
        if row.get("timestamp") is None:
            row["timestamp"] = int(time.time())

    try:
        analysis = get_rca_pipeline().analyze(rows)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    is_anomaly: bool = analysis.get("is_anomaly", False)

    if not is_anomaly:
        return WebhookResponse(
            triggered=False,
            n8n_notified=False,
            analysis_summary={
                "is_anomaly": False,
                "anomaly_score": analysis.get("anomaly_score"),
                "anomaly_probability": analysis.get("anomaly_probability"),
            },
            message="No anomaly detected — n8n webhook not triggered.",
        )

    # Build canonical payload
    service_name: str = (
        analysis.get("root_cause_service")
        or (req.rows[0].service_name if req.rows else "unknown")
    )
    anomaly_type = _derive_anomaly_type(analysis)
    confidence_score: float = analysis.get("anomaly_probability", 0.0)

    payload = build_payload(
        service_name=service_name,
        anomaly_type=anomaly_type,
        confidence_score=confidence_score,
        status="ANOMALY_DETECTED",
        extra={
            "anomaly_score": analysis.get("anomaly_score"),
            "root_cause_service": analysis.get("root_cause_service"),
            "root_cause_fault_type": analysis.get("root_cause_fault_type"),
        },
    )

    background_tasks.add_task(notify_n8n, payload)

    return WebhookResponse(
        triggered=True,
        n8n_notified=True,   # dispatched; actual delivery confirmed in logs
        payload=payload,
        analysis_summary={
            "is_anomaly": True,
            "anomaly_score": analysis.get("anomaly_score"),
            "anomaly_probability": analysis.get("anomaly_probability"),
            "root_cause_service": analysis.get("root_cause_service"),
            "root_cause_fault_type": analysis.get("root_cause_fault_type"),
        },
        message="Anomaly detected — n8n webhook notification dispatched.",
    )


@router.post(
    "/anomaly/notify",
    response_model=WebhookResponse,
    summary="Directly forward a pre-formed anomaly payload to n8n",
)
async def anomaly_notify(payload_in: AnomalyPayload):
    """
    Immediately POSTs a pre-formed anomaly payload to the n8n webhook URL.

    Use this endpoint to:
    - Test your n8n workflow without going through the ML pipeline.
    - Forward anomaly events detected by external systems (e.g. Prometheus
      alertmanager, Grafana alerts) into the same n8n notification pipeline.
    """
    payload = build_payload(
        service_name=payload_in.service_name,
        anomaly_type=payload_in.anomaly_type,
        confidence_score=payload_in.confidence_score,
        status=payload_in.status,
        timestamp=payload_in.timestamp,
    )

    notified = await notify_n8n(payload)

    return WebhookResponse(
        triggered=True,
        n8n_notified=notified,
        payload=payload,
        message=(
            "Payload forwarded to n8n successfully."
            if notified
            else "Payload built but n8n notification failed — check N8N_WEBHOOK_URL and server logs."
        ),
    )
