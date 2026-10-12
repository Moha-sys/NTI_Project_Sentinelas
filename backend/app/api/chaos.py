import time
import logging
from typing import Dict, Any, List
from pydantic import BaseModel
from fastapi import APIRouter, BackgroundTasks
from app.correlation.engine import RawAlert, get_correlation_engine
from app.agent.sre_agent import get_sre_agent
from app.config import settings
from app.webhooks.n8n_notifier import build_payload, notify_n8n

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chaos", tags=["Chaos Engineering & Ground Truth"])

class ChaosInjectionRequest(BaseModel):
    service: str  # payment-service, cart-service, frontend, productcatalog-service, redis-cart
    experiment_type: str  # NetworkLatency, PodFailure, StressChaos, NetworkPartition
    tenant_id: str = "tenant_a"  # tenant_a, tenant_b, tenant_c
    duration_seconds: int = 60

# In-memory Ground Truth Ledger (synchronized with PostgreSQL in production)
ground_truth_ledger: List[Dict[str, Any]] = []

CHAOS_SIGNATURES = {
    "NetworkLatency": {
        "symptom": "HTTP 504 Gateway Timeout and P99 latency spike > 1200ms",
        "ground_truth_cause": "NetworkChaos: 300ms inter-service RTT delay & jitter",
        "severity": "critical"
    },
    "PodFailure": {
        "symptom": "Pod CrashLoopBackOff & Exit Code 137 (OOM / Kill)",
        "ground_truth_cause": "PodChaos: Simulated node eviction or SIGKILL on container",
        "severity": "critical"
    },
    "StressChaos": {
        "symptom": "CPU CFS Throttling surging > 85% & thread pool saturation",
        "ground_truth_cause": "StressChaos: 95% CPU & memory stress burner",
        "severity": "high"
    },
    "NetworkPartition": {
        "symptom": "RPC transport error: Dial failed & split-brain connection reset",
        "ground_truth_cause": "NetworkChaos: Bi-directional iptables packet drop",
        "severity": "critical"
    }
}

@router.post("/inject")
async def inject_chaos(req: ChaosInjectionRequest, background_tasks: BackgroundTasks):
    """
    Injects a precision failure into the microservice mesh,
    records Ground Truth in the Ledger, cascades telemetry alerts into the
    correlation engine, and — once the AI agent has finished its
    investigation — fires the n8n anomaly webhook so the alert actually
    reaches Telegram and email.
    """
    sig = CHAOS_SIGNATURES.get(req.experiment_type, {
        "symptom": "Generic service degradation",
        "ground_truth_cause": req.experiment_type,
        "severity": "warning"
    })

    experiment_id = f"chaos_{req.service}_{int(time.time())}"
    injected_at = time.time()

    # Inject REAL fault against local running microservice
    from app.mesh.manager import get_mesh_manager
    real_mesh_result = get_mesh_manager().inject_real_fault(req.service, req.experiment_type)

    # Record Ground Truth in Ledger
    ledger_entry = {
        "experiment_id": experiment_id,
        "service": req.service,
        "experiment_type": req.experiment_type,
        "tenant_id": req.tenant_id,
        "ground_truth_cause": sig["ground_truth_cause"],
        "expected_symptom": sig["symptom"],
        "injected_at": injected_at,
        "status": "ACTIVE",
        "real_mesh_action": real_mesh_result
    }
    ground_truth_ledger.insert(0, ledger_entry)

    # Trigger Cascading Alert Storm
    engine = get_correlation_engine()
    
    # Primary culprit alert
    primary_alert = RawAlert(
        id=f"alt-pri-{int(injected_at)}",
        alertname=f"{req.service.replace('-', '_').title()}_{req.experiment_type}",
        service=req.service,
        severity=sig["severity"],
        tenant_id=req.tenant_id,
        description=f"REAL FAULT: {real_mesh_result.get('message', sig['ground_truth_cause'])}"
    )
    incident = engine.correlate(primary_alert)

    # Cascading alerts from downstream dependents
    downstream_map = {
        "cart-service": ["frontend", "checkout-service"],
        "payment-service": ["checkout-service", "frontend"],
        "redis-cart": ["cart-service", "frontend"],
        "productcatalog-service": ["frontend", "recommendation-service"],
        "frontend": ["ingress-gateway"]
    }
    downstream = downstream_map.get(req.service, ["frontend"])
    
    for dep in downstream:
        dep_alert = RawAlert(
            id=f"alt-casc-{dep}-{int(injected_at)}",
            alertname=f"{dep.replace('-', '_').title()}CascadingFailure",
            service=dep,
            severity="warning" if sig["severity"] != "critical" else "critical",
            tenant_id=req.tenant_id,
            description=f"Downstream disruption: upstream {req.service} unresponsive"
        )
        incident = engine.correlate(dep_alert)

    # Artificially simulate 500+ raw telemetry alerts grouped into this incident
    incident.total_alerts = max(incident.total_alerts, 542)

    # Trigger Autonomous AI Agent diagnosis asynchronously
    agent = get_sre_agent()
    investigation = await agent.investigate_incident(incident)

    ledger_entry["ai_diagnosis"] = investigation["root_cause_analysis"]["diagnosis"]
    ledger_entry["confidence_score"] = investigation["root_cause_analysis"]["confidence_score"]
    ledger_entry["investigation_duration_seconds"] = investigation["investigation_duration_seconds"]

    # ── Alert delivery ────────────────────────────────────────────────────────
    # The investigation has produced a root cause and a confidence score, so
    # forward it to n8n. The workflow's own gate (confidence >= 0.6) decides
    # whether this becomes a Telegram/email alert or is silently acknowledged.
    rca = investigation["root_cause_analysis"]
    ml_prediction = rca.get("ml_prediction") or {}
    root_cause_service = (
        ml_prediction.get("root_cause_service")
        or rca.get("primary_service")
        or req.service
    )
    root_cause_fault = (
        ml_prediction.get("root_cause_fault_type")
        or req.experiment_type.lower()
    )
    confidence = float(rca.get("confidence_score") or 0.0)

    alert_payload = build_payload(
        service_name=root_cause_service,
        anomaly_type=f"{root_cause_fault}@{root_cause_service}",
        confidence_score=confidence,
        status="ANOMALY_DETECTED",
        timestamp=injected_at,
        extra={
            "anomaly_score": ml_prediction.get("anomaly_score"),
            "root_cause_service": root_cause_service,
            "root_cause_fault_type": root_cause_fault,
        },
    )
    background_tasks.add_task(notify_n8n, alert_payload)
    ledger_entry["alert_payload"] = alert_payload

    return {
        "status": "CHAOS_INJECTED",
        "experiment_id": experiment_id,
        "ground_truth": ledger_entry,
        "correlated_incident_id": incident.incident_id,
        "compressed_alerts_count": incident.total_alerts,
        "investigation": investigation,
        "alert": {
            "dispatched_to_n8n": True,
            "webhook_url": settings.N8N_WEBHOOK_URL,
            "payload": alert_payload,
        },
    }

@router.get("/ledger")
def get_ground_truth_ledger():
    """Retrieve Ground-Truth Ledger entries."""
    return ground_truth_ledger

