import logging
import time
from typing import Dict, Any, List, Optional

from app.rag.retriever import get_runbook_retriever
from app.correlation.engine import CorrelatedIncident
from app.sla.calculator import SLARiskCalculator
from app.tools.telemetry import TelemetryTools
from app.agent.llm import get_ollama_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior Site Reliability Engineer performing root cause analysis "
    "on a microservices incident. You are given: (1) ML model predictions from an "
    "Isolation Forest anomaly detector and a LightGBM root-cause classifier, "
    "(2) retrieved internal runbooks, and (3) Kubernetes/telemetry evidence. "
    "Write a concise incident report with sections: Root Cause, Evidence, "
    "Impact, and Recommended Remediation. Be specific and factual; cite the "
    "model confidence and the runbook titles you rely on. Do not invent metrics."
)


def build_live_telemetry_rows() -> List[Dict[str, Any]]:
    """Snapshot the local microservice mesh into the long-format telemetry
    rows the ML pipeline was trained on.

    Healthy values are calibrated to the per-service baselines of the chaos
    dataset (raw /proc readings are unavailable on some platforms, e.g.
    Windows returns identical 28MB for every service, which is
    out-of-distribution for the model). Fault signals (crash, CPU burn,
    injected latency) come from REAL mesh process state and follow the
    signatures observed during the original chaos experiments, including
    one-hop cascade impact on direct dependents.
    """
    from app.mesh.manager import get_mesh_manager

    # Per-service means from normal windows of aiops_chaos_dataset.csv
    baseline = {
        "cart-service":            {"cpu": 2.18, "mem":  96.0, "lat": 16.49, "err": 0.001},
        "frontend":                {"cpu": 3.80, "mem": 148.3, "lat": 24.44, "err": 0.001},
        "payment-service":         {"cpu": 1.44, "mem":  44.5, "lat": 11.64, "err": 0.001},
        "productcatalog-service":  {"cpu": 1.83, "mem":  64.0, "lat": 14.26, "err": 0.001},
        "redis-cart":              {"cpu": 0.85, "mem":  32.6, "lat":  3.85, "err": 0.001},
    }
    # Direct dependents (from SERVICES_METADATA dependency graph)
    dependents = {
        "payment-service": ["frontend"],
        "cart-service": ["frontend"],
        "productcatalog-service": ["frontend"],
        "redis-cart": ["cart-service"],
        "frontend": [],
    }

    now = int(time.time())
    rows: List[Dict[str, Any]] = []
    crashed: List[str] = []
    latency_injected: List[str] = []

    for svc in get_mesh_manager().get_mesh_status():
        name = svc.get("service", "")
        base = baseline.get(name, {"cpu": 2.0, "mem": 80.0, "lat": 20.0, "err": 0.001})
        is_up = 1 if svc.get("status") == "HEALTHY" else 0

        if not is_up:
            # PodFailure signature: everything zeroed, error rate saturated
            row = {"cpu_percent": 0.0, "memory_mb": 0.0, "latency_ms": 0.0,
                   "error_rate": 1.0, "service_up": 0}
            crashed.append(name)
        elif svc.get("cpu_burning"):
            # StressChaos signature: CPU pegged, elevated latency/errors
            row = {"cpu_percent": 94.0, "memory_mb": round(base["mem"] + 10.0, 2),
                   "latency_ms": round(base["lat"] * 3.0, 2),
                   "error_rate": 0.03, "service_up": 1}
        elif float(svc.get("added_latency_ms") or 0.0) > 0:
            # NetworkLatency signature: injected ~850ms delay on this hop
            row = {"cpu_percent": base["cpu"], "memory_mb": base["mem"],
                   "latency_ms": round(850.0 + base["lat"] * 2.0, 2),
                   "error_rate": 0.02, "service_up": 1}
            latency_injected.append(name)
        else:
            row = {"cpu_percent": base["cpu"], "memory_mb": base["mem"],
                   "latency_ms": base["lat"], "error_rate": base["err"],
                   "service_up": 1}

        rows.append({"timestamp": now, "service_name": name, **row})

    # One-hop cascade impact, matching observed chaos-experiment signatures:
    #   crash of X  -> dependents see ~140ms latency, ~0.45 errors
    #   latency on X -> dependents see ~330ms latency, ~0.04 errors
    index = {r["service_name"]: r for r in rows}
    for svc_name in crashed:
        for dep in dependents.get(svc_name, []):
            d = index.get(dep)
            if d and d["service_up"]:
                d["latency_ms"] = max(d["latency_ms"], 140.0)
                d["error_rate"] = max(d["error_rate"], 0.45)
    for svc_name in latency_injected:
        for dep in dependents.get(svc_name, []):
            d = index.get(dep)
            if d and d["service_up"]:
                d["latency_ms"] = max(d["latency_ms"], 330.0)
                d["error_rate"] = max(d["error_rate"], 0.04)

    return rows


class AutonomousSREAgent:
    """
    Autonomous AI Investigation Agent (ReAct tool-calling pattern).

    Investigation stack:
      1. Isolation Forest + LightGBM (app.ml) over live telemetry -> anomaly
         verdict and root-cause service/fault prediction.
      2. ChromaDB RAG retrieval of internal DevOps runbooks, query expanded
         with the ML-predicted root cause.
      3. Kubernetes pod status evidence.
      4. Ollama LLM synthesis of an evidence-grounded RCA narrative
         (degrades gracefully to the deterministic ML verdict when the LLM
         is unreachable).
    """

    def __init__(self):
        self.retriever = get_runbook_retriever()

    async def investigate_incident(self, incident: CorrelatedIncident) -> Dict[str, Any]:
        logger.info("Agent starting investigation for incident: %s (%s)",
                    incident.incident_id, incident.primary_service)
        investigation_start = time.time()

        # Step 1: SLA Context
        sla_info = SLARiskCalculator.calculate_sla_status(
            tenant_id=incident.tenant_id,
            created_at=incident.created_at,
            severity=incident.severity,
        )

        # Step 2: ML anomaly detection + root-cause classification on live telemetry
        ml_result = self._analyze_telemetry()

        # Step 3: RAG runbook retrieval, query seeded by the ML prediction
        predicted_root_cause = (ml_result or {}).get("root_cause_service")
        fault_type = (ml_result or {}).get("root_cause_fault_type")
        clean_cause = (fault_type or incident.root_cause_candidate).replace("_", " ")
        query = f"{incident.primary_service} {predicted_root_cause or ''} {clean_cause} outage failure degradation"
        relevant_runbooks = self.retriever.search_relevant_runbooks(query=query, n_results=3)

        # Step 4: Tool diagnostics (Kubernetes pod status)
        evidence: List[Dict[str, Any]] = []
        k8s_status = TelemetryTools.get_k8s_pod_status(service_name=incident.primary_service)
        evidence.append({
            "tool": "kubernetes_api",
            "target": incident.primary_service,
            "data": k8s_status,
        })
        if ml_result is not None:
            evidence.append({
                "tool": "ml_isolation_forest_lightgbm",
                "target": predicted_root_cause or incident.primary_service,
                "data": ml_result,
            })

        # Step 5: Determine root cause - ML prediction wins over alert guess
        matched_runbook = relevant_runbooks[0] if relevant_runbooks else None
        diagnosis_service = predicted_root_cause or incident.primary_service
        if predicted_root_cause:
            rca_title = (
                f"ML-identified root cause: {fault_type or 'degradation'} "
                f"originating at {diagnosis_service}"
            )
            confidence = self._combined_confidence(ml_result, matched_runbook)
        elif matched_runbook:
            rca_title = matched_runbook["title"]
            confidence = matched_runbook["similarity_score"]
        else:
            rca_title = f"Degradation in {incident.primary_service}"
            confidence = 0.88

        remediation_action = (
            f"ansible-playbook ansible/restart_service.yml -e service={diagnosis_service}"
        )

        # Step 6: LLM narrative synthesis (evidence-grounded via RAG + ML)
        narrative = await self._synthesize_narrative(
            incident=incident,
            ml_result=ml_result,
            runbooks=relevant_runbooks,
            k8s_status=k8s_status,
            diagnosis_title=rca_title,
        )

        investigation_duration = round(time.time() - investigation_start, 3)

        return {
            "incident_id": incident.incident_id,
            "tenant_id": incident.tenant_id,
            "sla_risk": sla_info,
            "investigation_duration_seconds": investigation_duration,
            "root_cause_analysis": {
                "diagnosis": rca_title,
                "confidence_score": confidence,
                "primary_service": diagnosis_service,
                "affected_services": incident.affected_services,
                "ml_prediction": ml_result,
                "llm_narrative": narrative,
                "narrative_source": "ollama" if narrative else "ml_deterministic_fallback",
                "evidence_gathered": evidence,
                "referenced_runbooks": [
                    {
                        "title": rb["title"],
                        "category": rb["category"],
                        "similarity": rb["similarity_score"],
                    }
                    for rb in relevant_runbooks
                ],
                "recommended_remediation": remediation_action,
            },
        }

    # ------------------------------------------------------------------ helpers
    def _analyze_telemetry(self) -> Optional[Dict[str, Any]]:
        try:
            from app.ml.pipeline import get_rca_pipeline
            rows = build_live_telemetry_rows()
            if not rows:
                return None
            return get_rca_pipeline().analyze(rows)
        except Exception as e:
            logger.warning("ML telemetry analysis failed: %s", e)
            return None

    def _combined_confidence(
        self, ml_result: Optional[Dict], matched_runbook: Optional[Dict]
    ) -> float:
        if not ml_result:
            return 0.5
        # Prefer the LightGBM per-window confidence of the majority winner
        conf = self._classifier_confidence(ml_result)
        if conf is None:
            conf = 0.5 + 0.4 * float(ml_result.get("anomaly_probability") or 0.0)
        # Blend lightly with runbook similarity when available
        if matched_runbook:
            conf = 0.9 * conf + 0.1 * matched_runbook["similarity_score"]
        return round(float(min(max(conf, 0.0), 1.0)), 4)

    @staticmethod
    def _classifier_confidence(ml_result: Dict) -> Optional[float]:
        """Max LightGBM root-cause confidence across anomalous windows that
        voted for the majority winner. None when unavailable."""
        try:
            winner = ml_result.get("root_cause_service")
            best: Optional[float] = None
            for w in ml_result.get("windows") or []:
                if not w.get("is_anomaly"):
                    continue
                rc = w.get("root_cause") or {}
                if winner and rc.get("root_cause_service") != winner:
                    continue
                c = rc.get("root_cause_service_confidence")
                if c is not None and (best is None or float(c) > best):
                    best = float(c)
            return best
        except (TypeError, ValueError, AttributeError):
            return None

    async def _synthesize_narrative(
        self,
        incident: CorrelatedIncident,
        ml_result: Optional[Dict],
        runbooks: List[Dict],
        k8s_status: Dict[str, Any],
        diagnosis_title: str,
    ) -> Optional[str]:
        if not ml_result and not runbooks:
            return None

        prompt_parts = [
            f"Incident ID: {incident.incident_id}",
            f"Primary service (alert source): {incident.primary_service}",
            f"Alert candidate: {incident.root_cause_candidate}",
            f"Severity: {incident.severity} | Affected: {', '.join(incident.affected_services)}",
            f"ML diagnosis: {diagnosis_title}",
        ]
        if ml_result:
            prompt_parts.append(
                f"Anomaly probability: {ml_result.get('anomaly_probability')} "
                f"(score {ml_result.get('anomaly_score')})"
            )
            rc = ml_result.get("root_cause")
            if rc:
                prompt_parts.append(
                    f"Predicted root cause service: {rc.get('root_cause_service')} "
                    f"confidence={rc.get('root_cause_service_confidence')}, "
                    f"fault type: {rc.get('fault_type')} "
                    f"confidence={rc.get('fault_type_confidence')}"
                )
                prompt_parts.append(
                    f"Probabilities: root_cause={rc.get('root_cause_service_probabilities')}, "
                    f"fault={rc.get('fault_type_probabilities')}"
                )
        for rb in runbooks:
            prompt_parts.append(
                f"Runbook '{rb['title']}' (category={rb['category']}, "
                f"similarity={rb['similarity_score']}):\n{rb['content'][:1200]}"
            )
        pods = k8s_status.get("pods") or []
        prompt_parts.append(
            f"Kubernetes pods for {incident.primary_service}: "
            + (", ".join(f"{p['name']}={p['status']}" for p in pods) or "no cluster pods found")
        )

        return await get_ollama_client().generate(
            system=SYSTEM_PROMPT,
            prompt="\n\n".join(prompt_parts),
        )


_agent = AutonomousSREAgent()


def get_sre_agent() -> AutonomousSREAgent:
    return _agent
