import httpx
import logging
from typing import Dict, Any, Optional
from app.config import settings

logger = logging.getLogger(__name__)

class TelemetryTools:
    """
    Diagnostic toolset used by Autonomous SRE Agent to query observability backends:
    - Prometheus (PromQL metrics)
    - Grafana Loki (LogQL container logs)
    - Grafana Tempo (TraceQL distributed traces)
    - Kubernetes API (pods, deployments, events)
    """

    @classmethod
    async def query_prometheus(cls, query: str) -> Dict[str, Any]:
        """Execute a PromQL instant query against Prometheus."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{settings.PROMETHEUS_URL}/api/v1/query",
                    params={"query": query}
                )
                if resp.status_code == 200:
                    return resp.json().get("data", {})
                return {"error": f"Prometheus query failed with HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.warning(f"Prometheus query error: {e}")
            return {"error": str(e), "mock_data": {"resultType": "vector", "result": []}}

    @classmethod
    async def query_loki(cls, query: str, limit: int = 50) -> Dict[str, Any]:
        """Execute a LogQL query against Grafana Loki."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{settings.LOKI_URL}/loki/api/v1/query_range",
                    params={"query": query, "limit": limit}
                )
                if resp.status_code == 200:
                    return resp.json().get("data", {})
                return {"error": f"Loki query failed with HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            logger.warning(f"Loki query error: {e}")
            return {"error": str(e), "mock_data": {"resultType": "streams", "result": []}}

    @classmethod
    async def query_tempo(cls, trace_id: str) -> Dict[str, Any]:
        """Fetch trace details from Grafana Tempo."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{settings.TEMPO_URL}/api/traces/{trace_id}")
                if resp.status_code == 200:
                    return resp.json()
                return {"error": f"Tempo query failed with HTTP {resp.status_code}"}
        except Exception as e:
            logger.warning(f"Tempo query error: {e}")
            return {"error": str(e)}

    # Circuit breaker: after a connection failure, skip the K8s API for a while
    _k8s_unavailable_until: float = 0.0
    _K8S_RETRY_COOLDOWN_SECONDS: float = 120.0

    @classmethod
    def get_k8s_pod_status(cls, service_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Fetch pod statuses and restart counts from Kubernetes API."""
        import os
        import time as _time
        if _time.time() < cls._k8s_unavailable_until:
            return {"error": "kubernetes API cooling down after previous failure",
                    "service": service_name, "pods": []}
        try:
            from kubernetes import client, config
            loaded = False
            try:
                config.load_incluster_config()
                loaded = True
            except Exception:
                pass

            if not loaded:
                possible_configs = [
                    os.environ.get("KUBECONFIG"),
                    "/home/moha/.kube/config",
                    "/etc/rancher/k3s/k3s.yaml",
                    os.path.expanduser("~/.kube/config"),
                    "/root/.kube/config",
                ]
                for p in possible_configs:
                    if p and os.path.exists(p) and os.access(p, os.R_OK) and os.path.getsize(p) > 0:
                        try:
                            config.load_kube_config(config_file=p)
                            loaded = True
                            break
                        except Exception:
                            continue

            if not loaded:
                return {"error": "No valid kubeconfig found", "service": service_name, "pods": []}

            v1 = client.CoreV1Api()
            all_pods = v1.list_namespaced_pod(namespace=namespace).items
            clean_name = service_name.lower().replace("_", "-")
            alt_name = clean_name.replace("-", "")

            items = []
            for pod in all_pods:
                labels = pod.metadata.labels or {}
                app_label = (labels.get("app") or labels.get("app.kubernetes.io/name") or "").lower()
                pname = pod.metadata.name.lower()

                # Match by label or by name prefix
                if (app_label in [clean_name, alt_name] or
                    pname.startswith(clean_name) or
                    pname.startswith(alt_name)):
                    items.append({
                        "name": pod.metadata.name,
                        "phase": pod.status.phase,
                        "restart_count": sum(c.restart_count for c in (pod.status.container_statuses or [])),
                        "node": pod.spec.node_name or "local-node",
                        "status": pod.status.phase
                    })

            return {"service": service_name, "namespace": namespace, "pods": items}
        except Exception as e:
            import time as _time
            err = str(e)
            # Connection-level failures (no cluster / stale kubeconfig):
            # open the circuit breaker so subsequent calls return instantly.
            if any(k in err for k in ("Max retries exceeded", "ConnectionRefused",
                                      "NewConnectionError", "Timeout", "timed out")):
                cls._k8s_unavailable_until = (
                    _time.time() + cls._K8S_RETRY_COOLDOWN_SECONDS
                )
                logger.warning(
                    "Kubernetes API unreachable - pausing K8s queries for %ss",
                    int(cls._K8S_RETRY_COOLDOWN_SECONDS),
                )
            logger.debug(f"Kubernetes query notice: {e}")
            return {"error": err, "service": service_name, "pods": []}

