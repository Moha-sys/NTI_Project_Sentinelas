"""Read-only evidence fetchers. Each returns compact text; every call is size-capped."""
import json

from app.chat import docs_index, logbuffer


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def _trim(value, n=160):
    return value[:n] + "…" if isinstance(value, str) and len(value) > n else value


def _shared_process_note(rows) -> str:
    """If every service reports the same pid, cpu/memory are one process's usage repeated."""
    pids = {r.get("pid") for r in rows if r.get("pid") is not None}
    if len(rows) > 1 and len(pids) == 1:
        return ("NOTE: all services report the same pid, so they run inside one process. "
                "cpu_percent and memory_mb are that process's usage repeated for every service, "
                "not per-service values.\n")
    return ""


def fetch_status() -> str:
    from app.api.incidents import platform_settings
    from app.correlation.engine import get_correlation_engine
    from app.mesh.manager import get_mesh_manager

    rows = get_mesh_manager().get_mesh_status()
    incidents = get_correlation_engine().get_all_incidents()
    by_status: dict[str, int] = {}
    for inc in incidents:
        by_status[inc.status] = by_status.get(inc.status, 0) + 1
    summary = {
        "mesh_services": json.loads(json.dumps(rows, default=str)),
        "incident_counts_by_status": by_status,
        "autonomous_mode": platform_settings.autonomous_mode,
    }
    return _cap(_shared_process_note(rows) + json.dumps(summary, separators=(",", ":")), 3200)


def fetch_incidents(limit: int = 6) -> str:
    from app.correlation.engine import get_correlation_engine
    from app.sla.calculator import SLARiskCalculator

    items = []
    for inc in get_correlation_engine().get_all_incidents():
        try:
            data = {k: _trim(v) for k, v in inc.model_dump().items()}
            sla = SLARiskCalculator.calculate_sla_status(
                tenant_id=inc.tenant_id, created_at=inc.created_at, severity=inc.severity)
            data["sla_risk_score"] = sla.get("risk_score")
            items.append(data)
        except Exception as exc:  # one bad incident must not hide the others
            items.append({"error": f"could not read incident: {exc}"})
    items.sort(key=lambda d: str(d.get("created_at", "")), reverse=True)
    if not items:
        return "No incidents recorded."
    return _cap(json.dumps(items[:limit], default=str, separators=(",", ":")), 2500)


def fetch_metrics() -> str:
    from app.mesh.manager import get_mesh_manager

    mgr = get_mesh_manager()
    rows = mgr.get_mesh_status()
    raw = mgr.scrape_prometheus_metrics() or ""
    lines = [ln for ln in raw.splitlines() if ln and not ln.startswith("#")][:50]
    text = (_shared_process_note(rows)
            + "Per-service readings from the local mesh. latency_ms and error_rate are the mesh's own "
              "readings (how they are measured is not documented here). added_latency_ms is artificial "
              "delay injected by chaos experiments, not measured latency.\n"
            + json.dumps(rows, default=str, separators=(",", ":"))
            + "\nRaw Prometheus samples (first 50):\n" + "\n".join(lines))
    return _cap(text, 3200)


def _fmt_logs(rows) -> str:
    return "\n".join(f"{r['ts']} {r['level']} {r['logger']}: {r['msg']}" for r in rows)


def fetch_logs(question: str, n: int = 40) -> str:
    wants_errors = any(w in question.lower()
                       for w in ("error", "exception", "fail", "warn", "crash", "wrong", "problem"))
    if wants_errors:
        rows = logbuffer.recent(n=n, min_level="WARNING")
        if rows:
            return _cap(_fmt_logs(rows), 2800)
        context = logbuffer.recent(n=15, min_level="INFO")
        text = ("No WARNING or ERROR log lines have been recorded since the backend started "
                "(the buffer only holds lines since startup).")
        if context:
            text += " Latest INFO lines for context:\n" + _fmt_logs(context)
        return _cap(text, 2800)
    rows = logbuffer.recent(n=n, min_level="INFO")
    if not rows:
        return "The log buffer is empty (it only holds lines since the backend started)."
    return _cap(_fmt_logs(rows), 2800)


def fetch_docs(question: str) -> str:
    hits = docs_index.search_docs(question, k=4)
    return "\n\n".join(f"({h['file']} › {h['heading']})\n{h['text']}" for h in hits)
