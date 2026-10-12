"""n8n outbound webhook notifier.

Fires an HTTP POST to the configured N8N_WEBHOOK_URL whenever an anomaly
is confirmed by the ML pipeline.  The call is intentionally non-blocking:
failures are logged but never raise, so the core API path is unaffected.
"""
import logging
import time
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ── Payload schema ────────────────────────────────────────────────────────────

def build_payload(
    *,
    service_name: str,
    anomaly_type: str,
    confidence_score: float,
    status: str = "ANOMALY_DETECTED",
    timestamp: Optional[float] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Return the canonical Sentinelas anomaly payload sent to n8n."""
    payload = {
        "service_name": service_name,
        "anomaly_type": anomaly_type,
        "confidence_score": round(confidence_score, 4),
        "timestamp": timestamp if timestamp is not None else time.time(),
        "status": status,
    }
    if extra:
        payload.update(extra)
    return payload


# ── Fire-and-forget notifier ──────────────────────────────────────────────────

async def notify_n8n(payload: dict) -> bool:
    """POST *payload* to the n8n webhook URL defined in settings.

    The payload is stripped to the five canonical fields before sending so the
    n8n workflow always receives a clean, predictable body regardless of what
    internal extras are attached to the object.

    Returns True on HTTP 2xx, False on any error (network, timeout, non-2xx).
    Never raises.
    """
    url = settings.N8N_WEBHOOK_URL
    if not url:
        logger.debug("N8N_WEBHOOK_URL is not configured — skipping notification")
        return False

    # Send the five required fields to n8n
    outbound = {
        "service_name":     payload.get("service_name", "unknown"),
        "anomaly_type":     payload.get("anomaly_type", "unknown"),
        "confidence_score": payload.get("confidence_score", 0.0),
        "timestamp":        payload.get("timestamp", time.time()),
        "status":           payload.get("status", "ANOMALY_DETECTED"),
    }

    # Optional RCA context, forwarded when present so the n8n alert can name
    # the actual root cause instead of falling back to "Investigating...".
    for key in ("anomaly_score", "root_cause_service", "root_cause_fault_type"):
        if payload.get(key) is not None:
            outbound[key] = payload[key]

    logger.info(
        "Sending anomaly alert to n8n → %s | service=%s anomaly_type=%s score=%.4f",
        url,
        outbound["service_name"],
        outbound["anomaly_type"],
        outbound["confidence_score"],
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=outbound)
        if resp.is_success:
            logger.info(
                "n8n ACK [%s] service=%s anomaly_type=%s score=%.4f",
                resp.status_code,
                outbound["service_name"],
                outbound["anomaly_type"],
                outbound["confidence_score"],
            )
            return True
        else:
            logger.warning(
                "n8n returned non-2xx [%s] for service=%s: %s",
                resp.status_code,
                outbound["service_name"],
                resp.text[:200],
            )
            return False
    except httpx.TimeoutException:
        logger.warning("n8n notification timed out for service=%s", outbound["service_name"])
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("n8n notification failed for service=%s: %s", outbound["service_name"], exc)
        return False
