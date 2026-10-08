"""Tests for the Ollama LLM client and ML-enhanced SRE agent."""
import pytest

from app.agent.llm import OllamaClient, get_ollama_client


def test_singleton_client_configured():
    client = get_ollama_client()
    assert client.base_url.startswith("http")
    assert client.model


@pytest.mark.asyncio
async def test_generate_degrades_gracefully_when_unreachable():
    client = OllamaClient(base_url="http://localhost:1", model="missing-model")
    result = await client.generate(system="sys", prompt="hi")
    assert result is None


@pytest.mark.asyncio
async def test_generate_returns_text_or_none():
    """If Ollama is up, expect text; if down, None. Never raise."""
    client = get_ollama_client()
    result = await client.generate(
        system="Answer with one word.",
        prompt="Say OK",
        max_tokens=10,
    )
    assert result is None or isinstance(result, str)


def test_agent_investigation_includes_ml_and_runbooks(test_client):
    # Create an incident
    test_client.post("/api/v1/alerts/webhook", json={
        "id": "alt-ml-1",
        "alertname": "RedisConnectionFailed",
        "service": "cart-service",
        "severity": "critical",
        "tenant_id": "tenant_c",
    })
    incidents = test_client.get("/api/v1/incidents?tenant_id=tenant_c").json()
    assert len(incidents) > 0
    incident_id = incidents[0]["incident"]["incident_id"]

    resp = test_client.post(f"/api/v1/incidents/{incident_id}/diagnose")
    assert resp.status_code == 200
    rca = resp.json()["root_cause_analysis"]

    # Structural contract
    assert "diagnosis" in rca
    assert 0.0 <= rca["confidence_score"] <= 1.0
    assert "ml_prediction" in rca
    assert "llm_narrative" in rca
    assert rca["narrative_source"] in ("ollama", "ml_deterministic_fallback")
    assert len(rca["referenced_runbooks"]) > 0
    assert "recommended_remediation" in rca

    # When ML ran, its output must be well-formed
    ml = rca["ml_prediction"]
    if ml is not None:
        assert "is_anomaly" in ml
        assert "anomaly_probability" in ml


def test_agent_deterministic_fallback_without_llm(monkeypatch):
    """If Ollama is unreachable, agent still returns the ML verdict."""
    async def _unreachable(self, system, prompt, temperature=0.2, max_tokens=900):
        return None

    monkeypatch.setattr(OllamaClient, "generate", _unreachable)

    from app.agent.sre_agent import get_sre_agent
    from app.correlation.engine import RawAlert
    import time

    agent = get_sre_agent()
    alert = RawAlert(
        id="alt-fallback-1",
        alertname="PodCrashLooping",
        service="frontend",
        severity="critical",
        tenant_id="tenant_a",
        timestamp=time.time(),
    )
    from app.correlation.engine import get_correlation_engine
    incident = get_correlation_engine().correlate(alert)

    import asyncio
    result = asyncio.run(agent.investigate_incident(incident))
    rca = result["root_cause_analysis"]
    assert rca["narrative_source"] == "ml_deterministic_fallback"
    assert rca["llm_narrative"] is None
    assert rca["confidence_score"] > 0
