import json
import logging

from app.chat import docs_index, service
from app.chat.logbuffer import RingBufferHandler, redact
from app.chat.router import route


def test_router_picks_sources():
    assert "status" in route("is everything up?")
    assert "logs" in route("show me recent errors")
    assert "docs" in route("how do I start the project?")
    assert "metrics" in route("what is the cpu usage?")
    assert route("hello") == {"status", "docs"}


def test_redaction():
    token = "123456789:" + "A" * 35
    assert "[REDACTED_TOKEN]" in redact(f"bot {token} failed")
    assert "[REDACTED_EMAIL]" in redact("sent to person@example.com")
    assert "hunter2" not in redact("password=hunter2")


def test_ring_buffer_keeps_only_capacity():
    h = RingBufferHandler(capacity=3)
    for i in range(5):
        h.emit(logging.LogRecord("t", logging.INFO, "", 0, f"m{i}", None, None))
    assert [r["msg"] for r in h.records] == ["m2", "m3", "m4"]


def test_markdown_chunker_splits_on_headings():
    chunks = docs_index._chunk_markdown("# A\ntext a\n## B\ntext b")
    assert [c[0] for c in chunks] == ["A", "B"]


def _events(response):
    return [json.loads(l[6:]) for l in response.text.splitlines() if l.startswith("data: ")]


def test_chat_streams_answer_with_evidence(test_client, monkeypatch):
    async def fake_ctx(intents, question):
        return [service.Source("status", "mesh: 5/5 healthy")]

    async def fake_llm(messages):
        assert "mesh: 5/5 healthy" in messages[-1]["content"]
        yield "All "
        yield "healthy [S1]"

    monkeypatch.setattr(service, "gather_context", fake_ctx)
    monkeypatch.setattr(service, "stream_llm", fake_llm)
    events = _events(test_client.post("/api/v1/chat", json={"message": "is everything up?"}))
    assert events[0]["type"] == "sources"
    assert "healthy [S1]" in "".join(e["text"] for e in events if e["type"] == "token")
    assert events[-1]["type"] == "done"


def test_chat_degrades_when_llm_is_down(test_client, monkeypatch):
    async def fake_ctx(intents, question):
        return [service.Source("status", "mesh: 5/5 healthy")]

    async def broken_llm(messages):
        raise RuntimeError("ollama down")
        yield  # pragma: no cover

    monkeypatch.setattr(service, "gather_context", fake_ctx)
    monkeypatch.setattr(service, "stream_llm", broken_llm)
    events = _events(test_client.post("/api/v1/chat", json={"message": "status?"}))
    assert any(e["type"] == "error" for e in events)
    assert "mesh: 5/5 healthy" in "".join(e.get("text", "") for e in events)
    assert events[-1]["type"] == "done"


def test_chat_rejects_empty_message(test_client):
    assert test_client.post("/api/v1/chat", json={"message": ""}).status_code == 422