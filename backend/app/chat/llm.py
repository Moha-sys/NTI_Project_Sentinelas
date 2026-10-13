"""Streaming Ollama client. Model comes from CHAT_OLLAMA_MODEL, falling back to OLLAMA_MODEL."""
import json
import os
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv

from app.config import settings

load_dotenv(settings.BASE_DIR / ".env")
CHAT_MODEL = os.getenv("CHAT_OLLAMA_MODEL", settings.OLLAMA_MODEL)


async def stream_llm(messages: list[dict]) -> AsyncIterator[str]:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,          # [verify] disables thinking mode on models that have it
        "keep_alive": "30m",     # keep the model loaded between questions
        "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 500},
    }
    timeout = httpx.Timeout(connect=5.0, read=settings.OLLAMA_TIMEOUT_SECONDS, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{settings.OLLAMA_URL}/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="ignore")[:200]
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break