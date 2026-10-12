"""Ollama LLM client for evidence-grounded RCA narrative synthesis."""
import json
import logging
from typing import Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMPayloadError(RuntimeError):
    pass


class OllamaClient:
    """Thin async client for the local Ollama chat API."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT_SECONDS

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        system: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> Optional[str]:
        """Chat completion. Returns None on any transport/model error so
        callers can degrade gracefully to the deterministic ML output."""
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                if resp.status_code != 200:
                    logger.warning("Ollama HTTP %s: %s", resp.status_code, resp.text[:200])
                    return None
                data = resp.json()
                content = (data.get("message") or {}).get("content")
                if not content:
                    logger.warning("Ollama returned empty message: %s", json.dumps(data)[:300])
                    return None
                return content.strip()
        except Exception as e:
            logger.warning("Ollama call failed (%s: %s) - degrading to ML-only RCA",
                           type(e).__name__, e)
            return None


_ollama: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    global _ollama
    if _ollama is None:
        _ollama = OllamaClient()
    return _ollama
