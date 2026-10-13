"""In-memory ring buffer of recent app logs + secret redaction."""
import logging
import re
from collections import deque
from datetime import datetime

_SECRET_PATTERNS = [
    (re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b"), "[REDACTED_TOKEN]"),   # Telegram bot token
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "[REDACTED_KEY]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b(password|passwd|token|secret|api[_-]?key)\s*[=:]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[REDACTED_EMAIL]"),
]
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_NOISY = ("httpx", "httpcore", "chromadb")


def redact(text: str) -> str:
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


class RingBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 500):
        super().__init__(logging.INFO)
        self.records = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append({
                "ts": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "msg": redact(record.getMessage()),
            })
        except Exception:
            pass  # logging must never break the app


_handler: RingBufferHandler | None = None


def install() -> RingBufferHandler:
    global _handler
    if _handler is None:
        _handler = RingBufferHandler()
        logging.getLogger().addHandler(_handler)
    return _handler


def recent(n: int = 40, min_level: str = "INFO") -> list[dict]:
    handler = install()
    floor = _LEVELS.get(min_level, 20)
    rows = [r for r in handler.records
            if _LEVELS.get(r["level"], 20) >= floor and not r["logger"].startswith(_NOISY)]
    return rows[-n:]