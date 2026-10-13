"""Keyword intent router: decides which read-only sources to fetch. No LLM involved."""
import re

PATTERNS = {
    "status":    r"status|health|\bup\b|\bdown\b|running|alive|overview|everything|state|mesh|service",
    "incidents": r"incident|rca|root.?cause|alert|\bsla\b|approval|anomal|remediat",
    "metrics":   r"metric|latency|\bcpu\b|memory|\bram\b|\bp9\d|request|throughput|prometheus",
    "logs":      r"\blogs?\b|error|exception|warn|fail|crash|traceback|wrong|problem",
    "docs":      r"\bhow\b|what is|what's|explain|start|run|setup|install|\buse\b|why|architecture|"
                 r"n8n|telegram|runbook|dashboard|train|test|model|rag|chaos|sentinelas|\bdoc",
}


def route(question: str) -> set[str]:
    q = question.lower()
    found = {name for name, pat in PATTERNS.items() if re.search(pat, q)}
    return found or {"status", "docs"}