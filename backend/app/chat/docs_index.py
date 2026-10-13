"""Chunk project markdown docs and index them in a separate Chroma collection."""
import logging
import re
from pathlib import Path

import chromadb

from app.chat.logbuffer import redact
from app.config import settings

logger = logging.getLogger(__name__)

REPO_ROOT: Path = settings.BASE_DIR.parent
DOC_FILES = [
    "README.md", "AI_UNDERSTANDING.md", "DEVOPS_UNDERSTANDING.md",
    "DEMO_AND_UNDERSTANDING_GUIDE.md", "AI_ML_IMPLEMENTATION_REPORT.md", "n8n/README.md",
]
PERSIST_DIR: Path = settings.BASE_DIR / "data" / "chroma_docs"   # gitignored (backend/data/)
_collection = None


def _chunk_markdown(text: str, max_chars: int = 1000) -> list[tuple[str, str]]:
    chunks = []
    for part in re.split(r"(?m)^(?=#{1,3}\s)", text):
        part = part.strip()
        if not part:
            continue
        heading = part.splitlines()[0].lstrip("# ").strip()[:80]
        for i in range(0, len(part), max_chars):
            chunks.append((heading, part[i:i + max_chars]))
    return chunks


def _get_collection():
    global _collection
    if _collection is None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        _collection = client.get_or_create_collection("project_docs")
    return _collection


def index_docs() -> int:
    col = _get_collection()
    paths = [REPO_ROOT / f for f in DOC_FILES if (REPO_ROOT / f).exists()]
    paths += sorted(Path(settings.RUNBOOKS_DIR).glob("*.md"))
    ids, docs, metas = [], [], []
    for path in paths:
        rel = str(path.relative_to(REPO_ROOT))
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, (heading, chunk) in enumerate(_chunk_markdown(text)):
            ids.append(f"{rel}::{i}")
            docs.append(chunk)
            metas.append({"file": rel, "heading": heading})
    stale = set(col.get()["ids"]) - set(ids)
    if stale:
        col.delete(ids=list(stale))
    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    logger.info("Chat docs index: %d chunks from %d files", len(ids), len(paths))
    return len(ids)


def search_docs(query: str, k: int = 4) -> list[dict]:
    col = _get_collection()
    if col.count() == 0:
        index_docs()
    res = col.query(query_texts=[query], n_results=k)
    out = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        out.append({"file": meta["file"], "heading": meta["heading"], "text": redact(doc)})
    return out