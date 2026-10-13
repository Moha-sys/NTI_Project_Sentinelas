import json
import logging
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chat import docs_index
from app.chat.service import answer_stream

 # start capturing logs as soon as the router is imported
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat Assistant"])


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=8)


@router.post("")
async def chat(req: ChatRequest):
    history = [t.model_dump() for t in req.history]

    async def event_stream():
        async for event in answer_stream(req.message, history):
            yield f"data: {json.dumps(event)}\n\n"
        yield 'data: {"type":"done"}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/reindex")
def reindex():
    """Rebuild the docs index after documentation changes."""
    return {"chunks_indexed": docs_index.index_docs()}