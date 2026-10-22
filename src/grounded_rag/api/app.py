"""FastAPI service exposing /health and /chat."""

import logging
from typing import Annotated, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, StringConstraints

from grounded_rag.models import Answer
from grounded_rag.rag.engine import UpstreamError

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000


class Answerer(Protocol):
    def answer(self, question: str) -> Answer: ...


class Counter(Protocol):
    def count(self) -> int: ...


Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
]


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    question: Question


class SourceOut(BaseModel):
    doc_name: str
    page: int
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceOut]


class HealthResponse(BaseModel):
    status: str
    chunks: int


def _to_response(answer: Answer) -> ChatResponse:
    sources = [
        SourceOut(doc_name=s.doc_name, page=s.page, excerpt=s.excerpt, score=s.score)
        for s in answer.sources
    ]
    return ChatResponse(answer=answer.answer, refused=answer.refused, sources=sources)


def create_app(answerer: Answerer, counter: Counter) -> FastAPI:
    app = FastAPI(title="Grounded RAG", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            return HealthResponse(status="ok", chunks=counter.count())
        except Exception as exc:
            logger.exception("Health check failed")
            raise HTTPException(status_code=503, detail="Vector store unavailable") from exc

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            return _to_response(answerer.answer(request.question))
        except UpstreamError as exc:
            logger.exception("Upstream failure while answering")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def create_default_app() -> FastAPI:
    """Uvicorn factory: `uvicorn grounded_rag.api.app:create_default_app --factory`."""
    from grounded_rag.config import Settings
    from grounded_rag.factory import build_answerer, build_store

    settings = Settings()
    store = build_store(settings)
    return create_app(build_answerer(settings, store), store)
