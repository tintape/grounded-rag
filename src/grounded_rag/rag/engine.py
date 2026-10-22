"""Grounded answering: relevance gate -> generation -> citation validation."""

import logging
from dataclasses import dataclass
from typing import Protocol

from grounded_rag.models import (
    EMPTY_LIBRARY_MESSAGE,
    Answer,
    RetrievedChunk,
    Source,
)
from grounded_rag.rag.citations import cited_indices, rewrite_markers
from grounded_rag.rag.prompts import NOT_FOUND, SYSTEM_PROMPT, build_user_prompt
from grounded_rag.rag.sanitize import sanitize_answer

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> tuple[RetrievedChunk, ...]: ...
    def count(self) -> int: ...


class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class UpstreamError(RuntimeError):
    """The vector store or the language model failed; never answer in this case."""


@dataclass(frozen=True, slots=True)
class EngineConfig:
    top_k: int
    min_relevance: float
    excerpt_chars: int = 300


def _excerpt(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


class GroundedAnswerer:
    def __init__(self, retriever: Retriever, generator: Generator, config: EngineConfig) -> None:
        self._retriever = retriever
        self._generator = generator
        self._config = config

    def answer(self, question: str) -> Answer:
        query = question.strip()
        if not query:
            raise ValueError("Question must not be empty")
        hits = self._retrieve(query)
        relevant = tuple(h for h in hits if h.score >= self._config.min_relevance)
        if not relevant:
            if not hits and self._retriever.count() == 0:
                return Answer.refusal(EMPTY_LIBRARY_MESSAGE)
            logger.info("Refused before generation: best score below threshold")
            return Answer.refusal()
        raw = self._generate(query, relevant)
        return self._ground(raw, relevant)

    def _retrieve(self, query: str) -> tuple[RetrievedChunk, ...]:
        try:
            return self._retriever.search(query, self._config.top_k)
        except Exception as exc:
            raise UpstreamError("Document search is unavailable") from exc

    def _generate(self, query: str, hits: tuple[RetrievedChunk, ...]) -> str:
        try:
            return self._generator.generate(SYSTEM_PROMPT, build_user_prompt(query, hits))
        except Exception as exc:
            raise UpstreamError("The answer service is unavailable") from exc

    def _ground(self, raw: str, hits: tuple[RetrievedChunk, ...]) -> Answer:
        if NOT_FOUND in raw:
            return Answer.refusal()
        raw = sanitize_answer(raw)
        cited = cited_indices(raw, len(hits))
        if not cited:
            logger.info("Refused after generation: no valid citations")
            return Answer.refusal()
        sources: list[Source] = []
        page_to_number: dict[tuple[str, int], int] = {}
        mapping: dict[int, int] = {}
        for passage_number in cited:
            hit = hits[passage_number - 1]
            key = (hit.chunk.doc_name, hit.chunk.page)
            if key not in page_to_number:
                page_to_number[key] = len(sources) + 1
                sources.append(self._source(hit))
            mapping[passage_number] = page_to_number[key]
        return Answer(answer=rewrite_markers(raw, mapping), refused=False, sources=tuple(sources))

    def _source(self, hit: RetrievedChunk) -> Source:
        return Source(
            doc_name=hit.chunk.doc_name,
            page=hit.chunk.page,
            excerpt=_excerpt(hit.chunk.text, self._config.excerpt_chars),
            score=round(hit.score, 3),
        )
