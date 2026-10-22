"""Immutable domain models shared across ingestion, retrieval and the API."""

from dataclasses import dataclass, field
from enum import StrEnum

REFUSAL_MESSAGE = (
    "I couldn't find this in our documents. Please rephrase the question or contact a specialist."
)
EMPTY_LIBRARY_MESSAGE = "No documents have been loaded yet, so I can't answer from them."


@dataclass(frozen=True, slots=True)
class PageText:
    doc_name: str
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Source:
    doc_name: str
    page: int
    excerpt: str
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    answer: str
    refused: bool
    sources: tuple[Source, ...] = ()

    def __post_init__(self) -> None:
        if self.refused and self.sources:
            raise ValueError("A refused answer cannot carry sources")
        if not self.refused and not self.sources:
            raise ValueError("A grounded answer must cite at least one source")

    @classmethod
    def refusal(cls, message: str = REFUSAL_MESSAGE) -> "Answer":
        return cls(answer=message, refused=True, sources=())


class FileStatus(StrEnum):
    LOADED = "loaded"
    SKIPPED_NO_TEXT = "skipped_no_text"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FileIngestResult:
    doc_name: str
    status: FileStatus
    pages: int = 0
    chunks: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class IngestReport:
    results: tuple[FileIngestResult, ...] = field(default_factory=tuple)
    pruned: int = 0  # passages removed for PDFs no longer in the folder

    @property
    def loaded(self) -> int:
        return sum(1 for r in self.results if r.status is FileStatus.LOADED)

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks for r in self.results)

    @property
    def problems(self) -> tuple[FileIngestResult, ...]:
        return tuple(r for r in self.results if r.status is not FileStatus.LOADED)
