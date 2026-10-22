"""Extract clean, per-page text from a PDF, reporting unreadable files instead of hiding them."""

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from grounded_rag.models import FileStatus, PageText

logger = logging.getLogger(__name__)

MIN_PAGE_CHARS = 20
EDGE_LINES = 2  # header/footer candidates: first and last N lines of each page
REPEAT_RATIO = 0.6  # a line on >= 60% of pages' edges is treated as a header/footer

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class PdfReadResult:
    doc_name: str
    status: FileStatus
    pages: tuple[PageText, ...] = ()
    page_count: int = 0
    error: str | None = None


def clean_text(raw: str) -> str:
    """Normalise ligatures, re-join hyphenated words, collapse spaces, drop empty lines."""
    text = unicodedata.normalize("NFKC", raw)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = (_SPACES.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _line_key(line: str) -> str:
    return _DIGITS.sub("#", line.lower())


def _repeated_edge_lines(pages: list[list[str]]) -> set[str]:
    if len(pages) < 2:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counts.update({_line_key(line) for line in edges})
    threshold = max(2, round(len(pages) * REPEAT_RATIO))
    return {key for key, count in counts.items() if count >= threshold}


def _strip_repeated(pages: list[list[str]]) -> list[str]:
    repeated = _repeated_edge_lines(pages)
    return ["\n".join(ln for ln in lines if _line_key(ln) not in repeated) for lines in pages]


def read_pdf(path: Path) -> PdfReadResult:
    name = path.name
    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                return PdfReadResult(name, FileStatus.FAILED, error="PDF is password-protected")
            raw_pages = [clean_text(page.get_text("text", sort=True)) for page in doc]
    except Exception as exc:  # untrusted input: any parser error means "failed", not a crash
        logger.warning("Could not read %s: %s", name, exc)
        return PdfReadResult(name, FileStatus.FAILED, error=f"Unreadable PDF: {exc}")

    texts = _strip_repeated([text.splitlines() for text in raw_pages])
    pages = tuple(
        PageText(doc_name=name, page=number, text=text)
        for number, text in enumerate(texts, start=1)
        if len(text) >= MIN_PAGE_CHARS
    )
    status = FileStatus.LOADED if pages else FileStatus.SKIPPED_NO_TEXT
    return PdfReadResult(name, status, pages=pages, page_count=len(raw_pages))
