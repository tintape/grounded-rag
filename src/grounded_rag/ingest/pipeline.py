"""Load every PDF in a folder into the passage store, reporting per-file outcomes."""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from grounded_rag.ingest.chunker import chunk_pages, document_id
from grounded_rag.ingest.pdf_reader import read_pdf
from grounded_rag.models import Chunk, FileIngestResult, FileStatus, IngestReport

logger = logging.getLogger(__name__)


class DocumentSink(Protocol):
    def replace_document(self, doc_id: str, chunks: tuple[Chunk, ...]) -> int: ...
    def remove_documents_except(self, keep_doc_ids: set[str]) -> int: ...


def _store(sink: DocumentSink, result: FileIngestResult, chunks: tuple[Chunk, ...]):
    try:
        stored = sink.replace_document(document_id(result.doc_name), chunks)
    except Exception as exc:  # report and keep going; one bad upload must not stop the batch
        logger.exception("Failed to store %s", result.doc_name)
        return replace(result, status=FileStatus.FAILED, chunks=0, error=str(exc))
    return replace(result, chunks=stored)


def _ingest_file(path: Path, sink: DocumentSink, chunk_size: int, overlap: int) -> FileIngestResult:
    read = read_pdf(path)
    result = FileIngestResult(read.doc_name, read.status, pages=read.page_count, error=read.error)
    if read.status is not FileStatus.LOADED:
        # The file is present but unusable now: drop any older version so it can't be cited.
        return _store(sink, result, ())
    return _store(sink, result, chunk_pages(read.pages, chunk_size=chunk_size, overlap=overlap))


def ingest_folder(
    folder: Path, sink: DocumentSink, chunk_size: int, overlap: int, prune: bool = False
) -> IngestReport:
    """Ingest all PDFs in `folder`; with `prune`, also delete documents no longer in it."""
    if not folder.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {folder}")
    pdfs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    results = tuple(_ingest_file(p, sink, chunk_size, overlap) for p in pdfs)
    pruned = sink.remove_documents_except({document_id(p.name) for p in pdfs}) if prune else 0
    report = IngestReport(results=results, pruned=pruned)
    logger.info("Ingested %d/%d files, %d chunks", report.loaded, len(results), report.total_chunks)
    return report
