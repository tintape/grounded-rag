"""Load every PDF in a folder into the passage store, reporting per-file outcomes."""

import logging
from pathlib import Path
from typing import Protocol

from grounded_rag.ingest.chunker import chunk_pages, document_id
from grounded_rag.ingest.pdf_reader import read_pdf
from grounded_rag.models import Chunk, FileIngestResult, FileStatus, IngestReport

logger = logging.getLogger(__name__)


class DocumentSink(Protocol):
    def replace_document(self, doc_id: str, chunks: tuple[Chunk, ...]) -> int: ...


def _ingest_file(path: Path, sink: DocumentSink, chunk_size: int, overlap: int) -> FileIngestResult:
    read = read_pdf(path)
    if read.status is not FileStatus.LOADED:
        return FileIngestResult(read.doc_name, read.status, pages=read.page_count, error=read.error)
    chunks = chunk_pages(read.pages, chunk_size=chunk_size, overlap=overlap)
    try:
        stored = sink.replace_document(document_id(read.doc_name), chunks)
    except Exception as exc:  # report and keep going; one bad upload must not stop the batch
        logger.exception("Failed to store %s", read.doc_name)
        return FileIngestResult(read.doc_name, FileStatus.FAILED, read.page_count, error=str(exc))
    return FileIngestResult(read.doc_name, FileStatus.LOADED, read.page_count, chunks=stored)


def ingest_folder(folder: Path, sink: DocumentSink, chunk_size: int, overlap: int) -> IngestReport:
    if not folder.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {folder}")
    pdfs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    results = tuple(_ingest_file(p, sink, chunk_size, overlap) for p in pdfs)
    report = IngestReport(results=results)
    logger.info("Ingested %d/%d files, %d chunks", report.loaded, len(results), report.total_chunks)
    return report
