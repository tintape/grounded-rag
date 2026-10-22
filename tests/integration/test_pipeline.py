from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from grounded_rag.ingest.pipeline import ingest_folder
from grounded_rag.models import FileStatus
from grounded_rag.store import ChunkStore
from tests.fakes import KeywordEmbedding

TEXT = "Replace the HEPA filter every three months to keep suction strong."


@pytest.fixture
def store() -> ChunkStore:
    return ChunkStore(QdrantClient(location=":memory:"), "test", KeywordEmbedding())


def test_reports_every_file_outcome(make_pdf, tmp_path: Path, store):
    make_pdf("good.pdf", [f"Page one. {TEXT}", f"Page two. {TEXT}"])
    make_pdf("blank.pdf", ["", ""])
    (tmp_path / "broken.pdf").write_bytes(b"garbage")
    (tmp_path / "notes.txt").write_text("ignored")

    report = ingest_folder(tmp_path, store, chunk_size=256, overlap=32)

    statuses = {r.doc_name: r.status for r in report.results}
    assert statuses == {
        "blank.pdf": FileStatus.SKIPPED_NO_TEXT,
        "broken.pdf": FileStatus.FAILED,
        "good.pdf": FileStatus.LOADED,
    }
    assert report.total_chunks == store.count() == 2


def test_reingest_does_not_duplicate(make_pdf, tmp_path: Path, store):
    make_pdf("good.pdf", [TEXT])
    ingest_folder(tmp_path, store, chunk_size=256, overlap=32)
    ingest_folder(tmp_path, store, chunk_size=256, overlap=32)
    assert store.count() == 1


def test_store_failure_is_reported_not_raised(make_pdf, tmp_path: Path):
    class BrokenStore:
        def replace_document(self, doc_id, chunks):
            raise ConnectionError("embedding service down")

    make_pdf("good.pdf", [TEXT])
    report = ingest_folder(tmp_path, BrokenStore(), chunk_size=256, overlap=32)
    assert report.results[0].status is FileStatus.FAILED
    assert "embedding service down" in report.results[0].error


def test_missing_folder_raises(tmp_path: Path, store):
    with pytest.raises(FileNotFoundError):
        ingest_folder(tmp_path / "nope", store, chunk_size=256, overlap=32)
