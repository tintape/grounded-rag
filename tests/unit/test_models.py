import dataclasses

import pytest

from grounded_rag.models import (
    REFUSAL_MESSAGE,
    Answer,
    FileIngestResult,
    FileStatus,
    IngestReport,
    Source,
)


def make_source(page: int = 1) -> Source:
    return Source(doc_name="guide.pdf", page=page, excerpt="text", score=0.5)


def test_refusal_has_no_sources_and_standard_message():
    answer = Answer.refusal()
    assert answer.refused is True
    assert answer.sources == ()
    assert answer.answer == REFUSAL_MESSAGE


def test_grounded_answer_requires_sources():
    with pytest.raises(ValueError):
        Answer(answer="text", refused=False, sources=())


def test_refused_answer_cannot_carry_sources():
    with pytest.raises(ValueError):
        Answer(answer="no", refused=True, sources=(make_source(),))


def test_models_are_frozen():
    answer = Answer(answer="ok [1]", refused=False, sources=(make_source(),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.answer = "changed"


def test_ingest_report_totals():
    report = IngestReport(
        results=(
            FileIngestResult("a.pdf", FileStatus.LOADED, pages=3, chunks=7),
            FileIngestResult("b.pdf", FileStatus.SKIPPED_NO_TEXT, pages=2, chunks=0),
            FileIngestResult("c.pdf", FileStatus.FAILED, error="encrypted"),
        )
    )
    assert report.loaded == 1
    assert report.total_chunks == 7
    assert [r.doc_name for r in report.problems] == ["b.pdf", "c.pdf"]
