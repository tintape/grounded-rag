import json

import pytest
from qdrant_client import QdrantClient

from grounded_rag import cli, factory
from grounded_rag.models import Answer, Source
from grounded_rag.store import ChunkStore
from tests.fakes import KeywordEmbedding

GROUNDED = Answer("Hold Home and Spot [1].", False, (Source("guide.pdf", 4, "x", 0.9),))


class StubAnswerer:
    def answer(self, question: str) -> Answer:
        return Answer.refusal() if "CEO" in question else GROUNDED


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "")


def test_ingest_prints_per_file_report(monkeypatch, make_pdf, tmp_path, capsys):
    store = ChunkStore(QdrantClient(location=":memory:"), "t", KeywordEmbedding())
    monkeypatch.setattr(factory, "build_store", lambda settings: store)
    make_pdf("guide.pdf", ["Reset the robot by holding Home and Spot for ten seconds."])

    assert cli.main(["ingest", "--pdf-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "loaded" in out and "guide.pdf" in out
    assert "Library now holds 1 chunks" in out


def test_ask_prints_answer_and_sources(monkeypatch, capsys):
    monkeypatch.setattr(factory, "build_answerer", lambda settings, **kw: StubAnswerer())
    assert cli.main(["ask", "How do I reset?"]) == 0
    out = capsys.readouterr().out
    assert "Hold Home and Spot [1]." in out
    assert "[1] guide.pdf, page 4" in out


def test_eval_writes_report_without_judge(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(factory, "build_answerer", lambda settings, **kw: StubAnswerer())
    questions = tmp_path / "q.json"
    questions.write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "question": "Reset?",
                    "expected_sources": [{"doc_name": "guide.pdf", "page": 4}],
                },
                {"id": "b", "question": "CEO?", "answerable": False},
            ]
        )
    )
    code = cli.main(
        ["eval", "--questions", str(questions), "--out", str(tmp_path / "r"), "--no-judge"]
    )
    assert code == 0
    report = json.loads((tmp_path / "r" / "report.json").read_text())
    assert report["summary"]["refusal_accuracy"] == 1.0
    assert report["summary"]["source_hit_rate"] == 1.0


def test_missing_api_key_gives_config_error_exit_code(capsys):
    assert cli.main(["ask", "Reset?"]) == cli.EXIT_CONFIG_ERROR
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_missing_pdf_folder_is_reported(monkeypatch, tmp_path, capsys):
    store = ChunkStore(QdrantClient(location=":memory:"), "t", KeywordEmbedding())
    monkeypatch.setattr(factory, "build_store", lambda settings: store)
    assert cli.main(["ingest", "--pdf-dir", str(tmp_path / "missing")]) == cli.EXIT_UPSTREAM_ERROR
    assert "not found" in capsys.readouterr().err
