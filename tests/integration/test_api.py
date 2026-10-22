from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from grounded_rag.api.app import MAX_QUESTION_CHARS, create_app
from grounded_rag.models import Answer, Source
from grounded_rag.rag.engine import UpstreamError

GROUNDED = Answer(
    answer="Hold Home and Spot for 10 seconds [1].",
    refused=False,
    sources=(Source("product-guide.pdf", 4, "Factory reset: hold...", 0.81),),
)


@dataclass
class StubAnswerer:
    result: Answer | None = None
    error: Exception | None = None
    last_question: str | None = None

    def answer(self, question: str) -> Answer:
        self.last_question = question
        if self.error:
            raise self.error
        return self.result


@dataclass
class StubCounter:
    value: int = 42
    error: Exception | None = None

    def count(self) -> int:
        if self.error:
            raise self.error
        return self.value


def client(answerer=None, counter=None) -> TestClient:
    return TestClient(create_app(answerer or StubAnswerer(GROUNDED), counter or StubCounter()))


def test_health_reports_chunk_count():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks": 42}


def test_health_503_when_store_down():
    response = client(counter=StubCounter(error=ConnectionError())).get("/health")
    assert response.status_code == 503


def test_chat_returns_answer_with_sources_and_trims_question():
    answerer = StubAnswerer(GROUNDED)
    response = client(answerer).post("/chat", json={"question": "  How do I reset?  "})
    body = response.json()
    assert response.status_code == 200
    assert body["refused"] is False
    assert body["sources"][0] == {
        "doc_name": "product-guide.pdf",
        "page": 4,
        "excerpt": "Factory reset: hold...",
        "score": 0.81,
    }
    assert answerer.last_question == "How do I reset?"


def test_chat_refusal_is_200_with_no_sources():
    response = client(StubAnswerer(Answer.refusal())).post("/chat", json={"question": "CEO?"})
    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert response.json()["sources"] == []


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "   "}, {"question": "x" * (MAX_QUESTION_CHARS + 1)}],
)
def test_chat_rejects_invalid_questions(payload):
    assert client().post("/chat", json=payload).status_code == 422


def test_chat_503_on_upstream_failure():
    answerer = StubAnswerer(error=UpstreamError("The answer service is unavailable"))
    response = client(answerer).post("/chat", json={"question": "Reset?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "The answer service is unavailable"
