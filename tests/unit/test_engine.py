from dataclasses import dataclass, field

import pytest

from grounded_rag.models import EMPTY_LIBRARY_MESSAGE, REFUSAL_MESSAGE, Chunk, RetrievedChunk
from grounded_rag.rag.engine import EngineConfig, GroundedAnswerer, UpstreamError
from grounded_rag.rag.prompts import NOT_FOUND


def hit(doc: str, page: int, text: str, score: float, index: int = 0) -> RetrievedChunk:
    chunk = Chunk(f"{doc}-{page}-{index}", "doc", doc, page, index, text)
    return RetrievedChunk(chunk=chunk, score=score)


@dataclass
class FakeRetriever:
    hits: tuple[RetrievedChunk, ...] = ()
    size: int = 10
    error: Exception | None = None

    def search(self, query: str, top_k: int) -> tuple[RetrievedChunk, ...]:
        if self.error:
            raise self.error
        return self.hits[:top_k]

    def count(self) -> int:
        return self.size


@dataclass
class FakeGenerator:
    reply: str = ""
    error: Exception | None = None
    prompts: list[str] = field(default_factory=list)

    def generate(self, system: str, user: str) -> str:
        self.prompts.append(user)
        if self.error:
            raise self.error
        return self.reply


CONFIG = EngineConfig(top_k=5, min_relevance=0.3)
RESET = hit("product-guide.pdf", 4, "Hold Home and Spot for 10 seconds to reset.", 0.82)
BATTERY = hit("warranty.pdf", 2, "The battery is covered for 12 months.", 0.55)
NOISE = hit("faq.pdf", 1, "We ship to the US and Canada.", 0.12)


def make(reply: str = "", hits=(RESET, BATTERY, NOISE), **kwargs):
    generator = FakeGenerator(reply=reply, error=kwargs.pop("gen_error", None))
    retriever = FakeRetriever(hits=hits, **kwargs)
    return GroundedAnswerer(retriever, generator, CONFIG), generator


def test_answer_cites_only_used_sources():
    engine, _ = make("Hold Home and Spot for 10 seconds [1].")
    answer = engine.answer("How do I reset?")
    assert answer.refused is False
    assert [(s.doc_name, s.page) for s in answer.sources] == [("product-guide.pdf", 4)]
    assert answer.answer == "Hold Home and Spot for 10 seconds [1]."


def test_sources_are_renumbered_in_citation_order():
    engine, _ = make("Battery is 12 months [2]. Reset takes 10 s [1].")
    answer = engine.answer("Tell me things")
    assert [s.doc_name for s in answer.sources] == ["warranty.pdf", "product-guide.pdf"]
    assert answer.answer == "Battery is 12 months [1]. Reset takes 10 s [2]."


def test_low_relevance_hits_are_never_shown_to_the_model():
    engine, generator = make("Reset [1].")
    engine.answer("How do I reset?")
    assert "We ship to the US" not in generator.prompts[0]
    assert "[1] (product-guide.pdf, page 4)" in generator.prompts[0]


def test_refuses_without_calling_llm_when_nothing_is_relevant():
    engine, generator = make("should not be used", hits=(NOISE,))
    answer = engine.answer("What is the CEO's salary?")
    assert answer.refused is True
    assert answer.answer == REFUSAL_MESSAGE
    assert generator.prompts == []


def test_refuses_when_model_says_not_found():
    engine, _ = make(NOT_FOUND)
    assert engine.answer("Unrelated question").refused is True


def test_refuses_when_model_cites_a_passage_it_was_not_given():
    engine, _ = make("Reset with Home and Spot [1]. Also it costs $499 [9].")
    assert engine.answer("Reset?").refused is True


def test_refuses_when_model_cites_nothing_valid():
    engine, _ = make("It is probably 42 [0].")
    answer = engine.answer("Something")
    assert answer.refused is True
    assert answer.sources == ()


def test_refuses_when_model_answers_without_citations():
    engine, _ = make("A confident answer with no references.")
    assert engine.answer("Something").refused is True


def test_same_page_cited_twice_becomes_one_source_showing_both_passages():
    second = hit("product-guide.pdf", 4, "The ring flashes blue.", 0.9, index=1)
    engine, _ = make("Hold buttons [1]; ring flashes [2].", hits=(RESET, second))
    answer = engine.answer("Reset?")
    assert len(answer.sources) == 1
    assert answer.answer == "Hold buttons [1]; ring flashes [1]."
    assert "Hold Home and Spot" in answer.sources[0].excerpt
    assert "ring flashes blue" in answer.sources[0].excerpt
    assert answer.sources[0].score == 0.9


def test_not_found_word_inside_cited_answer_is_not_a_refusal():
    engine, _ = make("Hold Home and Spot [1]. NOT_FOUND")
    answer = engine.answer("Reset?")
    assert answer.refused is False
    assert "NOT_FOUND" not in answer.answer


def test_not_found_without_citations_is_a_refusal():
    engine, _ = make("Sorry, NOT_FOUND in the passages.")
    assert engine.answer("Reset?").refused is True


def test_empty_library_gets_specific_message():
    engine, _ = make(hits=(), size=0)
    assert engine.answer("Anything").answer == EMPTY_LIBRARY_MESSAGE


def test_excerpt_is_truncated():
    long_hit = hit("guide.pdf", 1, "word " * 200, 0.9)
    engine, _ = make("Answer [1].", hits=(long_hit,))
    assert len(engine.answer("q").sources[0].excerpt) <= CONFIG.excerpt_chars + 1


@pytest.mark.parametrize("question", ["", "   "])
def test_blank_question_rejected(question):
    engine, _ = make()
    with pytest.raises(ValueError):
        engine.answer(question)


def test_retrieval_failure_raises_upstream_error():
    engine, _ = make(error=ConnectionError("qdrant down"))
    with pytest.raises(UpstreamError):
        engine.answer("q")


def test_generation_failure_raises_upstream_error_not_answer():
    engine, _ = make(gen_error=TimeoutError("openai timeout"))
    with pytest.raises(UpstreamError):
        engine.answer("How do I reset?")


def test_answer_hitting_sanitizer_cap_becomes_refusal(monkeypatch):
    from grounded_rag.rag import sanitize

    monkeypatch.setattr(sanitize, "_MAX_PASSES", 0)
    engine, _ = make("Reset [1]. ![x](https://evil.example)")
    assert engine.answer("Reset?").refused is True
