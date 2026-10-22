import pytest
from qdrant_client import QdrantClient

from grounded_rag.ingest.chunker import chunk_pages, document_id
from grounded_rag.models import PageText
from grounded_rag.store import ChunkStore
from tests.fakes import KeywordEmbedding


@pytest.fixture
def store() -> ChunkStore:
    return ChunkStore(QdrantClient(location=":memory:"), "test", KeywordEmbedding())


def chunks_for(doc: str, *texts: str):
    pages = tuple(PageText(doc, n, t) for n, t in enumerate(texts, start=1))
    return chunk_pages(pages, chunk_size=256, overlap=32)


def test_empty_store_counts_zero_and_searches_nothing(store):
    assert store.count() == 0
    assert store.search("anything", top_k=3) == ()


def test_replace_document_is_idempotent(store):
    chunks = chunks_for("guide.pdf", "Battery lasts three hours.", "Reset with Home and Spot.")
    store.replace_document(document_id("guide.pdf"), chunks)
    store.replace_document(document_id("guide.pdf"), chunks)
    assert store.count() == 2


def test_replace_document_drops_stale_pages(store):
    doc_id = document_id("guide.pdf")
    store.replace_document(doc_id, chunks_for("guide.pdf", "Old page one.", "Old page two."))
    store.replace_document(doc_id, chunks_for("guide.pdf", "New single page."))
    assert store.count() == 1
    assert store.search("old page", top_k=5)[0].chunk.text == "New single page."


def test_documents_are_replaced_independently(store):
    store.replace_document(document_id("a.pdf"), chunks_for("a.pdf", "Alpha text."))
    store.replace_document(document_id("b.pdf"), chunks_for("b.pdf", "Beta text."))
    store.replace_document(document_id("a.pdf"), chunks_for("a.pdf", "Alpha v2."))
    assert store.count() == 2


def test_search_returns_best_match_with_metadata(store):
    store.replace_document(
        document_id("faq.pdf"),
        chunks_for("faq.pdf", "Shipping takes five business days.", "Warranty covers the battery."),
    )
    hits = store.search("battery warranty", top_k=2)
    best = hits[0]
    assert best.chunk.doc_name == "faq.pdf"
    assert best.chunk.page == 2
    assert best.score > hits[1].score
    assert 0.0 <= best.score <= 1.0
