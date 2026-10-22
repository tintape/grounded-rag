from grounded_rag.ingest.chunker import chunk_pages, document_id
from grounded_rag.models import PageText

SENTENCE = "Hold the Home and Spot buttons for ten seconds to reset the robot. "


def pages(*texts: str, doc: str = "guide.pdf") -> tuple[PageText, ...]:
    return tuple(PageText(doc, n, t) for n, t in enumerate(texts, start=1))


def test_short_pages_become_one_chunk_each():
    chunks = chunk_pages(pages("Short page one.", "Short page two."), chunk_size=128, overlap=16)
    assert [(c.page, c.index) for c in chunks] == [(1, 0), (2, 0)]


def test_chunks_never_cross_page_boundaries():
    chunks = chunk_pages(pages(SENTENCE * 40, SENTENCE * 40), chunk_size=128, overlap=16)
    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.page in (1, 2)
    page_one = [c for c in chunks if c.page == 1]
    assert [c.index for c in page_one] == list(range(len(page_one)))


def test_ids_are_deterministic_and_unique():
    first = chunk_pages(pages(SENTENCE * 40), chunk_size=128, overlap=16)
    second = chunk_pages(pages(SENTENCE * 40), chunk_size=128, overlap=16)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_doc_id_depends_only_on_file_name():
    assert document_id("a.pdf") == document_id("a.pdf")
    assert document_id("a.pdf") != document_id("b.pdf")
    chunks = chunk_pages(pages("Some text here.", doc="a.pdf"), chunk_size=128, overlap=16)
    assert chunks[0].doc_id == document_id("a.pdf")


def test_empty_input_gives_no_chunks():
    assert chunk_pages((), chunk_size=128, overlap=16) == ()
