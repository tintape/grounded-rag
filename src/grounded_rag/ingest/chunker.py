"""Split page text into retrieval passages that never cross a page boundary."""

import uuid

from llama_index.core.node_parser import SentenceSplitter

from grounded_rag.models import Chunk, PageText

_NAMESPACE = uuid.UUID("6f1d2c3a-9b7e-4c55-8a10-3f2e8d9c7b61")


def document_id(doc_name: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, doc_name))


def _chunk_id(doc_id: str, page: int, index: int) -> str:
    return str(uuid.uuid5(uuid.UUID(doc_id), f"{page}:{index}"))


def chunk_pages(pages: tuple[PageText, ...], chunk_size: int, overlap: int) -> tuple[Chunk, ...]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: list[Chunk] = []
    for page in pages:
        doc_id = document_id(page.doc_name)
        for index, text in enumerate(splitter.split_text(page.text)):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc_id, page.page, index),
                    doc_id=doc_id,
                    doc_name=page.doc_name,
                    page=page.page,
                    index=index,
                    text=text,
                )
            )
    return tuple(chunks)
