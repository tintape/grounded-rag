"""Qdrant-backed passage store (via LlamaIndex) with per-document replace semantics."""

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from grounded_rag.models import Chunk, RetrievedChunk

EMBED_BATCH_SIZE = 64


def _embedding_text(chunk: Chunk) -> str:
    # The document name carries topic signal ("warranty", "troubleshooting") worth embedding.
    return f"Document: {chunk.doc_name}\n{chunk.text}"


def _to_node(chunk: Chunk, embedding: list[float]) -> TextNode:
    return TextNode(
        id_=chunk.chunk_id,
        text=chunk.text,
        embedding=embedding,
        metadata={"doc_name": chunk.doc_name, "page": chunk.page, "index": chunk.index},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=chunk.doc_id)},
    )


def _from_node(node: TextNode, score: float) -> RetrievedChunk:
    meta = node.metadata
    chunk = Chunk(
        chunk_id=node.node_id,
        doc_id=node.ref_doc_id or "",
        doc_name=str(meta["doc_name"]),
        page=int(meta["page"]),
        index=int(meta["index"]),
        text=node.get_content(),
    )
    return RetrievedChunk(chunk=chunk, score=float(score))


class ChunkStore:
    def __init__(self, client: QdrantClient, collection: str, embedder: BaseEmbedding) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._vectors = QdrantVectorStore(collection_name=collection, client=client)

    def _exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def count(self) -> int:
        if not self._exists():
            return 0
        return self._client.count(self._collection, exact=True).count

    def replace_document(self, doc_id: str, chunks: tuple[Chunk, ...]) -> int:
        """Delete every stored passage of `doc_id`, then insert `chunks`. Returns chunks stored."""
        if self._exists():
            self._vectors.delete(ref_doc_id=doc_id)
        if not chunks:
            return 0
        texts = [_embedding_text(c) for c in chunks]
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            embeddings.extend(self._embedder.get_text_embedding_batch(batch))
        self._vectors.add([_to_node(c, e) for c, e in zip(chunks, embeddings, strict=True)])
        return len(chunks)

    def search(self, query: str, top_k: int) -> tuple[RetrievedChunk, ...]:
        if not self._exists():
            return ()
        query_embedding = self._embedder.get_query_embedding(query)
        result = self._vectors.query(
            VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=top_k)
        )
        nodes = result.nodes or []
        scores = result.similarities or [0.0] * len(nodes)
        return tuple(_from_node(n, s) for n, s in zip(nodes, scores, strict=True))
