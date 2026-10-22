"""Qdrant-backed passage store (via LlamaIndex) with per-document replace semantics."""

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from grounded_rag.models import Chunk, RetrievedChunk

EMBED_BATCH_SIZE = 64
DOC_ID_KEY = "doc_id"  # payload key LlamaIndex fills from the SOURCE relationship


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
        """Make the stored passages of `doc_id` equal to `chunks`. Returns chunks stored.

        Embeds first and upserts by deterministic id before deleting stale points, so a failed
        embedding call leaves the previous version intact and the document is never empty.
        """
        embeddings = self._embed(chunks)
        if chunks:
            self._vectors.add([_to_node(c, e) for c, e in zip(chunks, embeddings, strict=True)])
        self._delete_stale(doc_id, keep_ids=[c.chunk_id for c in chunks])
        return len(chunks)

    def remove_documents_except(self, keep_doc_ids: set[str]) -> int:
        """Delete passages of every document not in `keep_doc_ids`. Returns passages removed."""
        if not self._exists():
            return 0
        before = self.count()
        condition = models.FieldCondition(
            key=DOC_ID_KEY, match=models.MatchAny(any=sorted(keep_doc_ids))
        )
        self._client.delete(self._collection, models.Filter(must_not=[condition]))
        return before - self.count()

    def _embed(self, chunks: tuple[Chunk, ...]) -> list[list[float]]:
        texts = [_embedding_text(c) for c in chunks]
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            embeddings.extend(self._embedder.get_text_embedding_batch(batch))
        return embeddings

    def _delete_stale(self, doc_id: str, keep_ids: list[str]) -> None:
        if not self._exists():
            return
        same_doc = models.FieldCondition(key=DOC_ID_KEY, match=models.MatchValue(value=doc_id))
        keep = [models.HasIdCondition(has_id=keep_ids)] if keep_ids else []
        self._client.delete(self._collection, models.Filter(must=[same_doc], must_not=keep))

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
