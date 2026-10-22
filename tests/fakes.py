"""Offline stand-ins for external services (embeddings, LLM)."""

import hashlib
import math
import re

from llama_index.core.embeddings import BaseEmbedding

DIM = 128
_WORD = re.compile(r"[a-z0-9]+")


def _bag_of_words(text: str) -> list[float]:
    vector = [0.0] * DIM
    for word in _WORD.findall(text.lower()):
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
        vector[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class KeywordEmbedding(BaseEmbedding):
    """Deterministic bag-of-words embedding: shared words => higher cosine similarity."""

    def _get_text_embedding(self, text: str) -> list[float]:
        return _bag_of_words(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return _bag_of_words(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return _bag_of_words(query)
