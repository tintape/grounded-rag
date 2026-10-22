"""Wire real OpenAI + Qdrant components from Settings."""

import openai
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient

from grounded_rag.config import ConfigError, Settings
from grounded_rag.rag.engine import EngineConfig, GroundedAnswerer
from grounded_rag.store import ChunkStore

LLM_TIMEOUT_SECONDS = 30.0
LLM_MAX_RETRIES = 2


class OpenAIGenerator:
    """Chat completion via the official SDK, so any OpenAI-compatible endpoint works
    (OpenAI, OpenRouter, Azure-style gateways) with provider-prefixed model names."""

    def __init__(self, model: str, client: openai.OpenAI) -> None:
        self._model = model
        self._client = client

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


def make_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_key)
    try:
        return QdrantClient(path=settings.qdrant_path)
    except RuntimeError as exc:  # embedded mode takes an exclusive lock on the folder
        raise ConfigError(
            f"Embedded Qdrant at '{settings.qdrant_path}' is in use by another process "
            "(e.g. the API). Stop it, or run a Qdrant server and set QDRANT_URL."
        ) from exc


def build_store(settings: Settings, client: QdrantClient | None = None) -> ChunkStore:
    embedder = OpenAIEmbedding(
        model_name=settings.openai_embed_model,
        api_key=settings.require_api_key(),
        api_base=settings.openai_base_url,
    )
    return ChunkStore(client or make_qdrant_client(settings), settings.qdrant_collection, embedder)


def build_generator(settings: Settings) -> OpenAIGenerator:
    client = openai.OpenAI(
        api_key=settings.require_api_key(),
        base_url=settings.openai_base_url,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
    return OpenAIGenerator(settings.openai_chat_model, client)


def build_answerer(
    settings: Settings, store: ChunkStore | None = None, excerpt_chars: int = 300
) -> GroundedAnswerer:
    config = EngineConfig(
        top_k=settings.top_k, min_relevance=settings.min_relevance, excerpt_chars=excerpt_chars
    )
    return GroundedAnswerer(store or build_store(settings), build_generator(settings), config)
