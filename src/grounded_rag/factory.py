"""Wire real OpenAI + Qdrant components from Settings."""

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient

from grounded_rag.config import Settings
from grounded_rag.rag.engine import EngineConfig, GroundedAnswerer
from grounded_rag.store import ChunkStore

LLM_TIMEOUT_SECONDS = 30.0
LLM_MAX_RETRIES = 2


class OpenAIGenerator:
    def __init__(self, model: str, api_key: str) -> None:
        self._llm = OpenAI(
            model=model,
            temperature=0.0,
            api_key=api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )

    def generate(self, system: str, user: str) -> str:
        response = self._llm.chat(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=system),
                ChatMessage(role=MessageRole.USER, content=user),
            ]
        )
        return response.message.content or ""


def make_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=settings.qdrant_path)


def build_store(settings: Settings, client: QdrantClient | None = None) -> ChunkStore:
    embedder = OpenAIEmbedding(
        model=settings.openai_embed_model, api_key=settings.require_api_key()
    )
    return ChunkStore(client or make_qdrant_client(settings), settings.qdrant_collection, embedder)


def build_generator(settings: Settings) -> OpenAIGenerator:
    return OpenAIGenerator(settings.openai_chat_model, settings.require_api_key())


def build_answerer(
    settings: Settings, store: ChunkStore | None = None, excerpt_chars: int = 300
) -> GroundedAnswerer:
    config = EngineConfig(
        top_k=settings.top_k, min_relevance=settings.min_relevance, excerpt_chars=excerpt_chars
    )
    return GroundedAnswerer(store or build_store(settings), build_generator(settings), config)
