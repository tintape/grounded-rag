"""Runtime configuration loaded from environment variables / `.env`."""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when a required setting is missing at the point of use."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None  # any OpenAI-compatible endpoint, e.g. OpenRouter
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embed_model: str = "text-embedding-3-small"

    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: str = ".qdrant_local"
    qdrant_collection: str = "documents"

    top_k: int = Field(default=5, ge=1, le=20)
    min_relevance: float = Field(default=0.30, ge=0.0, le=1.0)
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0)

    pdf_dir: Path = Path("data/sample_pdfs")
    api_url: str = "http://localhost:8000"
    api_key: SecretStr | None = None  # when set, /chat requires the X-API-Key header

    @field_validator("qdrant_url", "openai_base_url", mode="before")
    @classmethod
    def _blank_url_is_none(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        return str(value).strip()

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @staticmethod
    def _secret(value: SecretStr | None) -> str | None:
        return value.get_secret_value() if value and value.get_secret_value() else None

    @property
    def chat_api_key(self) -> str | None:
        return self._secret(self.api_key)

    @property
    def qdrant_key(self) -> str | None:
        return self._secret(self.qdrant_api_key)

    def require_api_key(self) -> str:
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
            raise ConfigError("OPENAI_API_KEY is not set. Copy .env.example to .env and add it.")
        return self.openai_api_key.get_secret_value()
