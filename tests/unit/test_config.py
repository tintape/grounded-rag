import pytest
from pydantic import ValidationError

from grounded_rag.config import ConfigError, Settings


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_defaults_are_sane():
    settings = make()
    assert settings.top_k == 5
    assert 0 < settings.min_relevance < 1
    assert settings.chunk_overlap < settings.chunk_size


def test_blank_qdrant_url_means_embedded_mode():
    assert make(qdrant_url="").qdrant_url is None
    assert make(qdrant_url="  ").qdrant_url is None


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        make(chunk_size=128, chunk_overlap=128)


@pytest.mark.parametrize("field,value", [("top_k", 0), ("min_relevance", 1.5), ("chunk_size", 10)])
def test_out_of_range_values_rejected(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_require_api_key_raises_clear_error_when_missing():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        make(openai_api_key=None).require_api_key()


def test_require_api_key_returns_secret_value():
    assert make(openai_api_key="sk-test").require_api_key() == "sk-test"


def test_settings_are_immutable():
    settings = make()
    with pytest.raises(ValidationError):
        settings.top_k = 10
