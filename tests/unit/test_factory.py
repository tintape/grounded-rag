from qdrant_client import QdrantClient

from grounded_rag.config import Settings
from grounded_rag.factory import build_answerer, make_qdrant_client
from grounded_rag.rag.engine import GroundedAnswerer


def test_embedded_qdrant_when_no_url(tmp_path):
    settings = Settings(_env_file=None, qdrant_path=str(tmp_path / "q"))
    client = make_qdrant_client(settings)
    assert isinstance(client, QdrantClient)
    client.close()


def test_build_answerer_wires_components_without_network(tmp_path):
    settings = Settings(_env_file=None, openai_api_key="sk-test", qdrant_path=str(tmp_path / "q"))
    assert isinstance(build_answerer(settings), GroundedAnswerer)


def test_locked_embedded_store_gives_clear_config_error(tmp_path):
    import pytest

    from grounded_rag.config import ConfigError

    settings = Settings(_env_file=None, qdrant_path=str(tmp_path / "q"))
    first = make_qdrant_client(settings)
    try:
        with pytest.raises(ConfigError, match="QDRANT_URL"):
            make_qdrant_client(settings)
    finally:
        first.close()


def test_generator_talks_to_configured_base_url_and_model():
    import json

    import httpx
    import openai

    from grounded_rag.factory import OpenAIGenerator

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Reset [1]."},
                    }
                ],
            },
        )

    client = openai.OpenAI(
        api_key="sk-or-test",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    generator = OpenAIGenerator("openai/gpt-4.1-mini", client=client)

    assert generator.generate("system", "user") == "Reset [1]."
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["body"]["model"] == "openai/gpt-4.1-mini"
    assert seen["body"]["temperature"] == 0
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


def test_base_url_is_optional_and_blank_means_default():
    assert Settings(_env_file=None, openai_base_url="").openai_base_url is None
    url = "https://openrouter.ai/api/v1"
    assert Settings(_env_file=None, openai_base_url=url).openai_base_url == url
