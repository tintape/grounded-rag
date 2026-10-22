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
