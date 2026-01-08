import pytest
from fastapi.testclient import TestClient
from gilda import Annotation

from coda.inference.agent import (
    InferenceAgent,
    CodaToyInferenceAgent,
    InferenceServer,
)


@pytest.fixture
def toy_agent():
    """Fixture for CodaToyInferenceAgent."""
    return CodaToyInferenceAgent()


@pytest.fixture
def inference_server(toy_agent):
    """Fixture for InferenceServer with TestClient."""
    server = InferenceServer(toy_agent, host="0.0.0.0", port=5123)
    return server


@pytest.fixture
def client(inference_server):
    """Fixture for FastAPI TestClient."""
    return TestClient(inference_server.app)


class TestInferenceAgent:
    """Unit tests for InferenceAgent."""

    @pytest.mark.asyncio
    async def test_toy_agent_fever_detection(self, toy_agent):
        """Test that toy agent detects fever-related keywords."""
        chunk_id = "test-001"
        text = "The patient had a high fever and temperature."
        annotations = []

        result = await toy_agent.process_chunk(chunk_id, text, annotations)

        assert result["chunk_id"] == chunk_id
        assert "fever" in result["cod"].lower() or "infectious" in result["cod"].lower()
        assert result["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_toy_agent_cardiac_detection(self, toy_agent):
        """Test that toy agent detects cardiac-related keywords."""
        chunk_id = "test-002"
        text = "The patient complained of severe chest pain."
        annotations = []

        result = await toy_agent.process_chunk(chunk_id, text, annotations)

        assert result["chunk_id"] == chunk_id
        assert "cardiac" in result["cod"].lower() or "heart" in result["cod"].lower()
        assert result["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_toy_agent_unknown_case(self, toy_agent):
        """Test that toy agent returns unknown for unrecognized symptoms."""
        chunk_id = "test-003"
        text = "The patient had some issues."
        annotations = []

        result = await toy_agent.process_chunk(chunk_id, text, annotations)

        assert result["chunk_id"] == chunk_id
        assert "unknown" in result["cod"].lower()
        assert result["confidence"] < 0.5


class TestInferenceServer:
    """Integration tests for InferenceServer HTTP endpoints."""

    def test_health_endpoint(self, client):
        """Test the /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_infer_endpoint_fever(self, client):
        """Test the /infer endpoint with fever symptoms."""
        request_data = {
            "chunk_id": "http-test-001",
            "text": "Patient had high fever and elevated temperature.",
            "annotations": []
        }

        response = client.post("/infer", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert result["chunk_id"] == "http-test-001"
        assert "cod" in result
        assert "confidence" in result
        assert "fever" in result["cod"].lower() or "infectious" in result["cod"].lower()

    def test_infer_endpoint_cardiac(self, client):
        """Test the /infer endpoint with cardiac symptoms."""
        request_data = {
            "chunk_id": "http-test-002",
            "text": "The patient had chest pain and heart palpitations.",
            "annotations": []
        }

        response = client.post("/infer", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert result["chunk_id"] == "http-test-002"
        assert "cardiac" in result["cod"].lower() or "heart" in result["cod"].lower()

    def test_infer_endpoint_with_annotations(self, client):
        """Test the /infer endpoint with medical annotations."""
        request_data = {
            "chunk_id": "http-test-003",
            "text": "Patient had fever.",
            "annotations": [
                # Simplified annotation structure for testing
                {"text": "fever", "start": 12, "end": 17}
            ]
        }

        response = client.post("/infer", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert result["chunk_id"] == "http-test-003"
        assert "cod" in result
        assert result["confidence"] > 0

    def test_infer_endpoint_missing_fields(self, client):
        """Test the /infer endpoint with missing required fields."""
        request_data = {
            "chunk_id": "http-test-004",
            # Missing 'text' field
            "annotations": []
        }

        response = client.post("/infer", json=request_data)
        assert response.status_code == 422  # Validation error
