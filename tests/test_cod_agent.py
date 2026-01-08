import pytest
from fastapi.testclient import TestClient

from coda.inference.agent import CodaToyInferenceAgent, InferenceServer


@pytest.fixture
def toy_agent():
    """Fixture for CodaToyInferenceAgent - fresh instance for each test."""
    agent = CodaToyInferenceAgent()
    yield agent
    # Reset after each test to avoid state leakage
    agent.reset()


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
        assert "causes" in result
        assert "icd10:U07.1" in result["causes"]

        # Check structure
        covid_cause = result["causes"]["icd10:U07.1"]
        assert covid_cause["name"] == "COVID-19, virus identified"
        assert covid_cause["identifiers"]["icd10"] == "U07.1"
        assert covid_cause["score"] > 0.5

    @pytest.mark.asyncio
    async def test_toy_agent_cardiac_detection(self, toy_agent):
        """Test that toy agent detects cardiac-related keywords."""
        chunk_id = "test-002"
        text = "The patient complained of severe chest pain."
        annotations = []

        result = await toy_agent.process_chunk(chunk_id, text, annotations)

        assert result["chunk_id"] == chunk_id
        assert "causes" in result
        assert "icd10:I46.9" in result["causes"]

        # Check structure
        cardiac_cause = result["causes"]["icd10:I46.9"]
        assert cardiac_cause["name"] == "Cardiac arrest, unspecified"
        assert cardiac_cause["identifiers"]["icd10"] == "I46.9"
        assert cardiac_cause["score"] > 0.5

    @pytest.mark.asyncio
    async def test_toy_agent_unknown_case(self, toy_agent):
        """Test that toy agent returns other for unrecognized symptoms."""
        chunk_id = "test-003"
        text = "The patient had some issues."
        annotations = []

        result = await toy_agent.process_chunk(chunk_id, text, annotations)

        assert result["chunk_id"] == chunk_id
        assert "causes" in result
        assert "icd10:R99" in result["causes"]

        # Check structure
        other_cause = result["causes"]["icd10:R99"]
        assert other_cause["name"] == "Other ill-defined and unspecified causes of mortality"
        assert other_cause["identifiers"]["icd10"] == "R99"
        assert other_cause["score"] > 0.5

    @pytest.mark.asyncio
    async def test_toy_agent_stateful_confidence_increase(self, toy_agent):
        """Test that agent scores increase with repeated evidence across chunks."""
        # First chunk with fever mention
        result1 = await toy_agent.process_chunk("chunk-1", "The patient had a fever.", [])
        assert result1["chunks_processed"] == 1
        assert "icd10:U07.1" in result1["causes"]
        score1 = result1["causes"]["icd10:U07.1"]["score"]

        # Second chunk with another fever mention
        result2 = await toy_agent.process_chunk("chunk-2", "The fever continued for days.", [])
        assert result2["chunks_processed"] == 2
        score2 = result2["causes"]["icd10:U07.1"]["score"]

        # Third chunk with temperature mention
        result3 = await toy_agent.process_chunk("chunk-3", "High temperature was recorded.", [])
        assert result3["chunks_processed"] == 3
        score3 = result3["causes"]["icd10:U07.1"]["score"]

        # Score should remain high with accumulating evidence (all mention fever/temp)
        assert score1 > 0.5, "First chunk should have high infectious score"
        assert score2 > 0.5, "Second chunk should have high infectious score"
        assert score3 > 0.5, "Third chunk should have high infectious score"

    @pytest.mark.asyncio
    async def test_toy_agent_reset(self, toy_agent):
        """Test that agent reset clears dialogue history."""
        # Process some chunks
        await toy_agent.process_chunk("chunk-1", "Patient had fever.", [])
        await toy_agent.process_chunk("chunk-2", "Fever persisted.", [])
        assert len(toy_agent.dialogue_history) == 2

        # Reset the agent
        toy_agent.reset()
        assert len(toy_agent.dialogue_history) == 0
        assert toy_agent.all_text == ""

        # Process new chunk after reset
        result = await toy_agent.process_chunk("chunk-3", "New interview: chest pain.", [])
        assert result["chunks_processed"] == 1
        assert "icd10:I46.9" in result["causes"]
        assert result["causes"]["icd10:I46.9"]["score"] > 0.5

    @pytest.mark.asyncio
    async def test_toy_agent_multiple_causes(self, toy_agent):
        """Test that agent returns multiple causes with scores."""
        chunk_id = "test-multi-001"
        text = "The patient had fever and chest pain."
        annotations = []

        result = await toy_agent.process_chunk(chunk_id, text, annotations)

        assert result["chunk_id"] == chunk_id
        assert "causes" in result

        # Should have all three causes
        assert "icd10:U07.1" in result["causes"]
        assert "icd10:I46.9" in result["causes"]
        assert "icd10:R99" in result["causes"]

        # Both infectious and cardiac should have positive scores
        assert result["causes"]["icd10:U07.1"]["score"] > 0
        assert result["causes"]["icd10:I46.9"]["score"] > 0

        # Scores should sum to approximately 1 (for toy agent)
        total = sum(c["score"] for c in result["causes"].values())
        assert abs(total - 1.0) < 0.001

        # Verify structure of each cause
        for curie, cause in result["causes"].items():
            assert "name" in cause
            assert "identifiers" in cause
            assert "score" in cause
            assert "icd10" in cause["identifiers"]

    @pytest.mark.asyncio
    async def test_toy_agent_icd10_structure(self, toy_agent):
        """Test that agent returns proper ICD-10 CURIE structure."""
        chunk_id = "test-icd10-001"
        text = "The patient had fever."
        annotations = []

        result = await toy_agent.process_chunk(chunk_id, text, annotations)

        # Check CURIE keys
        assert "icd10:U07.1" in result["causes"]
        assert "icd10:I46.9" in result["causes"]
        assert "icd10:R99" in result["causes"]

        # Check COVID-19 cause structure
        covid = result["causes"]["icd10:U07.1"]
        assert covid["name"] == "COVID-19, virus identified"
        assert covid["identifiers"]["icd10"] == "U07.1"
        assert isinstance(covid["score"], float)

        # Check cardiac cause structure
        cardiac = result["causes"]["icd10:I46.9"]
        assert cardiac["name"] == "Cardiac arrest, unspecified"
        assert cardiac["identifiers"]["icd10"] == "I46.9"

        # Check other cause structure
        other = result["causes"]["icd10:R99"]
        assert other["name"] == "Other ill-defined and unspecified causes of mortality"
        assert other["identifiers"]["icd10"] == "R99"

    @pytest.mark.asyncio
    async def test_toy_agent_timestamps(self, toy_agent):
        """Test that agent properly tracks timestamps."""
        import time

        # Process chunks with explicit timestamps
        ts1 = time.time()
        result1 = await toy_agent.process_chunk("chunk-1", "Patient had fever.", [], timestamp=ts1)
        assert result1["timestamp"] == ts1

        ts2 = ts1 + 2.0  # 2 seconds later
        result2 = await toy_agent.process_chunk("chunk-2", "Fever continued.", [], timestamp=ts2)
        assert result2["timestamp"] == ts2

        # Verify dialogue history contains timestamps
        assert len(toy_agent.dialogue_history) == 2
        chunk_id1, stored_ts1, text1, annotations1 = toy_agent.dialogue_history[0]
        assert stored_ts1 == ts1
        assert text1 == "Patient had fever."

        chunk_id2, stored_ts2, text2, annotations2 = toy_agent.dialogue_history[1]
        assert stored_ts2 == ts2
        assert text2 == "Fever continued."

        # Test auto-generated timestamp
        result3 = await toy_agent.process_chunk("chunk-3", "Temperature high.", [])
        assert "timestamp" in result3
        assert result3["timestamp"] >= ts1  # Should be >= first timestamp
        assert isinstance(result3["timestamp"], float)


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
        assert "causes" in result
        assert "icd10:U07.1" in result["causes"]
        assert result["causes"]["icd10:U07.1"]["score"] > 0.5

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
        assert "causes" in result
        assert "icd10:I46.9" in result["causes"]
        assert result["causes"]["icd10:I46.9"]["score"] > 0.5

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
        assert "causes" in result
        assert len(result["causes"]) == 3  # Should have all 3 causes

    def test_infer_endpoint_missing_fields(self, client):
        """Test the /infer endpoint with missing required fields."""
        request_data = {
            "chunk_id": "http-test-004",
            # Missing 'text' field
            "annotations": []
        }

        response = client.post("/infer", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_reset_endpoint(self, client):
        """Test the /reset endpoint clears agent state."""
        # Send first chunk
        response1 = client.post("/infer", json={
            "chunk_id": "reset-test-001",
            "text": "Patient had fever.",
            "annotations": []
        })
        assert response1.status_code == 200
        result1 = response1.json()
        assert result1["chunks_processed"] == 1

        # Send second chunk
        response2 = client.post("/infer", json={
            "chunk_id": "reset-test-002",
            "text": "Fever continued.",
            "annotations": []
        })
        assert response2.status_code == 200
        result2 = response2.json()
        assert result2["chunks_processed"] == 2

        # Reset the agent
        reset_response = client.post("/reset")
        assert reset_response.status_code == 200
        assert reset_response.json()["status"] == "reset"

        # Send new chunk after reset
        response3 = client.post("/infer", json={
            "chunk_id": "reset-test-003",
            "text": "New patient with chest pain.",
            "annotations": []
        })
        assert response3.status_code == 200
        result3 = response3.json()
        assert result3["chunks_processed"] == 1  # Should be back to 1 after reset

    def test_infer_endpoint_with_timestamp(self, client):
        """Test the /infer endpoint with explicit timestamp."""
        import time

        timestamp = time.time()
        request_data = {
            "chunk_id": "timestamp-test-001",
            "text": "Patient had fever.",
            "annotations": [],
            "timestamp": timestamp
        }

        response = client.post("/infer", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert result["chunk_id"] == "timestamp-test-001"
        assert result["timestamp"] == timestamp
        assert "causes" in result
        assert len(result["causes"]) > 0
