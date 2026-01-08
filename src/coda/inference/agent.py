import asyncio
import logging
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from gilda import Annotation

logger = logging.getLogger(__name__)


class InferenceAgent:
    """Base class for cause-of-death inference agents."""

    async def process_chunk(self, chunk_id: str, text: str,
                           annotations: List[Annotation]) -> dict:
        """Process dialogue chunk and return inference results.

        Parameters
        ----------
        chunk_id : str
            Unique identifier for this chunk
        text : str
            Transcribed text
        annotations : List[Annotation]
            Grounded medical terms from text

        Returns
        -------
        dict with keys:
            - chunk_id: str
            - cod: str (cause of death)
            - confidence: float
            - reasoning: str (optional)
        """
        raise NotImplementedError


class CodaToyInferenceAgent(InferenceAgent):
    """Simple rule-based inference agent for testing."""

    async def process_chunk(self, chunk_id: str, text: str,
                           annotations: List[Annotation]) -> dict:
        # Simulate processing time
        await asyncio.sleep(0.5)

        # Simple keyword-based logic
        text_lower = text.lower()
        if "fever" in text_lower or "temperature" in text_lower:
            cod = "Infectious disease (suspected COVID-19)"
            confidence = 0.7
        elif "chest pain" in text_lower or "heart" in text_lower:
            cod = "Cardiac arrest"
            confidence = 0.6
        else:
            cod = "Unknown - insufficient information"
            confidence = 0.3

        return {
            "chunk_id": chunk_id,
            "cod": cod,
            "confidence": confidence
        }


class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    chunk_id: str
    text: str
    annotations: list


class InferenceServer:
    """FastAPI server for inference agent."""

    def __init__(self, agent: InferenceAgent, host: str = "0.0.0.0", port: int = 5123):
        self.agent = agent
        self.host = host
        self.port = port
        self.app = FastAPI(title="CODA Inference Agent")

        @self.app.post("/infer")
        async def infer(request: InferenceRequest):
            """Process dialogue chunk and return inference results."""
            try:
                result = await self.agent.process_chunk(
                    request.chunk_id,
                    request.text,
                    request.annotations
                )
                logger.info(f"Processed chunk {request.chunk_id}: {result['cod']}")
                return result
            except Exception as e:
                logger.error(f"Error processing chunk {request.chunk_id}: {e}", exc_info=True)
                raise

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy"}

    def run(self):
        """Start the inference server."""
        import uvicorn
        logger.info(f"Starting inference server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run the inference server with toy agent
    agent = CodaToyInferenceAgent()
    server = InferenceServer(agent, host="0.0.0.0", port=5123)
    server.run()
