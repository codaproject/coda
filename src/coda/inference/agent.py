import asyncio
import logging
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from gilda import Annotation

logger = logging.getLogger(__name__)


class InferenceAgent:
    """Base class for cause-of-death inference agents with dialogue history tracking."""

    def __init__(self):
        """Initialize the agent with empty dialogue history."""
        self.dialogue_history = []  # List of (chunk_id, timestamp, text, annotations) tuples
        self.all_text = ""  # Accumulated text from all chunks

    def reset(self):
        """Reset dialogue history for a new interview."""
        self.dialogue_history = []
        self.all_text = ""
        logger.info("Agent state reset for new interview")

    async def process_chunk(self, chunk_id: str, text: str,
                           annotations: List[Annotation], timestamp: float = None) -> dict:
        """Process dialogue chunk and return inference results.

        This method handles dialogue history tracking and delegates
        to the subclass `infer()` method for actual COD inference.

        Parameters
        ----------
        chunk_id : str
            Unique identifier for this chunk
        text : str
            Transcribed text
        annotations : List[Annotation]
            Grounded medical terms from text
        timestamp : float, optional
            Unix timestamp (seconds since epoch) when chunk was created

        Returns
        -------
        dict with keys:
            - chunk_id: str
            - cod: str (cause of death)
            - confidence: float
            - reasoning: str (optional)
            - chunks_processed: int
            - timestamp: float
        """
        # Use current time if no timestamp provided
        if timestamp is None:
            import time
            timestamp = time.time()

        # Add to dialogue history
        self.dialogue_history.append((chunk_id, timestamp, text, annotations))
        self.all_text += " " + text

        # Call subclass inference implementation
        result = await self.infer(chunk_id, text, annotations)

        # Ensure required fields and add metadata
        result["chunk_id"] = chunk_id
        result["timestamp"] = timestamp
        result["chunks_processed"] = len(self.dialogue_history)

        logger.info(f"Chunk {chunk_id}: {len(self.dialogue_history)} chunks processed, COD={result.get('cod', 'N/A')}")

        return result

    async def infer(self, chunk_id: str, text: str,
                   annotations: List[Annotation]) -> dict:
        """Perform COD inference based on current chunk and accumulated history.

        Subclasses must implement this method. The dialogue history is available
        via self.dialogue_history and self.all_text.

        Parameters
        ----------
        chunk_id : str
            Unique identifier for this chunk
        text : str
            Transcribed text for current chunk
        annotations : List[Annotation]
            Grounded medical terms from current chunk

        Returns
        -------
        dict with keys:
            - cod: str (cause of death)
            - confidence: float
            - reasoning: str (optional)
        """
        raise NotImplementedError


class CodaToyInferenceAgent(InferenceAgent):
    """Simple rule-based inference agent using accumulated dialogue history."""

    async def infer(self, chunk_id: str, text: str,
                   annotations: List[Annotation]) -> dict:
        """Perform COD inference based on accumulated dialogue history."""
        # Simulate processing time
        await asyncio.sleep(0.5)

        # Analyze accumulated evidence from all chunks
        all_text_lower = self.all_text.lower()

        # Count symptom mentions across entire dialogue
        fever_mentions = all_text_lower.count("fever") + all_text_lower.count("temperature")
        cardiac_mentions = (all_text_lower.count("chest pain") +
                          all_text_lower.count("heart") +
                          all_text_lower.count("cardiac"))

        # Determine COD based on cumulative evidence
        if fever_mentions > 0:
            cod = "Infectious disease (suspected COVID-19)"
            # Confidence increases with more mentions
            confidence = min(0.9, 0.5 + (fever_mentions * 0.1))
            reasoning = f"Fever/temperature mentioned {fever_mentions} time(s) across {len(self.dialogue_history)} chunk(s)"
        elif cardiac_mentions > 0:
            cod = "Cardiac arrest"
            confidence = min(0.9, 0.5 + (cardiac_mentions * 0.1))
            reasoning = f"Cardiac symptoms mentioned {cardiac_mentions} time(s) across {len(self.dialogue_history)} chunk(s)"
        else:
            cod = "Unknown"
            confidence = 0.3
            reasoning = f"Insufficient evidence after {len(self.dialogue_history)} chunk(s)"

        return {
            "cod": cod,
            "confidence": confidence,
            "reasoning": reasoning
        }


class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    chunk_id: str
    text: str
    annotations: list
    timestamp: float = None  # Optional timestamp


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
                    request.annotations,
                    request.timestamp
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

        @self.app.post("/reset")
        async def reset():
            """Reset agent state for new interview."""
            if hasattr(self.agent, 'reset'):
                self.agent.reset()
                logger.info("Agent state reset via API")
                return {"status": "reset", "message": "Agent state cleared"}
            else:
                return {"status": "not_supported", "message": "Agent does not support state reset"}

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
