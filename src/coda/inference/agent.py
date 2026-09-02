import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from gilda import Annotation
from coda.config import settings
from coda.metadata import Metadata

logger = logging.getLogger('coda.inference')


@dataclass
class SessionRuntime:
    agent: "InferenceAgent"
    lock: asyncio.Lock


class InferenceAgent:
    """Base class for cause-of-death inference agents with dialogue history tracking."""

    def __init__(self):
        """Initialize the agent with empty dialogue history."""
        self.dialogue_history = []  # List of (chunk_id, timestamp, text, annotations) tuples
        self.all_text = ""  # Accumulated text from all chunks
        self.metadata = Metadata()

    def reset(self):
        """Reset dialogue history for a new interview."""
        self.dialogue_history = []
        self.all_text = ""
        self.metadata = Metadata()
        logger.info("Agent state reset for new interview")

    def create_session_agent(self) -> "InferenceAgent":
        """Create a fresh agent instance for one session/generation."""
        return self.__class__()

    async def process_chunk(self, chunk_id: str, text: str,
                           annotations: List[Annotation], timestamp: float = None,
                           metadata: dict = None) -> dict:
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
            - timestamp: float
            - chunks_processed: int
            - causes: dict mapping cause names to scores
            - reasoning: str (optional)
        """
        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = time.time()

        # Carry per-interview metadata forward if provided
        if metadata is not None:
            self.metadata = Metadata.from_dict(metadata)

        # Add to dialogue history
        self.dialogue_history.append((chunk_id, timestamp, text, annotations))
        self.all_text += " " + text

        infer_start = time.perf_counter()
        # Call subclass inference implementation
        result = await self.infer(chunk_id, text, annotations)
        infer_s = time.perf_counter() - infer_start

        # Ensure required fields and add metadata
        result["chunk_id"] = chunk_id
        result["timestamp"] = timestamp
        result["chunks_processed"] = len(self.dialogue_history)
        result["timings"] = {"inference_s": round(infer_s, 3)}

        # Log top cause for monitoring
        causes = result.get('causes', {})
        if causes:
            top_curie = max(causes.items(), key=lambda x: x[1]['score'])[0]
            top_cause_name = causes[top_curie]['name']
            top_score = causes[top_curie]['score']
            logger.info(
                "Chunk %s: %d chunks processed in %.2fs, top cause=%s (%s, score=%.2f)",
                chunk_id, len(self.dialogue_history), infer_s,
                top_cause_name, top_curie, top_score
            )
        else:
            logger.info(
                "Chunk %s: %d chunks processed in %.2fs, no causes",
                chunk_id, len(self.dialogue_history), infer_s
            )

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
            - causes: dict mapping CURIE keys (e.g., "icd10:U07.1") to cause objects
              Each cause object has:
                - name: str (standard ICD-10 name)
                - identifiers: dict (e.g., {"icd10": "U07.1"})
                - score: float (typically probability, not required to sum to 1)
            - reasoning: str (optional explanation)
        """
        raise NotImplementedError


class CodaToyInferenceAgent(InferenceAgent):
    """Simple rule-based inference agent using accumulated dialogue history."""

    async def infer(self, chunk_id: str, text: str,
                    annotations: List[Annotation]) -> dict:
        """Perform COD inference based on accumulated dialogue history."""
        # Analyze accumulated evidence from all chunks
        all_text_lower = self.all_text.lower()

        # Count symptom mentions across entire dialogue
        fever_mentions = all_text_lower.count("fever") + all_text_lower.count("temperature")
        cardiac_mentions = (all_text_lower.count("chest pain") +
                            all_text_lower.count("heart") +
                            all_text_lower.count("cardiac"))
        total_mentions = fever_mentions + cardiac_mentions

        # Calculate three probabilities normalized to sum to 1
        if total_mentions > 0:
            infectious_score = fever_mentions / total_mentions
            cardiac_score = cardiac_mentions / total_mentions
            other_score = 1.0 - (infectious_score + cardiac_score)
        else:
            infectious_score = 0.0
            cardiac_score = 0.0
            other_score = 1.0

        # Build causes with ICD-10 codes as CURIEs
        causes = {
            "icd10:U07.1": {
                "name": "COVID-19, virus identified",
                "identifiers": {"icd10": "U07.1"},
                "score": infectious_score
            },
            "icd10:I46.9": {
                "name": "Cardiac arrest, unspecified",
                "identifiers": {"icd10": "I46.9"},
                "score": cardiac_score
            },
            "icd10:R99": {
                "name": "Other ill-defined and unspecified causes of mortality",
                "identifiers": {"icd10": "R99"},
                "score": other_score
            }
        }

        reasoning = (f"Based on accumulated dialogue, "
                     f"infectious-related mentions: {fever_mentions}, "
                     f"cardiac-related mentions: {cardiac_mentions}, "
                     f"total mentions: {total_mentions}.")

        return {
            "causes": causes,
            "reasoning": reasoning
        }


class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    chunk_id: str
    text: str
    annotations: list
    timestamp: float = None  # Optional timestamp
    # Structured per-interview metadata
    metadata: dict = None
    session_id: str = "default"
    session_generation: int = 0


class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    session_generation: Optional[int] = None


class InferenceServer:
    """FastAPI server for inference agent."""

    def __init__(
            self,
            agent: InferenceAgent,
            host: Optional[str] = None,
            port: Optional[int] = None,
    ):
        self.agent = agent
        self.host = host or settings.inference.host
        self.port = port if port is not None else settings.inference.port
        self.app = FastAPI(title="CODA Inference Agent")
        self._session_runtimes: dict[tuple[str, int], SessionRuntime] = {}
        self._session_runtimes_lock = asyncio.Lock()

        async def get_session_runtime(session_id: str,
                                      session_generation: int) -> SessionRuntime:
            key = (session_id, session_generation)
            async with self._session_runtimes_lock:
                runtime = self._session_runtimes.get(key)
                if runtime is None:
                    runtime = SessionRuntime(
                        agent=self.agent.create_session_agent(),
                        lock=asyncio.Lock(),
                    )
                    self._session_runtimes[key] = runtime
                return runtime

        @self.app.post("/infer")
        async def infer(request: InferenceRequest):
            """Process dialogue chunk and return inference results."""
            started = time.perf_counter()
            runtime = await get_session_runtime(
                request.session_id,
                request.session_generation,
            )
            try:
                async with runtime.lock:
                    result = await runtime.agent.process_chunk(
                        request.chunk_id,
                        request.text,
                        request.annotations,
                        request.timestamp,
                        request.metadata
                    )
                causes = result.get('causes', {})
                if causes:
                    top_curie = max(causes.items(), key=lambda x: x[1]['score'])[0]
                    top_cause_name = causes[top_curie]['name']
                    logger.info(
                        "Processed chunk %s in %.2fs: top cause=%s (%s)",
                        request.chunk_id, time.perf_counter() - started,
                        top_cause_name, top_curie
                    )
                else:
                    logger.info(
                        "Processed chunk %s in %.2fs: no causes",
                        request.chunk_id, time.perf_counter() - started
                    )
                return result
            except Exception as e:
                logger.error(
                    "Error processing chunk %s after %.2fs: %s",
                    request.chunk_id, time.perf_counter() - started, e,
                    exc_info=True
                )
                raise

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy"}

        @self.app.post("/reset")
        async def reset(request: Optional[ResetRequest] = None):
            """Reset agent state for one session or all sessions."""
            async with self._session_runtimes_lock:
                if request and request.session_id is not None:
                    if request.session_generation is None:
                        keys = [
                            key for key in self._session_runtimes
                            if key[0] == request.session_id
                        ]
                    else:
                        keys = [(request.session_id, request.session_generation)]
                else:
                    keys = list(self._session_runtimes.keys())
                for key in keys:
                    self._session_runtimes.pop(key, None)

            if request and request.session_id is not None:
                logger.info(
                    "Agent session reset via API for session=%s generation=%s",
                    request.session_id, request.session_generation
                )
                return {"status": "reset", "message": "Agent session cleared"}

            if hasattr(self.agent, 'reset'):
                self.agent.reset()
            logger.info("Agent state reset via API")
            return {"status": "reset", "message": "Agent state cleared"}

    def run(self):
        """Start the inference server."""
        import uvicorn
        logger.info(f"Starting inference server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CODA inference agent server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent",
                        default=getattr(settings.inference, "agent", "champs_prompted"),
                        help="Inference agent implementation "
                             "(champs_prompted | champs_finetuned)")
    parser.add_argument("--provider", default=settings.inference.llm.provider,
                        help="LLM provider (e.g. openai, ollama)")
    parser.add_argument("--model", default=settings.inference.llm.model,
                        help="LLM model name (e.g. gpt-5.4-mini, gpt-oss:20b)")
    parser.add_argument("--host", default=settings.inference.host,
                        help="Server host")
    parser.add_argument("--port", type=int, default=settings.inference.port,
                        help="Server port")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.agent == "champs_finetuned":
        from coda.inference.champs_finetuned import create_champs_finetuned_agent
        agent = create_champs_finetuned_agent()
        agent.ensure_model()
    else:
        from coda.inference.champs_prompted_agent import create_champs_prompted_agent
        agent = create_champs_prompted_agent(provider=args.provider, model=args.model)

    server = InferenceServer(agent, host=args.host, port=args.port)
    server.run()
