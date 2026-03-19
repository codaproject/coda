"""
CHAMPS LLM-based cause-of-death inference agent for CODA.

Wraps the CHAMPS COD classification logic as a CODA inference agent that works
with dialogue text (verbal autopsy narratives), using CODA's LLMClient
infrastructure for LLM calls.
"""

import csv
import logging
import os
from typing import Dict, Any, List

from gilda import Annotation

from coda.inference.agent import InferenceAgent, InferenceServer
from coda.llm_api.client import LLMClient
from coda.resources import get_resource_path

logger = logging.getLogger(__name__)

CHAMPS_RESOURCE_DIR = get_resource_path("champs")


def read_champs_resource(filename: str) -> str:
    """Read a text resource file from the champs resources directory."""
    path = os.path.join(CHAMPS_RESOURCE_DIR, filename)
    with open(path) as fh:
        return fh.read().strip()


def load_group_causes() -> List[str]:
    """Load CHAMPS group cause names from resource file."""
    text = read_champs_resource("group_causes.txt")
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_group_to_icd10() -> Dict[str, str]:
    """Load CHAMPS group-name-to-ICD10 mapping from TSV resource file."""
    path = os.path.join(CHAMPS_RESOURCE_DIR, "group_to_icd10.tsv")
    mapping = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mapping[row["cause_name"]] = row["icd10"]
    return mapping


CHAMPS_GROUP_CAUSES = load_group_causes()
CHAMPS_GROUP_TO_ICD10 = load_group_to_icd10()
SYSTEM_PROMPT = read_champs_resource("system_prompt.txt")
SCHEMA_GUIDANCE = read_champs_resource("schema_guidance.txt")
DIAGNOSIS_STANDARD = read_champs_resource("diagnosis_standard.txt")

# JSON schema for structured output
COD_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "top_causes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cause_name": {"type": "string"},
                    "probability": {"type": "number"},
                },
                "required": ["cause_name", "probability"],
            },
            "minItems": 1,
            "maxItems": 3,
        }
    },
    "required": ["top_causes"],
}


class ChampsLLMInferenceAgent(InferenceAgent):
    """CHAMPS LLM-based cause-of-death inference agent.

    Uses a CODA LLMClient to call an LLM with the CHAMPS COD classification
    prompt and structured output schema. Works with dialogue text only -
    no demographics, MITS, structured VA, or clinical abstraction sections.
    """

    def __init__(self, llm_client: LLMClient, label_schema: str = "group"):
        super().__init__()
        self.llm_client = llm_client
        self.label_schema = label_schema

        if label_schema == "group":
            self.allowed_causes = CHAMPS_GROUP_CAUSES
            self.cause_to_icd10 = CHAMPS_GROUP_TO_ICD10
            self.schema_guidance = SCHEMA_GUIDANCE
        else:
            raise ValueError(
                f"Unsupported label_schema '{label_schema}'. "
                "Only 'group' is currently supported."
            )

        allowed_str = ", ".join(self.allowed_causes)
        self.rendered_system_prompt = \
            SYSTEM_PROMPT.format(allowed_causes=allowed_str) + "\n\n" \
            + self.schema_guidance

    async def infer(
        self, chunk_id: str, text: str, annotations: List[Annotation]
    ) -> dict:
        """Perform COD inference using accumulated dialogue via LLM."""
        user_prompt = (
            f"## DIAGNOSIS STANDARD\n{DIAGNOSIS_STANDARD}\n\n"
            f"## INPUT\n"
            f"- case_id: {chunk_id}\n"
            f"- narrative:\n"
            f"  {self.all_text.strip()}"
        )

        try:
            response = self.llm_client.call_with_schema(
                system_prompt=self.rendered_system_prompt,
                user_prompt=user_prompt,
                schema=COD_OUTPUT_SCHEMA,
                schema_name="champs_cod_classification",
                temperature=0.1,
            )
        except Exception:
            logger.exception("LLM call failed for chunk %s", chunk_id)
            return {
                "causes": {},
                "reasoning": "LLM API call raised an exception.",
            }

        if response.get("api_failed"):
            logger.warning("LLM API failed for chunk %s", chunk_id)
            return {
                "causes": {},
                "reasoning": "LLM API call failed after retries.",
            }

        return self._parse_response(response)

    def _parse_response(self, response: Dict[str, Any]) -> dict:
        """Convert LLM structured response to CODA cause format."""
        causes = {}
        top_causes = response.get("top_causes", [])

        for entry in top_causes:
            cause_name = entry.get("cause_name", "")
            probability = float(entry.get("probability", 0.0))
            icd10 = self.cause_to_icd10.get(cause_name)

            if icd10 is None:
                logger.warning("Unknown cause name '%s' - skipping",
                               cause_name)
                continue

            curie = f"icd10:{icd10}"
            causes[curie] = {
                "name": cause_name,
                "identifiers": {"icd10": icd10},
                "score": probability,
            }

        reasoning = "CHAMPS LLM classification"

        return {"causes": causes, "reasoning": reasoning}


def create_champs_agent(
    provider: str = "ollama",
    model: str = "gpt-oss:20b",
    **kwargs,
) -> ChampsLLMInferenceAgent:
    """Create a ChampsLLMInferenceAgent with a new LLM client.

    Defaults to gpt-oss:20b running on a local Ollama instance.
    """
    from coda.llm_api import create_llm_client

    client = create_llm_client(provider=provider, model=model, **kwargs)
    return ChampsLLMInferenceAgent(llm_client=client)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    agent = create_champs_agent()
    server = InferenceServer(agent, host="0.0.0.0", port=5123)
    server.run()
