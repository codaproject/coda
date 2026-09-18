"""
CHAMPS LLM-based cause-of-death inference agent for CODA.

Wraps the CHAMPS COD classification logic as a CODA inference agent that works
with dialogue text (verbal autopsy narratives), using CODA's LLMClient
infrastructure for LLM calls.
"""

import asyncio
import csv
import difflib
import logging
import os
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional

from gilda import Annotation

from coda.inference.agent import InferenceAgent, InferenceServer
from coda.inference.probbase import load_questions
from coda.inference.voi import select_questions
from coda.llm_api.client import LLMClient
from coda.resources import get_resource_path
from coda.config import settings

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


def load_group_to_who_va():
    """Load CHAMPS group-name-to-WHO-VA-cause mapping from TSV resource file."""
    path = os.path.join(CHAMPS_RESOURCE_DIR, "group_to_who_va.tsv")
    mapping = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mapping[row["cause_name"]] = row["who_va"]
    return mapping


CHAMPS_GROUP_CAUSES = load_group_causes()
CHAMPS_GROUP_TO_ICD10 = load_group_to_icd10()
CHAMPS_GROUP_TO_WHO_VA = load_group_to_who_va()
SYSTEM_PROMPT = read_champs_resource("system_prompt.txt")
SCHEMA_GUIDANCE = read_champs_resource("schema_guidance.txt")
DIAGNOSIS_STANDARD = read_champs_resource("diagnosis_standard.txt")

DEFAULT_NUM_QUESTIONS = 3
DEFAULT_QUESTION_STYLE = "default"
DEFAULT_QUESTION_MODE = "llm"

# Where the follow-up questions come from. `llm` takes the ones the model
# writes, `voi` replaces them with the WHO VA questions whose answers are
# expected to be most informative about the causes currently in play.
QUESTION_MODES = ("llm", "voi")

# Schema-level phrasing of the questions field, one entry per question style.
QUESTION_SCHEMA_DESCRIPTIONS = {
    "default":
        "Exactly {num_questions} follow-up questions that would best "
        "differentiate the top causes and increase confidence in the "
        "underlying cause.",
    "simplified":
        "Exactly {num_questions} follow-up questions that would best "
        "differentiate the top causes and increase confidence in the "
        "underlying cause, each short and phrased in plain lay language for "
        "a family member with no medical training.",
}

QUESTION_STYLES = tuple(QUESTION_SCHEMA_DESCRIPTIONS)


def _validate_question_style(question_style):
    """Raise ValueError unless the style is one of QUESTION_STYLES."""
    if question_style not in QUESTION_STYLES:
        raise ValueError(f"Unknown question style {question_style!r}, expected "
                         f"one of: {', '.join(QUESTION_STYLES)}")


def _validate_question_mode(question_mode):
    """Raise ValueError unless the mode is one of QUESTION_MODES."""
    if question_mode not in QUESTION_MODES:
        raise ValueError(f"Unknown question mode {question_mode!r}, expected "
                         f"one of: {', '.join(QUESTION_MODES)}")


def load_question_style_guidance(question_style=DEFAULT_QUESTION_STYLE):
    """Read the Step 5 prompt body for a question style."""
    _validate_question_style(question_style)
    return read_champs_resource(f"question_style_{question_style}.txt")


def build_cod_output_schema(num_questions=DEFAULT_NUM_QUESTIONS,
                            question_style=DEFAULT_QUESTION_STYLE):
    """Build the JSON schema for structured output."""
    _validate_question_style(question_style)
    questions_description = QUESTION_SCHEMA_DESCRIPTIONS[question_style]
    return {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "1-2 sentence summary of the key evidence for the top cause.",
            },
            "top_causes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cause_name": {"type": "string"},
                        "probability": {
                            "type": "number", "minimum": 0, "maximum": 1,
                            "description": "Calibrated probability in [0,1]; the "
                                           "top_causes probabilities should sum to ~1.",
                        },
                    },
                    "required": ["cause_name", "probability"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": 3,
            },
            "questions": {
                "type": "array",
                "description": questions_description.format(
                    num_questions=num_questions),
                "items": {"type": "string"},
                "minItems": num_questions,
                "maxItems": num_questions,
            },
        },
        "required": ["reasoning", "top_causes", "questions"],
        "additionalProperties": False,
    }


class ChampsPromptedInferenceAgent(InferenceAgent):
    """CHAMPS LLM-based cause-of-death inference agent.

    Uses a CODA LLMClient to call an LLM with the CHAMPS COD classification
    prompt and structured output schema. Works with dialogue text only -
    no demographics, MITS, structured VA, or clinical abstraction sections.
    """

    def __init__(self, llm_client: LLMClient,
                 use_diagnosis_standard: bool = False,
                 llm_semaphore: asyncio.Semaphore | None = None,
                 num_questions=DEFAULT_NUM_QUESTIONS,
                 question_style=DEFAULT_QUESTION_STYLE,
                 question_mode=DEFAULT_QUESTION_MODE):
        super().__init__()
        _validate_question_mode(question_mode)
        self.llm_client = llm_client
        self.llm_semaphore = llm_semaphore or asyncio.Semaphore(1)

        self.allowed_causes = CHAMPS_GROUP_CAUSES
        self.cause_to_icd10 = CHAMPS_GROUP_TO_ICD10
        self.schema_guidance = SCHEMA_GUIDANCE
        self.use_diagnosis_standard = use_diagnosis_standard
        self.num_questions = num_questions
        self.question_style = question_style
        self.question_mode = question_mode
        self.output_schema = build_cod_output_schema(num_questions,
                                                     question_style)

        self.cause_to_who_va = CHAMPS_GROUP_TO_WHO_VA
        self.question_bank = load_questions() if question_mode == "voi" else []
        self.asked_questions = set()

        # Case-insensitive lookup for recovering labels that differ only in case.
        self._causes_by_lower = {c.lower(): c for c in self.allowed_causes}

        allowed_str = ", ".join(self.allowed_causes)
        # Spliced in before the format pass so its own {num_questions} renders
        question_guidance = load_question_style_guidance(question_style)
        self.rendered_system_prompt = \
            SYSTEM_PROMPT.replace("{question_guidance}", question_guidance) \
                         .format(allowed_causes=allowed_str,
                                 num_questions=num_questions) + "\n\n" \
            + self.schema_guidance

    async def infer(self, chunk_id: str, text: str,
                    annotations: List[Annotation]):
        """Perform COD inference using accumulated dialogue via LLM."""
        if self.use_diagnosis_standard:
            user_prompt = f"## DIAGNOSIS STANDARD\n{DIAGNOSIS_STANDARD}\n\n"
        else:
            user_prompt = ''
        user_prompt += (
            f"## INPUT\n"
            f"- narrative:\n"
            f"  {self.all_text.strip()}"
        )

        try:
            logger.info(f'Inferring causes up to chunk {chunk_id}...')
            started = time.perf_counter()
            async with self.llm_semaphore:
                response = await asyncio.to_thread(
                    self.llm_client.call_with_schema,
                    system_prompt=self.rendered_system_prompt,
                    user_prompt=user_prompt,
                    schema=self.output_schema,
                    schema_name="champs_cod_classification",
                    temperature=0,
                )
            logger.info("LLM call for chunk %s completed in %.2fs",
                        chunk_id, time.perf_counter() - started)
        except Exception:
            logger.exception("LLM call failed for chunk %s", chunk_id)
            return {
                "causes": {},
                "reasoning": "LLM API call raised an exception.",
                "questions": [],
            }

        if response.get("api_failed"):
            logger.warning("LLM API failed for chunk %s", chunk_id)
            return {
                "causes": {},
                "reasoning": "LLM API call failed after retries.",
                "questions": [],
            }

        return self._parse_response(response)

    # Similarity floor for fuzzy-matching a near-miss label to an allowed cause.
    # Tight enough that only trivial drift (plurals, case, "Neonatal" vs
    # "Perinatal") maps; genuinely off-list labels fall through and are skipped.
    FUZZY_MATCH_CUTOFF = 0.85

    def _match_cause(self, cause_name: str) -> Optional[str]:
        """Map an LLM-returned label to an allowed cause name, or None.

        Small models drift from the exact allowed wording (e.g. "Anemia" for
        "Anemias", "Neonatal" for "Perinatal asphyxia/hypoxia") even though the
        prompt asks for exact labels. Recover those near-misses instead of
        silently dropping the cause: exact, then case-insensitive, then fuzzy.
        """
        name = (cause_name or "").strip()
        if not name:
            return None
        if name in self.cause_to_icd10:
            return name
        ci = self._causes_by_lower.get(name.lower())
        if ci is not None:
            return ci
        close = difflib.get_close_matches(
            name, self.allowed_causes, n=1, cutoff=self.FUZZY_MATCH_CUTOFF)
        return close[0] if close else None

    def reset(self):
        """Reset dialogue history and asked questions for a new interview."""
        super().reset()
        self.asked_questions = set()

    def create_session_agent(self) -> "ChampsPromptedInferenceAgent":
        return ChampsPromptedInferenceAgent(
            llm_client=self.llm_client,
            use_diagnosis_standard=self.use_diagnosis_standard,
            llm_semaphore=self.llm_semaphore,
            num_questions=self.num_questions,
            question_style=self.question_style,
            question_mode=self.question_mode,
        )

    def _who_va_posterior(self, causes):
        """Restate the cause probabilities over the WHO VA causes.

        Causes with no WHO VA counterpart drop out and the rest are
        renormalized. Counterparts are shared, so two CHAMPS causes that the
        WHO VA list does not tell apart merge into one.
        """
        posterior = defaultdict(float)
        for cause in causes.values():
            who_va = self.cause_to_who_va.get(cause["name"])
            if who_va:
                posterior[who_va] += cause["score"]
        total = sum(posterior.values())
        if not total:
            return {}
        return {cause: score / total for cause, score in posterior.items()}

    def _select_questions(self, causes):
        """Pick the unasked questions that best separate the current causes."""
        posterior = self._who_va_posterior(causes)
        if not posterior:
            return []
        selected = select_questions(posterior, self.question_bank,
                                    self.num_questions, self.asked_questions)
        self.asked_questions.update(question.id for question in selected)
        return [question.text for question in selected]

    def _parse_response(self, response: Dict[str, Any]) -> dict:
        """Convert LLM structured response to CODA cause format."""
        causes = {}
        top_causes = response.get("top_causes", [])

        for entry in top_causes:
            cause_name = entry.get("cause_name", "")
            probability = float(entry.get("probability", 0.0))
            matched = self._match_cause(cause_name)

            if matched is None:
                logger.warning("Unknown cause name '%s' - skipping",
                               cause_name)
                continue
            if matched != cause_name:
                logger.info("Mapped cause name '%s' -> '%s'",
                            cause_name, matched)

            icd10 = self.cause_to_icd10[matched]
            curie = f"icd10:{icd10}"
            causes[curie] = {
                "name": matched,
                "identifiers": {"icd10": icd10},
                "score": probability,
            }

        total = sum(c["score"] for c in causes.values())
        if total > 0:
            for c in causes.values():
                c["score"] = round(c["score"] / total, 4)

        reasoning = response.get("reasoning", "")
        if self.question_mode == "voi":
            questions = self._select_questions(causes)
        else:
            questions = response.get("questions", [])

        return {"causes": causes, "reasoning": reasoning,
                "questions": questions}


def create_champs_prompted_agent(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> ChampsPromptedInferenceAgent:
    """Create a ChampsPromptedInferenceAgent with a new LLM client.

    Defaults to the configured inference provider/model.
    """
    from coda.llm_api import create_llm_client

    provider = provider or settings.inference.llm.provider
    model = model or settings.inference.llm.model
    client = create_llm_client(provider=provider, model=model, **kwargs)
    max_concurrency = int(settings.inference.get("max_concurrency", 1) or 1)
    num_questions = int(settings.inference.get("num_questions",
                                               DEFAULT_NUM_QUESTIONS))
    question_style = str(settings.inference.get("question_style",
                                                DEFAULT_QUESTION_STYLE))
    question_mode = str(settings.inference.get("question_mode",
                                               DEFAULT_QUESTION_MODE))
    return ChampsPromptedInferenceAgent(
        llm_client=client,
        llm_semaphore=asyncio.Semaphore(max_concurrency),
        num_questions=num_questions,
        question_style=question_style,
        question_mode=question_mode,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    agent = create_champs_prompted_agent()
    server = InferenceServer(
        agent,
        host=settings.inference.host,
        port=settings.inference.port,
    )
    server.run()
