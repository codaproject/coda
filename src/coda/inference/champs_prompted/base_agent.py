"""Base class and shared helpers of the CHAMPS prompted COD agents."""

import asyncio
import logging
import time
from pathlib import Path

from coda.inference.agent import InferenceAgent
from coda.llm_api.client import LLMClient

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def read_prompt(filename):
    """Read a prompt text file of the CHAMPS prompted agents."""
    return (PROMPTS_DIR / filename).read_text().strip()


CLINICAL_NOTE_GUIDANCE = read_prompt("clinical_note_guidance.txt")

DEFAULT_NUM_QUESTIONS = 3
DEFAULT_QUESTION_STYLE = "default"

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


def load_question_style_guidance(question_style=DEFAULT_QUESTION_STYLE):
    """Read the Step 5 prompt body for a question style."""
    _validate_question_style(question_style)
    return read_prompt(f"question_style_{question_style}.txt")


def question_schema(num_questions=DEFAULT_NUM_QUESTIONS,
                    question_style=DEFAULT_QUESTION_STYLE):
    """Return the JSON schema of the follow-up questions output field."""
    _validate_question_style(question_style)
    return {
        "type": "array",
        "description": QUESTION_SCHEMA_DESCRIPTIONS[question_style].format(
            num_questions=num_questions),
        "items": {"type": "string"},
        "minItems": num_questions,
        "maxItems": num_questions,
    }


class LLMCallFailed(Exception):
    """Raised when an LLM call raises or the client gives up after retries."""


def failed_result(reason):
    """Return the inference result reported when the LLM call failed."""
    return {"causes": {}, "reasoning": reason, "questions": []}


def normalize_scores(causes):
    """Rescale cause scores in place to sum to 1 and return the causes."""
    total = sum(c["score"] for c in causes.values())
    if total > 0:
        for c in causes.values():
            c["score"] = round(c["score"] / total, 4)
    return causes


def render_profile(profile):
    """Return INPUT list lines describing the decedent profile."""
    if profile is None:
        return ""
    lines = []
    if profile.stillbirth:
        lines.append("stillbirth: yes")
    elif profile.age:
        lines.append(f"age at death: {profile.age.value:g} {profile.age.unit}")
    if profile.sex:
        lines.append(f"sex: {profile.sex}")
    if profile.date_of_death:
        lines.append(f"date of death: {profile.date_of_death}")
    if profile.manner_of_death:
        manner = profile.manner_of_death.replace("_", " ")
        lines.append(f"manner of death: {manner}")
    if profile.birth_weight_g:
        lines.append(f"birth weight: {profile.birth_weight_g:g} g")
    if profile.hiv_status:
        lines.append(f"HIV status: {profile.hiv_status}")
    for name, unit, admission, death in [
            ("weight", "kg", profile.weight_kg_admission,
             profile.weight_kg_death),
            ("MUAC", "cm", profile.muac_cm_admission,
             profile.muac_cm_death)]:
        values = [f"{value:g} {unit} at {when}" for value, when in
                  [(admission, "admission"), (death, "death")] if value]
        if values:
            lines.append(f"{name}: {', '.join(values)}")
    return "".join(f"- {line}\n" for line in lines)


class ChampsPromptedInferenceAgent(InferenceAgent):
    """Base class for CHAMPS LLM-based cause-of-death inference agents.

    Uses a CODA LLMClient to call an LLM with a COD classification prompt.
    Works with dialogue text, the case profile and clinical notes - no MITS,
    structured VA, or clinical abstraction sections.

    Subclasses implement infer and set diagnosis_standard, the text put
    before the input when use_diagnosis_standard is on.
    """

    def __init__(self, llm_client: LLMClient,
                 use_diagnosis_standard: bool = False,
                 llm_semaphore: asyncio.Semaphore | None = None,
                 num_questions=DEFAULT_NUM_QUESTIONS,
                 question_style=DEFAULT_QUESTION_STYLE):
        super().__init__()
        self.llm_client = llm_client
        self.llm_semaphore = llm_semaphore or asyncio.Semaphore(1)

        self.use_diagnosis_standard = use_diagnosis_standard
        self.num_questions = num_questions
        self.question_style = question_style

    def build_user_prompt(self):
        """Return the user prompt with the case input accumulated so far."""
        if self.use_diagnosis_standard:
            user_prompt = f"## DIAGNOSIS STANDARD\n" \
                          f"{self.diagnosis_standard}\n\n"
        else:
            user_prompt = ''
        user_prompt += (
            f"## INPUT\n"
            f"{render_profile(self.metadata.profile)}"
            f"- narrative:\n"
            f"  {self.all_text.strip()}"
        )
        if self.clinical_notes:
            user_prompt += "\n\n" + "\n".join(
                f'<clinical_note filename="{note["filename"]}">\n'
                f'{note["text"].strip()}\n</clinical_note>'
                for note in self.clinical_notes)
        return user_prompt

    async def call_llm(self, chunk_id, method, **kwargs):
        """Run an LLM client method off the event loop and return its result.

        Raise LLMCallFailed if the call raises or the client gives up.
        """
        try:
            logger.info(f'Inferring causes up to chunk {chunk_id}...')
            started = time.perf_counter()
            async with self.llm_semaphore:
                response = await asyncio.to_thread(method, **kwargs)
            logger.info("LLM call for chunk %s completed in %.2fs",
                        chunk_id, time.perf_counter() - started)
        except Exception:
            logger.exception("LLM call failed for chunk %s", chunk_id)
            raise LLMCallFailed("LLM API call raised an exception.")
        if isinstance(response, dict) and response.get("api_failed"):
            logger.warning("LLM API failed for chunk %s", chunk_id)
            raise LLMCallFailed("LLM API call failed after retries.")
        return response

    def create_session_agent(self) -> "ChampsPromptedInferenceAgent":
        return self.__class__(
            llm_client=self.llm_client,
            use_diagnosis_standard=self.use_diagnosis_standard,
            llm_semaphore=self.llm_semaphore,
            num_questions=self.num_questions,
            question_style=self.question_style,
        )
