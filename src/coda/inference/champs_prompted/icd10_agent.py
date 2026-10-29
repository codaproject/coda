"""CHAMPS prompted agent choosing among CHAMPS group causes in ICD-10."""

import csv
import difflib
import logging
import os
from typing import Dict, Any, List, Optional

from gilda import Annotation

from coda.inference.champs_prompted.base_agent import (
    CLINICAL_NOTE_GUIDANCE,
    DEFAULT_NUM_QUESTIONS,
    DEFAULT_QUESTION_STYLE,
    ChampsPromptedInferenceAgent,
    LLMCallFailed,
    failed_result,
    load_question_style_guidance,
    normalize_scores,
    question_schema,
    read_prompt,
)
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
SYSTEM_PROMPT = read_prompt("icd10/system_prompt.txt")
SCHEMA_GUIDANCE = read_prompt("icd10/schema_guidance.txt")
DIAGNOSIS_STANDARD = read_prompt("icd10/diagnosis_standard.txt")

def build_cod_output_schema(num_questions=DEFAULT_NUM_QUESTIONS,
                            question_style=DEFAULT_QUESTION_STYLE):
    """Build the JSON schema for structured output."""
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
            "questions": question_schema(num_questions, question_style),
        },
        "required": ["reasoning", "top_causes", "questions"],
        "additionalProperties": False,
    }


class ChampsPromptedIcd10Agent(ChampsPromptedInferenceAgent):
    """Prompted agent choosing among CHAMPS group causes coded in ICD-10."""

    allowed_causes = CHAMPS_GROUP_CAUSES
    cause_codes = CHAMPS_GROUP_TO_ICD10
    coding_system = "icd10"
    system_prompt = SYSTEM_PROMPT
    schema_guidance = SCHEMA_GUIDANCE
    clinical_note_guidance = CLINICAL_NOTE_GUIDANCE
    diagnosis_standard = DIAGNOSIS_STANDARD

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_schema = build_cod_output_schema(self.num_questions,
                                                     self.question_style)

        # Case-insensitive lookup for recovering labels that differ only in case.
        self._causes_by_lower = {c.lower(): c for c in self.allowed_causes}

        allowed_str = ", ".join(self.allowed_causes)
        # Spliced in before the format pass so its own {num_questions} renders
        question_guidance = load_question_style_guidance(self.question_style)
        self.rendered_system_prompt = \
            self.system_prompt \
                .replace("{question_guidance}", question_guidance) \
                .format(allowed_causes=allowed_str,
                        num_questions=self.num_questions) + "\n\n" \
            + self.schema_guidance + "\n\n" \
            + self.clinical_note_guidance

    async def infer(self, chunk_id: str, text: str,
                    annotations: List[Annotation]):
        """Perform COD inference using accumulated dialogue via LLM."""
        try:
            response = await self.call_llm(
                chunk_id,
                self.llm_client.call_with_schema,
                system_prompt=self.rendered_system_prompt,
                user_prompt=self.build_user_prompt(),
                schema=self.output_schema,
                schema_name="champs_cod_classification",
                temperature=0,
            )
        except LLMCallFailed as e:
            return failed_result(str(e))
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
        if name in self.cause_codes:
            return name
        ci = self._causes_by_lower.get(name.lower())
        if ci is not None:
            return ci
        close = difflib.get_close_matches(
            name, self.allowed_causes, n=1, cutoff=self.FUZZY_MATCH_CUTOFF)
        return close[0] if close else None

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

            code = self.cause_codes[matched]
            causes[f"{self.coding_system}:{code}"] = {
                "name": matched,
                "identifiers": {self.coding_system: code},
                "score": probability,
            }

        reasoning = response.get("reasoning", "")
        questions = response.get("questions", [])

        return {"causes": normalize_scores(causes), "reasoning": reasoning,
                "questions": questions}
