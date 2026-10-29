"""CHAMPS prompted agents choosing ICD-11 codes from age-group label lists."""

import json
import logging
import os
import re
from functools import lru_cache

import yaml

from coda.inference.champs_prompted.base_agent import (
    CLINICAL_NOTE_GUIDANCE,
    PROMPTS_DIR,
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

NUM_RANKS = 5

# Upper age bounds in days of the CHAMPS age groups born alive
AGE_GROUP_LIMITS = [
    (1, "less24hours"),
    (7, "earlyneonate"),
    (28, "lateneonate"),
    (365.25, "infant"),
    (5 * 365.25, "child"),
]


AGE_GROUP_GUIDANCE = yaml.safe_load(
    (PROMPTS_DIR / "icd11" / "age_group_guidance.yaml").read_text())


def age_group(profile):
    """Return the CHAMPS age group of a decedent profile, or None."""
    if profile is None:
        return None
    if profile.stillbirth:
        return "stillbirth"
    days = profile.age.days if profile.age else None
    if days is None or days < 0:
        return None
    for limit, group in AGE_GROUP_LIMITS:
        if days < limit:
            return group
    return None


@lru_cache
def load_labels(group):
    """Return the ICD-11 code-to-name labels of an age group, or all labels."""
    name = f"icd_{group}_labels.json" if group else "icd_all_labels.json"
    with open(get_resource_path(os.path.join("champs", name))) as fh:
        return json.load(fh)


def rank_properties(codes):
    """Return schema properties for ranked codes and their probabilities."""
    properties = {}
    for i in range(1, NUM_RANKS + 1):
        properties[f"rank_{i}_code"] = {"type": "string", "enum": codes}
        properties[f"rank_{i}_prob"] = {"type": "number",
                                        "minimum": 0, "maximum": 1}
    return properties


def object_schema(properties):
    """Return a closed object schema requiring all its properties."""
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


class ChampsPromptedIcd11Agent(ChampsPromptedInferenceAgent):
    """Base class for prompted agents choosing ICD-11 codes by age group.

    The allowed codes and age-group guidance follow the decedent's CHAMPS
    age group, or all CHAMPS ICD-11 codes when the age group is unknown.
    Subclasses implement rank_codes.
    """

    diagnosis_standard = read_prompt("icd11/diagnosis_standard.txt")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question_guidance = \
            load_question_style_guidance(self.question_style) \
            .replace("{num_questions}", str(self.num_questions))
        self.question_schema = question_schema(self.num_questions,
                                               self.question_style)

    def fill_prompt(self, template, labels, group):
        """Fill a prompt template with the allowed codes and guidance."""
        label_list = "\n".join(f"  {code} | {name}"
                               for code, name in sorted(labels.items()))
        guidance = "\n".join(f"    - {item}"
                             for item in AGE_GROUP_GUIDANCE.get(group, [])) \
            or "No age group is known for this case."
        return template.replace("{label_list}", label_list) \
            .replace("{age_group_guidance}", guidance) \
            .replace("{question_guidance}", self.question_guidance)

    async def infer(self, chunk_id, text, annotations):
        """Rank ICD-11 codes for the case and return them as causes."""
        group = age_group(self.metadata.profile)
        labels = load_labels(group)
        logger.info("Ranking ICD-11 codes for age group %s (%d codes)",
                    group or "unknown", len(labels))
        try:
            ranks, reasoning = await self.rank_codes(chunk_id, labels, group)
        except LLMCallFailed as e:
            return failed_result(str(e))
        causes = {}
        for i in range(1, NUM_RANKS + 1):
            code = ranks.get(f"rank_{i}_code")
            if code not in labels or f"icd11:{code}" in causes:
                continue
            causes[f"icd11:{code}"] = {
                "name": labels[code],
                "identifiers": {"icd11": code},
                "score": float(ranks.get(f"rank_{i}_prob", 0.0)),
            }
        return {"causes": normalize_scores(causes), "reasoning": reasoning,
                "questions": ranks.get("questions", [])}

    async def rank_codes(self, chunk_id, labels, group):
        """Return the ranked codes response and the reasoning shown."""
        raise NotImplementedError


class ChampsPromptedIcd11SinglePassAgent(ChampsPromptedIcd11Agent):
    """ICD-11 agent reasoning and ranking codes in one structured call."""

    system_prompt = read_prompt("icd11/system_prompt.txt").replace(
        "{output_steps}", read_prompt("icd11/single_pass_output.txt"))

    async def rank_codes(self, chunk_id, labels, group):
        system_prompt = self.fill_prompt(self.system_prompt, labels, group) \
            + "\n\n" + CLINICAL_NOTE_GUIDANCE
        schema = object_schema({
            "reasoning": {"type": "string",
                          "description": "2-3 sentence clinical assessment."},
            **rank_properties(sorted(labels)),
            "questions": self.question_schema,
        })
        response = await self.call_llm(
            chunk_id,
            self.llm_client.call_with_schema,
            system_prompt=system_prompt,
            user_prompt=self.build_user_prompt(),
            schema=schema,
            schema_name="champs_icd11_classification",
            temperature=0,
        )
        return response, response.get("reasoning", "")


class ChampsPromptedIcd11TwoPassAgent(ChampsPromptedIcd11Agent):
    """ICD-11 agent reasoning in free text, then ranking codes in a second call.

    Mirrors the two-pass constrained pipeline of champs-llm: the first pass
    writes a clinical assessment with candidate codes, the second selects the
    final codes from the allowed list given that assessment.
    """

    pass1_system_prompt = read_prompt("icd11/system_prompt.txt").replace(
        "{output_steps}", read_prompt("icd11/pass1_output.txt"))
    pass2_system_prompt = read_prompt("icd11/pass2_system_prompt.txt")
    pass2_instruction = read_prompt("icd11/pass2_instruction.txt")
    PASS1_TEMPERATURE = 0.7
    PASS2_TEMPERATURE = 0.1

    async def rank_codes(self, chunk_id, labels, group):
        user_prompt = self.build_user_prompt()
        assessment = await self.call_llm(
            chunk_id,
            self.llm_client.call,
            user_prompt=user_prompt,
            system_prompt=self.fill_prompt(self.pass1_system_prompt,
                                           labels, group)
            + "\n\n" + CLINICAL_NOTE_GUIDANCE,
            temperature=self.PASS1_TEMPERATURE,
        )
        response = await self.call_llm(
            chunk_id,
            self.llm_client.call_with_schema,
            system_prompt=self.fill_prompt(self.pass2_system_prompt,
                                           labels, group),
            user_prompt=f"{user_prompt}\n\n## CLINICAL ASSESSMENT\n"
                        f"{assessment}\n\n"
                        + self.fill_prompt(self.pass2_instruction,
                                           labels, group),
            schema=object_schema({**rank_properties(sorted(labels)),
                                  "questions": self.question_schema}),
            schema_name="champs_icd11_codes",
            temperature=self.PASS2_TEMPERATURE,
        )
        return response, clinical_assessment(assessment)


def heading(name):
    """Return a pattern matching a heading or bold label, colon optional."""
    return re.compile(rf"(?:#+ *)?\**{name}\**:?\**", re.IGNORECASE)


ASSESSMENT_HEADING = heading("Clinical Assessment")
CODES_HEADING = heading("Candidate Codes")


def clinical_assessment(text):
    """Return the Clinical Assessment section of a first-pass response."""
    start = ASSESSMENT_HEADING.search(text)
    if start is None:
        return text.strip()
    end = CODES_HEADING.search(text, start.end())
    return text[start.end():end.start() if end else None].strip()
