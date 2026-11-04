"""Prompted agents choosing ICD-11 codes of adult causes of death."""

import csv
from functools import lru_cache

import yaml

from coda.inference.champs_prompted.base_agent import PROMPTS_DIR, read_prompt
from coda.inference.champs_prompted.icd11_agent import (
    ChampsPromptedIcd11SinglePassAgent,
    ChampsPromptedIcd11TwoPassAgent,
)
from coda.resources import get_resource_path

# Upper age bounds in days of the adult age groups
ADULT_AGE_GROUP_LIMITS = [
    (40 * 365.25, "young_adult"),
    (65 * 365.25, "middle_aged"),
    (float("inf"), "older_adult"),
]


@lru_cache
def load_adult_labels(sex=None):
    """Return the ICD-11 code-to-cause labels of adult causes for a sex.

    Causes restricted to one sex are left out for the other, all causes are
    kept when the sex is unknown.
    """
    with open(get_resource_path("adult_causes.csv")) as fh:
        return {row["icd11_code"]: row["cause"] for row in csv.DictReader(fh)
                if row["sex"] in ("any", sex)
                or sex not in ("female", "male")}


def load_adult_chain_labels():
    """Return the ICD-11 code-to-name labels of adult causal chain links."""
    with open(get_resource_path("adult_chain_causes.csv")) as fh:
        return {row["icd11_code"]: row["cause"]
                for row in csv.DictReader(fh)}


class AdultIcd11Mixin:
    """Adult causes and guidance for the ICD-11 prompted agents."""

    clinical_guidance = read_prompt("icd11_adult/clinical_guidance.txt")
    age_group_limits = ADULT_AGE_GROUP_LIMITS
    age_group_guidance = yaml.safe_load(
        (PROMPTS_DIR / "icd11_adult" / "age_group_guidance.yaml").read_text())
    chain_labels = load_adult_chain_labels()

    def allowed_labels(self, group):
        profile = self.metadata.profile
        return load_adult_labels(profile.sex if profile else None)


class AdultIcd11SinglePassAgent(AdultIcd11Mixin,
                                ChampsPromptedIcd11SinglePassAgent):
    """Adult ICD-11 agent reasoning and ranking codes in one call."""


class AdultIcd11TwoPassAgent(AdultIcd11Mixin, ChampsPromptedIcd11TwoPassAgent):
    """Adult ICD-11 agent reasoning in free text, then ranking codes."""
