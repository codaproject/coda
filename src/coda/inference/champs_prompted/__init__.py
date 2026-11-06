"""Prompted cause-of-death agents for child (CHAMPS) and adult deaths, and
their configured factory.
"""

import asyncio
from typing import Optional

from coda.config import settings
from coda.inference.champs_prompted.adult_agent import (
    AdultIcd11SinglePassAgent,
    AdultIcd11TwoPassAgent,
)
from coda.inference.champs_prompted.base_agent import (
    DEFAULT_NUM_QUESTIONS,
    DEFAULT_QUESTION_STYLE,
    ChampsPromptedInferenceAgent,
)
from coda.inference.champs_prompted.icd10_agent import ChampsPromptedIcd10Agent
from coda.inference.champs_prompted.icd11_agent import (
    ChampsPromptedIcd11Agent,
    ChampsPromptedIcd11SinglePassAgent,
    ChampsPromptedIcd11TwoPassAgent,
)
from coda.inference.population_agent import PopulationAgent

__all__ = [
    "AdultIcd11SinglePassAgent",
    "AdultIcd11TwoPassAgent",
    "ChampsPromptedIcd10Agent",
    "ChampsPromptedIcd11Agent",
    "ChampsPromptedIcd11SinglePassAgent",
    "ChampsPromptedIcd11TwoPassAgent",
    "ChampsPromptedInferenceAgent",
    "create_champs_prompted_agent",
]

# Agent class per (population, coding, passes) setting of
# inference.champs_prompted
AGENT_CLASSES = {
    ("child", "icd10", 1): ChampsPromptedIcd10Agent,
    ("child", "icd11", 1): ChampsPromptedIcd11SinglePassAgent,
    ("child", "icd11", 2): ChampsPromptedIcd11TwoPassAgent,
    ("adult", "icd11", 1): AdultIcd11SinglePassAgent,
    ("adult", "icd11", 2): AdultIcd11TwoPassAgent,
}


def create_champs_prompted_agent(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> PopulationAgent:
    """Create the child and adult prompted agents with a new LLM client.

    Each case goes to the agent of its profile's population, and to that of
    inference.champs_prompted.population when the profile gives none. The
    agent classes follow the coding and passes of inference.champs_prompted,
    and the LLM defaults to the configured inference provider/model.
    """
    from coda.llm_api import create_llm_client

    agent_settings = settings.inference.get("champs_prompted") or {}
    default = agent_settings.get("population", "child")
    coding = agent_settings.get("coding", "icd10")
    passes = int(agent_settings.get("passes", 1))
    keys = {"child": ("child", coding, 1 if coding == "icd10" else passes),
            "adult": ("adult", "icd11", passes)}
    for key in keys.values():
        if key not in AGENT_CLASSES:
            raise ValueError(f"Unsupported champs_prompted population "
                             f"{key[0]!r}, coding {key[1]!r} with passes "
                             f"{key[2]}")
    if default not in keys:
        raise ValueError(f"Unknown champs_prompted population {default!r}")

    provider = provider or settings.inference.llm.provider
    model = model or settings.inference.llm.model
    client = create_llm_client(provider=provider, model=model, **kwargs)
    shared = {
        "llm_client": client,
        "llm_semaphore": asyncio.Semaphore(
            int(settings.inference.get("max_concurrency", 1) or 1)),
        "num_questions": int(settings.inference.get("num_questions",
                                                    DEFAULT_NUM_QUESTIONS)),
        "question_style": str(settings.inference.get(
            "question_style", DEFAULT_QUESTION_STYLE)),
    }
    agents = {
        "child": AGENT_CLASSES[keys["child"]](**shared),
        "adult": AGENT_CLASSES[keys["adult"]](**shared),
    }
    return PopulationAgent(agents, default)
