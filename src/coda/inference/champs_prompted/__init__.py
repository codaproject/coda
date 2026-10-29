"""CHAMPS prompted cause-of-death agents and their configured factory."""

import asyncio
from typing import Optional

from coda.config import settings
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

__all__ = [
    "ChampsPromptedIcd10Agent",
    "ChampsPromptedIcd11Agent",
    "ChampsPromptedIcd11SinglePassAgent",
    "ChampsPromptedIcd11TwoPassAgent",
    "ChampsPromptedInferenceAgent",
    "create_champs_prompted_agent",
]

# Agent class per (coding, passes) setting of inference.champs_prompted
AGENT_CLASSES = {
    ("icd10", 1): ChampsPromptedIcd10Agent,
    ("icd11", 1): ChampsPromptedIcd11SinglePassAgent,
    ("icd11", 2): ChampsPromptedIcd11TwoPassAgent,
}


def create_champs_prompted_agent(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> ChampsPromptedInferenceAgent:
    """Create the configured CHAMPS prompted agent with a new LLM client.

    The agent class follows inference.champs_prompted (coding and passes),
    and the LLM defaults to the configured inference provider/model.
    """
    from coda.llm_api import create_llm_client

    agent_settings = settings.inference.get("champs_prompted") or {}
    coding = agent_settings.get("coding", "icd10")
    passes = 1 if coding == "icd10" else int(agent_settings.get("passes", 1))
    if (coding, passes) not in AGENT_CLASSES:
        raise ValueError(f"Unsupported champs_prompted coding {coding!r} "
                         f"with passes {passes}")

    provider = provider or settings.inference.llm.provider
    model = model or settings.inference.llm.model
    client = create_llm_client(provider=provider, model=model, **kwargs)
    max_concurrency = int(settings.inference.get("max_concurrency", 1) or 1)
    num_questions = int(settings.inference.get("num_questions",
                                               DEFAULT_NUM_QUESTIONS))
    question_style = str(settings.inference.get("question_style",
                                                DEFAULT_QUESTION_STYLE))
    return AGENT_CLASSES[(coding, passes)](
        llm_client=client,
        llm_semaphore=asyncio.Semaphore(max_concurrency),
        num_questions=num_questions,
        question_style=question_style,
    )
