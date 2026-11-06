"""Prompted cause-of-death agents for child (CHAMPS) and adult deaths, and
their configured factory.
"""

import asyncio
from typing import Optional

from coda.config import settings
from coda.inference.agent import InferenceAgent
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

__all__ = [
    "AdultIcd11SinglePassAgent",
    "AdultIcd11TwoPassAgent",
    "ChildAdultAgent",
    "ChampsPromptedIcd10Agent",
    "ChampsPromptedIcd11Agent",
    "ChampsPromptedIcd11SinglePassAgent",
    "ChampsPromptedIcd11TwoPassAgent",
    "ChampsPromptedInferenceAgent",
    "create_champs_prompted_agent",
]

# Agent class per (cohort, coding, passes) setting of
# inference.champs_prompted
AGENT_CLASSES = {
    ("child", "icd10", 1): ChampsPromptedIcd10Agent,
    ("child", "icd11", 1): ChampsPromptedIcd11SinglePassAgent,
    ("child", "icd11", 2): ChampsPromptedIcd11TwoPassAgent,
    ("adult", "icd11", 1): AdultIcd11SinglePassAgent,
    ("adult", "icd11", 2): AdultIcd11TwoPassAgent,
}


class ChildAdultAgent(InferenceAgent):
    """Infer with the child or adult agent of the case's cohort.

    The dialogue history is kept here and shared with whichever agent infers,
    so a case can change cohort at any point of the interview.
    """

    def __init__(self, agents, default):
        super().__init__()
        self.agents = agents
        self.default = default

    @property
    def cohort(self):
        """Return the cohort of the current case."""
        profile = self.metadata.profile
        return profile.cohort if profile and profile.cohort else self.default

    async def infer(self, chunk_id, text, annotations):
        agent = self.agents[self.cohort]
        agent.dialogue_history = self.dialogue_history
        agent.all_text = self.all_text
        agent.metadata = self.metadata
        agent.clinical_notes = self.clinical_notes
        return await agent.infer(chunk_id, text, annotations)

    def reset(self):
        super().reset()
        for agent in self.agents.values():
            agent.reset()

    def create_session_agent(self):
        return ChildAdultAgent(
            {cohort: agent.create_session_agent()
             for cohort, agent in self.agents.items()}, self.default)


def create_champs_prompted_agent(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> ChildAdultAgent:
    """Create the child and adult prompted agents with a new LLM client.

    Each case goes to the agent of its profile's cohort, and to that of
    inference.champs_prompted.cohort when the profile gives none. The
    agent classes follow the coding and passes of inference.champs_prompted,
    and the LLM defaults to the configured inference provider/model.
    """
    from coda.llm_api import create_llm_client

    agent_settings = settings.inference.get("champs_prompted") or {}
    default = agent_settings.get("cohort", "child")
    coding = agent_settings.get("coding", "icd10")
    passes = int(agent_settings.get("passes", 1))
    keys = {"child": ("child", coding, 1 if coding == "icd10" else passes),
            "adult": ("adult", "icd11", passes)}
    for key in keys.values():
        if key not in AGENT_CLASSES:
            raise ValueError(f"Unsupported champs_prompted cohort "
                             f"{key[0]!r}, coding {key[1]!r} with passes "
                             f"{key[2]}")
    if default not in keys:
        raise ValueError(f"Unknown champs_prompted cohort {default!r}")

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
    child = {}
    selection = settings.inference.get("question_selection", "llm")
    if selection != "llm":
        if coding != "icd11":
            raise ValueError(f"Question selection {selection!r} needs icd11 "
                             f"coding")
        from coda.inference.champs_prompted.who_questions import \
            WhoQuestions
        child["question_selector"] = WhoQuestions(
            selection, client,
            skip_logic=settings.inference.get("question_skip_logic", True))
    agents = {
        "child": AGENT_CLASSES[keys["child"]](**shared, **child),
        "adult": AGENT_CLASSES[keys["adult"]](**shared),
    }
    return ChildAdultAgent(agents, default)
