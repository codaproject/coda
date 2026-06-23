from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coda.inference.champs_llm_agent import (
        ChampsLLMInferenceAgent,
        create_champs_agent,
    )

__all__ = ["ChampsLLMInferenceAgent", "create_champs_agent"]


def __getattr__(name):
    if name in __all__:
        from coda.inference import champs_llm_agent

        return getattr(champs_llm_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
