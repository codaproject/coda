from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coda.inference.champs_prompted import (
        ChampsPromptedInferenceAgent,
        create_champs_prompted_agent,
    )

__all__ = ["ChampsPromptedInferenceAgent", "create_champs_prompted_agent"]


def __getattr__(name):
    if name in __all__:
        from coda.inference import champs_prompted

        return getattr(champs_prompted, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
