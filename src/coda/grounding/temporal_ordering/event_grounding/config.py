"""Configuration for the temporal-ordering event grounder.

Mirrors the yaml-backed config pattern used by the RAG grounder
(:mod:`coda.grounding.rag_grounder.config`). The SNOMED data directory used by
the event grounder can be provided either via the ``SNOMED_DATA_PATH``
environment variable (which takes precedence) or the
``grounder.snomed_data_path`` field in the yaml config. When neither is set,
event grounding is disabled — it is an optional module.
"""

from dataclasses import dataclass
import os
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "temporal_ordering_config.yaml"
_SNOMED_DATA_ENV_VAR = "SNOMED_DATA_PATH"


@dataclass
class TemporalOrderingGroundingConfig:
    # Absolute path to the SNOMED RF2 data directory used by the GILDA-based
    # event grounder, or None when grounding is disabled.
    snomed_data_path: Path | None

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_CONFIG_PATH) -> "TemporalOrderingGroundingConfig":
        path = Path(path)
        data = {}
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        grounder = data.get("grounder") or {}

        # The environment variable wins; fall back to the yaml field. An empty or
        # missing value in both cases means grounding is disabled.
        env_value = (os.getenv(_SNOMED_DATA_ENV_VAR) or "").strip()
        if env_value:
            # Env-provided paths are resolved against the current working dir.
            snomed_data_path: Path | None = Path(env_value).expanduser().resolve()
        else:
            yaml_value = (grounder.get("snomed_data_path") or "").strip()
            if yaml_value:
                # Yaml-relative paths resolve against the config file's directory,
                # matching the RAG grounder's convention.
                snomed_data_path = (path.parent / yaml_value).expanduser().resolve()
            else:
                snomed_data_path = None

        return cls(snomed_data_path=snomed_data_path)

    @classmethod
    def default(cls) -> "TemporalOrderingGroundingConfig":
        return cls.from_yaml(_DEFAULT_CONFIG_PATH)
