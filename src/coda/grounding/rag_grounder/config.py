from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "grounder_config" / "icd10_config.yaml"


@dataclass
class ExtractorConfig:
    prompt_config_path: str


@dataclass
class RetrieverConfig:
    ontology: str
    # TODO: embedding_model should eventually be read from neo4j metadata
    # stored at kg build time so that the term embeddings and query embeddings
    # uses the same configuration for the embeddings
    embedding_model: str
    top_k: int
    min_similarity: float


@dataclass
class RerankerConfig:
    enabled: bool
    prompt_config_path: str


@dataclass
class LLMConfig:
    model: str
    provider: str


@dataclass
class PromptConfig:
    use_schema: bool
    system_prompt: str
    user_prompt: str
    schema: Dict[str, Any]

    # Only used by extractor
    concept_key: str | None = None
    supporting_evidence_key: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "PromptConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            use_schema=data["use_schema"],
            system_prompt=data["system_prompt"],
            user_prompt=data["user_prompt"],
            schema=data["schema"],
            concept_key=data.get("concept_key"),
            supporting_evidence_key=data.get("supporting_evidence_key"),
        )


@dataclass
class RAGGrounderConfig:
    concept_type: str
    llm: LLMConfig
    extractor: ExtractorConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_CONFIG_PATH) -> "RAGGrounderConfig":
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        concept_type = data.get("concept_type", "disease")
        config = cls(
            concept_type=concept_type,
            llm=LLMConfig(**data.get("llm", {})),
            extractor=ExtractorConfig(**data.get("extractor", {})),
            retriever=RetrieverConfig(**data.get("retriever", {})),
            reranker=RerankerConfig(**data.get("reranker", {}))
        )
        # Resolve prompt_config_path (relative to the config file's directory)
        # to an absolute path so it opens regardless of the current working dir.
        config.extractor.prompt_config_path = str((path.parent / config.extractor.prompt_config_path).resolve())
        if config.reranker.enabled:
            config.reranker.prompt_config_path = str((path.parent / config.reranker.prompt_config_path).resolve())
        return config

    @classmethod
    def default(cls) -> "RAGGrounderConfig":
        return cls.from_yaml(_DEFAULT_CONFIG_PATH)
