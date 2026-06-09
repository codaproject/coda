from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


@dataclass
class ExtractorConfig:
    prompt_config_path: Path


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
    prompt_config_path: Path


@dataclass
class LLMConfig:
    model: str


@dataclass
class PromptConfig:
    use_schema: bool
    system_prompt: str
    user_prompt: str
    query_fields: Dict[str, str]
    schema: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path) -> "PromptConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            use_schema=data.get("use_schema", True),
            system_prompt=data.get("system_prompt") or "",
            user_prompt=data["user_prompt"],
            query_fields=data.get("query_fields", {"concept": "Concept", "supporting_evidence": "Supporting_Evidence"}),
            schema=data.get("schema", {}),
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
        config_dir = path.parent
        with open(path) as f:
            data = yaml.safe_load(f)
        concept_type = data.get("concept_type", "disease")
        extractor_data = data.get("extractor", {})
        reranker_data = data.get("reranker", {})
        return cls(
            concept_type=concept_type,
            llm=LLMConfig(**data.get("llm", {})),
            extractor=ExtractorConfig(
                prompt_config_path=(config_dir / extractor_data["prompt_config_path"]).resolve(),
            ),
            retriever=RetrieverConfig(**data.get("retriever", {})),
            reranker=RerankerConfig(
                prompt_config_path=(config_dir / reranker_data["prompt_config_path"]).resolve(),
            )
        )

    @classmethod
    def default(cls) -> "RAGGrounderConfig":
        return cls.from_yaml(_DEFAULT_CONFIG_PATH)
