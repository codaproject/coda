from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


@dataclass
class ExtractorConfig:
    concept_type: str


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
class LLMConfig:
    model: str


@dataclass
class RAGGrounderConfig:
    extractor: ExtractorConfig
    retriever: RetrieverConfig
    llm: LLMConfig

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_CONFIG_PATH) -> "RAGGrounderConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            extractor=ExtractorConfig(**data.get("extractor", {})),
            retriever=RetrieverConfig(**data.get("retriever", {})),
            llm=LLMConfig(**data.get("llm", {})),
        )

    @classmethod
    def default(cls) -> "RAGGrounderConfig":
        return cls.from_yaml(_DEFAULT_CONFIG_PATH)
