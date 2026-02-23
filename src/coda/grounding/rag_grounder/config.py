from dataclasses import dataclass
from typing import Optional

from .retrieval_term import TermStore


@dataclass
class RAGGrounderConfig:
    term_store: Optional[TermStore] = None
    model_name: str = "all-MiniLM-L6-v2"
    concept_type: str = "disease"
    retrieval_top_k: int = 10
    retrieval_min_similarity: float = 0.0
