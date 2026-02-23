"""
Generalized RAG-based grounder for custom retrieval terms.

This module provides a flexible grounding system that can work with any set of terms,
unlike the ICD-10 specific grounder. Terms and embeddings are kept in memory.
"""
from .retrieval_term import RetrievalTerm, TermStore
from .config import RAGGrounderConfig
from .core import RAGGrounderPipeline, PipelineResult, ProcessedConcept, EvidenceSpan

__all__ = [
    "RetrievalTerm",
    "TermStore",
    "RAGGrounderConfig",
    "RAGGrounderPipeline",
    "PipelineResult",
    "ProcessedConcept",
    "EvidenceSpan",
]
