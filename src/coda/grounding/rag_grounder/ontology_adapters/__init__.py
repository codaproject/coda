"""Ontology-specific adapters for RAG retrieval terms."""

from .icd10 import load_icd10_retrieval_terms
from .icd11 import load_icd11_retrieval_terms

__all__ = ["load_icd10_retrieval_terms", "load_icd11_retrieval_terms"]
