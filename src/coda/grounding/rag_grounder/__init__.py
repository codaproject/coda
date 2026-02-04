"""
Generalized RAG-based grounder for custom retrieval terms.

This module provides a flexible grounding system that can work with any set of terms,
unlike the ICD-10 specific grounder.
"""
from coda import CODA_BASE
from .retrieval_term import RetrievalTerm

# Create pystow module for rag_grounder data
RAG_GROUNDER_BASE = CODA_BASE.module("rag_grounder")

__all__ = ['RetrievalTerm', 'RAG_GROUNDER_BASE']
