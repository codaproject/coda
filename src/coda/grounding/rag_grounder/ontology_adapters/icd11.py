"""
ICD-11 ontology adapter for RAG retrieval terms.
Currently, we are using the linearization of the ICD-11 to create retrieval 
terms, processing only rows where IsResidual is False and Foundation URI is not 
None.
"""

from __future__ import annotations

import zipfile

import pandas as pd
from openacme import OPENACME_BASE

from ..retrieval_term import RetrievalTerm

ICD11_BASE = OPENACME_BASE.module("icd11")
ICD11_ZIP_URL = (
    "https://icdcdn.who.int/static/releasefiles/2025-01/"
    "SimpleTabulation-ICD-11-MMS-en.zip"
)
ICD11_FNAME = "SimpleTabulation-ICD-11-MMS-en.txt"


def _clean_text(value: object) -> str | None:
    """Return stripped text or None for empty/None values."""
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_icd11_retrieval_terms(
    prefix: str = "icd11",
    include_residual: bool = False,
) -> list[RetrievalTerm]:
    """
    Load ICD-11 terms as RetrievalTerm objects for RAG indexing.

    Parameters
    ----------
    prefix : str
        CURIE namespace prefix used in returned term IDs.
    include_residual : bool
        Whether to include residual categories ("other"/"unspecified").

    Returns
    -------
    list[RetrievalTerm]
        Deterministically ordered retrieval terms.
    """
    zip_path = ICD11_BASE.ensure(url=ICD11_ZIP_URL)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(ICD11_FNAME) as fh:
            df = pd.read_csv(fh, sep="\t")

    terms: list[RetrievalTerm] = []

    for _, row in df.iterrows():
        is_residual = bool(row.get("IsResidual", False))
        if is_residual and not include_residual:
            continue

        foundation_uri = _clean_text(row.get("Foundation URI"))
        title = _clean_text(row.get("Title"))
        if not foundation_uri or not title:
            continue

        foundation_id = foundation_uri.rstrip("/").split("/")[-1]
        cleaned_title = title.replace("- ", "").strip()

        terms.append(
            RetrievalTerm(
                id=f"{prefix}:{foundation_id}",
                name=cleaned_title,
                definition=None,
                synonyms=None,
            )
        )

    terms.sort(key=lambda term: term.id)
    return terms
