"""SapBERT embeddings for SNOMED CT surface-form nodes, cached to ``.npy``.

Each SNOMED surface form (a concept's FSN or one of its synonyms) becomes its
own KG node with its own vector, so the Neo4j vector index covers both FSNs and
synonyms. Embedding every surface form once is expensive, so — mirroring the
icd10 embedding pipeline in ``openacme`` — vectors are cached on disk and only
regenerated when the surface-form set changes.

Cache layout (in the ``coda:snomedct_embeddings`` pystow module):
  - ``embeddings.npy``            : float32 array, rows ordered by sorted node id
  - ``snomedct_surface_forms.json``: ``{node_id: text}`` that was embedded

The node id -> row mapping is implicit in the sorted keys of the JSON, matching
the convention openacme uses for icd10 (``get_code_index``).
"""
import json
import logging
from pathlib import Path

import numpy as np

from coda import CODA_BASE
from coda.embeddings.sapbert import DEFAULT_BATCH_SIZE, MODEL_NAME, SapBERTEncoder

logger = logging.getLogger(__name__)

SNOMED_EMBEDDINGS_BASE = CODA_BASE.module("snomedct_embeddings")
EMBEDDINGS_FILE = "embeddings.npy"
SURFACE_FORMS_FILE = "snomedct_surface_forms.json"


def get_node_index(surface_forms: dict[str, str]) -> dict:
    """Return the node-id <-> row-index mapping (rows ordered by sorted node id)."""
    node_ids = sorted(surface_forms.keys())
    return {
        "node_to_idx": {node_id: idx for idx, node_id in enumerate(node_ids)},
        "idx_to_node": node_ids,
    }


def _paths(base=None) -> tuple[Path, Path]:
    base = base or SNOMED_EMBEDDINGS_BASE
    root = Path(base.base)
    return root / EMBEDDINGS_FILE, root / SURFACE_FORMS_FILE


def load_embeddings(base=None) -> tuple[np.ndarray, dict[str, str]]:
    """Load the cached embeddings and their ``{node_id: text}`` map."""
    embeddings_file, surface_forms_file = _paths(base)
    embeddings = np.load(embeddings_file.as_posix())
    with open(surface_forms_file, encoding="utf-8") as f:
        surface_forms = json.load(f)
    return embeddings, surface_forms


def generate_snomedct_embeddings(
    node_texts: dict[str, str],
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    base=None,
) -> tuple[np.ndarray, dict[str, str]]:
    """Return SapBERT embeddings for ``{node_id: text}``, using the disk cache.

    If a cache exists for exactly this surface-form set it is reloaded;
    otherwise the vectors are (re)generated and saved. Identical strings are
    encoded once and shared across the nodes that use them.
    """
    embeddings_file, surface_forms_file = _paths(base)

    if embeddings_file.is_file() and surface_forms_file.is_file():
        cached_embeddings, cached_surface_forms = load_embeddings(base)
        if cached_surface_forms == node_texts:
            logger.info(
                "Reusing cached SNOMED embeddings for %d surface forms",
                len(node_texts),
            )
            return cached_embeddings, cached_surface_forms
        logger.info("SNOMED surface-form set changed; regenerating embeddings")

    node_ids = sorted(node_texts)

    # Encode each unique string once, then fan the vector out to every node
    # that shares it.
    unique_texts = list(dict.fromkeys(node_texts[n] for n in node_ids))
    logger.info(
        "Embedding %d unique strings for %d SNOMED surface forms with %s",
        len(unique_texts), len(node_ids), model_name,
    )
    encoder = SapBERTEncoder(model_name=model_name)
    vectors = encoder.encode(
        unique_texts, batch_size=batch_size, show_progress=True
    ).numpy().astype(np.float32)
    text_to_vec = {text: vectors[i] for i, text in enumerate(unique_texts)}

    embeddings = np.vstack([text_to_vec[node_texts[n]] for n in node_ids])

    root = Path((base or SNOMED_EMBEDDINGS_BASE).base)
    root.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file.as_posix(), embeddings)
    with open(surface_forms_file, "w", encoding="utf-8") as f:
        json.dump(node_texts, f)
    logger.info("Saved SNOMED embeddings %s to %s", embeddings.shape, root)

    return embeddings, node_texts
