"""
Build (or rebuild) the SapBERT ChromaDB vector database over our SNOMED + HP terms.

This is the companion builder for `sapbert_utils.py`: it collects the surface
forms (`collect_terms`), embeds each unique string once with `SapBERTEncoder`, and
indexes them into a persistent ChromaDB collection configured for cosine space.
Querying that collection at runtime is handled by `IndexQueryUtil` in
`sapbert_utils.py`.

Usage
-----
    python -m temporal_ordering.event_grounding.build_chromadb [--rebuild]
        [--batch-size N] [--snomed-root PATH] [--chroma-path PATH]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

from .sapbert_utils import (
    CHROMA_ADD_BATCH,
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_BATCH_SIZE,
    MODEL_NAME,
    SNOMED_DATA,
    SapBERTEncoder,
    _get_chroma_client,
    collect_terms,
)


def build_database(
    snomed_root: Path = SNOMED_DATA,
    chroma_path: Path = CHROMA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rebuild: bool = False,
    erase: bool = False,
    encoder: SapBERTEncoder | None = None,
    hp_grounder=None,
) -> None:
    """Construct (or rebuild) the SapBERT vector database over our SNOMED + HP terms.

    If `erase` is set, the entire `chroma_path` directory (all collections and
    files) is deleted before building. This is broader than `rebuild`, which
    only drops the `snomed_hp_sapbert` collection.

    `encoder` and `hp_grounder` default to the real SapBERT encoder and GILDA's
    default grounder; both can be injected (e.g. by tests) to build over a mock
    term set without touching the transformer weights or the full HP database.
    """
    if erase and Path(chroma_path).exists():
        logger.info(f"  Erasing existing database directory {chroma_path} (--erase) ...")
        shutil.rmtree(chroma_path)

    client = _get_chroma_client(chroma_path)

    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        if rebuild:
            logger.info(f"  Dropping existing collection '{COLLECTION_NAME}' (--rebuild) ...")
            client.delete_collection(COLLECTION_NAME)
        else:
            raise SystemExit(
                f"Collection '{COLLECTION_NAME}' already exists at {chroma_path}. "
                "Pass --rebuild to overwrite it."
            )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "model": MODEL_NAME},
    )

    records = collect_terms(snomed_root, hp_grounder=hp_grounder)
    if not records:
        raise SystemExit("No terms collected; nothing to index.")

    # Embed unique surface strings once, then reuse for any concept sharing them.
    texts = [r.text for r in records]
    unique_texts = list(dict.fromkeys(t.lower() for t in texts))
    logger.info(f"  Embedding {len(unique_texts):,} unique strings "
          f"({len(texts):,} surface forms total) ...")

    embedder = encoder if encoder is not None else SapBERTEncoder()
    # Preserve original casing for embedding (one representative per lowercased key).
    repr_text: dict[str, str] = {}
    for t in texts:
        repr_text.setdefault(t.lower(), t)
    encode_inputs = [repr_text[k] for k in unique_texts]

    vectors = embedder.encode(encode_inputs, batch_size=batch_size, show_progress=True)
    key_to_vec = {k: vectors[i].tolist() for i, k in enumerate(unique_texts)}

    logger.info(f"  Adding {len(records):,} records to ChromaDB collection '{COLLECTION_NAME}' ...")
    for start in tqdm(range(0, len(records), CHROMA_ADD_BATCH), desc="Indexing", unit="batch"):
        chunk = records[start:start + CHROMA_ADD_BATCH]
        collection.add(
            ids=[str(start + i) for i in range(len(chunk))],
            embeddings=[key_to_vec[r.text.lower()] for r in chunk],
            documents=[r.text for r in chunk],
            metadatas=[
                {
                    "concept_id": r.concept_id,
                    "db": r.db,
                    "name": r.name,
                    "status": r.status,
                }
                for r in chunk
            ],
        )

    logger.info(f"\nDone. Indexed {collection.count():,} vectors at {chroma_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--snomed-root", default=str(SNOMED_DATA))
    parser.add_argument("--chroma-path", default=str(CHROMA_PATH))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--rebuild", action="store_true", help="Overwrite an existing collection")
    parser.add_argument(
        "--erase",
        action="store_true",
        help="Delete the entire database directory before building (all collections)",
    )
    args = parser.parse_args()

    build_database(
        snomed_root=Path(args.snomed_root),
        chroma_path=Path(args.chroma_path),
        batch_size=args.batch_size,
        rebuild=args.rebuild,
        erase=args.erase,
    )


if __name__ == "__main__":
    main()
