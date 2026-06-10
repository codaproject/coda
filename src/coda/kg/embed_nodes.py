"""Precompute and store node embeddings in nodes TSV files."""
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from coda.kg.sources import write_tsv_gz

DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_COL = "embedding:float[]"


def embed_nodes(nodes_file: Path, model_name: str = DEFAULT_MODEL) -> None:
    """Encode the 'name' column of a nodes TSV.gz and write back with embeddings."""
    df = pd.read_csv(nodes_file, sep="\t", low_memory=False)
    model = SentenceTransformer(model_name)
    names = df["name"].fillna("").tolist()
    embeddings = model.encode(names, normalize_embeddings=True, show_progress_bar=True)
    df[EMBEDDING_COL] = [";".join(e.astype(str).tolist()) for e in embeddings]
    write_tsv_gz(df, nodes_file)
