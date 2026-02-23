"""
High-level assembly of cross-ontology mapping graphs.

Composes a base mapping (e.g. SEMRA SSSOM) with extension mappings (e.g. ICD-11 to ICD-10)
and enriches the result with definitions from DefinitionSources.
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

import networkx as nx
import pandas as pd

from .source.base import DefinitionSource
from .source.icd10 import ICD10DefinitionSource

from coda import CODA_BASE

logger = logging.getLogger(__name__)

# Constants
SEMRA_PROCESSED_SSSOM_URL = "https://zenodo.org/records/15826693/files/processed.sssom.tsv.gz"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_ICD11_EDGES_PATH = _REPO_ROOT / "kg" / "icd11_edges.tsv"
DEFAULT_ICD11_NODES_PATH = _REPO_ROOT / "kg" / "icd11_nodes.tsv"


def load_base_mapping(path: Union[Path, str]) -> nx.MultiDiGraph:
    """Load base mapping from SSSOM file (e.g. SEMRA processed.sssom.tsv)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Mapping graph not found at {path}. Download from:\n{SEMRA_PROCESSED_SSSOM_URL}\n"
            "Then set mapping_graph_path to the downloaded file."
        )
    logger.info("Loading mapping graph from %s...", path)
    compression = "gzip" if path.suffix == ".gz" else None
    df = pd.read_csv(path, sep="\t", low_memory=False, compression=compression)

    columns = ["subject_id", "subject_label", "predicate_id", "object_id", "object_label"]
    available = [c for c in columns if c in df.columns]
    df = df[available].copy()
    df = df.dropna(subset=["subject_id", "object_id"])
    df["predicate_id"] = df["predicate_id"].fillna("skos:relatedMatch")
    df = df.drop_duplicates(subset=["subject_id", "object_id"])

    edges = [
        (r.subject_id, r.object_id, {"predicate_id": r.predicate_id})
        for r in df.itertuples()
    ]
    G = nx.MultiDiGraph()
    G.add_edges_from(edges)

    seen = set()
    for _, row in df.iterrows():
        for node_id, label_col in [
            (row.subject_id, "subject_label"),
            (row.object_id, "object_label"),
        ]:
            if node_id not in seen:
                seen.add(node_id)
                label = getattr(row, label_col, "") if hasattr(row, label_col) else ""
                G.nodes[node_id]["label"] = label if pd.notna(label) else ""
                G.nodes[node_id]["namespace"] = (
                    node_id.split(":")[0] if ":" in node_id else ""
                )

    logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_reachable_curies(
    graph: nx.MultiDiGraph,
    curie: str,
    exact_match_only: bool = True,
) -> list[str]:
    """Return CURIEs in the same connected component as curie (excluding curie itself)."""
    if curie not in graph:
        return []

    if exact_match_only:
        exact_edges = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("predicate_id") == "skos:exactMatch"
        ]
        G_traverse = nx.Graph()
        G_traverse.add_edges_from(exact_edges)
    else:
        G_traverse = nx.Graph(graph)

    reachable = nx.node_connected_component(G_traverse, curie)
    return [n for n in reachable if n != curie]


# Backward compatibility
load_mapping_graph_from_sssom = load_base_mapping
load_mapping_graph = load_base_mapping


def extend_graph(
    base_mapping_graph: nx.MultiDiGraph,
    extension_mapping_graph: nx.MultiDiGraph,
) -> nx.MultiDiGraph:
    """Compose base and extension graphs. Returns a new graph."""
    return nx.compose(base_mapping_graph, extension_mapping_graph)


def add_definitions_to_graph(
    graph: nx.MultiDiGraph,
    definition_sources: List[DefinitionSource],
) -> None:
    """Add definitions from sources to graph nodes. Mutates graph in place."""
    ns_to_definitions: dict = {}
    for src in definition_sources:
        ns_to_definitions[src.namespace] = src.load_definitions()

    def lookup(curie: str, ns: str) -> str:
        defs = ns_to_definitions.get(ns)
        if not defs:
            return ""
        return defs.get(curie, "") or defs.get(
            curie.split(":")[-1] if ":" in curie else curie, ""
        )

    count = 0
    for node_id in graph.nodes:
        ns = graph.nodes[node_id].get("namespace", "")
        if not ns:
            ns = node_id.split(":")[0] if ":" in node_id else ""
        definition = lookup(node_id, ns)
        graph.nodes[node_id]["definition"] = definition or ""
        if definition:
            count += 1

    if count:
        logger.info(f"Added definitions to {count} nodes from definition sources")


def load_extension_from_nodes_and_edges_tsv(
    edges_path: Union[Path, str],
    nodes_path: Optional[Union[Path, str]] = None,
    start_col: str = ":START_ID",
    end_col: str = ":END_ID",
    type_col: str = ":TYPE",
    id_col: str = "id:ID",
    name_col: str = "name",
) -> nx.MultiDiGraph:
    """Load an extension mapping graph from Neo4j-style edges TSV and optional nodes TSV.

    Edges TSV: :START_ID, :END_ID, :TYPE (or configurable).
    Nodes TSV (optional): id:ID, name (or configurable). If provided, enriches node labels.
    """
    edges_path = Path(edges_path)
    if not edges_path.exists():
        raise FileNotFoundError(f"Extension edges file not found: {edges_path}")

    # Infer nodes path if not provided: icd11_edges.tsv -> icd11_nodes.tsv
    if nodes_path is None:
        nodes_path = edges_path.parent / (edges_path.stem.replace("_edges", "_nodes") + ".tsv")
    else:
        nodes_path = Path(nodes_path)

    logger.info("Loading extension mapping from %s...", edges_path)
    df = pd.read_csv(edges_path, sep="\t", low_memory=False)
    for col in (start_col, end_col, type_col):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {edges_path}")

    df = df.dropna(subset=[start_col, end_col])
    df = df.drop_duplicates(subset=[start_col, end_col])

    edges = [
        (row[start_col], row[end_col], {"predicate_id": str(row[type_col])})
        for _, row in df.iterrows()
    ]
    G = nx.MultiDiGraph()
    G.add_edges_from(edges)

    # Add node attributes (namespace from CURIE prefix)
    for node_id in G.nodes:
        ns = node_id.split(":")[0] if ":" in node_id else ""
        G.nodes[node_id]["namespace"] = ns
        G.nodes[node_id]["label"] = ""

    # Enrich labels from nodes TSV if available
    if nodes_path.exists():
        logger.info("Loading node labels from %s...", nodes_path)
        nodes_df = pd.read_csv(nodes_path, sep="\t", low_memory=False)
        id_col_actual = id_col if id_col in nodes_df.columns else "id"
        name_col_actual = name_col if name_col in nodes_df.columns else "name"
        if name_col_actual not in nodes_df.columns:
            logger.warning("Nodes file %s has no '%s' column, skipping label enrichment", nodes_path, name_col)
        else:
            for _, row in nodes_df.iterrows():
                nid = row.get(id_col_actual)
                if pd.isna(nid) or nid not in G:
                    continue
                name = row.get(name_col_actual)
                if pd.notna(name) and str(name).strip():
                    G.nodes[nid]["label"] = str(name).strip()
            label_count = sum(1 for n in G.nodes if G.nodes[n].get("label"))
            logger.info(f"Enriched {label_count} labels from nodes file")

    logger.info(f"Loaded extension: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# Backward compatibility
load_extension_from_edges_tsv = load_extension_from_nodes_and_edges_tsv


def compose(
    base: nx.MultiDiGraph,
    *extensions: nx.MultiDiGraph,
) -> nx.MultiDiGraph:
    """Compose base mapping with one or more extension graphs.

    Uses nx.compose: nodes and edges from extensions are merged into base.
    Later extensions overwrite node attributes when keys collide.
    """
    result = base
    for ext in extensions:
        result = extend_graph(result, ext)
    return result


def assemble(
    base_path: Union[Path, str],
    extension_paths: Optional[List[Union[Path, str]]] = None,
    definition_sources: Optional[List[DefinitionSource]] = None,
) -> nx.MultiDiGraph:
    """Assemble a complete cross-ontology mapping graph.

    1. Load base mapping from SSSOM (e.g. SEMRA)
    2. Load extension mappings from edges TSVs (e.g. ICD-11 to ICD-10)
    3. Compose base + extensions
    4. Add definitions from definition sources

    Parameters
    ----------
    base_path : Path or str
        Path to base mapping SSSOM file (e.g. processed.sssom.tsv)
    extension_paths : list of Path or str, optional
        Paths to extension edges TSVs (Neo4j format: :START_ID, :END_ID, :TYPE).
        Node labels are loaded from matching *_nodes.tsv if present (e.g. icd11_edges.tsv -> icd11_nodes.tsv).
    definition_sources : list of DefinitionSource, optional
        Sources to add definitions to the composed graph

    Returns
    -------
    nx.MultiDiGraph
        Composed graph with definitions added (mutated in place)
    """
    graph = load_base_mapping(base_path)

    if extension_paths:
        extensions = [load_extension_from_nodes_and_edges_tsv(p) for p in extension_paths]
        graph = compose(graph, *extensions)

    if definition_sources:
        add_definitions_to_graph(graph, definition_sources)

    return graph


def main() -> None:
    """CLI entry point: builds and prints stats for the default cross-ontology mapping."""
    base_path = CODA_BASE.join("kg", name="processed.sssom.tsv.gz", ensure_exists=False)
    if not base_path.exists():
        logger.info("Base SSSOM not found; downloading from Zenodo...")
        base_path = CODA_BASE.ensure("kg", name="processed.sssom.tsv.gz", url=SEMRA_PROCESSED_SSSOM_URL)

    extension_paths = [DEFAULT_ICD11_EDGES_PATH] if DEFAULT_ICD11_EDGES_PATH.exists() else None
    if not extension_paths:
        logger.warning("ICD-11 edges not found at %s, skipping extension", DEFAULT_ICD11_EDGES_PATH)
    definition_sources = [ICD10DefinitionSource()]
    graph = assemble(
        base_path=base_path,
        extension_paths=extension_paths,
        definition_sources=definition_sources,
    )
    out_path = CODA_BASE.join("kg", name="assembled_mapping.graphml", ensure_exists=True)
    nx.write_graphml(graph, out_path)
    logger.info("Saved assembled mapping to %s", out_path)
    print(f"Assembled mapping: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


if __name__ == "__main__":
    main()
