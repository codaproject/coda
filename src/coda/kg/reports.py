"""Descriptive build reports written to kg/reports/ on every build.

These are regenerated each build and not version controlled. The QC checks
that also emit reports (missing_nodes.tsv, duplicate_nodes.tsv) live in
processor_util; this module holds the descriptive ones.
"""
import csv
import gzip
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from coda import __version__
from coda.kg.sources import KGSourceExporter, REPORTS_BASE

logger = logging.getLogger(__name__)


def _namespace(curie: str) -> str:
    """Return the namespace prefix of a CURIE (the part before the colon)."""
    return curie.split(":", 1)[0]


def _read_tsv_gz(path):
    """Yield (header, row_iterator) for a gzipped TSV file."""
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        yield header
        yield from reader


def generate_source_summary(exporters: list[KGSourceExporter]):
    """Write source_summary.tsv: one row per source with node/edge counts.

    A quick sanity table to confirm each source contributed the expected
    volume and the right node labels and edge types.
    """
    records = []
    for exporter in tqdm(exporters, desc="summarizing sources", unit="source"):
        # Only the label/type columns are needed; reading just these avoids
        # mixed-dtype warnings on the wider annotation columns and is faster.
        labels_col = pd.read_csv(
            exporter.nodes_file, sep="\t", usecols=[":LABEL"]
        )[":LABEL"]
        types_col = pd.read_csv(
            exporter.edges_file, sep="\t", usecols=[":TYPE"]
        )[":TYPE"]
        labels = sorted(labels_col.dropna().unique().tolist())
        types = sorted(types_col.dropna().unique().tolist())
        records.append({
            "source": exporter.name,
            "n_nodes": len(labels_col),
            "n_edges": len(types_col),
            "node_labels": ";".join(str(x) for x in labels),
            "edge_types": ";".join(str(x) for x in types),
            "nodes_file_mb": round(
                exporter.nodes_file.stat().st_size / 1e6, 3
            ),
            "edges_file_mb": round(
                exporter.edges_file.stat().st_size / 1e6, 3
            ),
        })
    df = pd.DataFrame(records).sort_values("source")
    REPORTS_BASE.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        REPORTS_BASE.joinpath("source_summary.tsv"), sep="\t", index=False
    )
    logger.info("Wrote source summary for %d sources", len(records))


def generate_namespace_bridge_matrix(exporters: list[KGSourceExporter]):
    """Write namespace_bridge_matrix.tsv: edge counts by namespace pair.

    Counts edges per (source, start namespace, end namespace), e.g. how many
    mondo to omim edges MONDO contributes, to show how the ontologies link up.
    """
    counts: Counter = Counter()
    for exporter in tqdm(
        exporters, desc="building bridge matrix", unit="source"
    ):
        rows = _read_tsv_gz(exporter.edges_file)
        header = next(rows)
        start_idx = header.index(":START_ID")
        end_idx = header.index(":END_ID")
        for row in rows:
            counts[(
                exporter.name,
                _namespace(row[start_idx]),
                _namespace(row[end_idx]),
            )] += 1
    df = pd.DataFrame(
        [
            {
                "source": source,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "count": count,
            }
            for (source, start_ns, end_ns), count in counts.items()
        ]
    ).sort_values(
        ["source", "count", "start_ns", "end_ns"],
        ascending=[True, False, True, True],
    )
    REPORTS_BASE.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        REPORTS_BASE.joinpath("namespace_bridge_matrix.tsv"),
        sep="\t",
        index=False,
    )
    logger.info("Wrote namespace bridge matrix with %d rows", len(df))


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_rows(path) -> int:
    """Count data rows (excluding header) in a gzipped TSV file."""
    with gzip.open(path, "rt") as f:
        return max(sum(1 for _ in f) - 1, 0)


def generate_build_manifest(
    exporters: list[KGSourceExporter], build_seconds: float | None = None
):
    """Write build_manifest.json for provenance and diffing builds.

    Records the CODA version, build timestamp, and per-source file sizes, row
    counts, and SHA-256 checksums so builds can be compared over time.
    """
    sources = []
    for exporter in tqdm(exporters, desc="manifest", unit="source"):
        sources.append({
            "source": exporter.name,
            "nodes_file": exporter.nodes_file.name,
            "n_nodes": _count_rows(exporter.nodes_file),
            "nodes_bytes": exporter.nodes_file.stat().st_size,
            "nodes_sha256": _sha256(exporter.nodes_file),
            "edges_file": exporter.edges_file.name,
            "n_edges": _count_rows(exporter.edges_file),
            "edges_bytes": exporter.edges_file.stat().st_size,
            "edges_sha256": _sha256(exporter.edges_file),
        })
    manifest = {
        "coda_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": (
            round(build_seconds, 1) if build_seconds is not None else None
        ),
        "n_sources": len(sources),
        "total_nodes": sum(s["n_nodes"] for s in sources),
        "total_edges": sum(s["n_edges"] for s in sources),
        "sources": sources,
    }
    REPORTS_BASE.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_BASE.joinpath("build_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote build manifest for %d sources", len(sources))


def generate_reports(
    exporters: list[KGSourceExporter], build_seconds: float | None = None
):
    """Generate all descriptive build reports under kg/reports/."""
    generate_source_summary(exporters)
    generate_namespace_bridge_matrix(exporters)
    generate_build_manifest(exporters, build_seconds=build_seconds)
