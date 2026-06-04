from abc import ABC, abstractmethod
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent.parent.parent
KG_BASE = REPO_ROOT.joinpath('kg')
# Directory for human-readable build reports and QC statistics. These are
# regenerated on every build and are not version controlled.
REPORTS_BASE = KG_BASE.joinpath('reports')


class KGSourceExporter(ABC):
    """Base class for knowledge graph sources."""

    name: str = NotImplemented

    def __init__(self):
        self.nodes_file: Path = KG_BASE / f'{self.name}_nodes.tsv.gz'
        self.edges_file: Path = KG_BASE / f'{self.name}_edges.tsv.gz'

    @abstractmethod
    def export(self):
        """Export the source to Neo4j compatible TSV files for nodes and edges."""
