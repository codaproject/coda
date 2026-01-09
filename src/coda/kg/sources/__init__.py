from abc import ABC, abstractmethod
from pathlib import Path

HERE = Path(__file__).parent
KG_BASE = HERE.parent.parent.parent.joinpath('kg')


class KGSourceExporter(ABC):
    """Base class for knowledge graph sources."""

    name: str = NotImplemented

    def __init__(self):
        self.nodes_file: Path = KG_BASE / f'{self.name}_nodes.tsv'
        self.edges_file: Path = KG_BASE / f'{self.name}_edges.tsv'

    @abstractmethod
    def export(self):
        """Export the source to Neo4j compatible TSV files for nodes and edges."""
