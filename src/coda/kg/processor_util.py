import logging
import polars
from coda.kg.sources import KGSourceExporter, KG_BASE

logger = logging.getLogger(__name__)


class DuplicateNodeIDError(ValueError):
    """Raised when a duplicate node ID is found in a node file."""


def check_duplicated_nodes(exporters: list[KGSourceExporter], strict: bool = True):
    """Check for duplicated node IDs in the exporters, and resolves overlap when possible.

    Parameters
    ----------
    exporters : list[KGSourceExporter]
        List of exporters to check.
    strict : bool = True
        If to raise an exception if two nodes found with conflicting information

    Raises
    ------
    DuplicateNodeIDError
        If duplicate node IDs are found with conflicting information.
    """
    logger.info("checking for duplicated nodes...")
    nodes_and_sources: dict[str, set[str]] = (
        {}
    )  ## maps each node to the set of sources it comes from
    duplicate_ids = set()  ## set of duplicated ids
    all_nodes: dict[str, dict[str, dict]] = (
        {}
    )  ## each sources representation of all nodes
    frames: dict[str, polars.DataFrame] = {}  ## data frames for all nodes
    for exporter in exporters:
        frames[exporter.name] = polars.read_csv(exporter.nodes_file, separator="\t")
        all_nodes[exporter.name] = {}
        for node in frames[exporter.name].iter_rows(named=True):
            node_id = node.get("id:ID", "")
            all_nodes[exporter.name][node_id] = node
            if node_id not in nodes_and_sources:
                nodes_and_sources[node_id] = {exporter.name}
            else:
                nodes_and_sources[node_id].add(exporter.name)
                duplicate_ids.add(node_id)
    all_node_attributes = set()
    joined_nodes = []
    conflicting_nodes_count: int = 0
    logger.info("Attempting to automatically resolve duplicated nodes...")
    for duplicate_id in duplicate_ids:
        joined_node = {}
        for source in nodes_and_sources[duplicate_id]:
            node_rep = all_nodes[source][duplicate_id]
            for key in node_rep.keys():
                all_node_attributes.add(key)
                if key not in joined_node:
                    joined_node[key] = node_rep[key]
                elif joined_node.get(key) is None:
                    joined_node[key] = node_rep[key]
                elif joined_node.get(key) == node_rep.get(key):
                    pass
                else:
                    conflicting_nodes_count += 1
                    logger.warning(
                        f"{duplicate_id} has conflicting information in {key} attribute from {' '.join(list(nodes_and_sources[duplicate_id]))}"
                    )
        joined_node["source:string[]"] = ";".join(list(nodes_and_sources[duplicate_id]))
        joined_nodes.append(joined_node)
    if conflicting_nodes_count > 0 and strict:
        raise DuplicateNodeIDError(
            f"found conflicting information in {conflicting_nodes_count} nodes..."
        )
    logger.info("Removing resolved duplicated from node resource files...")
    if len(joined_nodes) > 0:
        joined_df = polars.from_dicts(joined_nodes)
        for exporter in exporters:
            frame = frames[exporter.name]
            frame = frame.join(joined_df, on="id:ID", how="anti")
            frame.sort("id:ID").write_csv(exporter.nodes_file, separator="\t")
        node_file = KG_BASE / f"combined_nodes.tsv"
        joined_df.write_csv(node_file, separator="\t")
