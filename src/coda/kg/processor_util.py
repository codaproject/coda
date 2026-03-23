import logging
import pandas as pd
import csv
from tqdm import tqdm
import os
from coda.kg.sources import KGSourceExporter, KG_BASE

logger = logging.getLogger(__name__)


class DuplicateNodeIDError(ValueError):
    """Raised when a duplicate node ID is found in a node file."""


class MissingNodeIDError(ValueError):
    """Raised when a non-existent node ID referenced in a relationship file."""


def check_duplicated_nodes(exporters: list[KGSourceExporter], strict: bool = True):
    """Check for duplicated node IDs in the exporters, and resolves overlap when possible.

    Parameters
    ----------
    exporters : list[KGSourceExporter]
        List of exporters to check.
    strict : bool
        If to raise an exception if two nodes found with conflicting information.
        Default: True

    Raises
    ------
    DuplicateNodeIDError
        If duplicate node IDs are found with conflicting information.
    """
    logger.info("Checking for duplicated nodes...")
    # Maps each node to the set of sources it comes from
    nodes_and_sources: dict[str, set[str]] = {}
    # Set of duplicated ids
    duplicate_ids = set()
    # Each source's representation of all nodes
    all_nodes: dict[str, dict[str, dict]] = {}
    # Data frames for all nodes
    frames: dict[str, pd.DataFrame] = {}
    for exporter in exporters:
        frames[exporter.name] = pd.read_csv(exporter.nodes_file, sep="\t")
        all_nodes[exporter.name] = {}
        for node in frames[exporter.name].to_dict(orient="records"):
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
                        f"{duplicate_id} has conflicting information in "
                        f"{key} attribute from "
                        f"{' '.join(list(nodes_and_sources[duplicate_id]))}"
                    )
        joined_node["source:string[]"] = \
            ";".join(list(nodes_and_sources[duplicate_id]))
        joined_nodes.append(joined_node)
    if conflicting_nodes_count > 0 and strict:
        raise DuplicateNodeIDError(
            f"found conflicting information in {conflicting_nodes_count} nodes..."
        )
    logger.info("Removing resolved duplicated from node resource files...")
    if len(joined_nodes) > 0:
        joined_df = pd.DataFrame(joined_nodes)
        for exporter in exporters:
            frame = frames[exporter.name]
            frame = frame[~frame["id:ID"].isin(joined_df["id:ID"])]
            frame.sort_values("id:ID").to_csv(exporter.nodes_file, sep="\t", index=False)
        node_file = KG_BASE / "combined_nodes.tsv"
        joined_df.to_csv(node_file, sep="\t", index=False)


def check_missing_node_ids_in_edges(exporters, strict: bool = True):
    """Ensure every node ID referenced in the edges file exists in the
    exporters or combined_nodes node resource files.

    Parameters
    ----------
    exporters : list[KGSourceExporter]
        List of exporters to check.
    strict : bool = True
        If to raise an exception if a node reference is missing or just
        raise a warning
    Raises
    ------
    MissingNodeIDError
        If the head or tail of a node is not present in the set of nodes.
    """
    node_ids = set()
    # Get all node ids
    for exporter in tqdm(exporters, desc="loading all graph nodes",
                         unit="source"):
        with open(exporter.nodes_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            id_index = header.index("id:ID")
            for row in reader:
                id_value = row[id_index]
                node_ids.add(id_value)
    # Also check file that stores combined nodes just in case
    combined_nodes_path = KG_BASE.joinpath("combined_nodes.tsv")
    if os.path.exists(combined_nodes_path):
        with open(combined_nodes_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            id_index = header.index("id:ID")
            for row in reader:
                id_value = row[id_index]
                node_ids.add(id_value)
    # Check that all nodes exist in the edge file
    records: list = []
    for exporter in tqdm(exporters, desc="checking exporter edge existence",
                         unit="source"):
        tqdm.write(f"Checking {exporter.name} edges")
        with open(exporter.edges_file, mode="r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            start_id_index = header.index(":START_ID")
            end_id_index = header.index(":END_ID")
            type_index = header.index(":TYPE")
            msg = ("Edge ({start})-[{type}]->({end}) references "
                   "missing node ID {missing_id}.")
            for row in tqdm(reader, unit="edges", leave=False):
                start_id_value = row[start_id_index]
                end_id_value = row[end_id_index]
                type_value = row[type_index]
                for node_id in (start_id_value, end_id_value):
                    if node_id not in node_ids:
                        record = {
                            "start": start_id_value,
                            "type": type_value,
                            "end": end_id_value,
                            "missing_id": node_id,
                        }
                        if strict:
                            raise MissingNodeIDError(msg.format(**record))
                        else:
                            records.append(record)
                            logger.warning(msg.format(**record))
            if len(records) > 0:
                logger.info("Edges with missing node IDs found, wrote "
                            "list to missing_edges.tsv")
                df = pd.DataFrame(records)
                df.to_csv("missing_edges.tsv", sep="\t", index=False)
