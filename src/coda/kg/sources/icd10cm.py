"""Knowledge graph source for ICD10-CM."""

import os
from pathlib import Path
from typing import IO
import zipfile
import certifi

import pystow
from lxml import etree
import pandas as pd
from openacme.icd10.map_definitions import _ensure_umls_files
from coda.kg.sources import KGSourceExporter, write_tsv_gz


ICD10CM_BASE_URL = \
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/"
ICD10CM_TABLE_URL = ICD10CM_BASE_URL + "icd10cm-table-and-index-2027.zip"
ICD10CM_TABLE_FILE = "icd10cm-tabular_-2027.xml"
ICD10CM_CODE_URL = ICD10CM_BASE_URL + "icd10cm-code-descriptions-2027.zip"
ICD10CM_CODE_FILE = "icd10cm-order-2027.txt"

ICD10CM_BASE = pystow.module("icd10cm")

NOTE_FIELDS: list[str] = [
    "includes",
    "inclusionTerm",
    "excludes1",
    "excludes2",
    "codeAlso",
    "useAdditionalCode",
    "codeFirst",
    "sevenChrNote",
]


def open_zipped_file(url, file_name) -> IO[bytes]:
    """Return an open file handler for a file inside a zip file."""
    os.environ["SSL_CERT_FILE"] = certifi.where()
    zip_path = ICD10CM_BASE.ensure(url=url)
    with zipfile.ZipFile(zip_path, "r") as zf:
        matches = [p for p in zf.namelist() if Path(p).name == file_name]
        if not matches:
            raise ValueError(f"{file_name} not found in zip from {url}")
        return zf.open(matches[0])


def load_valid_codes() -> dict[str, str]:
    """Load a list of valid codes and if they are billable."""
    codes_to_billable = {}
    with open_zipped_file(ICD10CM_CODE_URL, ICD10CM_CODE_FILE) as f:
        for line in f:
            if len(line) < 16:
                continue
            code = line[6:13].strip().decode()
            billable = "true" if line[14] == ord("1") else "false"
            codes_to_billable[code] = billable
        return codes_to_billable


def extract_note_fields(node: etree._Element):
    tags_dict = {f"{x}:string[]": None for x in NOTE_FIELDS}
    for tag in NOTE_FIELDS:
        tag_info = node.find(tag)
        if tag_info is not None:
            list_rep = [x.text or "" for x in tag_info.findall("note")]
            tags_dict[f"{tag}:string[]"] = ";".join(list_rep)
    return tags_dict


def synthesize_extensions(icd10_cm_code, desc, scd, valid_codes):
    """Yield (node_dict, edge) for each synthesized 7th-char extension."""
    for ext in scd.findall("extension"):
        char = ext.get("char")
        if "." not in icd10_cm_code:
            padded = icd10_cm_code + "."
        else:
            padded = icd10_cm_code
        padded = f"{padded:X<7}"
        ext_code = padded + char
        bare_code = ext_code.replace(".", "")
        # There are some codes the logic would suggest exist but medically
        # do not ex S06.396 which does not have S06.396D because it would
        # be fatal
        if bare_code in valid_codes:
            yield (
                {
                    "id:ID": f'icd10cm:{ext_code}',
                    ":LABEL": "icd10cm",
                    "kind": "code",
                    "billable:boolean": valid_codes[bare_code],
                    "description": f"{desc}, {ext.text}",
                    **{f"{k}:string[]": None for k in NOTE_FIELDS},
                },
                {":START_ID": f'icd10cm:{icd10_cm_code}',
                 ":END_ID": f'icd10cm:{ext_code}',
                 ":TYPE": "is_a"},
            )


def extract_icd10cm_table(valid_codes):
    """extract the node and edge set for ICD10-CM"""
    with open_zipped_file(ICD10CM_TABLE_URL, ICD10CM_TABLE_FILE) as fh:
        tree = etree.parse(fh)

    nodes: list[dict] = []
    edges: list[dict] = []

    for chapter in tree.findall("chapter"):
        chapter_code = chapter.find("name").text
        nodes.append(
            {
                "id:ID": f'icd10cm:{chapter_code}',
                ":LABEL": "icd10cm",
                "kind": chapter.tag,
                "description": chapter.find("desc").text,
                "billable:boolean": "false",
                **extract_note_fields(chapter),
            }
        )

        for block in chapter.findall("section"):
            block_code = block.get("id")
            tag = "range" if "-" in block_code else block.tag
            nodes.append(
                {
                    "id:ID": f'icd10cm:{block_code}',
                    ":LABEL": "icd10cm",
                    "kind": tag,
                    "description": block.find("desc").text,
                    "billable:boolean": "false",
                    **extract_note_fields(block),
                }
            )
            edges.append(
                {":START_ID": f'icd10cm:{block_code}',
                 ":END_ID": f'icd10cm:{chapter_code}',
                 ":TYPE": "is_a"}
            )

            # Extract nested codes with a stack
            stack = [(diag, block_code, None)
                     for diag in block.findall("diag")]

            while stack:
                code, parent_code, inherited_scd = stack.pop()
                icd10_cm_code = code.find("name").text
                desc = code.find("desc").text

                # Parse note fields (include, exlude, code also etc) added to
                # node as lists
                tags_dict = extract_note_fields(code)

                # Fail safe to prevent adding the same ID as both a section
                # and code
                if icd10_cm_code != block_code:
                    nodes.append(
                        {
                            "id:ID": f'icd10cm:{icd10_cm_code}',
                            ":LABEL": "icd10cm",
                            "kind": "code",
                            "billable:boolean": valid_codes.get(
                                icd10_cm_code.replace(".", ""), "false"
                            ),
                            "description": desc,
                            **tags_dict,
                        }
                    )
                    edges.append(
                        {
                            ":START_ID": f'icd10cm:{icd10_cm_code}',
                            ":END_ID": f'icd10cm:{parent_code}',
                            ":TYPE": "is_a",
                        }
                    )

                # Check for seven character definition from either
                # this code or its ancestor
                scd = code.find("sevenChrDef") or inherited_scd

                # Case 1: node has descendants -> add to stack
                children = code.findall("diag")
                if children:
                    stack += [(child, icd10_cm_code, scd)
                              for child in children]
                # Case 2: lead node -> check for seventh character descendants
                elif scd is not None:
                    for node_dict, edge in \
                            synthesize_extensions(icd10_cm_code, desc or "",
                                                  scd, valid_codes):
                        nodes.append(node_dict)
                        edges.append(edge)

    # check to make sure all codes and sections are present
    _validate_nodes(nodes, valid_codes)
    return nodes, edges


def _validate_nodes(nodes, valid_codes):
    """Validate that all codes were extracted and no extras exist."""
    found_codes = set(
        row["id:ID"].removeprefix("icd10cm:").replace(".", "")
        for row in nodes
        if row["kind"] == "code" or row["kind"] == "section"
    )
    ref_codes = set(valid_codes.keys())
    missing = ref_codes - found_codes
    extra = found_codes - ref_codes

    if len(missing) > 0:
        raise ValueError(
            f"Extracted nodes missing {len(missing)} codes \n "
            f"Example missing codes {sorted(missing)[:10]}"
        )
    if len(extra) > 0:
        raise ValueError(
            f"Extracted nodes have {len(extra)} extra codes \n "
            f"Example extra codes {sorted(extra)[:10]}"
        )


def get_icd10cm_nodes_edges():
    """Get the ICD10-CM nodes and edges as lists of dicts."""
    valid_codes = load_valid_codes()
    nodes, edges = extract_icd10cm_table(valid_codes)
    return nodes, edges


def get_mapping_edges(valid_icd10cm_codes):
    """Build maps_to edges from ICD10-CM codes to WHO ICD10 codes.

    Pairs are taken from UMLS MRCONSO where an ICD10-CM and a WHO ICD10
    code share a concept (CUI). ``valid_icd10cm_codes`` is the set of
    unprefixed ICD10-CM codes to keep.
    """
    mrconso_path, _ = _ensure_umls_files()
    mrconso = pd.read_csv(Path(mrconso_path), sep="|", header=None,
                          # CUI, SAB (source), CODE
                          usecols=[0, 11, 13],
                          names=["CUI", "SAB", "CODE"])

    icd10 = mrconso[mrconso.SAB == "ICD10"][["CUI", "CODE"]]
    icd10cm = mrconso[mrconso.SAB == "ICD10CM"][["CUI", "CODE"]]

    # Pair WHO and CM codes that share a UMLS concept (CUI)
    xwalk = icd10.merge(icd10cm, on="CUI", suffixes=("_icd10", "_icd10cm"))
    xwalk = xwalk[xwalk["CODE_icd10cm"].isin(valid_icd10cm_codes)]
    xwalk = xwalk.drop_duplicates(subset=["CODE_icd10", "CODE_icd10cm"])

    return [{":START_ID": f"icd10cm:{row.CODE_icd10cm}",
             ":END_ID": f"icd10:{row.CODE_icd10}",
             ":TYPE": "maps_to"}
            for row in xwalk.itertuples(index=False)]


class Icd10CmExporter(KGSourceExporter):
    name = "icd10cm"

    def export(self):
        nodes, edges = get_icd10cm_nodes_edges()
        # Add ICD10-CM -> WHO ICD10 mapping edges
        valid_ids = {n["id:ID"].removeprefix("icd10cm:") for n in nodes}
        edges.extend(get_mapping_edges(valid_ids))

        nodes_df = pd.DataFrame.from_records(nodes)
        write_tsv_gz(nodes_df.sort_values("id:ID"), self.nodes_file)

        edges_df = pd.DataFrame.from_records(edges)
        write_tsv_gz(edges_df.sort_values([":START_ID", ":END_ID"]),
                     self.edges_file)


if __name__ == "__main__":
    exporter = Icd10CmExporter()
    exporter.export()
