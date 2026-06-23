"""Knowledge graph source for ICD10-CM."""

import os
from pathlib import Path
from typing import Tuple, Generator, IO
import zipfile
import certifi

import pystow
from lxml import etree
import pandas as pd
from openacme.icd10.map_definitions import _ensure_umls_files


os.environ["SSL_CERT_FILE"] = certifi.where()

ICD10CM_BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2027/"
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
    """Return an open file hanler for a file inside a zip file."""
    zip_path = pystow.module("icd10-cm").ensure(url=url)
    with zipfile.ZipFile(zip_path, "r") as zf:
        matches = [p for p in zf.namelist() if Path(p).name == file_name]
        if not matches:
            raise ValueError(f"{file_name} not found in zip from {url}")
        return zf.open(matches[0])


def load_valid_codes() -> dict[str,str]:
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
    """Yield (node_dict, edge_tuple) for each synthesized 7th-char extension."""
    for ext in scd.findall("extension"):
        char = ext.get("char")
        padded = icd10_cm_code + "." if not "." in icd10_cm_code else icd10_cm_code
        padded = f"{padded:X<7}"
        ext_code = padded + char
        # There are some codes the logic would sugest exist but medically
        # do not ex S06.396 which does not have S06.396D because it would
        # be fatal

        if ext_code.replace(".", "") in valid_codes:
            yield (
                {
                    "id:ID": f'icd10cm:{ext_code}',
                    ":LABEL": "icd10cm",
                    "kind": "code",
                    "billable:boolean": valid_codes.get(ext_code.replace(".", "")),
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
    """helper function to validate that all codes were extracted and no extras"""
    found_codes = set(
        row["id:ID"].replace(".", "")
        for row in nodes
        if row["kind"] == "code" or row["kind"] == "section"
    )
    ref_codes = set(valid_codes.keys())
    missing = ref_codes - found_codes
    extra = found_codes - ref_codes

    if len(missing) > 0:
        raise ValueError(
            f"Extracted nodes missing {len(missing)} codes \n Example missing codes {sorted(missing)[:10]}"
        )
    if len(extra) > 0:
        raise ValueError(
            f"Extracted nodes have {len(extra)} extra codes \n Example extra codes {sorted(extra)[:10]}"
        )


def get_icd10cm_nodes_edges():
    """Get the ICD10-CM nodes and edges as dataframes."""
    valid_codes = load_valid_codes()
    nodes, edges = extract_icd10cm_table(valid_codes)
    return nodes, edges


def get_mapping_edges():
    # Parse into data frames and write out
    nodes_df = pd.DataFrame.from_records(nodes)
    edges_df = pd.DataFrame.from_records(edges)

    # get valid codes map code -> name
    valid_codes = {x['id:ID'] : x['description'] for x in nodes}

    mrconso_path, mrdef_path = _ensure_umls_files()
    mrconso_path = Path(mrconso_path)
    mrdef_path = Path(mrdef_path)


    mrconso = pd.read_csv(mrconso_path, sep="|", header=None,
                          # CUI, SAB (source), CODE
                          usecols=[0, 11, 13],
                          names=["CUI", "SAB", "CODE"])

    icd10 = mrconso[mrconso.SAB == "ICD10"][["CUI", "CODE"]]
    icd10cm = mrconso[mrconso.SAB == "ICD10CM"][["CUI", "CODE"]]

    xwalk = icd10.merge(icd10cm, on="CUI", suffixes=("_icd10", "_icd10cm"))
    # xwalk now has WHO code, CM code, and shared CUI for each pair
    xwalk = xwalk[xwalk['CODE_icd10cm'].isin(valid_codes)]
    grouped_xwalk = xwalk.drop_duplicates(subset=['CODE_icd10', 'CODE_icd10cm'])
    grouped_xwalk[':TYPE'] = "maps_to"
    grouped_xwalk = grouped_xwalk.rename(columns={'CODE_icd10' : ':END_ID' ,
                                                  'CODE_icd10cm': ':START_ID'}).drop(columns=["CUI"])
    grouped_xwalk[':START_ID'] = "icd10cm:" + grouped_xwalk[":START_ID"]
    grouped_xwalk[':END_ID'] = "icd10:" + grouped_xwalk[":END_ID"]
    # icd_nodes = pd.read_csv("kg/icd10_nodes.tsv", sep='\t').drop_duplicates(subset=['id:ID'])['id:ID'].to_list()
    # grouped_xwalk = grouped_xwalk[grouped_xwalk[':END_ID'].isin(icd_nodes)]

    edges_df[':START_ID'] = "icd10cm:" + edges_df[":START_ID"]
    edges_df[':END_ID'] = "icd10cm:" + edges_df[":END_ID"]
    nodes_df['id:ID'] = 'icd10cm:' + nodes_df['id:ID']
    edges_df = pd.concat([edges_df, grouped_xwalk])


if __name__ == "__main__":
    nodes, edges = get_icd10cm_nodes_edges()

    nodes_df.to_csv("kg/icd10_cm_nodes.tsv", sep="\t", index=False)
    edges_df.to_csv("kg/icd10_cm_edges.tsv", sep="\t", index=False)
