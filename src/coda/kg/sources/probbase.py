"""
This module gets the list of VA questions from the "probbase" associated
with InterVA and construct nodes for the questions themselves, and
build edges between the questions and the causes they relate to
using the information in the probbase.
"""

__all__ = ["ProbBaseExporter"]

import pandas as pd

from coda.kg.sources import KGSourceExporter

PROBBASE_URL = (
    "https://github.com/verbal-autopsy-software/interva/raw/"
    "refs/heads/main/src/interva/data/probbase.xls"
)


def process_va_col(col_name):
    assert col_name.startswith('b_')
    code = col_name[2:]
    if code.endswith("00"):
        code = code[:-2]
    else:
        code = f"{code[:2]}.{code[2:]}"
    return f"who.va:VAs-{code}"


class ProbBaseExporter(KGSourceExporter):
    name = "probBase"

    def export(self):
        df = pd.read_excel(PROBBASE_URL, sheet_name="probbase")
        id_column = "who_2016"
        name_column = "qdesc"
        prop_columns = ["indic", "sdesc", "ilab", "subst", "samb"]
        df["who_curie"] = df[id_column].apply(lambda x: f"who.va.q:{x}")
        va_question_cols = {
            col: process_va_col(col) for col in df.columns if col.startswith("b_")
        }
        nodes = df[["who_curie", name_column, *prop_columns]]
        nodes[":LABEL"] = "who.va.q"
        nodes = nodes.rename(columns={"who_curie": "id:ID", name_column: "name"})
        nodes = nodes.dropna(subset=["indic"])

        nodes.sort_values("id:ID").drop_duplicates().to_csv(
            self.nodes_file, sep="\t", index=False
        )

        edges = []
        for _, row in df.iterrows():
            if pd.isna(row["indic"]):
                continue
            node_curie = row["who_curie"]
            for col, va_curie in va_question_cols.items():
                edge = [
                    node_curie,
                    va_curie,
                    "probbase_rel",
                    row[col],
                ]
                edges.append(edge)
        edge_df = pd.DataFrame(
            edges, columns=[":START_ID", ":END_ID", ":TYPE", "value"]
        )
        edge_df = pd.concat(
            [edge_df.drop(columns=[":TYPE"]), edge_df[":TYPE"].apply(pd.Series)],
            axis=1,
        )
        edge_df = edge_df.sort_values([":START_ID", ":END_ID"])
        edge_df.drop_duplicates().to_csv(self.edges_file, sep="\t", index=False)


if __name__ == "__main__":
    exporter = ProbBaseExporter()
    exporter.export()
