import json
from pathlib import Path

import pandas as pd

from coda.kg.sources import KGSourceExporter

# NOTE: This exporter assumes that MeSH country nodes are already present in the KG (via a separate module).
# Country nodes are NOT created here, only referenced via CURIEs.

HERE = Path(__file__).parent
COUNTRY_MAPPING_FILE =  HERE.parent.parent/"resources"/"wdi_mesh_country_mapping.json"
with open(COUNTRY_MAPPING_FILE, "rb") as file:
    LOCATION_MESH_MAPPING = json.load(file)

class WDIExporter(KGSourceExporter):
    name = "wdi"

    def export(self):
        # Load data
        dev_df, health_df, mesh_df = self._load_data()

        # Combine datasets (deduplicate overlapping series)
        df = self._combine_data(dev_df, health_df)

        # Normalize country names
        df = self._normalize_countries(df)

        # Ground countries to MeSH
        df = self._ground_countries(df, mesh_df)

        # Build graph
        nodes_df, edges_df = self._build_graph(df, mesh_df)

        # Write output
        nodes_df.to_csv(self.nodes_file, sep="\t", index=False)
        edges_df.to_csv(self.edges_file, sep="\t", index=False)

    # Data loading

    def _load_data(self):
        dev_df = pd.read_csv(HERE / "world_dev_indicator_data.tsv.gz", sep="\t")
        health_df = pd.read_csv(HERE / "world_health_indicator_data.tsv.gz", sep="\t")
        kg_dir = self.nodes_file.parent
        mesh_df = pd.read_csv(kg_dir / "mesh_hierarchy_nodes.tsv", sep="\t")

        return dev_df, health_df, mesh_df

    # Combine datasets

    def _combine_data(self, dev_df, health_df):
        """
        Merge dev + health datasets
        Remove overlapping Series Codes
        """
        dev_codes = set(dev_df["Series Code"])
        health_df = health_df[
            ~health_df["Series Code"].isin(dev_codes)
        ]

        df = pd.concat([dev_df, health_df], ignore_index=True)
        return df

    # Normalize country names

    def _normalize_countries(self, df):
        df["Country Name"] = (
            df["Country Name"]
            .map(LOCATION_MESH_MAPPING)
            .fillna(df["Country Name"])
        )
        return df

    # Ground countries to MeSH

    def _ground_countries(self, df, mesh_df):
        """
        Keep only rows that can be grounded to MeSH geoloc nodes
        """

        df = pd.merge(
            df,
            mesh_df,
            left_on="Country Name",
            right_on="name",
            how="inner",
        )[df.columns]

        return df

    # Build graph

    def _build_graph(self, df, mesh_df):
        nodes = set()
        edges = []

        for _, row in df.iterrows():
            country_name = row["Country Name"]
            series_code = row["Series Code"]
            series_name = row["Series Name"]

            # Get country CURIE (from mesh, NOT CREATED HERE)
            country_info = mesh_df[mesh_df["name"] == country_name]
            if country_info.empty:
                continue

            country_curie = country_info.iloc[0]["id:ID"]

            # Indicator node
            indicator_curie = f"wdi:{series_code}"
            nodes.add((indicator_curie, "wdi", series_name))

            # Extract year data 
            year_data = self._extract_year_data(row)
            if not year_data:
                continue

            # Edge 
            edges.append(
                (
                    country_curie,
                    indicator_curie,
                    "has_indicator",
                    json.dumps(year_data),
                )
            )

        # Convert to DataFrames
        nodes_df = pd.DataFrame(
            list(nodes),
            columns=["id:ID", ":LABEL", "name"],
        ).sort_values("id:ID")

        edges_df = pd.DataFrame(
            edges,
            columns=[":START_ID", ":END_ID", ":TYPE", "years_data"],
        ).sort_values([":START_ID", ":END_ID"])

        return nodes_df, edges_df

    # Extract year data

    def _extract_year_data(self, row):
        year_data = {}

        for col, val in row.items():
            if not isinstance(col, str):
                continue

            # Match columns like "2019 [YR2019]"
            if len(col) >= 4 and col[:4].isdigit():
                try:
                    year = col[:4]
                    year_data[year] = round(float(val), 3)
                except (ValueError, TypeError):
                    continue

        return year_data


if __name__ == "__main__":
    exporter = WDIExporter()
    exporter.export()