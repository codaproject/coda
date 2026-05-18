"""WHO Mortality Database exporter for the CODA knowledge graph.

Ingests:
  - Morticd10_part6.csv (ICD-10 mortality data, 2021 onwards)
  - pop.csv (mid-year population and live births)
  - country_codes.csv (WHO country code → name mapping)

Produces:
  - who_mortality_nodes.tsv  (country nodes with population properties)
  - who_mortality_edges.tsv  (country → icd10 cause edges with death counts)

Download the source files manually from:
  https://www.who.int/data/data-collection-tools/who-mortality-database

Place the extracted CSVs in the src/coda/resources/ directory.
"""

import logging
from pathlib import Path

import pandas as pd

# from coda import CODA_BASE
from coda.kg.sources import KGSourceExporter, KG_BASE

logger = logging.getLogger(__name__)
HERE = Path(__file__).parent
WHO_MORTALITY_BASE = HERE.parent.parent/"resources"

# Filenames after extracting the ZIPs
MORT_PART6_FILENAME = "Morticd10_part6.csv"
COUNTRY_CODES_FILENAME = "country_codes.csv"
POP_FILENAME = "pop.csv"

# ICD-10 List codes to include (skip 101 = tabulation, UE1 = Portugal special)
VALID_LISTS = {"103", "104", "10M"}

# Sex code mapping
SEX_MAP = {1: "male", 2: "female", 9: "unspecified"}

# Standard age group labels for Frmat 00 (Deaths2–Deaths26)
AGE_GROUPS = [
    "0", "1", "2", "3", "4",
    "5-9", "10-14", "15-19", "20-24", "25-29",
    "30-34", "35-39", "40-44", "45-49", "50-54",
    "55-59", "60-64", "65-69", "70-74", "75-79",
    "80-84", "85-89", "90-94", "95+", "unknown",
]

DEATH_COLS = [f"Deaths{i}" for i in range(2, 27)]
POP_COLS = [f"Pop{i}" for i in range(2, 27)]


def _normalise_icd10_cause(cause: str) -> str:
    
    """Normalise a WHO ICD-10 cause code to match KG node format.
    - Insert dot for 4-char codes: A000 -> a00.0
    - 3-char codes keep as-is
    """
    cause = cause.strip()
    if len(cause) == 4 and cause[0].isalpha() and cause[1:].isalnum():
        return f"{cause[:3]}.{cause[3]}"
    return cause


def _pick_most_granular(df: pd.DataFrame) -> pd.DataFrame:
    
    """For each (Country, Year), keep only the most granular List.

    Priority: 104 > 10M > 103. If a country-year has List 104 data,
    its List 103 rows are dropped to avoid double-counting.
    """

    list_priority = {"104": 0, "10M": 1, "103": 2}
    df = df.copy()
    df["_list_priority"] = df["List"].map(list_priority)
    best = df.groupby(["Country", "Year"])["_list_priority"].min().reset_index()
    best = best.rename(columns={"_list_priority": "_best"})
    df = df.merge(best, on=["Country", "Year"])
    df = df[df["_list_priority"] == df["_best"]].drop(columns=["_list_priority", "_best"])
    return df


def _load_country_codes(path: Path) -> dict[int, str]:
    
    """Load WHO country code -> name mapping."""

    df = pd.read_csv(path, encoding="latin1")
    return dict(zip(df["country"].astype(int), df["name"].str.strip()))


def _build_population_lookup(pop_path: Path) -> dict[int, dict]:
    
    """Build a lookup: country_code -> {year_sex -> {pop_total, age_pops, live_births}}.

    Returns a nested dict for attaching population to country nodes.
    """

    df = pd.read_csv(pop_path, encoding="latin1")
    # Filter to national-level data
    df = df[df["Admin1"].isna()]
    # Filter for data starting 2021
    df = df[df["Year"] >= 2021]

    pop_data: dict[int, list] = {}
    for _, row in df.iterrows():
        country = int(row["Country"])
        year = int(row["Year"])
        sex = SEX_MAP.get(int(row["Sex"]), str(int(row["Sex"])))
        pop_total = row.get("Pop1")
        live_births = row.get("Lb")

        age_pops = [row.get(col) for col in POP_COLS]
        age_pops = [int(v) if pd.notna(v) else None for v in age_pops]

        entry = {
            "year": year,
            "sex": sex,
            "population_total": int(pop_total) if pd.notna(pop_total) else None,
            "age_populations": age_pops,
            "live_births": int(live_births) if pd.notna(live_births) else None,
        }

        pop_data.setdefault(country, []).append(entry)

    return pop_data


class WhoMortalityExporter(KGSourceExporter):
    name = "who_mortality"

    def export(self):
        # Resolve file paths
        mort_path = WHO_MORTALITY_BASE / MORT_PART6_FILENAME
        country_codes_path = WHO_MORTALITY_BASE / COUNTRY_CODES_FILENAME
        pop_path = WHO_MORTALITY_BASE / POP_FILENAME

        for path, desc in [
            (mort_path, "Morticd10_part6"),
            (country_codes_path, "country_codes"),
            (pop_path, "pop"),
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{desc} not found at {path}. Download from:\n"
                    "https://www.who.int/data/data-collection-tools/who-mortality-database\n"
                    f"Extract and place the CSV in {WHO_MORTALITY_BASE}"
                )

        # --- Load reference data ---
        country_names = _load_country_codes(Path(country_codes_path))
        pop_data = _build_population_lookup(Path(pop_path))
        logger.info("Loaded %d country codes, population data for %d countries",
                     len(country_names), len(pop_data))

        # --- Load mortality data ---
        logger.info("Loading mortality data from %s ...", mort_path)
        df = pd.read_csv(Path(mort_path), encoding="latin1", low_memory=False)
        original_rows = len(df)

        # Treat List and Cause as strings
        df["List"] = df["List"].astype(str).str.strip()
        df["Cause"] = df["Cause"].astype(str).str.strip()

        # Filter to national-level data
        df = df[df["Admin1"].isna() & df["SubDiv"].isna()]
        logger.info("National-level rows: %d of %d", len(df), original_rows)

        # Filter to valid ICD-10 lists
        df = df[df["List"].isin(VALID_LISTS)]
        logger.info("After List filter (103/104/10M): %d rows", len(df))

        # Pick most granular list per country-year
        df = _pick_most_granular(df)
        logger.info("After granularity dedup: %d rows", len(df))

        # Drop the all-causes summary row (Cause = AAA)
        df = df[df["Cause"].str.upper() != "AAA"]

        # Normalise cause codes
        df["cause_normalised"] = df["Cause"].map(_normalise_icd10_cause)

        # --- Build nodes (countries) ---
        countries_in_data = set(df["Country"].unique())
        countries_in_pop = set(pop_data.keys())
        all_countries = countries_in_data | countries_in_pop

        nodes = []
        for country_code in sorted(all_countries):
            name = country_names.get(country_code, "")
            node_id = f"who_mortality:{country_code}"

            # Serialise population data as JSON string
            pop_entries = pop_data.get(country_code, [])
            pop_json = str(pop_entries) if pop_entries else ""

            nodes.append([node_id, "who_mortality", name, pop_json])

        nodes_df = pd.DataFrame(nodes, columns=["id:ID", ":LABEL", "name", "population_data"])
        nodes_df.to_csv(self.nodes_file, sep="\t", index=False)
        logger.info("Wrote %d country nodes to %s", len(nodes_df), self.nodes_file)

        # --- Build edges (country → icd10 cause) ---
        edges = []
        for _, row in df.iterrows():
            country_code = int(row["Country"])
            start_id = f"who_mortality:{country_code}"
            end_id = f"icd10:{row['cause_normalised']}"

            year = int(row["Year"])
            sex = SEX_MAP.get(int(row["Sex"]), str(int(row["Sex"])))
            deaths_total = int(row["Deaths1"]) if pd.notna(row["Deaths1"]) else 0

            # Age-specific deaths as semicolon-separated values
            age_deaths = []
            for col in DEATH_COLS:
                val = row.get(col)
                age_deaths.append(str(int(val)) if pd.notna(val) else "")
            age_deaths_str = ";".join(age_deaths)

            edges.append([
                start_id,
                end_id,
                "has_mortality",
                year,
                sex,
                deaths_total,
                ";".join(AGE_GROUPS),
                age_deaths_str,
            ])

        edges_df = pd.DataFrame(edges, columns=[
            ":START_ID", ":END_ID", ":TYPE",
            "year:int", "sex", "deaths_total:int",
            "age_groups:string[]", "age_deaths:string[]",
        ])
        edges_df = edges_df.sort_values([":START_ID", ":END_ID", "year:int"])
        edges_df.to_csv(self.edges_file, sep="\t", index=False)
        logger.info("Wrote %d mortality edges to %s", len(edges_df), self.edges_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    WhoMortalityExporter().export()