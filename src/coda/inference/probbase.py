"""The InterVA-5 probbase, read as a bank of yes/no verbal autopsy questions.

Each question carries the probability of a yes answer given each WHO VA
cause. The matrix ships as a resource TSV holding the published grades
verbatim, rebuilt with ``python -m coda.inference.probbase``.
"""

__all__ = ["Question", "load_questions"]

import csv
from collections import namedtuple

from coda.resources import get_resource_path

PROBBASE_URL = (
    "https://github.com/verbal-autopsy-software/interva/raw/"
    "refs/heads/main/src/interva/data/probbase.xls"
)

RESOURCE_FILE = "probbase_who_va.tsv"

CAUSE_PREFIX = "VAs-"

# Numeric reading of the probbase grades, following InterVA-5.
GRADE_PROBABILITIES = {
    "": 0.0,
    "N": 0.0,
    "E": 0.00001,
    "D-": 0.0001,
    "D": 0.0005,
    "D+": 0.001,
    "C-": 0.002,
    "C": 0.005,
    "C+": 0.01,
    "B-": 0.02,
    "B": 0.05,
    "B+": 0.1,
    "A-": 0.2,
    "A": 0.5,
    "A+": 0.8,
    "I": 1.0,
}

# A question, and P(yes | cause) keyed by WHO VA cause id
Question = namedtuple("Question", ["id", "text", "likelihoods"])


def load_questions():
    """Read the question bank from the resource TSV."""
    questions = []
    with open(get_resource_path(RESOURCE_FILE), newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            likelihoods = {
                cause: GRADE_PROBABILITIES[grade]
                for cause, grade in row.items()
                if cause.startswith(CAUSE_PREFIX)
            }
            questions.append(
                Question(row["question_id"], row["question"], likelihoods)
            )
    return questions


def write_resource():
    """Rebuild the resource TSV from the published probbase spreadsheet.

    Needs the ``kg_build`` extra for the Excel reader. The first spreadsheet
    row names the cause each column stands for, the second holds InterVA's
    unconditional prior, and every row below is one question.
    """
    import pandas as pd

    df = pd.read_excel(PROBBASE_URL, sheet_name="probbase")
    causes = {
        col: CAUSE_PREFIX + str(df.iloc[0][col]).split()[0]
        for col in df.columns if col.startswith("b_")
    }
    table = df.iloc[2:].dropna(subset=["who_2016"])
    table = table[["who_2016", "qdesc", *causes]]
    table = table.rename(columns={"who_2016": "question_id",
                                  "qdesc": "question", **causes})
    table["question"] = table["question"].str.strip()
    table.to_csv(get_resource_path(RESOURCE_FILE), sep="\t", index=False)


if __name__ == "__main__":
    write_resource()
