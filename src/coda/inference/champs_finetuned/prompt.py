"""Menu-style classification prompt for the MedGemma CHAMPS classifier."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from coda.inference.champs_prompted.icd11_agent import age_group

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    keep_trailing_newline=False,
    undefined=StrictUndefined,
)


def render_message(*, role: str, **kwargs: str) -> dict[str, str]:
    template = jinja_env.get_template(f"{role}.j2")
    return {"role": role, "content": template.render(**kwargs)}


@dataclass(frozen=True)
class MenuPrompt:
    ANSWER_PREFIX: ClassVar[str] = "Answer:"
    LEADING_NUMBER: ClassVar[re.Pattern[str]] = re.compile(r"^\W*(\d+)")
    ANSWER_NUMBER: ClassVar[re.Pattern[str]] = re.compile(
        r"answer\s*:?\s*\**\s*(\d+)", re.IGNORECASE
    )
    TRAILING_NUMBER: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![\d-])(\d+)[\s.*:)\]]*$"
    )

    messages: list[dict[str, str]]
    menu_labels: tuple[str, ...]

    @staticmethod
    def render(menu_labels: Sequence[str]) -> str:
        return "\n".join(
            f"{number}. {label}" for number, label in enumerate(menu_labels, start=1)
        )

    @classmethod
    def format_answer(cls, number: int) -> str:
        return f"{cls.ANSWER_PREFIX} {number}"

    def label_for(self, number: str) -> str | None:
        index = int(number)
        return (
            self.menu_labels[index - 1] if 1 <= index <= len(self.menu_labels) else None
        )

    def decode_answer(self, text: str) -> str | None:
        text = text.strip()
        if not text:
            return None

        if answers := self.ANSWER_NUMBER.findall(text):
            return self.label_for(answers[-1])

        for pattern in (self.LEADING_NUMBER, self.TRAILING_NUMBER):
            if (match := pattern.search(text)) and (label := self.label_for(match[1])):
                return label

        return None


# CHAMPS age group descriptions, by age group
AGE_GROUPS = {
    "stillbirth": "stillbirth",
    "less24hours": "death in the first 24 hours",
    "earlyneonate": "early neonate (1 to 6 days)",
    "lateneonate": "late neonate (7 to 27 days)",
    "infant": "infant (28 days to less than 12 months)",
    "child": "child (12 months to less than 60 months)",
}
SEXES = {"female": "female", "male": "male",
         "ambiguous": "indeterminate or ambiguous"}
HIV_ABSTRACTIONS = {"positive": "HIV infection: Yes",
                    "negative": "HIV infection: No",
                    "exposed": "HIV exposure: Yes"}


def age_components(age):
    """Return the CHAMPS age fields of an age, as parts that add up."""
    value = age.value
    if age.unit == "years":
        return [("Age in Years", int(value))]
    if age.unit == "months":
        years, months = divmod(int(value), 12)
        return [("Age in Years", years), ("Age in Months", months)]
    if age.unit == "days":
        return [("Age in Years", 0), ("Age in Months", 0),
                ("Age in days", float(value))]
    if age.unit == "hours":
        days, hours = divmod(float(value), 24)
        return [("Age in Years", 0), ("Age in Months", 0),
                ("Age in days", days), ("Age in hours", hours)]
    return []


def render_profile_evidence(profile):
    """Return the CHAMPS demographics lines of a decedent profile."""
    if profile is None:
        return ""
    fields = []
    if profile.sex:
        fields.append(("Sex", SEXES[profile.sex]))
    if profile.date_of_death:
        fields.append(("De-Identified Date of Death", profile.date_of_death))
    if profile.age and not profile.stillbirth:
        fields += age_components(profile.age)
    if group := age_group(profile):
        fields.append(("Age Grouping of the CHAMPS Case", AGE_GROUPS[group]))
    if profile.weight_kg_death:
        fields.append(("Weight in kilograms", profile.weight_kg_death))
    if profile.muac_cm_death:
        fields.append(("Mid-upper arm circumference in centimeters",
                       profile.muac_cm_death))
    if profile.birth_weight_g:
        fields.append(("Birth weight (grams)", profile.birth_weight_g))
    return "\n".join(f"- {label}: {value}" for label, value in fields)


def render_profile_abstraction(profile):
    """Return the CHAMPS clinical abstraction lines of a decedent profile."""
    if profile is None:
        return ""
    lines = []
    if line := HIV_ABSTRACTIONS.get(profile.hiv_status):
        lines.append(line)
    if profile.weight_kg_admission:
        lines.append(f"Weight (kilograms): {profile.weight_kg_admission:g}")
    if profile.muac_cm_admission:
        lines.append("Mid-upper arm circumference (cm): "
                     f"{profile.muac_cm_admission:g}")
    return "\n".join(lines)


def build_menu_prompt(
    menu_codes: Sequence[str],
    label_descriptions: dict[str, str],
    *,
    narrative: str,
    evidence: str = "",
    clinical: str = "",
) -> MenuPrompt:
    """Build a numbered multiple-choice prompt over the candidate cause codes."""
    codes = tuple(menu_codes)
    display_labels = [label_descriptions[code] for code in codes]
    labels = MenuPrompt.render(display_labels)

    messages = [
        render_message(role="system"),
        render_message(
            role="user",
            labels=labels,
            text=narrative,
            evidence=evidence,
            secondary_text=clinical,
        ),
    ]
    return MenuPrompt(messages=messages, menu_labels=codes)
