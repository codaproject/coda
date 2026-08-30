"""Menu-style classification prompt for the MedGemma CHAMPS classifier."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from jinja2 import Environment, FileSystemLoader, StrictUndefined

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
