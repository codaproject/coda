"""Structured per-interview metadata.

Shape: metadata -> case_id, profile -> age. Profile holds decedent
attributes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Days per age unit
DAYS_PER_UNIT = {
    "hours": 1 / 24,
    "days": 1.0,
    "weeks": 7.0,
    "months": 365.25 / 12,
    "years": 365.25,
}


@dataclass
class Age:
    value: Optional[float] = None
    unit: Optional[str] = None

    def in_unit(self, unit: str) -> Optional[float]:
        if self.value is None or self.unit not in DAYS_PER_UNIT:
            return None
        return self.value * DAYS_PER_UNIT[self.unit] / DAYS_PER_UNIT[unit]

    @property
    def hours(self) -> Optional[float]:
        return self.in_unit("hours")

    @property
    def days(self) -> Optional[float]:
        return self.in_unit("days")

    @property
    def weeks(self) -> Optional[float]:
        return self.in_unit("weeks")

    @property
    def months(self) -> Optional[float]:
        return self.in_unit("months")

    @property
    def years(self) -> Optional[float]:
        return self.in_unit("years")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["Age"]:
        if not d or d.get("value") is None:
            return None
        return cls(value=d.get("value"), unit=d.get("unit"))

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit}


def _measurement(value) -> Optional[float]:
    """Return a positive measurement as a float, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


@dataclass
class Profile:
    age: Optional[Age] = None
    stillbirth: bool = False
    # ISO date, YYYY-MM-DD
    date_of_death: Optional[str] = None
    # One of SEXES
    sex: Optional[str] = None
    # One of MANNERS_OF_DEATH
    manner_of_death: Optional[str] = None
    birth_weight_g: Optional[float] = None
    # One of HIV_STATUSES
    hiv_status: Optional[str] = None
    weight_kg_admission: Optional[float] = None
    weight_kg_death: Optional[float] = None
    muac_cm_admission: Optional[float] = None
    muac_cm_death: Optional[float] = None

    SEXES = ("female", "male", "ambiguous")
    MANNERS_OF_DEATH = ("disease", "accident", "self_harm", "assault",
                        "pending_investigation", "unknown")
    HIV_STATUSES = ("negative", "positive", "exposed", "unknown")
    MEASUREMENTS = ("birth_weight_g", "weight_kg_admission", "weight_kg_death",
                    "muac_cm_admission", "muac_cm_death")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["Profile"]:
        if not d:
            return None
        sex = d.get("sex")
        manner = d.get("manner_of_death")
        hiv = d.get("hiv_status")
        profile = cls(
            age=Age.from_dict(d.get("age")),
            stillbirth=bool(d.get("stillbirth", False)),
            date_of_death=d.get("date_of_death") or None,
            sex=sex if sex in cls.SEXES else None,
            manner_of_death=manner if manner in cls.MANNERS_OF_DEATH else None,
            hiv_status=hiv if hiv in cls.HIV_STATUSES else None,
            **{name: _measurement(d.get(name)) for name in cls.MEASUREMENTS},
        )
        if profile == cls():
            return None
        return profile

    def to_dict(self) -> dict:
        return {
            "age": self.age.to_dict() if self.age else None,
            "stillbirth": self.stillbirth,
            "date_of_death": self.date_of_death,
            "sex": self.sex,
            "manner_of_death": self.manner_of_death,
            "hiv_status": self.hiv_status,
            **{name: getattr(self, name) for name in self.MEASUREMENTS},
        }


@dataclass
class Metadata:
    case_id: Optional[str] = None
    profile: Optional[Profile] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Metadata":
        if not d:
            return cls()
        return cls(case_id=d.get("case_id") or None,
                   profile=Profile.from_dict(d.get("profile")))

    def to_dict(self) -> dict:
        return {"case_id": self.case_id,
                "profile": self.profile.to_dict() if self.profile else None}

    def is_empty(self) -> bool:
        return self.case_id is None and self.profile is None
