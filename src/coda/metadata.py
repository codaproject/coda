"""Structured per-interview metadata passed to inference agents.

Shape: metadata -> profile -> age. Profile holds decedent attributes;
further top-level fields (date, location) can be added over time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Age:
    value: Optional[float] = None
    unit: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["Age"]:
        if not d or d.get("value") is None:
            return None
        return cls(value=d.get("value"), unit=d.get("unit"))

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit}


@dataclass
class Profile:
    age: Optional[Age] = None
    stillbirth: bool = False

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["Profile"]:
        if not d:
            return None
        age = Age.from_dict(d.get("age"))
        stillbirth = bool(d.get("stillbirth", False))
        if age is None and not stillbirth:
            return None
        return cls(age=age, stillbirth=stillbirth)

    def to_dict(self) -> dict:
        return {
            "age": self.age.to_dict() if self.age else None,
            "stillbirth": self.stillbirth,
        }


@dataclass
class Metadata:
    profile: Optional[Profile] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Metadata":
        if not d:
            return cls()
        return cls(profile=Profile.from_dict(d.get("profile")))

    def to_dict(self) -> dict:
        return {"profile": self.profile.to_dict() if self.profile else None}

    def is_empty(self) -> bool:
        return self.profile is None
