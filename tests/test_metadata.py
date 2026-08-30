import pytest

from coda.metadata import Age, Metadata


def test_age_conversion():
    a = Age(value=2, unit="years")
    assert a.years == pytest.approx(2.0)
    assert a.months == pytest.approx(24.0, abs=0.1)
    assert a.days == pytest.approx(730.5, abs=0.5)


def test_age_hours():
    a = Age(value=2.5, unit="days")
    assert a.hours == pytest.approx(60.0)
    assert a.days == pytest.approx(2.5)


def test_age_unknown_unit():
    assert Age(value=5, unit="fortnights").days is None
    assert Age().days is None


def test_empty():
    m = Metadata.from_dict(None)
    assert m.is_empty()
    assert m.to_dict() == {"profile": None}


def test_age_roundtrip():
    m = Metadata.from_dict(
        {"profile": {"age": {"value": 6, "unit": "months"}, "stillbirth": False}}
    )
    assert not m.is_empty()
    assert m.profile.age.value == 6
    assert m.profile.age.unit == "months"
    assert m.to_dict()["profile"]["age"] == {"value": 6, "unit": "months"}


def test_stillbirth_only():
    m = Metadata.from_dict({"profile": {"age": None, "stillbirth": True}})
    assert not m.is_empty()
    assert m.profile.stillbirth
    assert m.profile.age is None


def test_all_blank_is_empty():
    m = Metadata.from_dict({"profile": {"age": None, "stillbirth": False}})
    assert m.is_empty()
