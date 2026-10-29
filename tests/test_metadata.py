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
    assert m.to_dict() == {"case_id": None, "profile": None}


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


def test_death_details_roundtrip():
    d = {"date_of_death": "2026-03-14", "sex": "female",
         "manner_of_death": "accident"}
    m = Metadata.from_dict({"profile": d})
    assert not m.is_empty()
    profile = m.to_dict()["profile"]
    assert {k: profile[k] for k in d} == d


def test_invalid_choices_dropped():
    m = Metadata.from_dict(
        {"profile": {"sex": "other", "manner_of_death": "unclear"}})
    assert m.is_empty()


def test_clinical_fields_roundtrip():
    d = {"birth_weight_g": 2900.0, "hiv_status": "exposed",
         "weight_kg_admission": 6.9, "weight_kg_death": None,
         "muac_cm_admission": 11.9, "muac_cm_death": None}
    m = Metadata.from_dict({"profile": d})
    profile = m.to_dict()["profile"]
    assert {k: profile[k] for k in d} == d


def test_invalid_clinical_values_dropped():
    m = Metadata.from_dict({"profile": {"hiv_status": "maybe",
                                        "weight_kg_death": -1,
                                        "muac_cm_admission": "x"}})
    assert m.is_empty()
