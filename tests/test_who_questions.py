import pytest

from coda.inference.champs_prompted.who_questions import age_fields, \
    evidence_in_text, is_negated, parse_answer, profile_fields
from coda.metadata import Profile

YES_NO = {"type": "select one", "choices": ["yes", "no", "dk", "ref"]}
CARE = {"type": "select all that apply",
        "choices": ["pharmacy", "private_hospital", "DK", "Ref"]}
DAYS = {"type": "integer", "choices": None}


def test_age_fields():
    assert age_fields(3)["age_neonate_days"] == 3
    child = age_fields(400)
    assert child["age_group"] == "child"
    assert child["age_child_unit"] == "years"
    assert child["age_child_years"] == 1
    assert age_fields(None)["age_group"] is None


def test_profile_fields():
    profile = Profile.from_dict({"stillbirth": True, "sex": "ambiguous"})
    fields = profile_fields(profile)
    assert fields["age_group"] == "neonate"
    assert fields["Id10019"] == "undetermined"
    assert fields["Id10114"] == "yes"


def test_parse_answer():
    assert parse_answer(YES_NO, "Yes") == "yes"
    assert parse_answer(YES_NO, "ref") is None
    assert parse_answer(YES_NO, "yes no") is None
    assert parse_answer(YES_NO, "dk") is None
    assert parse_answer(CARE, "pharmacy, private_hospital") == \
        "pharmacy private_hospital"
    assert parse_answer(DAYS, "5") == "5"
    assert parse_answer(DAYS, "2.5") is None
    assert parse_answer(DAYS, "five") is None


def test_evidence_in_text():
    text = "She had a high fever, for five days."
    assert evidence_in_text("high fever for five", text)
    assert not evidence_in_text("high temperature", text)
    assert not evidence_in_text("", text)


def test_is_negated():
    assert is_negated("She had no diarrhea")
    assert is_negated("he didn't cough")
    assert not is_negated("He died at home")


class FakeClient:
    def call_with_schema(self, schema, **kwargs):
        return {q: {"evidence": "a high fever", "answer": "yes"}
                for q in schema["properties"]}


def test_interview():
    pytest.importorskip("codalg")
    from coda.inference.champs_prompted.who_questions import WhoQuestions
    questions = WhoQuestions("voi", FakeClient(), skip_logic=False)
    interview = questions.start()
    profile = Profile.from_dict({"age": {"value": 14, "unit": "months"}})
    causes = {"icd11:1C62.Z": {"score": 0.6}, "icd11:CA40.Z": {"score": 0.4}}
    labels, stop = interview.update("She had a high fever.", profile, causes,
                                    3)
    features = interview.interview.features.loc["case"]
    answered = features.index[features == "yes"].drop("Id10013")
    assert len(answered) > 3
    assert len(labels) == 3
    assert not set(labels) & set(questions.fields.loc[answered, "label"])
    assert interview.update("", profile, {"icd10:A00": {"score": 1}},
                            3) == ([], False)
