import numpy as np
import pytest

from coda.inference.champs_prompted.who_questions import age_features, \
    age_fields, asks_topic, changes_top_cause, dont_know, \
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


def test_age_features():
    features = age_features(400)
    assert features["ageInYears"] == 1
    assert features["ageInMonthsByYear"] == 13
    assert features["isChild"] == 1
    assert features["isNeonatal"] == 0
    assert features["age_group_subcat"] is None
    assert age_features(2)["age_group_subcat"] == "neonate 72h"
    assert age_features(None) == {}


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


def test_dont_know():
    assert dont_know(YES_NO) == "dk"
    assert dont_know(CARE) == "DK"
    assert dont_know(DAYS) is None
    assert parse_answer(DAYS, "five") is None


def test_evidence_in_text():
    text = "She had a high fever, for five days."
    assert evidence_in_text("high fever for five", text)
    assert not evidence_in_text("high temperature", text)
    assert not evidence_in_text("", text)


def test_asks_topic():
    text = "Interviewer: How was her breathing? Respondent: Fast. Skin fine."
    assert asks_topic(text, ("breath",))
    assert not asks_topic(text, ("skin",))


def test_is_negated():
    assert is_negated("She had no diarrhea")
    assert is_negated("he didn't cough")
    assert not is_negated("He died at home")


class FakeClient:
    def call_with_schema(self, schema, **kwargs):
        return {q: {"evidence": "a high fever", "answer": "yes"}
                for q in schema["properties"]}


def test_changes_top_cause():
    fever = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert changes_top_cause(np.array([0.6, 0.4]), fever, ["A", "B"])
    assert not changes_top_cause(np.array([0.95, 0.05]), fever, ["A", "B"])
    assert not changes_top_cause(np.array([0.6, 0.4]), fever, ["A", "A"])
    rare = np.array([[0.99, 0.01], [0.5, 0.5]])
    assert not changes_top_cause(np.array([0.9, 0.1]), rare, ["A", "B"])


CAUSES = {"icd11:1C62.Z": {"score": 0.6}, "icd11:CA40.Z": {"score": 0.4}}


def test_interview():
    pytest.importorskip("codalg")
    from coda.inference.champs_prompted.who_questions import WhoQuestions
    questions = WhoQuestions("voi", FakeClient(), skip_logic=False)
    interview = questions.start()
    profile = Profile.from_dict({"age": {"value": 14, "unit": "months"}})
    labels, stop = interview.update("She had a high fever.", profile,
                                    CAUSES, 3)
    features = interview.interview.features.loc["case"]
    answered = features.index[features == "yes"].drop("Id10013")
    assert len(answered) > 3
    assert len(labels) == 3
    assert not set(labels) & set(questions.fields.loc[answered, "label"])
    assert interview.update("", profile, {"icd10:A00": {"score": 1}},
                            3) == ([], False)
    questions = WhoQuestions("voi", FakeClient(), skip_logic=False,
                             probabilities="lightgbm")
    interview = questions.start()
    interview.update("She had a high fever.", profile, {}, 3)
    assert interview.probabilities().loc["case"].sum() == pytest.approx(1)


class NoAnswerClient:
    def call_with_schema(self, schema, **kwargs):
        return {q: {"evidence": "I don't know", "answer": "no_answer"}
                for q in schema["properties"]}


def test_no_answer():
    pytest.importorskip("codalg")
    from coda.inference.champs_prompted.who_questions import WhoQuestions
    questions = WhoQuestions("voi", NoAnswerClient(), skip_logic=False)
    cough = "Did he have a cough? I don't know."
    assert questions.asks_about(cough, "Id10153")
    assert not questions.asks_about(cough, "Id10397")
    assert not questions.asks_about("She had a cough.", "Id10153")
    interview = questions.start()
    interview.update("I don't know.", None, CAUSES, 3)
    asked = interview.interview.asked_questions.loc["case"]
    assert set(asked.index[asked]) <= set(questions.calculated)


def test_review_of_systems():
    pytest.importorskip("codalg")
    from coda.inference.champs_prompted.who_questions import \
        REVIEW_OF_SYSTEMS, WhoQuestions
    profile = Profile.from_dict({"age": {"value": 14, "unit": "months"}})
    questions = WhoQuestions("baq", NoAnswerClient(), skip_logic=False,
                             review_of_systems=True,
                             region={"hiv": "high", "malaria": "very_low"})
    assert questions.region == {"Id10002": "high", "Id10003": "veryl"}
    labels, stop = questions.start().update("I don't know.", profile,
                                            CAUSES, 3)
    assert labels == [q for q, _, _ in REVIEW_OF_SYSTEMS[:3]]
    assert not stop
    questions = WhoQuestions("baq", FakeClient(), skip_logic=False,
                             review_of_systems=True)
    labels, _ = questions.start().update("She had a high fever.", profile,
                                         CAUSES, 3)
    assert not set(labels) & {q for q, _, _ in REVIEW_OF_SYSTEMS}
