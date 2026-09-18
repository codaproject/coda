import pytest
from unittest.mock import MagicMock

from coda.inference.champs_prompted_agent import ChampsPromptedInferenceAgent
from coda.inference.probbase import Question, load_questions
from coda.inference.voi import (
    binary_entropy,
    information_gain,
    select_questions,
)

UNIFORM = {"VAs-01.01": 0.5, "VAs-01.05": 0.5}
FLAT = Question("flat", "flat?", {"VAs-01.01": 0.5, "VAs-01.05": 0.5})
WEAK = Question("weak", "weak?", {"VAs-01.01": 0.6, "VAs-01.05": 0.4})
STRONG = Question("strong", "strong?", {"VAs-01.01": 0.9, "VAs-01.05": 0.1})
DECISIVE = Question("decisive", "decisive?", {"VAs-01.01": 1.0,
                                              "VAs-01.05": 0.0})


def test_binary_entropy():
    assert binary_entropy(0.5) == 1.0
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0


def test_no_gain_from_a_question_that_does_not_separate_causes():
    assert information_gain(UNIFORM, FLAT.likelihoods) == 0.0


def test_full_gain_from_a_question_that_settles_the_cause():
    assert information_gain(UNIFORM, DECISIVE.likelihoods) == 1.0


def test_gain_follows_how_far_apart_the_likelihoods_are():
    assert information_gain(UNIFORM, WEAK.likelihoods) < \
        information_gain(UNIFORM, STRONG.likelihoods)


def test_questions_are_ranked_by_gain():
    selected = select_questions(UNIFORM, [WEAK, STRONG], 2)
    assert [q.id for q in selected] == ["strong", "weak"]


def test_excluded_questions_are_not_selected():
    selected = select_questions(UNIFORM, [WEAK, STRONG], 2, exclude={"strong"})
    assert [q.id for q in selected] == ["weak"]


def test_question_bank_loads():
    questions = load_questions()
    assert len(questions) > 300
    fever = [q for q in questions if q.id == "W610147-o"][0]
    assert fever.text.endswith("did (s)he have a fever?")
    assert 0.0 < fever.likelihoods["VAs-01.05"] <= 1.0


RESPONSE = {
    "reasoning": "",
    "top_causes": [
        {"cause_name": "Malaria", "probability": 0.6},
        {"cause_name": "Lower respiratory infections", "probability": 0.4},
    ],
    "questions": ["a question the llm wrote?"],
}


def make_agent(**kwargs):
    return ChampsPromptedInferenceAgent(llm_client=MagicMock(), **kwargs)


def test_llm_mode_keeps_the_questions_the_model_wrote():
    result = make_agent()._parse_response(RESPONSE)
    assert result["questions"] == ["a question the llm wrote?"]


def test_voi_mode_replaces_them_with_who_va_questions():
    agent = make_agent(question_mode="voi")
    questions = agent._parse_response(RESPONSE)["questions"]
    assert len(questions) == 3
    assert "a question the llm wrote?" not in questions
    assert all(q in {question.text for question in agent.question_bank}
               for q in questions)


def test_a_question_is_not_proposed_twice():
    agent = make_agent(question_mode="voi")
    first = agent._parse_response(RESPONSE)["questions"]
    second = agent._parse_response(RESPONSE)["questions"]
    assert not set(first) & set(second)


def test_asked_questions_clear_on_reset():
    agent = make_agent(question_mode="voi")
    first = agent._parse_response(RESPONSE)["questions"]
    agent.reset()
    assert agent._parse_response(RESPONSE)["questions"] == first


def test_causes_with_no_who_va_counterpart_drop_out():
    posterior = make_agent(question_mode="voi")._who_va_posterior(
        {"a": {"name": "Malaria", "score": 0.5},
         "b": {"name": "Undetermined", "score": 0.5}})
    assert posterior == {"VAs-01.05": 1.0}


def test_causes_sharing_a_counterpart_merge():
    posterior = make_agent(question_mode="voi")._who_va_posterior(
        {"a": {"name": "Perinatal asphyxia/hypoxia", "score": 0.3},
         "b": {"name": "Neonatal encephalopathy", "score": 0.3},
         "c": {"name": "Neonatal sepsis", "score": 0.4}})
    assert posterior == pytest.approx({"VAs-10.02": 0.6, "VAs-10.04": 0.4})


def test_no_questions_when_no_cause_is_covered():
    agent = make_agent(question_mode="voi")
    assert agent._select_questions({"a": {"name": "Other", "score": 1.0}}) == []


def test_mode_carries_into_session_agent():
    agent = make_agent(question_mode="voi").create_session_agent()
    assert agent.question_mode == "voi"
    assert agent.question_bank


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        make_agent(question_mode="telepathy")
