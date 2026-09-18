from coda.inference.probbase import Question, load_questions
from coda.inference.voi import (
    binary_entropy,
    information_gain,
    select_questions,
)


def make_question(qid, **likelihoods):
    return Question(qid, f"question {qid}?", likelihoods)


UNIFORM = {"VAs-01.01": 0.5, "VAs-01.05": 0.5}


def test_binary_entropy():
    assert binary_entropy(0.5) == 1.0
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0


def test_no_gain_from_a_question_that_does_not_separate_causes():
    flat = make_question("flat", **{"VAs-01.01": 0.5, "VAs-01.05": 0.5})
    assert information_gain(UNIFORM, flat.likelihoods) == 0.0


def test_full_gain_from_a_question_that_settles_the_cause():
    decisive = make_question("decisive", **{"VAs-01.01": 1.0, "VAs-01.05": 0.0})
    assert information_gain(UNIFORM, decisive.likelihoods) == 1.0


def test_gain_follows_how_far_apart_the_likelihoods_are():
    weak = make_question("weak", **{"VAs-01.01": 0.6, "VAs-01.05": 0.4})
    strong = make_question("strong", **{"VAs-01.01": 0.9, "VAs-01.05": 0.1})
    assert information_gain(UNIFORM, weak.likelihoods) < \
        information_gain(UNIFORM, strong.likelihoods)


def test_questions_are_ranked_by_gain():
    weak = make_question("weak", **{"VAs-01.01": 0.6, "VAs-01.05": 0.4})
    strong = make_question("strong", **{"VAs-01.01": 0.9, "VAs-01.05": 0.1})
    selected = select_questions(UNIFORM, [weak, strong], 2)
    assert [q.id for q in selected] == ["strong", "weak"]


def test_excluded_questions_are_not_selected():
    weak = make_question("weak", **{"VAs-01.01": 0.6, "VAs-01.05": 0.4})
    strong = make_question("strong", **{"VAs-01.01": 0.9, "VAs-01.05": 0.1})
    selected = select_questions(UNIFORM, [weak, strong], 2, exclude={"strong"})
    assert [q.id for q in selected] == ["weak"]


def test_question_bank_loads():
    questions = load_questions()
    assert len(questions) > 300
    fever = [q for q in questions if q.id == "W610147-o"][0]
    assert fever.text.endswith("did (s)he have a fever?")
    assert 0.0 < fever.likelihoods["VAs-01.05"] <= 1.0
