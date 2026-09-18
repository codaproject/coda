"""Choosing which question to ask next by expected information gain.

The worth of a yes/no question is the mutual information between its answer
and the cause of death, under the cause probabilities believed so far. That is
highest for a question whose answer is unpredictable overall but predictable
once the cause is known, which is what it means for an answer to discriminate
between the causes still in play.
"""

__all__ = ["binary_entropy", "information_gain", "select_questions"]

from math import log2


def binary_entropy(p):
    """Entropy in bits of a yes/no outcome with probability p of yes."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def information_gain(posterior, likelihoods):
    """Bits the answer to one question is expected to reveal about the cause.

    `posterior` holds the current probability of each cause and `likelihoods`
    the probability of a yes answer given each cause. Causes the question does
    not speak to are taken as a certain no.
    """
    answer_entropy = binary_entropy(
        sum(p * likelihoods.get(cause, 0.0) for cause, p in posterior.items())
    )
    conditional_entropy = sum(
        p * binary_entropy(likelihoods.get(cause, 0.0))
        for cause, p in posterior.items()
    )
    return answer_entropy - conditional_entropy


def select_questions(posterior, questions, num_questions, exclude=()):
    """Return the questions worth most under `posterior`, best first.

    Each question is scored on its own, so the selection is greedy and a batch
    of them may well cover the same ground twice.
    """
    ranked = sorted(
        (q for q in questions if q.id not in exclude),
        key=lambda q: information_gain(posterior, q.likelihoods),
        reverse=True,
    )
    return ranked[:num_questions]
