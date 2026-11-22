"""Two-regime hypergraph lookahead and top-K bundle SVM hinge-margin policy.

Implements adaptive WHO VA question selection and interview stopping:
  * Regime 1 (P(c*) <= explore_threshold): mutual information over a blended
    posterior / prior-deweighted likelihood belief with 2-step parent -> child
    hypergraph lookahead extracted from the XForm XML.
  * Regime 2 (P(c*) > explore_threshold): top-K bundle 2-sided Crammer-Singer
    hinge verification over active rival causes, stopping only when
    P(c*) > stop_threshold and no open rival bundle violates margin M.
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

_PROB_EPS = 1e-12
_GATE_PROB_MIN = 0.01
_DEFAULT_MARGIN = float(np.log(0.97 / 0.03))
_SELECTED_RE = re.compile(
    r"(?<!not\()selected\(\s*([/\w.-]+)\s*,\s*['\"]([^'\"]+)['\"]\s*\)"
)
_EQUALS_RE = re.compile(r"([/\w.-]+)\s*(?<![!<>=])=\s*['\"]([^'\"]+)['\"]")


@dataclass(frozen=True, slots=True)
class HypergraphHingeConfig:
    """Hyperparameters of the two-regime hypergraph hinge-margin policy."""

    explore_threshold: float = 0.7
    stop_threshold: float = 0.95
    margin: float = _DEFAULT_MARGIN
    bundle_size: int = 4
    rival_threshold: float = 0.005
    lookahead_weight: float = 0.5
    explore_blend: float = 0.5
    min_information: float = 1e-7
    hinge_tolerance: float = 1e-3
    min_hinge_score: float = 1e-5


def mutual_information(
    probabilities: np.ndarray, distribution: np.ndarray
) -> np.ndarray:
    """Return the mutual information of the answer and cause for each case."""
    answer_entropy = entropy(probabilities @ distribution, axis=1)
    conditional_entropy = probabilities @ entropy(distribution, axis=1)
    return answer_entropy - conditional_entropy


def _normalize_priors(priors: Sequence[float], num_causes: int) -> np.ndarray:
    arr = np.asarray(priors, dtype=np.float64)
    if arr.shape != (num_causes,) or (arr < 0.0).any() or arr.sum() <= 0.0:
        msg = ("Priors must be non-negative with one entry per cause and "
               "positive sum.")
        raise ValueError(msg)
    return arr / arr.sum()


def _top_k_column_sum(matrix: np.ndarray, k: int) -> np.ndarray:
    n_rows = matrix.shape[0]
    if n_rows == 0 or k <= 0:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    if k >= n_rows:
        return matrix.sum(axis=0)
    partitioned = np.partition(matrix, n_rows - k, axis=0)
    return partitioned[-k:].sum(axis=0)


def _verified_stop_scores(
    ig_row: np.ndarray, min_information: float
) -> np.ndarray:
    """Shift a verified case's scores below min_information, keeping order."""
    max_score = float(ig_row.max()) if ig_row.size > 0 else 0.0
    if max_score < min_information:
        return ig_row
    return ig_row - (max_score + 1.0)


@lru_cache(maxsize=8)
def extract_parent_children(xform_xml: str) -> dict[str, list[str]]:
    """Extract the positive parent -> children question gates of an XForm."""
    if not xform_xml:
        return {}
    try:
        root = ET.fromstring(xform_xml)
    except ET.ParseError:
        return {}

    binds: list[tuple[str, set[str]]] = []
    field_nodesets: dict[str, str] = {}
    for elem in root.iter():
        if not elem.tag.endswith("bind"):
            continue
        nodeset = elem.attrib.get("nodeset")
        if not nodeset:
            continue
        name = nodeset.rsplit("/", 1)[-1]
        field_nodesets[name] = nodeset
        relevant = elem.attrib.get("relevant")
        if relevant:
            parents = _positive_parents(relevant)
            if parents:
                binds.append((nodeset, parents))

    children_of: dict[str, set[str]] = {}
    for field_name, nodeset in field_nodesets.items():
        for bind_nodeset, parents in binds:
            if bind_nodeset == nodeset \
                    or nodeset.startswith(f"{bind_nodeset}/"):
                for parent in parents:
                    if parent != field_name:
                        children_of.setdefault(parent, set()).add(field_name)

    return {
        parent: sorted(children) for parent, children in children_of.items()
    }


def _positive_parents(relevant: str) -> set[str]:
    parents: set[str] = set()
    for match in _SELECTED_RE.finditer(relevant):
        parents.add(match.group(1).rsplit("/", 1)[-1])
    for match in _EQUALS_RE.finditer(relevant):
        parents.add(match.group(1).rsplit("/", 1)[-1])
    return parents


class HypergraphHingePolicy:
    """Two-regime hypergraph lookahead and top-K bundle hinge-margin policy.

    Duck-types codalg's ``AnswerDistributions`` / ``InterviewPolicy``
    interface so it works directly with ``codalg@v0.3.0`` as well as unit
    tests without ``codalg``.
    """

    def __init__(
        self,
        causes: Sequence[str],
        distributions: Mapping[str, np.ndarray],
        *,
        questionnaire_id: str | None = None,
        priors: Sequence[float] | None = None,
        config: HypergraphHingeConfig | None = None,
    ) -> None:
        self.causes = list(causes)
        self.questions = list(distributions)
        self.answer_distributions = {
            q: np.asarray(dist, dtype=np.float64)
            for q, dist in distributions.items()
        }
        self.questionnaire_id = questionnaire_id
        self.priors: np.ndarray | None = (
            _normalize_priors(priors, len(self.causes))
            if priors is not None else None
        )
        self.config = config or HypergraphHingeConfig()
        self._log_distributions = {
            question: np.log(np.maximum(distribution, _PROB_EPS))
            for question, distribution in self.answer_distributions.items()
        }
        self._boost_cache: dict[
            str, tuple[dict[str, list[str]], np.ndarray]
        ] = {}

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | Path,
        *,
        config: HypergraphHingeConfig | None = None,
    ) -> "HypergraphHingePolicy":
        """Return the policy with codalg's pretrained answer distributions."""
        from codalg.interview import AnswerDistributions

        base = AnswerDistributions.from_pretrained(name_or_path)
        priors = getattr(base, "priors", None)
        return cls(
            base.causes,
            base.answer_distributions,
            questionnaire_id=base.questionnaire_id,
            priors=priors,
            config=config,
        )

    def cause_probabilities(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Return the probabilities of the policy's causes, summing to 1."""
        probabilities = probabilities.reindex(
            columns=self.causes, fill_value=0.0
        )
        total = probabilities.sum(axis=1)
        if (total == 0).any():
            cases = total.index[total == 0].tolist()
            msg = f"Cases {cases} have no probability on the policy's causes."
            raise ValueError(msg)
        return probabilities.div(total, axis=0)

    def __call__(self, probabilities: pd.DataFrame, interview) -> pd.DataFrame:
        """Return the score of each question in each case."""
        p_post = self.cause_probabilities(probabilities).to_numpy(
            dtype=np.float64
        )
        p_lik = self._prior_flat_likelihood(p_post)
        blend = self.config.explore_blend
        b_explore = (1.0 - blend) * p_post + blend * p_lik

        eligible = ~interview.asked_questions.reindex(
            columns=self.questions, fill_value=True
        ).to_numpy(dtype=bool)
        children_of, child_boost = self._hypergraph(interview)
        ig_scores = self._exploration_scores(
            b_explore, eligible, children_of, interview
        )

        scores = ig_scores.copy()
        for row_idx in range(len(p_post)):
            row_score = self._regime2_scores(
                p_post[row_idx],
                p_lik[row_idx],
                eligible[row_idx],
                child_boost,
                ig_scores[row_idx],
            )
            if row_score is not None:
                scores[row_idx] = row_score

        result = pd.DataFrame(
            scores, index=probabilities.index, columns=self.questions
        )
        return result[interview.asked_questions.columns]

    def stopped_cases(
        self, probabilities: pd.DataFrame, scores: pd.DataFrame
    ) -> pd.Series:
        """Return True for each case that is verified or has no information."""
        if scores.empty or scores.shape[1] == 0:
            return pd.Series(True, index=probabilities.index)
        return scores.max(axis=1) < self.config.min_information

    def _prior_flat_likelihood(self, p_post: np.ndarray) -> np.ndarray:
        if self.priors is None:
            return p_post
        raw = p_post / np.maximum(self.priors, _PROB_EPS)
        total = raw.sum(axis=1, keepdims=True)
        return np.divide(raw, total, out=p_post.copy(), where=total > 0.0)

    def _hypergraph(
        self, interview
    ) -> tuple[dict[str, list[str]], np.ndarray]:
        form_id = interview.questionnaire.id
        cached = self._boost_cache.get(form_id)
        if cached is not None:
            return cached

        raw_graph = extract_parent_children(interview.questionnaire.xform)
        question_set = set(self.questions)
        children_of = {
            parent: [child for child in children if child in question_set]
            for parent, children in raw_graph.items()
            if parent in question_set
        }
        children_of = {p: ch for p, ch in children_of.items() if ch}

        q_index = {q: idx for idx, q in enumerate(self.questions)}
        boost = np.zeros(
            (len(self.questions), len(self.causes)), dtype=np.float64
        )
        for parent, children in children_of.items():
            p_idx = q_index[parent]
            for child in children:
                dist = self.answer_distributions[child]
                med = np.median(dist, axis=0)
                lr = np.log(np.maximum(dist, _PROB_EPS)) - np.log(
                    np.maximum(med, _PROB_EPS)
                )
                boost[p_idx] = np.maximum(
                    boost[p_idx], np.maximum(lr, 0.0).max(axis=1)
                )

        self._boost_cache[form_id] = (children_of, boost)
        return children_of, boost

    def _exploration_scores(
        self,
        belief: np.ndarray,
        eligible: np.ndarray,
        children_of: Mapping[str, list[str]],
        interview,
    ) -> np.ndarray:
        q_index = {q: idx for idx, q in enumerate(self.questions)}
        ig = np.column_stack(
            [
                mutual_information(belief, self.answer_distributions[q])
                for q in self.questions
            ]
        )
        if not children_of or self.config.lookahead_weight <= 0.0:
            return ig

        fields = interview.questionnaire.fields
        for parent, children in children_of.items():
            p_idx = q_index[parent]
            child_indices = [q_index[child] for child in children]
            max_child_ig = ig[:, child_indices].max(axis=1)
            p_gate = self._gate_probability(belief, parent, fields)
            active = eligible[:, p_idx] & (p_gate > _GATE_PROB_MIN)
            ig[active, p_idx] += (
                self.config.lookahead_weight
                * p_gate[active]
                * max_child_ig[active]
            )
        return ig

    def _gate_probability(
        self, belief: np.ndarray, question: str, fields: pd.DataFrame
    ) -> np.ndarray:
        dist = self.answer_distributions[question]
        marginal = belief @ dist
        if question in fields.index:
            choices = fields.loc[question, "choices"]
            if isinstance(choices, list) and "yes" in choices:
                yes_idx = choices.index("yes")
                if yes_idx < marginal.shape[1]:
                    return marginal[:, yes_idx]
        return 1.0 - marginal[:, -1]

    def _regime2_scores(
        self,
        p_post: np.ndarray,
        p_lik: np.ndarray,
        eligible: np.ndarray,
        child_boost: np.ndarray,
        ig_row: np.ndarray,
    ) -> np.ndarray | None:
        cfg = self.config
        c_star = int(np.argmax(p_post))
        if p_post[c_star] <= cfg.explore_threshold:
            return None

        sv_weight = np.maximum(p_post, p_lik)
        sv_weight[c_star] = 0.0
        sv_idx = np.flatnonzero(sv_weight > cfg.rival_threshold)
        if sv_idx.size == 0:
            return (
                _verified_stop_scores(ig_row, cfg.min_information)
                if p_post[c_star] > cfg.stop_threshold
                else None
            )

        log_post = np.log(np.maximum(p_post, _PROB_EPS))
        lead = log_post[c_star] - log_post[sv_idx]
        exp_pos_llr = self._expected_positive_llr(
            c_star, sv_idx, eligible, child_boost
        )
        bundle_sum = _top_k_column_sum(exp_pos_llr, cfg.bundle_size)
        margin_gap = cfg.margin - lead + bundle_sum
        bundle_hinge = np.maximum(0.0, margin_gap) * sv_weight[sv_idx]

        if bundle_hinge.max() < cfg.hinge_tolerance:
            return (
                _verified_stop_scores(ig_row, cfg.min_information)
                if p_post[c_star] > cfg.stop_threshold
                else None
            )

        active_w = np.where(margin_gap > 0.0, sv_weight[sv_idx], 0.0)
        q_score = exp_pos_llr @ active_w
        if q_score.max() > cfg.min_hinge_score:
            return q_score
        return None

    def _expected_positive_llr(
        self,
        c_star: int,
        sv_idx: np.ndarray,
        eligible: np.ndarray,
        child_boost: np.ndarray,
    ) -> np.ndarray:
        exp_pos_llr = np.zeros(
            (len(self.questions), sv_idx.size), dtype=np.float64
        )
        for q_idx, question in enumerate(self.questions):
            if not eligible[q_idx]:
                continue
            dist = self.answer_distributions[question]
            log_dist = self._log_distributions[question]
            lr = (
                log_dist[sv_idx, :]
                - log_dist[c_star, :][np.newaxis, :]
                + child_boost[q_idx, sv_idx][:, np.newaxis]
            )
            exp_pos_llr[q_idx] = (
                np.maximum(lr, 0.0) * dist[sv_idx, :]
            ).sum(axis=1)
        return exp_pos_llr
