"""Follow-up questions from the WHO VA 2016 form, ranked by information gain.

An LLM codes the form answers the transcript gives, and a codalg policy ranks
the questions still open given the agent's ICD-11 cause probabilities.
"""

import logging
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from coda.inference.champs_prompted.base_agent import read_prompt

logger = logging.getLogger(__name__)

QUESTIONNAIRE = "who_va_2016"
POLICIES = ("voi", "baq")
CASE = "case"
# Rounds of answer coding per update, each round codes the best questions
# that earlier rounds left open
MAX_ROUNDS = 3
# Questions coded per question suggested
CANDIDATES_PER_QUESTION = 4
# Questions about the respondent, the deceased's social background, records
# and access to care
BACKGROUND = {"Id10008", "Id10009", "Id10020", "Id10022", "Id10051",
              "Id10052", "Id10063", "Id10064", "Id10065", "Id10429",
              "Id10430", "Id10435", "Id10437", "Id10438", "Id10442",
              "Id10450", "Id10451", "Id10452", "Id10453", "Id10454",
              "Id10455", "Id10456", "Id10458", "Id10459", "Id10462",
              "Id10463"}
# Form answers of a stillbirth
STILLBIRTH = {"Id10104": "no", "Id10109": "no", "Id10110": "no",
              "Id10114": "yes"}
SEXES = {"female": "female", "male": "male", "ambiguous": "undetermined"}
AGE_FIELDS = ("age_group", "age_neonate_days", "age_child_unit",
              "age_child_days", "age_child_months", "age_child_years",
              "age_adult")
SEX_FIELD = "Id10019"
BIRTH_WEIGHT_FIELD = "Id10366"
DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = DAYS_PER_YEAR / 12
# Words that make a quote a denial
NEGATIONS = {"no", "not", "never", "none", "nothing", "neither", "nor",
             "without", "didn", "wasn", "weren", "hadn", "couldn", "don",
             "doesn", "isn", "t"}
SYSTEM_PROMPT = read_prompt("who_answers.txt")
# Words in more form questions than this do not identify a question's topic
MAX_TOPIC_WORD_QUESTIONS = 15
# Answers less likely than this are not expected to change the top cause
MIN_ANSWER_PROBABILITY = 0.1


def use_bundled_java():
    """Point JAVA_HOME at the Java runtime of jdk4py when it is installed."""
    try:
        import jdk4py
    except ImportError:
        return
    os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)


def age_fields(days):
    """Return the form's age fields for an age at death in days."""
    fields = dict.fromkeys(AGE_FIELDS)
    if days is None or days < 0:
        return fields
    if days < 28:
        fields.update(age_group="neonate", age_neonate_days=int(days))
    elif days < 12 * DAYS_PER_YEAR:
        fields["age_group"] = "child"
        if days >= DAYS_PER_YEAR:
            unit, value = "years", days / DAYS_PER_YEAR
        elif days >= DAYS_PER_MONTH:
            unit, value = "months", days / DAYS_PER_MONTH
        else:
            unit, value = "days", days
        fields.update(age_child_unit=unit, **{f"age_child_{unit}": int(value)})
    else:
        fields.update(age_group="adult", age_adult=int(days / DAYS_PER_YEAR))
    return fields


def profile_fields(profile):
    """Return the form fields a decedent profile gives."""
    if profile is None:
        return age_fields(None) | {SEX_FIELD: None}
    days = 0 if profile.stillbirth else \
        profile.age.days if profile.age else None
    fields = age_fields(days) | {SEX_FIELD: SEXES.get(profile.sex)}
    if profile.birth_weight_g:
        fields[BIRTH_WEIGHT_FIELD] = str(round(profile.birth_weight_g))
    if profile.stillbirth:
        fields.update(STILLBIRTH)
    return fields


def changes_top_cause(probabilities, distribution, categories):
    """Return whether a likely answer would change the most probable category.

    The distribution gives each cause's answer probabilities, causes by
    answers, the probabilities and categories those of the causes.
    """
    joint = pd.DataFrame(probabilities[:, np.newaxis] * distribution) \
        .groupby(list(categories)).sum()
    likely = joint.sum() >= MIN_ANSWER_PROBABILITY
    top = joint.sum(axis=1).idxmax()
    return bool((joint.loc[:, likely].idxmax() != top).any())


def words(text):
    """Return the lowercase words of a text joined by single spaces."""
    return " ".join(re.findall(r"\w+", text.lower()))


def evidence_in_text(evidence, text):
    """Return whether a quote appears word for word in the text."""
    quote = words(evidence)
    return bool(quote) and f" {quote} " in f" {words(text)} "


def is_negated(evidence):
    """Return whether a quote contains a negation."""
    return bool(NEGATIONS & set(words(evidence).split()))


# Choices that give no answer
NON_ANSWERS = {"dk", "ref"}
UNKNOWN = "unknown"
# Answer of a respondent who says they do not know
DONT_KNOW = "dont_know"


def answer_hint(field):
    """Return the kind of answer a form field takes, as shown to the LLM."""
    choices = field["choices"]
    if isinstance(choices, list):
        names = [c for c in choices if c.lower() not in NON_ANSWERS]
        if field["type"] == "select all that apply":
            return "any of, separated by spaces: " + " ".join(names) \
                + f" | {DONT_KNOW} | {UNKNOWN}"
        return "one of: " + " | ".join(names) + f" | {DONT_KNOW} | {UNKNOWN}"
    if field["type"] == "integer":
        return f"a whole number | {DONT_KNOW} | {UNKNOWN}"
    return f"a number | {DONT_KNOW} | {UNKNOWN}"


def parse_answer(field, answer):
    """Return an LLM answer as the field's form value, or None if invalid."""
    answer = str(answer).strip().lower()
    choices = field["choices"]
    if isinstance(choices, list):
        names = {c.lower(): c for c in choices
                 if c.lower() not in NON_ANSWERS}
        given = answer.replace(",", " ").split()
        if field["type"] != "select all that apply" and len(given) != 1:
            return None
        if not given or any(g not in names for g in given):
            return None
        return " ".join(names[g] for g in given)
    try:
        number = float(answer)
    except ValueError:
        return None
    if number < 0:
        return None
    if field["type"] == "integer":
        return str(int(number)) if number.is_integer() else None
    return f"{number:g}"


def answer_schema(field):
    """Return the JSON schema of the answer to a form field."""
    if field["type"] == "select one":
        names = [c for c in field["choices"] if c.lower() not in NON_ANSWERS]
        return {"type": "string", "enum": [*names, DONT_KNOW, UNKNOWN]}
    return {"type": "string"}


def answers_schema(fields):
    """Return the JSON schema of an evidence and answer per form field."""
    properties = {
        name: {"type": "object",
               "properties": {"evidence": {"type": "string"},
                              "answer": answer_schema(field)},
               "required": ["evidence", "answer"],
               "additionalProperties": False}
        for name, field in fields.iterrows()}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


class WhoQuestions:
    """WHO VA 2016 question policy and answer coding shared by interviews."""

    def __init__(self, policy, llm_client, skip_logic=True):
        from codalg.interview import BAQPolicy, VOIPolicy
        from codalg.questionnaires import Questionnaire
        if policy not in POLICIES:
            raise ValueError(f"Unknown question policy {policy!r}, expected "
                             f"one of: {', '.join(POLICIES)}")
        policy_class = BAQPolicy if policy == "baq" else VOIPolicy
        self.policy = policy_class.from_pretrained(QUESTIONNAIRE)
        self.fields = Questionnaire.load(QUESTIONNAIRE).fields
        # ICD-11 category of each cause, its code before the dot
        self.categories = [c.split(".")[0] for c in self.policy.causes]
        counts = Counter(w for label in self.fields["label"]
                         for w in set(words(label).split()))
        self.common_words = {w for w, n in counts.items()
                             if n > MAX_TOPIC_WORD_QUESTIONS}
        self.llm_client = llm_client
        self.skip_logic = skip_logic
        if skip_logic:
            use_bundled_java()

    def start(self):
        """Return a new interview."""
        return WhoInterview(self)

    def asks_about(self, evidence, question):
        """Return whether a quote shares a topic word with a question."""
        label = set(words(self.fields.loc[question, "label"]).split())
        return bool((label - self.common_words)
                    & set(words(evidence).split()))

    def code_answers(self, text, questions):
        """Return the form values the text gives for the questions.

        A question the respondent says they do not know has DONT_KNOW.
        """
        listed = "\n".join(
            f"{q}: {self.fields.loc[q, 'label']} "
            f"[{answer_hint(self.fields.loc[q])}]" for q in questions)
        response = self.llm_client.call_with_schema(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"## TRANSCRIPT\n{text.strip()}\n\n"
                        f"## QUESTIONS\n{listed}",
            schema=answers_schema(self.fields.loc[questions]),
            schema_name="who_va_answers",
            temperature=0,
        )
        if response.get("api_failed"):
            raise RuntimeError("Answer coding failed after retries")
        values = {}
        for question in questions:
            item = response.get(question)
            if not isinstance(item, dict) or \
                    str(item.get("answer")).strip().lower() == UNKNOWN:
                continue
            answer = str(item["answer"]).strip().lower()
            value = DONT_KNOW if answer == DONT_KNOW else \
                parse_answer(self.fields.loc[question], answer)
            if value is None:
                logger.info("Invalid answer %r to %s", item.get("answer"),
                            question)
            elif not evidence_in_text(item.get("evidence", ""), text):
                logger.info("Answer to %s has no evidence in the transcript",
                            question)
            elif value == "no" and not is_negated(item["evidence"]):
                logger.info("Answer no to %s has no denial in its evidence",
                            question)
            elif value == DONT_KNOW and \
                    not self.asks_about(item["evidence"], question):
                logger.info("Answer %s to %s has no question in its evidence",
                            DONT_KNOW, question)
            else:
                values[question] = value
        return values


class WhoInterview:
    """WHO VA 2016 form answers and question selection of one interview."""

    def __init__(self, questions):
        from codalg.interview import Interview
        self.questions = questions
        prefilled = pd.DataFrame(index=[CASE],
                                 columns=[*AGE_FIELDS, SEX_FIELD],
                                 dtype=object)
        self.interview = Interview.start(
            prefilled, questions.policy.questions,
            questionnaire=QUESTIONNAIRE, use_rules=questions.skip_logic)

    def update(self, text, profile, causes, num_questions):
        """Code the answers in the text and return the next questions.

        Only the best-ranked questions are coded, so that those suggested
        are not already answered. Return the labels of the best questions,
        and whether the policy would stop asking.
        """
        self.set_fields(profile_fields(profile))
        probabilities = self.probabilities(causes)
        if probabilities is None:
            return [], False
        coded = set()
        for _ in range(MAX_ROUNDS):
            ranked, stop = self.rank(probabilities)
            candidates = ranked[:CANDIDATES_PER_QUESTION * num_questions]
            questions = [q for q in candidates if q not in coded]
            if not questions:
                break
            coded.update(questions)
            values = self.questions.code_answers(text, questions)
            if not values:
                break
            unknown = [q for q, v in values.items() if v == DONT_KNOW]
            self.interview.record(
                pd.Series(unknown, index=[CASE] * len(unknown), dtype=object),
                pd.DataFrame({q: [v] for q, v in values.items()
                              if v != DONT_KNOW},
                             index=[CASE], dtype=object))
        else:
            ranked, stop = self.rank(probabilities)
        labels = self.questions.fields["label"]
        return list(labels[ranked[:num_questions]]), stop

    def set_fields(self, fields):
        """Write form fields, marking the policy's questions answered."""
        asked = self.interview.asked_questions
        for field, value in fields.items():
            if field in asked.columns:
                if value is None:
                    continue
                asked.loc[CASE, field] = True
            self.interview.features.loc[CASE, field] = value

    def probabilities(self, causes):
        """Return the ICD-11 cause probabilities, or None if none are known."""
        scores = {c.split(":", 1)[1]: cause["score"]
                  for c, cause in causes.items() if c.startswith("icd11:")}
        if not any(scores.get(c) for c in self.questions.policy.causes):
            return None
        return pd.DataFrame(scores, index=[CASE])

    def rank(self, probabilities):
        """Return the open questions from best to worst and whether to stop."""
        policy = self.questions.policy
        asked = self.interview.asked_questions
        is_open = ~asked.loc[CASE]
        if self.interview.rules is not None:
            is_open &= self.interview.rules.eligible_questions(
                self.interview.features).loc[CASE]
        labels = self.questions.fields["label"]
        # Entry fields that follow a question have non-question labels
        questions = [q for q in is_open.index[is_open] if q not in BACKGROUND
                     and labels[q].rstrip().endswith("?")]
        scores = policy(probabilities, self.interview)
        ranked = scores.loc[CASE, questions].sort_values(ascending=False)
        # Stop once no open question could change the top cause category
        cause_probabilities = policy.cause_probabilities(probabilities) \
            .loc[CASE].to_numpy()
        stop = not any(
            changes_top_cause(cause_probabilities,
                              policy.answer_distributions[q],
                              self.questions.categories)
            for q in questions)
        return list(ranked.index), stop
