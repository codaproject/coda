"""Follow-up questions from the WHO VA 2016 form, ranked by information gain.

An LLM codes the form answers the transcript gives, and a codalg policy ranks
the questions still open given ICD-11 cause probabilities, those of the
agent or those a LightGBM model trained on CHAMPS gives the answers.
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
CLASSIFIER = "champs"
POLICIES = ("voi", "baq")
PROBABILITIES = ("llm", "lightgbm")
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
# CHAMPS age subcategories and the age in days each ends at
AGE_SUBCATEGORIES = (("neonate 72h", 3), ("neonate >72h", 28),
                     ("infant 6m", DAYS_PER_YEAR / 2),
                     ("infant >6m", DAYS_PER_YEAR))
# Form fields of the mortality of the region, high, low or very low
REGION_FIELDS = {"hiv": "Id10002", "malaria": "Id10003"}
REGION_VALUES = {"high": "high", "low": "low", "very_low": "veryl"}
ILLNESS = "During the illness that led to death, "
# Open questions asked before the policy's, each with the word stems of its
# topic and the form questions whose answers cover it
REVIEW_OF_SYSTEMS = (
    ("Could you tell me about the illness or events that led to her/his "
     "death?", ("ill", "happen"),
     ("Id10147", "Id10153", "Id10181", "Id10188", "Id10219", "Id10284",
      "Id10286")),
    (ILLNESS + "what did you notice about her/his general condition?",
     ("condition",),
     ("Id10244", "Id10268", "Id10214", "Id10281", "Id10286", "Id10284")),
    (ILLNESS + "what did you notice about her/his breathing?", ("breath",),
     ("Id10153", "Id10159", "Id10166", "Id10168", "Id10172", "Id10112")),
    (ILLNESS + "what did you notice about her/his feeding and belly "
     "(abdomen)?", ("feed", "belly", "abdom", "eat"),
     ("Id10181", "Id10188", "Id10193", "Id10194", "Id10200", "Id10273")),
    (ILLNESS + "what did you notice about how (s)he behaved or responded?",
     ("behav", "respon"),
     ("Id10214", "Id10219", "Id10281", "Id10286", "Id10208", "Id10278")),
    (ILLNESS + "what did you notice about her/his skin or body?",
     ("skin",),
     ("Id10233", "Id10240", "Id10265", "Id10268", "Id10289", "Id10244")),
    (ILLNESS + "did (s)he seem to have any pain?", ("pain",),
     ("Id10174", "Id10194", "Id10210", "Id10264")),
)
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


def age_features(days):
    """Return the classifier's age features for an age at death in days.

    These are the ages the form calculates and those of CHAMPS, which splits
    the age into whole years, months and days.
    """
    if days is None or days < 0:
        return {}
    days = int(days)
    years, in_year = divmod(days, DAYS_PER_YEAR)
    months, in_month = divmod(in_year, DAYS_PER_MONTH)
    neonate = int(days < 28)
    adult = int(days >= 12 * DAYS_PER_YEAR)
    groups = {"isNeonatal": neonate, "isChild": 1 - neonate - adult,
              "isAdult": adult}
    return {
        "ageInDays": days, "ageInDaysNeonate": days,
        "ageInYears": int(years), "ageInYears2": int(years),
        "ageInYearsRemain": in_year, "ageInMonths": int(months),
        "ageInMonthsRemain": int(in_month),
        "ageInMonthsByYear": int(years * 12 + months),
        **{f"{name}{suffix}": value for name, value in groups.items()
           for suffix in ("", "1", "2")},
        "age_years": int(years), "age_months": int(months),
        "age_days": int(in_month),
        "age_group_subcat": next(
            (name for name, end in AGE_SUBCATEGORIES if days < end), None),
    }


def age_days(profile):
    """Return a decedent profile's age at death in days, or None."""
    if profile is None:
        return None
    if profile.stillbirth:
        return 0
    return profile.age.days if profile.age else None


def profile_fields(profile):
    """Return the form fields a decedent profile gives."""
    if profile is None:
        return age_fields(None) | {SEX_FIELD: None}
    fields = age_fields(age_days(profile)) | \
        {SEX_FIELD: SEXES.get(profile.sex)}
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


def asks_topic(text, stems):
    """Return whether a question in the text has a word with a stem given."""
    questions = re.findall(r"[^.?!]*\?", text)
    return any(w.startswith(stems) for question in questions
               for w in words(question).split())


def is_negated(evidence):
    """Return whether a quote contains a negation."""
    return bool(NEGATIONS & set(words(evidence).split()))


# Choices that give no answer
NON_ANSWERS = {"dk", "ref"}
UNKNOWN = "unknown"
# Answer to a question asked whose reply gives no form answer
NO_ANSWER = "no_answer"


def dont_know(field):
    """Return the form field's don't know choice, or None if it has none."""
    choices = field["choices"]
    if not isinstance(choices, list):
        return None
    return next((c for c in choices if c.lower() == "dk"), None)


def answer_hint(field):
    """Return the kind of answer a form field takes, as shown to the LLM."""
    choices = field["choices"]
    if isinstance(choices, list):
        names = [c for c in choices if c.lower() not in NON_ANSWERS]
        if field["type"] == "select all that apply":
            return "any of, separated by spaces: " + " ".join(names) \
                + f" | {NO_ANSWER} | {UNKNOWN}"
        return "one of: " + " | ".join(names) + f" | {NO_ANSWER} | {UNKNOWN}"
    if field["type"] == "integer":
        return f"a whole number | {NO_ANSWER} | {UNKNOWN}"
    return f"a number | {NO_ANSWER} | {UNKNOWN}"


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
        return {"type": "string", "enum": [*names, NO_ANSWER, UNKNOWN]}
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

    def __init__(self, policy, llm_client, skip_logic=True,
                 review_of_systems=False, probabilities="llm", region=None):
        """Load the policy and the form.

        With review_of_systems, open review-of-systems questions come
        before the policy's. The probabilities are the agent's with llm, or
        the LightGBM model's with lightgbm. The region gives the LightGBM
        model the mortality, high, low or very_low, of hiv and malaria where
        the deaths happen.
        """
        from codalg.interview import BAQPolicy, VOIPolicy
        from codalg.questionnaires import Questionnaire
        if policy not in POLICIES:
            raise ValueError(f"Unknown question policy {policy!r}, expected "
                             f"one of: {', '.join(POLICIES)}")
        if probabilities not in PROBABILITIES:
            raise ValueError(f"Unknown cause probabilities {probabilities!r},"
                             f" expected one of: {', '.join(PROBABILITIES)}")
        policy_class = BAQPolicy if policy == "baq" else VOIPolicy
        self.policy = policy_class.from_pretrained(QUESTIONNAIRE)
        questionnaire = Questionnaire.load(QUESTIONNAIRE)
        self.fields = questionnaire.fields.join(
            questionnaire.labels.rename("label")).fillna({"label": ""})
        self.calculated = questionnaire.calculated
        self.classifier = None
        if probabilities == "lightgbm":
            from codalg.classification.lightgbm import LightGBMClassifier
            self.classifier = LightGBMClassifier.from_pretrained(CLASSIFIER)
        # ICD-11 category of each cause, its code before the dot
        self.categories = [c.split(".")[0] for c in self.policy.causes]
        counts = Counter(w for label in self.fields["label"]
                         for w in set(words(label).split()))
        self.common_words = {w for w, n in counts.items()
                             if n > MAX_TOPIC_WORD_QUESTIONS}
        self.llm_client = llm_client
        self.skip_logic = skip_logic
        self.review_of_systems = review_of_systems
        self.region = {REGION_FIELDS[name]: REGION_VALUES[value]
                       for name, value in (region or {}).items() if value}
        if skip_logic:
            use_bundled_java()

    def start(self):
        """Return a new interview."""
        return WhoInterview(self)

    def asks_about(self, evidence, question):
        """Return whether a quote asks a question sharing its topic word."""
        label = set(words(self.fields.loc[question, "label"]).split())
        return "?" in evidence and bool((label - self.common_words)
                                        & set(words(evidence).split()))

    def code_answers(self, text, questions):
        """Return the form values the text gives for the questions.

        A question asked whose reply gives no form answer has NO_ANSWER.
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
            value = NO_ANSWER if answer == NO_ANSWER else \
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
            elif value == NO_ANSWER and \
                    not self.asks_about(item["evidence"], question):
                logger.info("Answer %s to %s has no question in its evidence",
                            NO_ANSWER, question)
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
        # The form calculates these, so they are never asked
        calculated = questions.calculated.intersection(
            questions.policy.questions)
        self.interview.asked_questions[calculated] = True
        self.ages = {}
        self.causes = {}

    def update(self, text, profile, causes, num_questions):
        """Code the answers in the text and return the next questions.

        Only the best-ranked questions are coded, so that those suggested
        are not already answered, along with the form questions of the
        review-of-systems questions suggested. Return the open
        review-of-systems questions followed by the labels of the best
        questions, and whether the policy would stop asking. A question asked
        without a form answer gets its don't know choice, or stays without
        a value. Without cause probabilities, return no questions.
        """
        self.set_fields(profile_fields(profile))
        self.ages = age_features(age_days(profile))
        self.causes = causes
        if self.probabilities() is None:
            return [], False
        fields = self.questions.fields
        coded = set()
        for _ in range(MAX_ROUNDS):
            ranked, stop = self.rank()
            candidates = ranked[:CANDIDATES_PER_QUESTION * num_questions]
            reviews = self.open_reviews(text, ranked)[:num_questions]
            covering = [q for _, _, fields in reviews for q in fields
                        if q in ranked]
            questions = [q for q in dict.fromkeys(covering + candidates)
                         if q not in coded]
            if not questions:
                break
            coded.update(questions)
            values = self.questions.code_answers(text, questions)
            if not values:
                break
            values = {q: dont_know(fields.loc[q]) if v == NO_ANSWER else v
                      for q, v in values.items()}
            unknown = [q for q, v in values.items() if v is None]
            self.interview.record(
                pd.Series(unknown, index=[CASE] * len(unknown), dtype=object),
                pd.DataFrame({q: [v] for q, v in values.items()
                              if v is not None},
                             index=[CASE], dtype=object))
        else:
            ranked, stop = self.rank()
        reviews = [question for question, _, _ in
                   self.open_reviews(text, ranked)]
        labels = reviews + list(fields["label"][ranked])
        return labels[:num_questions], stop and not reviews

    def open_reviews(self, text, ranked):
        """Return the open review-of-systems questions.

        One is open while the text asks no question on its topic, none of its
        form questions is asked and some are among the ranked ones.
        """
        if not self.questions.review_of_systems:
            return []
        asked = self.interview.asked_questions.loc[CASE]
        return [(question, stems, fields)
                for question, stems, fields in REVIEW_OF_SYSTEMS
                if not asks_topic(text, stems)
                and not asked[list(fields)].any()
                and set(fields) & set(ranked)]

    def set_fields(self, fields):
        """Write form fields, marking the policy's questions answered."""
        asked = self.interview.asked_questions
        for field, value in fields.items():
            if field in asked.columns:
                if value is None:
                    continue
                asked.loc[CASE, field] = True
            self.interview.features.loc[CASE, field] = value

    def probabilities(self):
        """Return the ICD-11 cause probabilities, or None if none are known.

        These are the classifier's given the answers if there is one, else
        the agent's.
        """
        classifier = self.questions.classifier
        if classifier is not None:
            context = pd.DataFrame(self.questions.region | self.ages,
                                   index=[CASE])
            return classifier.classify(
                pd.concat([self.interview.features, context], axis=1))
        scores = {c.split(":", 1)[1]: cause["score"]
                  for c, cause in self.causes.items()
                  if c.startswith("icd11:")}
        if not any(scores.get(c) for c in self.questions.policy.causes):
            return None
        return pd.DataFrame(scores, index=[CASE])

    def rank(self):
        """Return the open questions from best to worst and whether to stop."""
        policy = self.questions.policy
        probabilities = self.probabilities()
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
