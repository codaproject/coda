import pytest

from coda.inference.champs_prompted.adult_agent import (
    ADULT_AGE_GROUP_LIMITS,
    AdultIcd11SinglePassAgent,
    load_adult_chain_labels,
    load_adult_labels,
)
from coda.inference.champs_prompted.icd11_agent import (
    ChampsPromptedIcd11SinglePassAgent,
    ChampsPromptedIcd11TwoPassAgent,
    age_group,
    clinical_assessment,
    load_labels,
)
from coda.metadata import Profile

RANKS = {"rank_1_code": "1F40.Y", "rank_1_prob": 0.6,
         "rank_2_code": "5B51", "rank_2_prob": 0.2,
         "rank_3_code": "1C41", "rank_3_prob": 0.1,
         "rank_4_code": "1A40.0", "rank_4_prob": 0.05,
         "rank_5_code": "CA40.Z", "rank_5_prob": 0.05,
         "questions": ["q1", "q2", "q3"]}
INFANT = {"profile": {"age": {"value": 11, "unit": "months"}}}
OLDER_MAN = {"profile": {"age": {"value": 70, "unit": "years"},
                         "sex": "male"}}


class FakeLLMClient:
    def __init__(self, ranks=RANKS):
        self.ranks = ranks
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return "**Clinical Assessment:** Severe malaria.\n**Candidate Codes:**"

    def call_with_schema(self, **kwargs):
        self.calls.append(kwargs)
        return {"reasoning": "Severe malaria.", **self.ranks}


@pytest.mark.parametrize("profile, group", [
    ({"stillbirth": True}, "stillbirth"),
    ({"age": {"value": 5, "unit": "hours"}}, "less24hours"),
    ({"age": {"value": 23, "unit": "hours"}}, "less24hours"),
    ({"age": {"value": 24, "unit": "hours"}}, "earlyneonate"),
    ({"age": {"value": 3, "unit": "days"}}, "earlyneonate"),
    ({"age": {"value": 2, "unit": "weeks"}}, "lateneonate"),
    ({"age": {"value": 11, "unit": "months"}}, "infant"),
    ({"age": {"value": 4, "unit": "years"}}, "child"),
    ({"age": {"value": 6, "unit": "years"}}, None),
    ({}, None),
])
def test_age_group(profile, group):
    assert age_group(Profile.from_dict(profile)) == group


def test_unknown_age_group_uses_all_labels():
    assert set(load_labels("infant")) < set(load_labels(None))


@pytest.mark.asyncio
async def test_single_pass_ranks_age_group_codes():
    client = FakeLLMClient()
    agent = ChampsPromptedIcd11SinglePassAgent(client)

    result = await agent.process_chunk("c1", "Child had fever.", [],
                                       metadata=INFANT)

    [call] = client.calls
    enum = call["schema"]["properties"]["rank_1_code"]["enum"]
    assert enum == sorted(load_labels("infant"))
    assert list(result["causes"])[0] == "icd11:1F40.Y"
    assert result["causes"]["icd11:1F40.Y"]["identifiers"] == \
        {"icd11": "1F40.Y"}
    assert result["reasoning"] == "Severe malaria."


@pytest.mark.asyncio
async def test_single_pass_causal_chain():
    client = FakeLLMClient({**RANKS, "chain": ["1F40.0", "8A66", "8A66"]})
    agent = ChampsPromptedIcd11SinglePassAgent(client)

    result = await agent.process_chunk("c1", "Child had fever.", [],
                                       metadata=INFANT)

    [call] = client.calls
    chain_enum = call["schema"]["properties"]["chain"]["items"]["enum"]
    assert chain_enum == sorted(load_labels("chain"))
    assert "8A66 | " in call["system_prompt"]
    assert [c["identifiers"]["icd11"] for c in result["chain"]] == \
        ["1F40.Y", "8A66"]


@pytest.mark.asyncio
async def test_two_pass_codes_from_first_pass_assessment():
    client = FakeLLMClient()
    agent = ChampsPromptedIcd11TwoPassAgent(client)

    result = await agent.process_chunk("c1", "Child had fever.", [],
                                       metadata=INFANT)

    first, second = client.calls
    assert "schema" not in first
    assert "Severe malaria." in second["user_prompt"]
    assert result["reasoning"] == "Severe malaria."
    assert len(result["causes"]) == 5


@pytest.mark.parametrize("label", [
    "**Clinical Assessment:**",
    "**Clinical Assessment**:",
    "**Clinical Assessment**",
    "### Clinical Assessment:",
    "## Clinical Assessment",
])
def test_clinical_assessment_headings(label):
    codes = label.replace("Clinical Assessment", "Candidate Codes")
    text = f"{label}\nSevere malaria.\n\n{codes}\n1. 1F40"
    assert clinical_assessment(text) == "Severe malaria."


def test_clinical_assessment_without_heading():
    assert clinical_assessment(" Severe malaria. ") == "Severe malaria."


@pytest.mark.parametrize("years, group", [
    (12, "young_adult"),
    (40, "middle_aged"),
    (70, "older_adult"),
])
def test_adult_age_group(years, group):
    profile = Profile.from_dict({"age": {"value": years, "unit": "years"}})
    assert age_group(profile, ADULT_AGE_GROUP_LIMITS) == group


def test_adult_labels_by_sex():
    assert load_adult_labels("female")["JB60"] == "Maternal"
    assert "JB60" not in load_adult_labels("male")
    assert "2C82.Z" not in load_adult_labels("female")
    assert len(load_adult_labels(None)) == 35


@pytest.mark.asyncio
async def test_adult_single_pass_ranks_adult_codes():
    client = FakeLLMClient({"rank_1_code": "8B20", "rank_1_prob": 1.0,
                            "chain": ["CA71.0", "CB41.0Z"]})
    agent = AdultIcd11SinglePassAgent(client)

    result = await agent.process_chunk("c1", "He collapsed suddenly.", [],
                                       metadata=OLDER_MAN)

    [call] = client.calls
    enum = call["schema"]["properties"]["rank_1_code"]["enum"]
    assert enum == sorted(load_adult_labels("male"))
    assert "Cardiovascular causes dominate" in call["system_prompt"]
    assert "{" not in call["system_prompt"]
    assert result["causes"]["icd11:8B20"]["name"] == "Stroke"
    chain_enum = call["schema"]["properties"]["chain"]["items"]["enum"]
    assert chain_enum == sorted(load_adult_chain_labels())
    assert [c["identifiers"]["icd11"] for c in result["chain"]] == \
        ["8B20", "CA71.0", "CB41.0Z"]
