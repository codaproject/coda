import pytest

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


class FakeLLMClient:
    def __init__(self):
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return "**Clinical Assessment:** Severe malaria.\n**Candidate Codes:**"

    def call_with_schema(self, **kwargs):
        self.calls.append(kwargs)
        return {"reasoning": "Severe malaria.", **RANKS}


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
