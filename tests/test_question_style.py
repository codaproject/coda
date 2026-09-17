import pytest
from unittest.mock import MagicMock

from coda.inference.champs_prompted_agent import (
    ChampsPromptedInferenceAgent,
    build_cod_output_schema,
)


def make_agent(**kwargs):
    return ChampsPromptedInferenceAgent(llm_client=MagicMock(), **kwargs)


def test_default_style_prompt():
    prompt = make_agent().rendered_system_prompt
    assert "Step 5: Propose exactly 3 follow-up questions" in prompt
    assert "lay terms" not in prompt


def test_simplified_style_prompt():
    prompt = make_agent(question_style="simplified").rendered_system_prompt
    assert "Step 5: Propose exactly 3 follow-up questions" in prompt
    assert "no medical training" in prompt
    assert "fits or shaking" in prompt


def test_num_questions_renders_in_style_body():
    prompt = make_agent(num_questions=5,
                        question_style="simplified").rendered_system_prompt
    assert "exactly 5 follow-up questions" in prompt
    assert "{num_questions}" not in prompt


def test_schema_description_follows_style():
    default = build_cod_output_schema(question_style="default")
    simplified = build_cod_output_schema(question_style="simplified")
    assert "lay language" not in \
        default["properties"]["questions"]["description"]
    assert "lay language" in \
        simplified["properties"]["questions"]["description"]


def test_style_carries_into_session_agent():
    agent = make_agent(question_style="simplified").create_session_agent()
    assert agent.question_style == "simplified"
    assert "no medical training" in agent.rendered_system_prompt


def test_unknown_style_rejected():
    with pytest.raises(ValueError):
        make_agent(question_style="chatty")
    with pytest.raises(ValueError):
        build_cod_output_schema(question_style="chatty")
