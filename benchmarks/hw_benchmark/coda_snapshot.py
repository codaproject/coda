"""CODA's CHAMPS inference request, as this benchmark issues it.

Reads CODA's checked-in CHAMPS resources (src/coda/resources/champs) directly for
the system prompt, schema guidance, and allowed causes, so the benchmark never
drifts from coda.inference.champs_llm_agent. COD_OUTPUT_SCHEMA below mirrors that
agent's schema, kept inline (rather than imported) so the benchmark stays free of
CODA's inference dependencies; keep it in sync if CODA's schema changes. The infer
helpers issue the constrained-decoding calls directly (Ollama's native
format=schema and the OpenAI-compatible response_format json_schema, strict) to
measure raw backend latency.
"""
import json
from pathlib import Path

CHAMPS = Path(__file__).resolve().parents[2] / "src" / "coda" / "resources" / "champs"

COD_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "1-2 sentence summary of the key evidence for the top cause.",
        },
        "top_causes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cause_name": {"type": "string"},
                    "probability": {
                        "type": "number", "minimum": 0, "maximum": 1,
                        "description": "Calibrated probability in [0,1]; the "
                                       "top_causes probabilities should sum to ~1.",
                    },
                },
                "required": ["cause_name", "probability"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 3,
        },
        "questions": {
            "type": "array",
            "description": (
                "Exactly 3 follow-up questions that would best differentiate "
                "the top causes and increase confidence in the underlying cause."
            ),
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["reasoning", "top_causes", "questions"],
    "additionalProperties": False,
}


def build_system_prompt():
    causes = [c.strip() for c in
        (CHAMPS / "group_causes.txt").read_text().splitlines() if c.strip()]
    system_prompt = (CHAMPS / "system_prompt.txt").read_text()
    schema_guidance = (CHAMPS / "schema_guidance.txt").read_text()
    return system_prompt.format(allowed_causes=", ".join(causes)) + "\n\n" + schema_guidance


def user_prompt(narrative):
    return f"## INPUT\n- narrative:\n  {narrative}"


def ollama_infer(model, system, user, host="http://localhost:11434",
                 temperature=0.0, think=None):
    from ollama import Client
    client = Client(host=host, timeout=300.0)
    kwargs = {} if think is None else {"think": think}
    resp = client.chat(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        format=COD_OUTPUT_SCHEMA,
        options={"temperature": temperature},
        **kwargs)
    return json.loads(resp.message.content)


def openai_infer(model, system, user, base_url, api_key="local", temperature=0.0):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "champs_cod_classification",
                            "schema": COD_OUTPUT_SCHEMA, "strict": True}},
        temperature=temperature)
    return json.loads(resp.choices[0].message.content)
