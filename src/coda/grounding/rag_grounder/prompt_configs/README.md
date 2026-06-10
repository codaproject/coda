# Prompt Configs

YAML files that define the LLM prompts used by the extractor and reranker. Each config is referenced by path in the main config YAML.

## Fields

| Field | Required | Description |
|-------|----------|-------------|
| `use_schema` | yes | If `true`, calls the LLM with structured output (JSON schema). If `false`, parses raw JSON from the response text. |
| `system_prompt` | no | System prompt. Supports `{concept_type}` interpolation (extractor only). |
| `user_prompt` | yes | User prompt. Extractor supports `{concept_type}` and `{text}`; reranker supports `{concept}`, `{evidence_text}`, and `{retrieved_terms}`. |
| `query_fields` | yes | Maps internal field names to the LLM's output field names. Must include `concept` and `supporting_evidence`. |
| `schema` | if `use_schema: true` | JSON schema for structured output. |

## `query_fields`

Because different prompts use different field names in their output, `query_fields` tells the extractor which keys to read:

```yaml
query_fields:
  concept: Disease               # key for the concept name in LLM output
  supporting_evidence: Supporting Evidence  # key for the evidence list
```

## Provided configs

| File | Description |
|------|-------------|
| `extractor_default.yaml` | General-purpose extractor. Parametric on `concept_type`. Uses structured output. |
| `extractor_medcoder.yaml` | Verbatim prompt from the MedCodER paper for ICD-10 disease extraction. Uses raw JSON output (`use_schema: false`). |
| `reranker_default.yaml` | General-purpose reranker. Uses structured output. |

## Custom configs

Create a new YAML file following the structure of an existing config, then point to it from your grounder config (e.g. `grounder_config/icd10_config.yaml`):

```yaml
concept_type: procedure

extractor:
  prompt_config_path: "../prompt_configs/my_extractor.yaml"
```
