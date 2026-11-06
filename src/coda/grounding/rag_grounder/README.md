# RAG Grounder

A grounding module that extracts concepts from clinical text and maps them to ontology terms using a three-step pipeline: LLM extraction → neo4j vector search retrieval → LLM re-ranking.

## Usage

```python
from coda.grounding.rag_grounder import RagGrounder

grounder = RagGrounder()   # reads config from the global settings (grounder.rag)

# Returns the top ScoredMatch per extracted concept
matches = grounder.ground(text)

# Returns span-level Annotation objects (Gilda-compatible)
annotations = grounder.annotate(text)

# Returns the full pipeline output as a dict
result = grounder.process(text)
# {"text": ..., "Concepts": [{"Concept": ..., "supporting_evidence": [...], "matched_terms": [(RetrievalTerm, score), ...]}, ...]}
```

## Configuration

The grounder reads the `grounder.rag` namespace of the global config
(`config/settings.yaml`, loaded via `coda.config`). See that file for the full
set of options and their documented alternatives.

```yaml
grounder:
  rag:
    concept_type: disease
    llm:
      provider: openai
      model: gpt-4o-mini
    extractor:
      type: hunflair               # LLM | hunflair
      prompt: extractor_default    # file stem in prompts/
    retriever:
      ontology: icd10              # ontology loaded into neo4j
      embedding_model: all-MiniLM-L6-v2
      top_k: 10
      min_similarity: 0.0
    reranker:
      enabled: true
      prompt: reranker_default     # file stem in prompts/
```

Prompts are selected by their filename stem in `prompts/`, described under
Prompts below. Override any value by editing `config/settings.yaml` or with a
`CODA_`-prefixed env var (e.g. `CODA_GROUNDER__RAG__RETRIEVER__ONTOLOGY=icd11`).
For ICD-11 grounding, set `retriever.ontology: icd11` and `extractor.type: LLM`
(the settings file's comments point this out).

## Module structure

| File | Role |
|------|------|
| `grounder.py` | `RagGrounder` — public entry point, orchestrates the pipeline |
| `extractor.py` | LLM-based concept and evidence extraction |
| `retriever.py` | Neo4j vector search retrieval |
| `reranker.py` | LLM-based re-ranking of retrieved terms |
| `types.py` | `RetrievalTerm` dataclass |
| `utils.py` | Evidence span finding |

Settings live in the repo-root `config/settings.yaml`, loaded by
`coda.config`, and prompts in `prompts/`.

## Prompts

The LLM extractor and the reranker each take their prompt from a YAML file in
`prompts/`. Settings name the file by its stem, in
`grounder.rag.extractor.prompt` and `grounder.rag.reranker.prompt`.

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `use_schema` | yes | If `true`, calls the LLM with structured output (JSON schema). If `false`, parses raw JSON from the response text. |
| `system_prompt` | no | System prompt. Supports `{concept_type}` interpolation (extractor only). |
| `user_prompt` | yes | User prompt. Extractor supports `{concept_type}` and `{text}`; reranker supports `{concept}`, `{evidence_text}`, and `{retrieved_terms}`. |
| `concept_key` | extractor only | The key for the concept name in the LLM's output. |
| `supporting_evidence_key` | extractor only | The key for the evidence list in the LLM's output. |
| `schema` | if `use_schema: true` | JSON schema for structured output. |

### Output field keys (extractor)

Because different prompts use different field names in their output,
`concept_key` and `supporting_evidence_key` tell the extractor which keys to
read:

```yaml
concept_key: Disease                       # key for the concept name in LLM output
supporting_evidence_key: Supporting Evidence  # key for the evidence list
```

### Provided prompts

| File | Description |
|------|-------------|
| `extractor_default.yaml` | General-purpose extractor. Parametric on `concept_type`. Uses structured output. |
| `extractor_medcoder.yaml` | Verbatim prompt from the MedCodER paper for ICD-10 disease extraction. Uses raw JSON output (`use_schema: false`). |
| `reranker_default.yaml` | General-purpose reranker. Uses structured output. |

### Custom prompts

Add a YAML file to `prompts/` following the structure of an existing one, then
select it by its filename stem:

```yaml
# prompts/my_extractor.yaml  ->  selected as `my_extractor`
grounder:
  rag:
    extractor:
      prompt: my_extractor
```

## Requirements

- A running neo4j instance at `bolt://localhost:7687` with the target ontology indexed as a vector index
- OpenAI API key (or configure a different LLM client via the `llm_client` argument)
- `pip install coda[rag]`
