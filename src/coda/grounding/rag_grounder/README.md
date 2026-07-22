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
      prompt: extractor_default    # key into config/prompts/
    retriever:
      ontology: icd10              # ontology loaded into neo4j
      embedding_model: all-MiniLM-L6-v2
      top_k: 10
      min_similarity: 0.0
    reranker:
      enabled: true
      prompt: reranker_default     # key into config/prompts/
```

Prompt configs are referenced by key (a filename stem under `config/prompts/`),
not by path. Override any value by editing `config/settings.yaml` or with a
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

Configuration and prompts live in the repo-root `config/` directory
(`config/settings.yaml` and `config/prompts/`), loaded by `coda.config`.

## Requirements

- A running neo4j instance at `bolt://localhost:7687` with the target ontology indexed as a vector index
- OpenAI API key (or configure a different LLM client via the `llm_client` argument)
- `pip install coda[rag]`
