# RAG Grounder

A grounding module that extracts concepts from clinical text and maps them to ontology terms using a three-step pipeline: LLM extraction → neo4j vector search retrieval → LLM re-ranking.

## Usage

```python
from coda.grounding.rag_grounder import RagGrounder

grounder = RagGrounder()                          # uses default config
grounder = RagGrounder(config_path="my_config.yaml")  # custom config

# Returns the top ScoredMatch per extracted concept
matches = grounder.ground(text)

# Returns span-level Annotation objects (Gilda-compatible)
annotations = grounder.annotate(text)

# Returns the full pipeline output as a dict
result = grounder.process(text)
# {"text": ..., "Concepts": [{"Concept": ..., "supporting_evidence": [...], "matched_terms": [(RetrievalTerm, score), ...]}, ...]}
```

## Configuration

Configuration is a YAML file with four sections. See `grounder_config/icd10_config.yaml` for a full example.

```yaml
concept_type: disease

llm:
  model: gpt-4o-mini

extractor:
  prompt_config_path: "../prompt_configs/extractor_default.yaml"

retriever:
  ontology: icd10          # ontology loaded into neo4j
  embedding_model: all-MiniLM-L6-v2
  top_k: 10
  min_similarity: 0.0

reranker:
  prompt_config_path: "prompt_configs/reranker_default.yaml"
```

Paths in `prompt_config_path` are resolved relative to the config file's location.

## Module structure

| File | Role |
|------|------|
| `grounder.py` | `RagGrounder` — public entry point, orchestrates the pipeline |
| `extractor.py` | LLM-based concept and evidence extraction |
| `retriever.py` | Neo4j vector search retrieval |
| `reranker.py` | LLM-based re-ranking of retrieved terms |
| `config.py` | Dataclasses for config and `PromptConfig` |
| `types.py` | `RetrievalTerm` dataclass |
| `utils.py` | Evidence span finding |
| `grounder_config/` | Built-in grounder configs (`icd10_config.yaml`, `icd11_config.yaml`) |
| `prompt_configs/` | YAML prompt configs for extractor and reranker |

## Requirements

- A running neo4j instance at `bolt://localhost:7687` with the target ontology indexed as a vector index
- OpenAI API key (or configure a different LLM client via the `llm_client` argument)
- `pip install coda[rag]`
