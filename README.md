CODA: Cause of Death Determination Assistant
============================================

This repository implements the Cause of Death Determination Assistant (CODA)
application which automates cause of death determination via an AI-assisted
interview process.

Installation
------------

Install directly from GitHub

```bash
pip install git+https://github.com/codaproject/coda.git
```

Or clone and install locally

```bash
git clone https://github.com/codaproject/coda.git
cd coda
pip install -e .
```

The knowledge graph data under `kg/` is stored with Git LFS. Before building
the KG image, install [Git LFS](https://git-lfs.com/) for your platform and run:

```bash
git lfs install
git lfs pull
```

Without the LFS objects, the `*.tsv.gz` files are small text pointers and the
Neo4j import step will fail. Git LFS is not required when installing only the
Python package or when using the pre-built Docker images.

Modules
-------

- `coda.app`: Browser-based web application.
- `coda.dialogue`: Dialogue processing including transcription and
  management of grounding to ontologies.
- `coda.grounding`: Models for grounding transcribed dialogue to
  medical terminologies and ontologies.
- `coda.inference`: Base classes and wrappers for cause of death
  inference engines.
- `coda.kg`: Code to build and interact with the CODA Knowledge Graph
  which draws on multiple fragmented sources to assemble terminologies,
  ontologies, prior knowledge and data.
- `coda.resources`: Version controlled, pre-processed or curated
  resource files.

CODA Knowledge Graph
--------------------
The CODA Knowledge Graph integrates multiple data sources to create a comprehensive
medical knowledge base. The following table summarizes the content and structure
contributed by each source:

| Source | Node Types | Edge Types | Semantics                                                                                                                                                                                        |
|--------|-----------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **ICD-10** | `icd10`: Disease classification codes | `is_a` (hierarchical) | WHO International Classification of Diseases, 10th revision. Provides standardized disease codes with hierarchical relationships.                                                                |
| **ICD-11** | `icd11`: Disease classification codes | `is_a` (hierarchy)<br>`maps_to` (ICD-11 to ICD-10) | WHO ICD-11 revision with mappings to ICD-10. Enables cross-version code translation.                                                                                                             |
| **ACME** | `icd10`: ICD-10 codes and code ranges | `causes` (causal relationships from Table D)<br>`part_of_range` (code to range membership) | CDC's WHO ICD-10 ACME decision tables encoding causal relationships between diseases for underlying cause of death determination. Sourced from [openacme](https://github.com/gyorilab/openacme). |
| **PHMRC** | `phmrc`: Verbal autopsy terms | `maps_to` (ICD-10 to PHMRC) | Population Health Metrics Research Consortium terms used in VA data collection, mapped to ICD-10 codes.                                                                                          |
| **WHO VA** | `who.va`: VA cause categories | `is_a` (hierarchy)<br>`maps_to` (ICD-10 to WHO VA) | WHO Verbal Autopsy cause categories with hierarchical structure and ICD-10 code range mappings.                                                                                                  |
| **ProbBase** | `who.va.q`: VA interview questions | `probbase_rel` (questions to causes) | InterVA probability base linking VA interview questions to WHO VA causes with probability values.                                                                                                |
| **HPO** | `hp`: Phenotypes<br>`omim`, `orpha`, `decipher`: Diseases | `has_phenotype` (disease to phenotype) | Human Phenotype Ontology annotations linking diseases to clinical phenotypes with evidence, frequency, and onset metadata.                                                                       |
| **MeSH** | `mesh`: Diseases, pathogens, geographic locations | `isa` (hierarchical) | Medical Subject Headings hierarchy filtered to diseases, pathogens, and geographic locations.                                                                                                    |
| **MONDO** | `mondo`: Diseases | `skos:exactMatch` (MONDO to icd10/icd11/mesh/omim/orpha) | Mondo Disease Ontology providing cross-references that bridge MONDO disease terms to ICD-10, ICD-11, MeSH, OMIM, and Orphanet identifiers used across the KG.                                     |
| **WDI** | `wdi`: Development/health indicators | `has_indicator` (country to indicator) | World Bank World Development Indicators and World Health Indicators linked to country nodes, with time-series data stored as year-value mappings on edges.                                       |
| **WHO Mortality** | `who_mortality`: Country nodes with population data | `has_mortality` (country to ICD-10 cause) | WHO Mortality Database providing national death counts by ICD-10 cause from 2021 onwards, broken down by year, sex, and age group. Country nodes carry population and birth data.                |

Running CODA using Docker
-------------------------

CODA supports two primary run paths:

- local source startup via `./startup.sh`
- containerized deployment via `docker compose up`

For remote hosts and shared environments, prefer the Docker Compose path.

The default containerized deployment includes all three services:

- **kg** (`coda.kg`) - Neo4j knowledge graph
- **inference** (`coda.inference`) - inference API
- **app** (`coda.app`) - web application

### Running without cloning the repository

If you have Docker installed, you can run CODA without cloning the repository.
Download the Compose file and example environment file, then configure your
settings:

```bash
curl -O https://raw.githubusercontent.com/codaproject/coda/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/codaproject/coda/main/.env.example
cp .env.example .env
# Edit .env for either OpenAI or Ollama
docker compose pull
docker compose up --no-build
```

This pulls the pre-built images from Docker Hub and starts the full stack
without requiring the repository's Dockerfiles or build context.

### Running with Docker Compose from source

Source builds include the KG image and therefore require the Git LFS data:

```bash
git lfs install
git lfs pull
cp .env.example .env
# Edit .env for either OpenAI or Ollama
docker compose up --build
```

You can verify the KG files were downloaded before building:

```bash
file kg/*.tsv.gz
```

The files should be reported as gzip-compressed data, not ASCII text.

Both Compose paths publish these endpoints by default:

- CODA web UI at `http://localhost:8000`
- CODA app health check at `http://localhost:8000/health`
- inference API documentation at `http://localhost:5123/docs`
- inference health check at `http://localhost:5123/health`
- Neo4j browser at `http://localhost:7474`
- Neo4j Bolt at `bolt://localhost:7687`

If you change `APP_PORT`, `INFERENCE_PORT`, `NEO4J_HTTP_PORT`, or
`NEO4J_BOLT_PORT`, replace the corresponding port in these URLs.

The Compose path is env-driven. The main runtime variables are:

- `APP_HOST`, `APP_PORT`
- `INFERENCE_HOST`, `INFERENCE_PORT`
- `INFERENCE_LLM_PROVIDER`, `INFERENCE_LLM_MODEL`
- `OPENAI_API_KEY`
- `OLLAMA_BASE_URL` when using Ollama
- `RAG_LLM_PROVIDER`, `RAG_LLM_MODEL`, `RAG_ONTOLOGY`,
  `RAG_USE_RERANKER`
- `CODA_KG_URL` when Neo4j is outside the standard deployment topology
- `NEO4J_HTTP_PORT`, `NEO4J_BOLT_PORT`

`INFERENCE_URL` is wired automatically for Compose and normally does not need to
be set there. The previous `CODA_INFERENCE_URL` name remains supported as a
compatibility alias, but new configurations should use `INFERENCE_URL`.
`CODA_DEVICE` and the `COMPUTE_DEVICE` image build argument are documented in
`.env.example` for GPU deployments.

Minimal `.env` examples:

OpenAI-backed inference:

```bash
OPENAI_API_KEY=your-openai-api-key-here
INFERENCE_LLM_PROVIDER=openai
INFERENCE_LLM_MODEL=gpt-5.4-mini
```

Ollama-backed inference:

```bash
INFERENCE_LLM_PROVIDER=ollama
INFERENCE_LLM_MODEL=llama3.2
```

The RAG grounder has its own provider and model settings. They default to
OpenAI with `gpt-4o-mini` and are used only when RAG is selected in the app:

```bash
RAG_LLM_PROVIDER=openai
RAG_LLM_MODEL=gpt-4o-mini
RAG_ONTOLOGY=icd10
RAG_USE_RERANKER=true
```

For local source startup, Ollama defaults to `http://localhost:11434`.

For Docker Compose, CODA defaults to
`http://host.docker.internal:11434`, because `localhost` inside a container
refers to the container itself. On Linux, Ollama must also listen on an address
reachable from Docker rather than only on `127.0.0.1`. For example:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

For a systemd-managed Ollama installation, add the environment variable to the
service and restart it:

```bash
sudo systemctl edit ollama
```

Add:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

When using `systemctl edit`, place those lines in the override section below
the generated comments. Lines left in the commented instruction area are not
saved, and systemd will report that the new contents are empty.

Then apply the change:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Verify that Ollama is no longer bound only to `127.0.0.1`:

```bash
ss -ltnp | grep ':11434'
```

Only expose that port on a trusted host/network. You can override the endpoint
for either deployment path with `OLLAMA_BASE_URL`.

Before starting CODA, confirm the configured model is available:

```bash
ollama pull llama3.2
ollama list
```

### Building and running the knowledge graph only

To build and run just the CODA knowledge graph, first download its Git LFS
assets:

```bash
git lfs install
git lfs pull
docker build --tag coda.kg:latest -f Dockerfile.kg .
docker run -it -p 7687:7687 -p 7474:7474 coda.kg:latest
```

Running CODA locally with Python
---------------------------------

From a cloned checkout with the package installed, copy `.env.example` to
`.env`, update the values you need, and start the local stack:

```bash
cp .env.example .env
# Edit .env for either OpenAI or Ollama
./startup.sh
```

`startup.sh` uses the same runtime env contract as Docker Compose and launches
the inference service and web app together. With the default ports, open the
web UI at `http://localhost:8000`; the inference health endpoint is
`http://localhost:5123/health`.

Alternatively, you can start the services individually. Start the inference
agent after exporting the values from `.env`:

```bash
set -a
source .env
set +a
python -m coda.inference.agent
```

The Python modules read environment variables, but do not load `.env`
themselves. The commands above export the file before starting the service.
The inference service uses:

- `INFERENCE_HOST` / `INFERENCE_PORT`
- `INFERENCE_LLM_PROVIDER` / `INFERENCE_LLM_MODEL`

You can still override them explicitly:

```bash
python -m coda.inference.agent --host 0.0.0.0 --port 5123 --provider openai --model gpt-5.4-mini
```

Then, in a separate terminal, start the web application:

```bash
set -a
source .env
set +a
python -m coda.app
```

This uses `APP_HOST`, `APP_PORT`, and `INFERENCE_URL` from the exported
environment. The optional RAG grounder also reads the `RAG_*` settings and
`CODA_KG_URL`. With the default ports, open `http://localhost:8000`.

Remote deployment notes
-----------------------

The browser UI uses microphone access via `getUserMedia`. Browsers generally
allow this on `localhost` and on secure origins served over HTTPS.

For remote deployments:

- run the app behind a reverse proxy that terminates TLS
- expose the app over HTTPS
- keep `INFERENCE_URL` pointed at the internal inference service
- restrict the inference and Neo4j ports with host firewall rules or private
  networking

The provided Compose stack runs the services over plain HTTP inside the Docker
network. That is suitable behind a reverse proxy, but not as the final
internet-facing setup by itself. Compose also publishes ports `5123`, `7474`,
and `7687` on the Docker host by default, and the current KG image has Neo4j
authentication disabled. Do not expose those ports directly to an untrusted
network.
