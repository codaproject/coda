CODA: Cause of Death Determination Assistant
============================================

This repository implements the Cause of Death Determination Assistant (CODA)
application which automates cause of death determination via an AI-assisted
interview process.

Installation
------------

CODA requires Python 3.9 or newer. We recommend installing into a virtual
environment to keep dependencies isolated from your system Python (this is
especially important on macOS, where `pip install` against the system
`python3` is discouraged).

```bash
git clone https://github.com/codaproject/coda.git
cd coda
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m nltk.downloader stopwords
```

Activate the environment (`source .venv/bin/activate`) in every new shell
before running the commands below. On Windows, use `.venv\Scripts\activate`
instead.

The `nltk.downloader stopwords` step fetches an NLTK corpus that `gilda`
loads at import time. It is stored under `~/nltk_data` and only needs to be
run once per machine.

Alternatively, install directly from GitHub (into an active virtual
environment):

```bash
pip install git+https://github.com/codaproject/coda.git
python -m nltk.downloader stopwords
```

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

| Source | Node Types | Edge Types | Semantics |
|--------|-----------|------------|-----------|
| **ICD-10** | `icd10`: Disease classification codes | `is_a` (hierarchical) | WHO International Classification of Diseases, 10th revision. Provides standardized disease codes with hierarchical relationships. |
| **ICD-11** | `icd11`: Disease classification codes | `is_a` (hierarchy)<br>`maps_to` (ICD-11 to ICD-10) | WHO ICD-11 revision with mappings to ICD-10. Enables cross-version code translation. |
| **ACME** | `icd10`: ICD-10 codes and code ranges | `causes` (causal relationships from Table D)<br>`part_of_range` (code to range membership) | CDC's WHO ICD-10 ACME decision tables encoding causal relationships between diseases for underlying cause of death determination. Sourced from [openacme](https://github.com/gyorilab/openacme). |
| **PHMRC** | `phmrc`: Verbal autopsy terms | `maps_to` (ICD-10 to PHMRC) | Population Health Metrics Research Consortium terms used in VA data collection, mapped to ICD-10 codes. |
| **WHO VA** | `who.va`: VA cause categories | `is_a` (hierarchy)<br>`maps_to` (ICD-10 to WHO VA) | WHO Verbal Autopsy cause categories with hierarchical structure and ICD-10 code range mappings. |
| **ProbBase** | `who.va.q`: VA interview questions | `probbase_rel` (questions to causes) | InterVA probability base linking VA interview questions to WHO VA causes with probability values. |
| **HPO** | `hp`: Phenotypes<br>`omim`: Diseases | `has_phenotype` (disease to phenotype) | Human Phenotype Ontology annotations linking diseases to clinical phenotypes with evidence, frequency, and onset metadata. |
| **MeSH** | `mesh`: Diseases, pathogens, geographic locations | `isa` (hierarchical) | Medical Subject Headings hierarchy filtered to diseases, pathogens, and geographic locations. |
| **WDI** | `wdi`: Development/health indicators | `has_indicator` (country to indicator) | World Bank World Development Indicators and World Health Indicators linked to country nodes, with time-series data stored as year-value mappings on edges. |

Running CODA using Docker
-------------------------

### Running without cloning the repository

If you have Docker installed, you can run CODA without cloning the repository.
Download the Docker Compose file and the example environment file, then configure
your settings:

```bash
curl -O https://raw.githubusercontent.com/codaproject/coda/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/codaproject/coda/main/.env.example
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
docker-compose up
```

This pulls pre-built images from Docker Hub and starts all services.

### Running with Docker compose

The easiest way to run CODA is with Docker compose, which starts all services:

```bash
docker-compose up
```

optionally with --build as an additional argument to build the images locally.

This starts three services:
- **kg** (`coda.kg`) - Neo4j knowledge graph on ports 7474 (browser) and 7687 (bolt)
- **inference** (`coda.inference`) - Inference agent on port 5123
- **app** (`coda.app`) - Web application on port 8000

Access the web UI at http://localhost:8000 and Neo4j browser at http://localhost:7474.

### Building and running the knowledge graph only

To build and run just the CODA knowledge graph:

```bash
docker build --tag coda.kg:latest -f Dockerfile.kg .
docker run -it -p 7687:7687 -p 7474:7474 coda.kg:latest
```

Running CODA locally with Python
---------------------------------

After installing the package, you can run CODA locally. Make sure your
`OPENAI_API_KEY` environment variable is set.

The quickest way is to use the startup script, which launches both the inference
agent and web application and reports when the system is ready:

```bash
./startup.sh
```

Alternatively, you can start the services individually. Start the inference agent:

```bash
python -m coda.inference.agent
```

This runs the inference server on port 5123. You can specify the LLM provider
and model:

```bash
python -m coda.inference.agent --provider openai --model gpt-5.4-mini
```

Then, in a separate terminal, start the web application:

```bash
python -m coda.app
```

The web UI will be available at http://localhost:8000.

Using a local LLM with Ollama
-----------------------------

CODA can run inference entirely on-device through
[Ollama](https://ollama.com), which is a practical default for development,
offline demos, and deployments where sending dictation to a cloud provider is
undesirable. No `OPENAI_API_KEY` is required in this mode.

### 1. Install Ollama

On macOS with Homebrew:

```bash
brew install ollama
```

On Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On Windows: download the installer from
[https://ollama.com/download](https://ollama.com/download).

### 2. Start the Ollama daemon

The daemon exposes a local HTTP API on `http://127.0.0.1:11434` and must be
running whenever CODA performs inference.

On macOS (runs now and on every login):

```bash
brew services start ollama
```

Or as a one-shot foreground process (any platform):

```bash
ollama serve
```

Verify it's reachable:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

### 3. Pull a model

Pick a model sized for your machine. `llama3.2` (≈ 2 GB, 3 B parameters) is
fast on any modern laptop and accurate enough for CODA's dictation-length
inputs. `gpt-oss:20b` (≈ 13 GB) is more capable but requires substantial
memory.

```bash
ollama pull llama3.2          # recommended default
# or
ollama pull gpt-oss:20b       # larger, slower, more capable
```

List available models at any time:

```bash
ollama list
```

### 4. Point CODA's inference agent at Ollama

Start the inference agent with the Ollama provider and the model you pulled:

```bash
python -m coda.inference.agent --provider ollama --model llama3.2
```

If you use the startup script, edit `startup.sh` so the inference line reads:

```bash
python -m coda.inference.agent --provider ollama --model llama3.2 &
```

Then run `./startup.sh` as usual. The web UI at http://localhost:8000 will
display live cause-of-death probabilities driven entirely by the local model.

### Troubleshooting

- **`model 'X' not found (status code: 404)`** — the inference agent is
  configured for a model that isn't pulled. Run `ollama pull X` or pass
  `--model <name-of-a-pulled-model>` when starting the agent.
- **`Failed to connect to Ollama`** — the daemon isn't running. Start it with
  `brew services start ollama` (macOS) or `ollama serve` (any platform).
- **Inference is slow** — smaller models respond faster. Try `llama3.2`
  (3 B) or `llama3.2:1b` (1 B) instead of larger variants.
- **Freeing disk space** — `ollama rm <model>` removes a pulled model;
  `brew uninstall ollama` (macOS) removes the daemon itself.

### Stopping Ollama

```bash
brew services stop ollama     # macOS, keeps Ollama installed
# or
pkill ollama                   # any platform, if running as `ollama serve`
```
