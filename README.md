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

Running CODA using Docker
-------------------------

### Running with Docker compose

The easiest way to run CODA is with Docker compose, which starts all services:

```bash
docker-compose up --build
```

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
