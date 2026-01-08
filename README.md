CODA: Cause of Death Determination Assistant
============================================

This repository implements the Cause of Death Determination Assistant (CODA)
application which automates cause of death determination via an AI-assisted
interview process.

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
| **ACME** | `icd10`: Disease codes | `is_a` (hierarchical) | Automated Classification of Medical Entities. Alternative ICD-10 structure for cause of death determination. |
| **PHMRC** | `phmrc`: Verbal autopsy terms | `maps_to` (ICD-10 to PHMRC) | Population Health Metrics Research Consortium terms used in VA data collection, mapped to ICD-10 codes. |
| **WHO VA** | `who.va`: VA cause categories | `is_a` (hierarchy)<br>`maps_to` (ICD-10 to WHO VA) | WHO Verbal Autopsy cause categories with hierarchical structure and ICD-10 code range mappings. |
| **ProbBase** | `who.va.q`: VA interview questions | `probbase_rel` (questions to causes) | InterVA probability base linking VA interview questions to WHO VA causes with probability values. |
| **HPO** | `hp`: Phenotypes<br>`omim`: Diseases | `has_phenotype` (disease to phenotype) | Human Phenotype Ontology annotations linking diseases to clinical phenotypes with evidence, frequency, and onset metadata. |

Dockerization
-------------
The Dockerfile builds the CODA Knowledge Graph in a neo4j graph database.

```bash
docker build --tag coda:latest .
```

the KG container can then be run with

```bash
docker run -it -p 7687:7687 -p 7474:7474 coda:latest
```
