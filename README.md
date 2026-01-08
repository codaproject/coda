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
