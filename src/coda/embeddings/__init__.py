"""Shared, reusable embedding encoders.

Framework-neutral embedding utilities usable across the codebase (the SNOMED KG
exporter in :mod:`coda.kg`, the event grounder in :mod:`coda.grounding`, etc.)
without pulling in gilda, chromadb, or a Neo4j driver.
"""
