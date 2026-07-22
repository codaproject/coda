"""Shared, framework-neutral SNOMED CT utilities.

This subpackage holds parsing primitives for the SNOMED CT RF2 release format
that are useful across the codebase (the GILDA-based event grounder in
:mod:`coda.grounding` and the knowledge-graph exporter in :mod:`coda.kg`). It
deliberately depends on nothing beyond the standard library so either consumer
can import it without pulling in gilda, pandas, or a Neo4j driver.
"""
