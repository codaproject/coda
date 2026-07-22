"""
LLM-based concept extraction from text.
"""
from abc import ABC, abstractmethod
import json
import logging
from typing import Any, Dict, Mapping

from coda.llm_api import LLMClient

logger = logging.getLogger(__name__)

class Extractor(ABC):
    """ Base class for different type of extractors """

    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        ...

class LLMExtractor(Extractor):
    """Extract concepts and supporting evidence from text using LLM."""

    def __init__(
        self,
        concept_type: str,
        prompt_config: Mapping[str, Any],
        llm_client: LLMClient,
    ):
        self.llm_client = llm_client
        self.concept_type = concept_type
        self.config = prompt_config

    def extract(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"Concepts": []}

        concept_type_cap = self.concept_type.capitalize()
        system_prompt = self.config["system_prompt"].format(concept_type=concept_type_cap)
        user_prompt = self.config["user_prompt"].format(
            concept_type=concept_type_cap,
            text=text,
        )

        logger.debug(
            "--- Extractor Input ---\n[System Prompt]\n%s\n\n[User Prompt]\n%s",
            system_prompt, user_prompt,
        )

        try:
            if self.config["use_schema"]:
                response_json = self.llm_client.call_with_schema(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=self.config["schema"],
                    schema_name=f"{self.concept_type}_extraction",
                    max_retries=3,
                    retry_delay=1.0,
                )
                logger.debug("--- Extractor Raw Output ---\n%s", json.dumps(response_json, indent=2))
                if response_json.get("api_failed", False):
                    logger.error("LLM API call failed")
                    return {"Concepts": []}
                concepts_raw = response_json.get("Concepts", [])
            else:
                response_text = self.llm_client.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                try:
                    logger.debug("--- Extractor Raw Output ---\n%s", json.dumps(json.loads(response_text), indent=2))
                except (json.JSONDecodeError, TypeError):
                    logger.debug("--- Extractor Raw Output ---\n%s", response_text)
                try:
                    concepts_raw = json.loads(response_text)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to parse LLM response as JSON")
                    return {"Concepts": []}

            if not isinstance(concepts_raw, list):
                logger.warning("Unexpected response: concepts is not a list")
                return {"Concepts": []}

            concept_field = self.config.get("concept_key")
            evidence_field = self.config.get("supporting_evidence_key")
            concepts = []
            for c in concepts_raw:
                if not isinstance(c, dict):
                    continue
                concept = c.get(concept_field, "")
                evidence = c.get(evidence_field, [])
                if concept:
                    concepts.append({"Concept": concept, "Supporting_Evidence": evidence})
            return {"Concepts": concepts}

        except Exception as e:
            logger.error(f"Failed to extract concepts: {e}", exc_info=True)
            return {"Concepts": []}


class Hunflair2Extractor(Extractor):
    """ Extracts concepts and diseases from text using the Hunflair2 disease NER """

    def __init__(
            self,
            concept_type: str
    ):
        from flair.nn import Classifier
        from wtpsplit_lite import SaT

        self.concept_type = concept_type
        self.tagger = Classifier.load("hunflair2")
        self.splitter = SaT("sat-3l-sm") # This is a more robust sentence splitter than Hunflair

    def extract(self, text: str) -> Dict[str, Any]:
        """ Runs Hunflair NER pipeline.

        Returns the standard ``{"Concepts": [...]}`` payload. In addition to the
        contract-required ``Concept``/``Supporting_Evidence`` keys, each concept
        carries its ``sentence_index`` and the ``start``/``end`` character offsets
        of the span *within its sentence*. The returned dict also includes an
        ordered ``Sentences`` list. These extra keys are additive and ignored by
        consumers that only rely on the ``Extractor`` contract.
        """
        # List with the output of the NER
        ret = []
        # Ordered list of the sentences produced by the splitter
        sentences: list[str] = []

        from flair.data import Sentence

        # For now, we only do diseases
        if self.concept_type == "disease":
            sents = [Sentence(s) for s in self.splitter.split(text)]
            self.tagger.predict(sentences=sents)

            for sentence_index, sent in enumerate(sents):
                sentences.append(sent.text)
                for label in sent.get_labels("ner"):
                    if label.value != "Disease":
                        continue
                    span = label.data_point
                    ret.append({
                        "Concept": span.text,
                        "Supporting_Evidence": [sent.text],
                        "sentence_index": sentence_index,
                        "start": span.start_position,
                        "end": span.end_position,
                    })

        return {"Concepts": ret, "Sentences": sentences}
