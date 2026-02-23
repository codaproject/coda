import json
import logging
from pathlib import Path
from typing import Dict, Optional

from .base import DefinitionSource

logger = logging.getLogger(__name__)


def _extract_definitions_from_icd10_data(data: dict) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for code, obj in data.items():
        if not isinstance(obj, dict):
            continue
        definition = obj.get("definition") or obj.get("definition_text", "")
        if not isinstance(definition, str) or not definition.strip():
            continue
        definition = definition.strip()
        result[f"icd10:{code}"] = definition
    return result


class ICD10DefinitionSource(DefinitionSource):
    def __init__(self, path: Optional[Path] = None):
        self.path = path

    @property
    def namespace(self) -> str:
        return "icd10"

    def load_definitions(self) -> Dict[str, str]:
        if self.path is not None and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            try:
                from openacme.icd10.map_definitions import map_icd10_to_definitions
            except ImportError as e:
                raise ImportError(
                    "ICD-10 definitions require openacme. Install with: pip install openacme"
                ) from e
            output_json = str(self.path) if self.path else None
            data = map_icd10_to_definitions(output_json=output_json)

        result = _extract_definitions_from_icd10_data(data)
        logger.info(f"Loaded {len(result) // 2} ICD-10 definitions")
        return result
