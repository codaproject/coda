import json
import logging
from pathlib import Path
from typing import Dict

from .base import DefinitionSource

logger = logging.getLogger(__name__)


class ICD11DefinitionSource(DefinitionSource):
    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def namespace(self) -> str:
        return "icd11"

    def load_definitions(self) -> Dict[str, str]:
        raise NotImplementedError("ICD11 definitions are not yet implemented")
