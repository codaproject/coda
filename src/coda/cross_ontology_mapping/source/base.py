from abc import ABC, abstractmethod
from typing import Dict


class DefinitionSource(ABC):
    @property
    @abstractmethod
    def namespace(self) -> str:
        ...

    @abstractmethod
    def load_definitions(self) -> Dict[str, str]:
        ...
