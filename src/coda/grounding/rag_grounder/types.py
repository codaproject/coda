from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RetrievalTerm:
    id: str
    name: str
    definition: Optional[str] = None
    synonyms: Optional[List[str]] = None
