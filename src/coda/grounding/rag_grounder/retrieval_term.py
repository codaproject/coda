from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RetrievalTerm:
    id: str
    name: str
    definition: Optional[str] = None
    synonyms: Optional[List[str]] = None


@dataclass
class TermStore:
    terms: List[RetrievalTerm]
    embeddings: np.ndarray
