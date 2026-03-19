import gilda

from . import BaseGrounder

DEFAULT_NAMESPACES = ['MESH', 'DOID', 'HP']


class GildaGrounder(BaseGrounder):
    """Wrapper for using Gilda as the grounding system."""

    def __init__(self, namespaces: list = None):
        super().__init__()
        self.namespaces = DEFAULT_NAMESPACES \
            if namespaces is None else namespaces

    def ground(self, text: str) -> list:
        return gilda.ground(text, namespaces=self.namespaces)

    def annotate(self, text: str) -> list:
        return gilda.annotate(text, namespaces=self.namespaces)