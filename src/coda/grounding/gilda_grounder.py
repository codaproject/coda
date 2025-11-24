import gilda

from . import BaseGrounder


class GildaGrounder(BaseGrounder):
    def ground(self, text: str) -> list:
        return gilda.ground(text)

    def annotate(self, text: str) -> list:
        return gilda.annotate(text)