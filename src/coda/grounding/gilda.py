from . import BaseGrounder


class GildaGrounder(BaseGrounder):
    def __init__(self, gilda_instance):
        self.gilda = gilda_instance

    def ground(self, text: str) -> list:
        return self.gilda.ground(text)

    def annotate(self, text: str) -> list:
        return self.gilda.annotate(text)