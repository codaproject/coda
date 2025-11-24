from gilda_grounder import ScoredMatch, Annotation


class BaseGrounder:
    def ground(self, text: str) -> list[ScoredMatch]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def annotate(self, text: str) -> list[Annotation]:
        raise NotImplementedError("This method should be overridden by subclasses.")