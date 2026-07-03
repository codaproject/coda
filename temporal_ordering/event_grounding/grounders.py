from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Optional, Sequence

from temporal_ordering.event_grounding.sapbert_utils import load_semantic_grounder


from .snomed_rf2_utils import make_gilda_grounder
from ..modeling import EventTimeline, GroundingTerm, TimelineComponent, TimelineLayer, Event

from pathlib import Path
import logging
logger = logging.getLogger(__name__)


class Grounder(ABC):
    """ Base class for the event grounder """

    @abstractmethod
    def ground_event(self, event:Event, context:Optional[str]) -> Event:
        ...

    def __call__(self, timeline: EventTimeline) -> EventTimeline:
        """ Grounds events in an event timeline with a set of predefined grounders """

        return EventTimeline(
            components= [
                TimelineComponent(
                    layers= [
                        TimelineLayer(
                            events= [self.ground_event(e, context=None) for e in l.events]
                        )
                        for l in c.layers
                    ]
                ) for c in timeline.components
            ]
        )
    
class SequentialGrounder(Grounder):
    """ Applies grounders sequentially until one grounder returns a match """

    def __init__(self, grounders:Sequence[Grounder]):
        self._grounders = grounders

    def ground_event(self, event: Event, context: Optional[str]) -> Event:
        for grounder in self._grounders:
            # Try a grounder
            event = grounder.ground_event(event, context)
            # If the event was grounded successfully, shortcircuit the loop
            if event.grounding:
                return event
        # Otherwise, return event, ungrounded
        return event
    

class GildaGrounder(Grounder):

    def __init__(self, data_path: Path) -> None:
        logger.info("Creating GILDA grounder")
        self.gilda = make_gilda_grounder(data_path)

    def ground_event(self, event:Event, context:Optional[str]) -> Event:
        match = self.gilda.ground_best(event.text, context)
        # If the grounder provides a result, add it, otherwise, return the original event        
        if match:
            return replace(event,
                                  grounding= GroundingTerm(
                                        db=match.term.db,
                                        id=match.term.id,
                                        name=match.term.text
                                    )
                                )

        return event
    
class SapBERTGrounder(Grounder):

    def __init__(self) -> None:
        self._query_util = load_semantic_grounder()

    def ground_event(self, event: Event, context:Optional[str]) -> Event:
        if context:
            query = f"{event.text} - {context}"
        else:
            query = event.text

        result = self._query_util.ground([query], top_k=1)[0]

        if result:
            candidate = result[0]
            return replace(event,
                           grounding= GroundingTerm(
                               db= candidate.db,
                               id= candidate.identifier,
                                name= candidate.name
                           ))

        return event
    
    