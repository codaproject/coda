import sys
from dataclasses import dataclass

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # Python 3.10 fallback
    from enum import Enum

    class StrEnum(str, Enum):
        """Minimal ``enum.StrEnum`` backport for Python 3.10.

        Mixing ``str`` with ``Enum`` gives value-based construction; delegating
        ``__str__`` to ``str`` makes ``str(member)`` return the member's value,
        matching the 3.11+ ``StrEnum`` behaviour the rest of the code relies on.
        """

        __str__ = str.__str__

from gilda import Term

class ClinicalEvent(StrEnum):
    SYMPTOM = "symptom"                       
    DISEASE = "disease"                                     
    CARE_SEEKING = "care_seeking_event"   
    MEDICAL_INTERVENTION = "medical_intervention" 
    DIAGNOSTIC = "diagnostic"           
    DETERIORATION = "deterioration"        
    HEALTH_EVENT = "health_event"         
                           
class TimeBreakCategory(StrEnum):
    ABSOLUTE = "absolute"       
    DURATION = "duration"       
    SEQUENCING = "sequencing"     
    RELATIVE_TIME = "relative_time"  
    ONSET = "onset"          
    PROXIMITY_DEATH = "proximity_death"
    FREQUENCY = "frequency"                       


@dataclass
class StatementGraphNode:
    index: int                          # : integer starting at 1
    statement: str                      # verbatim text of the statement
    time_break_marker: str | None    #: the explicit temporal expression string, or null
    time_break_category: TimeBreakCategory | None   #: one of the categories listed above, or null

@dataclass
class StatementGraphEdge:
    source: int
    dest: int

@dataclass
class StatementGraph:
    nodes: list[StatementGraphNode]
    edges: list[StatementGraphEdge]

@dataclass
class GroundingTerm:
    db: str
    id: str
    name: str

@dataclass
class Event:
    source_statement: int # Index of the source statement in the graph
    type_: ClinicalEvent  # Event type
    text: str   # Event text
    grounding: GroundingTerm | None # Grounding term, when available

@dataclass
class TimelineLayer:
    """Events at the same chronological depth within a component.

    The narrative gives no ordering between them, so they are treated as concurrent."""
    events: list[Event]

@dataclass
class TimelineComponent:
    """An independent narrative thread (a connected component of the statement DAG).

    Its layers are ordered chronologically."""
    layers: list[TimelineLayer]

@dataclass
class EventTimeline:
    """One timeline per connected component of the temporal-order graph."""
    components: list[TimelineComponent]

@dataclass
class VATimeline:
    statement_graph: StatementGraph
    event_timeline: EventTimeline
    va_narrative: str | None