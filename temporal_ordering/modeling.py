from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

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
    time_break_marker:Optional[str]    #: the explicit temporal expression string, or null
    time_break_category:Optional[TimeBreakCategory]   #: one of the categories listed above, or null

@dataclass
class StatementGraphEdge:
    source: int
    dest: int

@dataclass
class StatementGraph:
    nodes: list[StatementGraphNode]
    edges: list[StatementGraphEdge]

@dataclass
class Event:
    source_statement: int # Index of the source statement in the graph
    type_: ClinicalEvent  # Event type
    text: str   # Event text

@dataclass
class EventTimeline:
    """ The timeline has different components, that correspond to the connected components in the temporal order graph.
      Each inner list represents events happening in parallel"""
    components:list[list[Event]]

@dataclass
class VATimeline:
    statement_graph: StatementGraph
    event_timeline: EventTimeline
    va_narrative: Optional[str]