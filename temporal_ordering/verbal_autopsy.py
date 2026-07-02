from collections import defaultdict
import json
import re
from typing import Optional, cast

from langchain.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import networkx as nx

from dataclasses import asdict

from .modeling import ClinicalEvent, Event, EventTimeline, StatementGraph, StatementGraphEdge, StatementGraphNode, TimeBreakCategory, VATimeline



# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

STATEMENT_SYSTEM_PROMPT = """\
You are an expert clinical analyst specialising in verbal autopsies.
Your output must be valid JSON and nothing else — no markdown fences, no preamble, no commentary.
"""

STATEMENT_HUMAN_PROMPT = """\
The following text is the narration of a health event, also called a verbal autopsy.

Your task is to:
1. Split the text into individual statements and keep the verbatim text of each.
2. Arrange the statements as a PARTIAL ORDER — i.e. a directed acyclic graph (DAG) where
   an edge from statement A to statement B means "A happened before B".
3. If the temporal order between two statements is ambiguous, do NOT add an edge between
   them — they are considered concurrent (parallel).
4. For each statement, if it contains an EXPLICIT temporal expression that anchors or shifts
   the event in time, record that expression verbatim in "time_break_marker" and classify it
   in "time_break_category". Use the following (non-exhaustive) categories as a guide — treat
   them as intuitions, not a closed list, and generalise to similar phrasings:

     - "absolute"        : a calendar/clock anchor that fixes an absolute point in time
                           (e.g. "on 3 March", "in 2019", "last Tuesday", "at 6 am",
                           "during the rainy season")
     - "duration"        : how long a state or event lasted
                           (e.g. "for three days", "for two weeks", "over several months",
                           "throughout the night")
     - "sequencing"      : ordering cues that chain events without a fixed time
                           (e.g. "then", "after that", "subsequently", "later", "afterwards",
                           "meanwhile", "finally")
     - "relative_time"   : a shift measured from a previous reference point
                           (e.g. "the next day", "two days later", "a week earlier",
                           "the following morning", "soon after")
     - "onset"           : markers of when a condition began or how abruptly it arose
                           (e.g. "started", "began", "suddenly", "initially", "at first",
                           "gradually", "from birth")
     - "proximity_death" : timing expressed relative to the death itself
                           (e.g. "days before death", "before she died", "shortly before
                           passing", "on the day of death", "in her final hours")
     - "frequency"       : recurrence or repetition over time
                           (e.g. "every day", "intermittently", "on and off", "twice")

   If a statement carries no explicit temporal expression — i.e. its ordering is implicit,
   entailed by the narrative flow alone — set both "time_break_marker" and
   "time_break_category" to null. If an expression fits more than one category, pick the one
   that best captures its primary temporal function.

Return a single JSON object with exactly two keys:

"nodes" — a list of objects, one per statement, each with:
  - "index"               : integer starting at 1
  - "statement"           : verbatim text of the statement
  - "time_break_marker"   : the explicit temporal expression string, or null
  - "time_break_category" : one of the categories listed above, or null

"edges" — a list of objects representing the partial order, each with:
  - "from" : index of the earlier statement
  - "to"   : index of the later statement

Rules:
- Every node must appear in at least one edge unless it is the only node.
- Do NOT add an edge when the ordering is genuinely uncertain.
- The graph must be a DAG (no cycles).

Verbal autopsy text:
{verbal_autopsy}
"""

ENTITY_SYSTEM_PROMPT = """\
You are an expert clinical analyst specialising in verbal autopsies.
Your output must be valid JSON and nothing else — no markdown fences, no preamble, no commentary.
"""

ENTITY_HUMAN_PROMPT = """\
Below are numbered statements extracted from a verbal autopsy narrative.

For each statement, identify ALL medical entity mentions and classify them into exactly
one of these categories. Treat the examples as intuitions, not a closed list — generalise
to similar mentions:

  - "symptom"               : clinical symptoms reported or observed
                              (e.g., fever, cough, chest pain, weight loss, swelling)
  - "disease"               : named diseases, conditions, or diagnoses
                              (e.g., tuberculosis, hypertension, cirrhosis, diabetes)
  - "care_seeking_event"    : actions taken to obtain care, or contact with the health system
                              (e.g., visited a clinic, went to hospital, consulted a doctor,
                              saw a traditional healer, was referred, sought treatment,
                              called an ambulance, admitted to a ward)
  - "medical_intervention"  : treatments, procedures, surgeries, or medications administered
                              (e.g., surgery, chemotherapy, antibiotics, IV fluids, dressing)
  - "diagnostic"            : diagnostic tests, procedures, or clinical findings
                              (e.g., blood test, X-ray, biopsy, ECG, kidney function results)
  - "deterioration"         : worsening of condition or clinical decline over time
                              (e.g., condition worsened, grew weaker, symptoms intensified,
                              stopped eating, became bedridden, lost consciousness, no
                              improvement despite treatment)
  - "health_event"          : other significant health occurrences not covered above —
                              including DEATH at the end
                              (e.g., seizure, fall, collapse, onset of labour, death)

Boundary guidance:
- Prefer "care_seeking_event" for the ACT of seeking/accessing care, and
  "medical_intervention" for the treatment actually delivered.
- Prefer "deterioration" when the mention conveys a CHANGE for the worse; use "symptom"
  for a static symptom mention and "health_event" for a discrete acute occurrence.

Return a JSON object with a single key "statement_entities" — a list with one entry PER
statement (include every statement, even those with no entities), each entry having:
  - "statement_index" : integer index of the statement
  - "entities"        : list of objects, each with:
      - "type" : one of the five categories above
      - "text" : verbatim mention as it appears in the statement

Statements:
{statements_json}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Chain builders
# ─────────────────────────────────────────────────────────────────────────────

def _strip_thinking(msg) -> str:
    """Remove <think>…</think> reasoning blocks emitted by thinking-mode models."""
    content = msg.content if hasattr(msg, "content") else str(msg)
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def build_temporal_statements_ordering_chain(llm: BaseChatModel):
    """Chain that turns an `verbal autopsy` string into a partial-order graph JSON."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", STATEMENT_SYSTEM_PROMPT),
        ("human", STATEMENT_HUMAN_PROMPT),
    ])
    return prompt | llm | RunnableLambda(_strip_thinking) | JsonOutputParser()


def build_envent_extractor_chain(llm):
    """Chain that maps a JSON list of statements to typed clinically relevant events."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_SYSTEM_PROMPT),
        ("human", ENTITY_HUMAN_PROMPT),
    ])
    return prompt | llm | RunnableLambda(_strip_thinking) | JsonOutputParser()


def build_temporal_order(
        va_narrative:str,
        ordering_chain:Runnable,
        events_chain:Runnable,
        include_va_in_output:bool = False) -> VATimeline:
    
    """ Runs the chains to build a temporal ordering of the statements in a verbal autopsy and then extracts clinically relevant events """
    
    # Get the statements order
    graph = ordering_chain.invoke({"verbal_autopsy": va_narrative})

    # Get the events for each statement
    statements_json = [{"index": n["index"], "statement": n["statement"]} for n in graph["nodes"]],
    events_data = events_chain.invoke({"statements_json": statements_json})
    
    # Build a timeline of the events
    timeline = build_event_timeline(graph, events_data)
    
    ret = {
        "graph":              graph,
        "statement_entities": events_data["statement_entities"],
        "event_timeline":     timeline,
    }

    # Only return the input in the output if requested
    if include_va_in_output:
            ret["verbal_autopsy"] = va_narrative

    # Make this dictionary into our data model
    structured_output = build_data_model(ret)

    return structured_output




def _make_event_timelines(timeline) -> EventTimeline:
    components = {}
    for data_layer in timeline:
        c = data_layer["component"]
        if c not in components:
            components[c] = {}
        component = components[c]

        l = data_layer["layer"]
        if l not in component:
            component[l] = []
        layer = component[l]

        layer.append(
            [Event(
                source_statement=event['source_statement'],  
                type_= ClinicalEvent(event['type']),
                text= event['text']
            )
            for event in data_layer['events']]
        )

    # Make sure we don't have unordered lists in the event timeline
    r_components = []
    for k in sorted(components.keys()):
        c = components[k]
        r_layers = []
        for v in sorted(c.keys()):
            r_layers.append(c[v])
        r_components.append(r_layers)

    return EventTimeline(components=r_components)


def build_data_model(data) -> VATimeline:
    graph = data['graph']
    
    ret = VATimeline(
        va_narrative = data.get("verbal_autopsy"),
        
        statement_graph = StatementGraph(
            nodes = [
                StatementGraphNode(
                    index = n['index'],
                    statement=n['statement'],
                    time_break_category=TimeBreakCategory(n['time_break_category']) if n['time_break_category'] else None,
                    time_break_marker=n['time_break_marker']
                )
                for n in graph['nodes']
            ],
            edges = [StatementGraphEdge(source=e['from'], dest=e['to']) for e in graph['edges']],
        ),

        event_timeline= _make_event_timelines(data['event_timeline'])
    )

    return ret



def build_event_timeline(graph: dict, entity_data: dict) -> list[dict]:
    """Merge statement-level ordering with extracted entities into a layered timeline.

    Returns a list of dicts {component, layer, source_statements, events}, ordered
    by (component, layer).

    The statement DAG can split into several disconnected sub-graphs — independent
    narrative threads the LLM never linked. Each weakly-connected component becomes
    its own timeline (its own ``component`` id) with layers numbered from 0, so
    threads are never merged onto a shared layer just because they sit at the same
    depth from their respective roots. Components are ordered by their lowest
    statement index, which tracks narration order.

    Within a component, every event stays in the layer of the statement it came
    from, so a death sits at its natural chronological position rather than being
    forced to the end. Anything the DAG places after the death (e.g. a post-mortem
    exam) follows in a later layer, and anything that happened before the death —
    even if narrated afterwards — lands earlier, because ordering is governed
    entirely by the DAG's edges, not by narration order.
    """

    # Build a networkx DiGraph from the statement-level graph dict.
    G = nx.DiGraph()
    for node in graph["nodes"]:
        G.add_node(node["index"])
    for edge in graph.get("edges", []):
        G.add_edge(edge["from"], edge["to"])

    events_by_stmt: dict[int, list[dict]] = {
        entry["statement_index"]: [
            {**e, "source_statement": entry["statement_index"]}
            for e in entry.get("entities", [])
        ]
        for entry in entity_data.get("statement_entities", [])
    }

    components = sorted(nx.weakly_connected_components(G), key=min)


    timeline: list[dict] = []
    for comp_idx, comp_nodes in enumerate(components):
        sub = cast(nx.DiGraph, G.subgraph(comp_nodes))

        # Topological sort
        layers = {}
        for node in nx.topological_sort(sub):
            preds = list(sub.predecessors(node))
            layers[node] = 0 if not preds else max(layers[p] for p in preds) + 1

        layer_stmts: dict[int, list[int]] = defaultdict(list)
        for stmt_idx, layer in layers.items():
            layer_stmts[layer].append(stmt_idx)

        for layer_idx in sorted(layer_stmts):
            stmt_indices = sorted(layer_stmts[layer_idx])
            events = [e for s in stmt_indices for e in events_by_stmt.get(s, [])]
            timeline.append({
                "component": comp_idx,
                "layer": layer_idx,
                "source_statements": stmt_indices,
                "events": events,
            })

    return timeline

def _build_llm(model:str, base_url:str, api_key:Optional[str] = None, bearer_token:Optional[str] = None) -> ChatOpenAI:
    """ Instantiates a ChatModel to send messages to the inference service """
    # In case that we're dealing with AWS Bedrock, we need to handle it a little differently
    if "bedrock" in base_url.lower():
        if not bearer_token:
            raise ValueError("Bearer token needed for bedrock api access")
        # Bedrock Mantle accepts either Authorization or x-api-key, not both.
        # ChatOpenAI sends api_key as Authorization: Bearer, which is sufficient.
        return ChatOpenAI(
            base_url=base_url,
            api_key=SecretStr(bearer_token),
            model=model,
            temperature=0.0,
        )
    else:
        return ChatOpenAI(
            base_url=base_url,
            api_key= SecretStr(api_key) if api_key else None,
            model=model,
            temperature=0.0,
        )