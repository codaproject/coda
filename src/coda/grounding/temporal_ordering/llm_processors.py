from collections import defaultdict
import json
import re
from typing import Any, cast

from openai import OpenAI
import networkx as nx


from .modeling import ClinicalEvent, Event, EventTimeline, StatementGraph, StatementGraphEdge, StatementGraphNode, TimeBreakCategory, TimelineComponent, TimelineLayer, VATimeline


from .event_grounding.grounders import GildaGrounder, Grounder, SapBERTGrounder, SequentialGrounder
from .event_grounding.snomed_kg_utils import snomedct_terms_available

# Event grounding is optional: both the GILDA and SapBERT grounders source their
# SNOMED CT terms from the CODA KG, so grounding is wired up only when the KG was
# built with SNOMED CT (a licensed resource). When the KG has no `snomedct`
# nodes, `grounder` stays None and events are left ungrounded.
if snomedct_terms_available():
    gilda_grounder = GildaGrounder()
    sapbert_grounder = SapBERTGrounder()
    grounder: SequentialGrounder | None = SequentialGrounder([gilda_grounder, sapbert_grounder])
else:
    grounder = None

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

EVENT_SYSTEM_PROMPT = """\
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

def _strip_thinking(content: str) -> str:
    """Remove <think>…</think> reasoning blocks emitted by thinking-mode models."""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _parse_json(content: str) -> Any:
    """Parse a JSON payload from a model response, tolerating markdown fences."""
    text = _strip_thinking(content)
    # Some models wrap JSON in ```json … ``` fences despite instructions not to.
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


def _complete_json(client: OpenAI, model: str, system_prompt: str, human_prompt: str) -> Any:
    """Call the chat completions API and parse the JSON response."""
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ],
    )
    return _parse_json(response.choices[0].message.content or "")


def build_temporal_order(
        va_narrative:str,
        client:OpenAI,
        model:str,
        include_va_in_output:bool = False) -> VATimeline:

    """ Runs the LLM to build a temporal ordering of the statements in a verbal autopsy and then extracts clinically relevant events """

    # Get the statements order
    graph = _complete_json(
        client, model, STATEMENT_SYSTEM_PROMPT,
        STATEMENT_HUMAN_PROMPT.format(verbal_autopsy=va_narrative),
    )

    # Get the events for each statement
    statements_json = [{"index": n["index"], "statement": n["statement"]} for n in graph["nodes"]]
    events_data = _complete_json(
        client, model, EVENT_SYSTEM_PROMPT,
        ENTITY_HUMAN_PROMPT.format(statements_json=statements_json),
    )

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
    structured_output = build_data_model(ret, grounder)

    return structured_output




def _make_event_timelines(timeline) -> EventTimeline:
    # ``timeline`` is already ordered by (component, layer) — see build_event_timeline —
    # so grouping layers by component preserves chronological order within each.
    components: dict[int, list[TimelineLayer]] = defaultdict(list)
    for data_layer in timeline:
        events = [
            Event(
                source_statement=event["source_statement"],
                type_=ClinicalEvent(event["type"]),
                text=event["text"],
                grounding=None
            )
            for event in data_layer["events"]
        ]
        components[data_layer["component"]].append(TimelineLayer(events))

    return EventTimeline(
        components=[
            TimelineComponent(layers=components[c]) for c in sorted(components)
        ]
    )


def build_data_model(data, grounder: Grounder | None = None) -> VATimeline:
    graph = data['graph']
    
    event_timeline = _make_event_timelines(data['event_timeline'])
    # Ground events if grounder is provided
    if grounder:
        event_timeline = grounder(event_timeline)

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

        event_timeline= event_timeline
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

def build_llm(base_url:str, api_key: str | None = None, bearer_token: str | None = None) -> OpenAI:
    """ Instantiates an OpenAI client to send messages to the inference service """
    # In case that we're dealing with AWS Bedrock, we need to handle it a little differently
    if "bedrock" in base_url.lower():
        if not bearer_token:
            raise ValueError("Bearer token needed for bedrock api access")
        # Bedrock Mantle accepts either Authorization or x-api-key, not both.
        # The OpenAI client sends api_key as Authorization: Bearer, which is sufficient.
        key = bearer_token
    else:
        key = api_key

    # The OpenAI client requires a non-empty api_key; use a placeholder when the
    # inference service does not need one.
    return OpenAI(base_url=base_url, api_key=key or "not-needed")
