# Agent Architecture

This document describes the agent layer of IndicationScout: how agents are structured, how they call tools, and how to add new ones.

For the overall system architecture see [ARCHITECTURE.md](ARCHITECTURE.md).

## BaseAgent

**File**: [src/indication_scout/agents/base.py](../src/indication_scout/agents/base.py)

`BaseAgent` defines a single async interface used by class-style agents:

```python
class BaseAgent(ABC):
    @abstractmethod
    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        ...
```

Some current agents (e.g. `ClinicalTrialsAgent`) are not class-based — they are
exposed as `build_*_agent(llm, ...)` + `run_*_agent(agent, ...)` function pairs
that return a compiled LangGraph `create_react_agent` graph.

## File Layout

Each agent lives in its own subpackage with three files:

| File | Purpose |
|------|---------|
| `agents/<name>/<name>_agent.py` | `build_<name>_agent` (compiles ReAct graph) and `run_<name>_agent` (invokes + assembles output). System prompt lives here. |
| `agents/<name>/<name>_tools.py` | `build_<name>_tools(...)` returning a list of `@tool(response_format="content_and_artifact")` functions wrapping data source client methods. |
| `agents/<name>/<name>_output.py` | Agent-specific output Pydantic model. |

Data source models (e.g. `Trial`, `WhitespaceResult`, `IndicationLandscape`,
`TrialOutcomes`) live in `models/` and are referenced by the agent output model
— never duplicated.

## ClinicalTrialsAgent

### Agent module

**File**: [src/indication_scout/agents/clinical_trials/clinical_trials_agent.py](../src/indication_scout/agents/clinical_trials/clinical_trials_agent.py)

Two functions:

```python
def build_clinical_trials_agent(llm, date_before: date | None = None):
    """Compile a LangGraph ReAct agent."""

async def run_clinical_trials_agent(
    agent, drug_name: str, disease_name: str
) -> ClinicalTrialsOutput:
    """Invoke the agent and assemble a ClinicalTrialsOutput from the run."""
```

**LLM**: provided by the caller (the supervisor passes
`settings.llm_model` via `ChatAnthropic`).

**Agent creation** (uses LangGraph's prebuilt `create_react_agent`):

```python
return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)
```

The agent is invoked with:

```python
result = await agent.ainvoke(
    {"messages": [HumanMessage(content=f"Analyze {drug_name} in {disease_name}")]}
)
```

**System prompt branching strategy** — the LLM decides tool order:

1. Typically start with `detect_whitespace`
2. Always call `get_terminated` alongside `search_trials` and `get_landscape`
3. Finish with `finalize_analysis`, passing a 2–3 sentence plain-text summary

`finalize_analysis` is a tool whose only job is to terminate the loop and
carry the summary back as the artifact for the `summary` field.

**Result assembly**: `run_clinical_trials_agent` walks the message history
from `agent.ainvoke()` and reads the typed artifact off each `ToolMessage`:

- Tools use `@tool(response_format="content_and_artifact")` so each tool
  returns `(text_for_LLM, typed_pydantic_artifact)`. The artifact rides
  alongside the message; no `model_dump()` round-trip is needed.
- A `field_map` maps tool name → output field
  (`detect_whitespace → whitespace`, `search_trials → trials`,
  `get_landscape → landscape`, `get_terminated → terminated`,
  `finalize_analysis → summary`).
- For each `ToolMessage`, the artifact is assigned to its mapped field.
- If the same tool is called twice, the later artifact overwrites the
  earlier one. The system prompt guides the LLM to call each tool once.
- A `ClinicalTrialsOutput` is constructed from the assembled artifacts.

### Tools

**File**: [src/indication_scout/agents/clinical_trials/clinical_trials_tools.py](../src/indication_scout/agents/clinical_trials/clinical_trials_tools.py)

Entry point:
`build_clinical_trials_tools(date_before: date | None = None) -> list`

Returns five LangChain `@tool` functions. The `date_before` parameter is
captured via closure so it flows to every client call without being exposed
as a tool parameter to the LLM.

| Tool | Arguments | Client Method | Artifact |
|------|-----------|---------------|----------|
| `detect_whitespace` | `drug: str, indication: str` | `ClinicalTrialsClient.detect_whitespace()` | `WhitespaceResult` |
| `search_trials` | `drug: str, indication: str` | `ClinicalTrialsClient.search_trials()` | `list[Trial]` |
| `get_landscape` | `indication: str` | `ClinicalTrialsClient.get_landscape(top_n=10)` | `IndicationLandscape` |
| `get_terminated` | `drug: str, indication: str` | `ClinicalTrialsClient.get_terminated()` | `TrialOutcomes` |
| `finalize_analysis` | `summary: str` | — (loop terminator) | `str` |

Design rules:

- Tools accept primitive types (strings) so the LLM can provide arguments directly.
- Tools use `response_format="content_and_artifact"`: they return a
  `(content_str, artifact)` tuple. The string goes into the LLM's message
  context; the typed Pydantic artifact is assembled into the final output.
- Each tool creates its own client session: `async with ClinicalTrialsClient() as client:`.
- `get_landscape` passes `top_n=10` to keep the response within manageable LLM context.
- Every tool that touches an indication first calls
  `services.disease_helper.resolve_mesh_id(indication)`. If the resolver
  returns `None`, the tool logs a warning and returns an empty artifact —
  the agent still receives a valid (empty) shape and continues. The
  resolved D-number is forwarded as `target_mesh_id` so the client
  post-filters Essie's recall-first results.

### Output model

**File**: [src/indication_scout/agents/clinical_trials/clinical_trials_output.py](../src/indication_scout/agents/clinical_trials/clinical_trials_output.py)

```python
class ClinicalTrialsOutput(BaseModel):
    whitespace: WhitespaceResult | None = None
    landscape: IndicationLandscape | None = None
    trials: list[Trial] = []
    terminated: TrialOutcomes = Field(default_factory=TrialOutcomes)
    summary: str = ""
```

- References models from `models/model_clinical_trials.py`.
- Fields are None/empty when the agent chose not to call the corresponding tool. This is expected behavior, not an error.
- `terminated` is always a `TrialOutcomes` (never `None`) — empty scopes are empty lists.

### Data source models

**File**: [src/indication_scout/models/model_clinical_trials.py](../src/indication_scout/models/model_clinical_trials.py)

These are the contracts between `ClinicalTrialsClient` and the agents. All have the `coerce_nones` validator.

**Trial-level models**:

| Model | Key fields |
|-------|------------|
| `Intervention` | `intervention_type`, `intervention_name`, `description` |
| `PrimaryOutcome` | `measure`, `time_frame` |
| `MeshTerm` | `id` (D-number), `term` |
| `Trial` | `nct_id`, `title`, `brief_summary`, `phase`, `overall_status`, `why_stopped`, `indications`, `mesh_conditions` (list[MeshTerm]), `mesh_ancestors` (list[MeshTerm]), `interventions` (list[Intervention]), `sponsor`, `enrollment`, `start_date`, `completion_date`, `primary_outcomes` (list[PrimaryOutcome]) |

**Whitespace models**:

| Model | Key fields |
|-------|------------|
| `IndicationDrug` | `nct_id`, `drug_name`, `indication`, `phase`, `status`. Has `from_trial()` classmethod. |
| `WhitespaceResult` | `is_whitespace`, `no_data`, `exact_match_count`, `drug_only_trials`, `indication_only_trials`, `indication_drugs` (list[IndicationDrug]) |

**Landscape models**:

| Model | Key fields |
|-------|------------|
| `CompetitorEntry` | `sponsor`, `drug_name`, `drug_type`, `max_phase`, `trial_count` (int), `statuses` (set[str]), `total_enrollment` (int), `most_recent_start` (date \| None) |
| `RecentStart` | `nct_id`, `sponsor`, `drug`, `phase` |
| `IndicationLandscape` | `total_trial_count`, `competitors` (list[CompetitorEntry]), `phase_distribution` (dict[str, int]), `recent_starts` (list[RecentStart]) |

**Trial-outcome models**:

| Model | Key fields |
|-------|------------|
| `TerminatedTrial` | `nct_id`, `title`, `drug_name`, `indication`, `phase`, `why_stopped`, `stop_category` (safety/efficacy/business/enrollment/other/unknown), `enrollment`, `sponsor`, `start_date`, `termination_date`, `mesh_conditions` (list[MeshTerm]) |
| `TrialOutcomes` | `drug_wide`, `indication_wide`, `pair_specific`, `pair_completed` — each `list[TerminatedTrial]` |

## Data Flow

```
build_clinical_trials_agent(llm, date_before)
    |
    +-- build_clinical_trials_tools(date_before)   # 5 @tool functions
    +-- create_react_agent(model, tools, prompt)   # LangGraph ReAct agent
                       |
run_clinical_trials_agent(agent, drug, disease)
    |
    +-- agent.ainvoke({"messages": [HumanMessage(...)]})  # ReAct loop begins
    |       |
    |       +-- LLM calls detect_whitespace
    |       |       +-- resolve_mesh_id(indication)
    |       |       +-- ClinicalTrialsClient.detect_whitespace(target_mesh_id=...)
    |       |       +-- artifact: WhitespaceResult
    |       |
    |       +-- LLM calls search_trials, get_landscape, get_terminated
    |       |       +-- each resolves MeSH, hits client, returns typed artifact
    |       |
    |       +-- LLM calls finalize_analysis(summary="...")
    |               +-- artifact: summary string; terminates the loop
    |
    +-- walk message history, read .artifact off each ToolMessage
    +-- assemble ClinicalTrialsOutput (whitespace, landscape, trials, terminated, summary)
    +-- return ClinicalTrialsOutput
```

## How to Call

```python
from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)
from langchain_anthropic import ChatAnthropic
from datetime import date

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
agent = build_clinical_trials_agent(llm, date_before=date(2023, 1, 1))
output = await run_clinical_trials_agent(agent, "semaglutide", "NASH")

# Access structured data
output.whitespace   # WhitespaceResult | None
output.landscape    # IndicationLandscape | None
output.trials       # list[Trial]
output.terminated   # TrialOutcomes (drug_wide / indication_wide / pair_specific / pair_completed)
output.summary      # str — natural language assessment
```

## Dependencies

```toml
"langchain-core"
"langgraph"
"langchain-anthropic"
```

## Test Layout

```
tests/
+-- unit/agents/
|   +-- test_clinical_trials_tools.py    # mocked client, verifies each tool returns correct artifacts
|   +-- test_clinical_trials_agent.py    # tests result assembly with fake message histories
+-- integration/agents/
    +-- test_clinical_trials_tools.py    # hits real ClinicalTrials.gov API
    +-- test_clinical_trials_agent.py    # hits real ClinicalTrials.gov + Anthropic APIs
```

**Unit tests** mock `ClinicalTrialsClient` and verify:

- Each tool returns a `(content, artifact)` tuple with the correct types
- `date_before` flows through the closure to client calls
- Result assembly correctly reads artifacts off `ToolMessage` objects

**Integration tests** verify end-to-end:

- Tools return correct data from real API
- Agent produces correct structured output for known drug-disease pairs
- Agent handles nonexistent drugs/diseases gracefully

## Adding a New Agent

To add a new agent (e.g. `MechanismAgent`):

1. **Create `agents/mechanism/mechanism_output.py`** — output Pydantic model referencing models from `models/`. Include `coerce_nones` validator.
2. **Create `agents/mechanism/mechanism_tools.py`** — `build_mechanism_tools(...)` returning a list of `@tool(response_format="content_and_artifact")` functions wrapping data source client methods. Tools accept primitives, return `(content, artifact)`. Each tool manages its own client session.
3. **Create `agents/mechanism/mechanism_agent.py`** — `build_mechanism_agent(llm, ...)` that compiles a `create_react_agent`, plus `run_mechanism_agent(agent, ...)` that invokes the graph and walks the message history to assemble the typed output. Include a `finalize_analysis` tool to carry the summary.
4. **Add unit tests** in `tests/unit/agents/test_mechanism_*.py`.
5. **Add integration tests** in `tests/integration/agents/test_mechanism_*.py`.

Key patterns to follow:

- Tools accept primitive types, return `(content, artifact)` via `response_format="content_and_artifact"`
- Tools create their own client sessions (`async with Client() as c:`)
- Use closures to capture config that the LLM should not see (e.g. `date_before`)
- Result assembly walks message history and reads `.artifact` off `ToolMessage` objects
- Output model fields default to None/empty for tools that were not called
- All Pydantic models that ingest external data include the `coerce_nones` validator

## Agent Catalogue

| Agent | File | Status | Data Source |
|-------|------|--------|-------------|
| ClinicalTrialsAgent | `agents/clinical_trials/clinical_trials_agent.py` | Implemented | ClinicalTrialsClient |
| LiteratureAgent | `agents/literature/literature_agent.py` | Implemented | RetrievalService (PubMed + pgvector) |
| MechanismAgent | `agents/mechanism/mechanism_agent.py` | Implemented | OpenTargetsClient |
| SupervisorAgent | `agents/supervisor/supervisor_agent.py` | Implemented | Coordinates all sub-agents |
