# Agent Architecture Evolution

## Overview

Both the clinical trials and literature agents went through similar evolutionary arcs,
converging on the same architectural patterns as understanding deepened.

---

## Evolution of Understanding

### Stage 1 — `BaseAgent` + `create_agent()` (react)

The first agents used LangChain's `create_agent()` with a flat list of tools and a system prompt.
The agent was a class (`BaseAgent` subclass) with a `run()` method.

Key characteristics:
- Tools were module-level globals or built via closure (v2)
- Tools returned `model_dump()` dicts — Pydantic objects were serialized immediately
- After `agent.ainvoke()`, a `_parse_result()` method walked the message history,
  JSON-parsed each `ToolMessage`, and reconstructed Pydantic models from scratch
- The summary came from the final `AIMessage` in the history
- No explicit state management — everything lived in the message list

Pain points:
- `_parse_result()` was fragile — depended on JSON round-trips and `ToolMessage.name` matching
- Pydantic objects were destroyed and reconstructed unnecessarily
- No typed state — hard to know what had been collected at any point

---

### Stage 2 — LangGraph `StateGraph` (langgraph)

Replaced `create_agent` with an explicit `StateGraph` with three nodes:
`agent_node → tools_node → assemble_node` (loop back via `tools_condition`).

Key characteristics:
- Typed `State` (Pydantic `BaseModel`) holds structured results alongside messages
- `tools_node` has a dispatch table mapping tool names to state fields + parsers
- Still used `json.loads(msg.content)` + `model_validate()` in the dispatch table
- `assemble_node` extracts the final `AIMessage` text as the summary
- Tools still returned `model_dump()` dicts

Improvement over v2:
- Typed state makes it clear what has been collected
- Cleaner separation between tool execution and result storage
- No more `_parse_result()` — state is built incrementally

Remaining friction:
- JSON round-trip still happening: tool returns Pydantic → `ToolNode` serializes to JSON
  string → `tools_node` parses back to dict → `model_validate()` reconstructs Pydantic
- Drug/indication passed as LLM args — LLM could theoretically pass wrong values

---

### Stage 3 — `response_format="content_and_artifact"` (artifact)

Discovered LangChain's `response_format="content_and_artifact"` decorator option.
Tools return `tuple[str, <PydanticObject>]` — the string goes into `ToolMessage.content`
(what the LLM reads), and the raw Python object is stored in `ToolMessage.artifact`.

Key characteristics:
- `tools_node` reads `msg.artifact` directly — no `json.loads`, no `model_validate`
- Pydantic objects survive the tool call intact
- `tool_handlers` dispatch table simplified to a plain `artifact_fields` dict
- Drug/injection still LLM-provided (not yet injected via closure)

For clinical trials this is the right architecture because the LLM still needs to make
decisions (whitespace branch vs trials branch).

---

### Stage 4 — Linear Pipeline (current literature agent)

Realized that the literature agent's 4-step sequence is fully deterministic —
no branching, no LLM decision-making needed between steps.
Removed the LLM from orchestration entirely.

Key characteristics:
- No `agent_node`, no `ToolNode`, no `tools_condition`
- Each step is a plain async node that calls a service method directly
- State flows linearly: `expand → fetch → search → synthesize → assemble`
- `llm` parameter removed from `build_literature_graph()`
- Summary comes from `EvidenceSummary.summary` (already produced by `svc.synthesize`
  which calls the LLM internally via `query_llm`)
- No tool wrappers needed for the pipeline nodes

Key insight: "agent" doesn't always mean LLM-driven. When the flow is deterministic,
a pipeline is simpler, more reliable, and cheaper.

---

## Key Lessons

1. **Avoid unnecessary JSON round-trips.** Tools that return Pydantic objects should use
   `response_format="content_and_artifact"` so objects survive the tool call intact.

2. **Inject context via closure.** Drug name, indication, db session, drug profile —
   anything already known at graph-build time should be captured in the closure,
   not passed as LLM tool arguments. The LLM should only decide *which* tools to call.

3. **Not everything needs an LLM orchestrator.** If the sequence is fixed and deterministic,
   a linear pipeline is the right choice. Use an LLM agent only when genuine branching
   or decision-making is required (e.g. clinical trials whitespace branching).

4. **Typed state is essential.** LangGraph's `StateGraph` with a Pydantic state model
   makes it clear what has been collected at each step, and avoids the fragile
   message-history-walking approach of `_parse_result()`.

5. **The LLM's view vs the code's view.** `ToolMessage.content` is what the LLM sees
   (needs to be human-readable and informative). `ToolMessage.artifact` is what the
   code uses (the raw Python object). Keep these concerns separate.

---

## Session Summary — 2026-04-06

### Literature Agent

- Added `semantic_search` tool and wired it into `tools_node` and `LiteratureOutput`
- Fixed `semantic_search` handler in `tools_node` (was incorrectly using `lambda d: d["pmids"]`)
- Added `synthesize` tool from the bak
- Evolved tools to use `response_format="content_and_artifact"` — all four tools
  (`expand_search_terms`, `fetch_and_cache`, `semantic_search`, `synthesize`) now return
  `tuple[str, artifact]`
- `tools_node` simplified to read `msg.artifact` — no JSON parsing
- Rewrote agent as a **linear pipeline** — removed LLM orchestration entirely:
  - `expand_node`, `fetch_node`, `search_node`, `synthesize_node`, `assemble_node`
  - `llm` parameter removed from `build_literature_graph()`
  - `summarize_node` removed — summary read directly from `EvidenceSummary.summary`
- Fixed `LiteratureState.semantic_search_results`: `list[str]` → `list[dict]`
- Added `evidence_summary: EvidenceSummary | None = None` to state and output
- Updated unit tests to match pipeline structure (one test per node, shared `_make_svc()`)
- Updated integration test to assert on `semantic_search_results`, `evidence_summary`, and `summary`
- Fixed `agent_tester.py` — removed `llm`, removed old `run_literature_agent_artifact`

### Clinical Trials Agent

- Wrote `response_format="content_and_artifact"` rewrite plan to
  `for_me/clinical_trials_artifact_rewrite.md`
- Updated `clinical_trials_tools.py` — drug/indication injected via closure,
  all tools use `response_format="content_and_artifact"` (work in progress, not merged to agent yet)
