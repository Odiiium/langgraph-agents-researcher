# Deep Research Agent

A multi-step research agent built on **LangGraph**. Given a user question it plans
research tasks, gathers evidence from the web with a deep research sub-agent,
analyzes and quality-checks the findings in a loop, and synthesizes a final,
source-attributed answer. Every stage is protected by guardrails and every run is
traced (LangSmith / Langfuse) and logged to disk.

## What the agent can do

- Turn a free-form question into a structured **research plan**.
- Run an autonomous **research sub-agent** with web search + calculator + clock tools.
- **Analyze** collected evidence into claims with confidence scores and contradictions.
- **Self-check** the research/analysis and loop back for more research or re-analysis
  (up to `MAX_ITERATIONS`).
- **Synthesize** a final answer that only cites collected sources.
- **Guard** the whole pipeline: reject bad input, prompt injection, empty plans,
  ungrounded research, agent failures, and runaway iteration — and always terminate
  cleanly with the reason recorded.
- Persist results and guardrail errors to `results/<timestamp>/` and stage logs to `logs/`.

## Tools

The research sub-agent (`source/tools.py`) has three tools:

| Tool | Purpose |
|------|---------|
| `search_tool` | Web search via **Tavily** (`topic`: general/news/finance, tunable depth). |
| `calculator_tool` | Exact math via **sympy**. |
| `get_current_datetime` | Current time for an IANA timezone. |

Tools validate their arguments and return a uniform `ERROR: ...` string on failure
instead of raising; the agent is instructed to treat that as a failed call.

## Graph

Nodes are plain objects (`source/nodes.py`) registered by a generic `Pipeline`
(`source/pipeline.py`). Guardrail outcomes route through `GuardrailRouter` /
`CheckerRouter`. A rendered diagram is written to `graph.png` on every run.

```
START → pre_planner ─(blocked)───────────────────────────┐
   │(continue)  [EmptyQuery, InputLength, PromptInjection] │
current_time → planner → research ─(blocked)──────────────┤
   [EmptyPlan, researcher_response]  │(continue)           │
                              analyzer ─(blocked)──────────┤
                         [ResearchGrounding]  │(continue)  ▼
                                          checker ─(pass)→ synthesizer → save_result → END
                                             │(research/analysis)         (state + guardrail_error + console)
                                     research / analyzer
```

- **Hard guardrails** (empty/too-long/injection query, empty plan, ungrounded research,
  agent failure) → route straight to `save_result` (no answer, error recorded).
- **Soft guardrail** (`ResearchIterationGuardrail`) → at the iteration limit the run still
  goes to `synthesizer` for a best-effort answer, with the guardrail saved as an error.

Each node uses a task-appropriate OpenAI model (planner/analyzer `gpt-5`, checker
`o4-mini`, synthesizer `gpt-5-mini`, injection check `gpt-4o`, research agent
`openai:gpt-5-mini`); all model ids live in `source/config.py`.

## State

State is a single `ResearchState` `TypedDict` (`source/models.py`) threaded through the
graph; each node returns a partial update that LangGraph merges:

```
user_query, current_time, iteration_count,
plan, research, analysis, check_result, final_answer,
guardrail_violation
```

Structured fields are **pydantic** models (`ResearchPlan`, `ResearchResult`,
`AnalysisResult`, `CheckResult`, `SynthesizedAnswer`, `GuardrailViolationInfo`), which
also define the LLMs' structured outputs. `save_result` persists each run to
`results/<timestamp>/`:

- `research_result.json` / `research_result.md` — the final answer (if produced)
- `guardrail_error.json` — the guardrail that fired (if any)
- `run.json` — the full ending state

## LangSmith

The stack is native LangChain/LangGraph, so LangSmith traces automatically once the
env vars are set — no code changes. `main.py` calls `load_dotenv()` before running.
Set in `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Langfuse is also wired as a callback handler (`LANGFUSE_*` keys) and runs in parallel.

On top of tracing, key stages are logged to disk (`source/logging_config.py`):
chosen **node**, chosen **action** (router), **tool call**, **observation**, and the
**updated state** — plus every guardrail event.

- `query` runs log to `logs/<date_time>.log`
- `scenarios` runs log per scenario to `logs/_scenarios/<name>.log`

## Setup

```bash
cd grind_agents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` (git-ignored) with the required keys:

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=deep-research-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# optional (parallel tracer)
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_BASE_URL=...
```

## Running

Two entry points (run from the `grind_agents/` directory):

```bash
# 1) single user query
python source/main.py --query "What is the current price of Bitcoin and why?"

# 2) the demo scenarios
python source/main.py --scenarios
```

Both forms of invocation work: `python source/main.py ...` and `python -m source.main ...`.

Both print the answer (or guardrail reason) to the console, write results to
`results/`, and write logs to `logs/`.

## Demo tasks

`python source/main.py --scenarios` runs six queries, each exercising a different path:

| Scenario | Expected outcome |
|----------|------------------|
| `successful` | full run → synthesized answer |
| `partial` | loops to the iteration limit → best-effort answer + `ResearchIterationGuardrail` recorded |
| `guardrail_error` | prompt injection → hard stop, no answer |
| `input_length_error` | query > 2000 chars → `InputLengthGuardrail` hard stop |
| `successful_2` | full run → synthesized answer |
| `empty_query` | empty query → `EmptyQueryGuardrail` hard stop |

`successful*` and `guardrail_*` outcomes are deterministic; `partial` depends on the
checker reaching `MAX_ITERATIONS`.
