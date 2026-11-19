# Week 2, Day 4: Agent Teams, Swarms, and a No-LLM LangGraph Dispatcher

## Lessons Covered
- Lesson 71: The Difference Between Agent Teams, Agent Swarms and Agent Fleets
- Lesson 72: A LangGraph Dispatcher with No LLM - Fan-Out, Rate Limits and Retries
- Lesson 73: Build the Dispatcher with a Stand-In - max_concurrency and RetryPolicy
- Lesson 74: Run a Swarm of 5 Claim Agents on GPT-6 Luna and Trace It in Langfuse
- Lesson 75: Score the Swarm in Langfuse and Compare 5 Claim Agents with 25
- Lesson 76: Add a Swarm of Claim Agents and an Evaluation Scoreboard to UnderCover

## Core Idea
LangGraph is a general-purpose dependency-graph orchestrator, not just an LLM-calling framework — you can build a pure-code "dispatcher" (no LLM at all) that fans out work to a bounded pool of concurrent workers with automatic retries, and then drop an entire agent (itself a LangGraph graph) into one of those worker nodes, getting scalable, resilient agent swarms almost for free.

## Frameworks Introduced
- **Teams vs. Swarms vs. Fleets** (informal, evolving industry terms):
  - When to use: as a shared vocabulary, not a rigid taxonomy — these terms mean different things to different vendors.
  - How: **Agent team** = a handful (3-4) of collaborating agents with distinct roles (e.g., front-end, back-end, QA, DevOps), usually one acting as manager/coordinator, sharing a task list and communicating via markdown files or message-passing tools (e.g., Claude Agent Teams in Claude Code). **Agent swarm** = a large number (20-100+) of mostly-identical agent instances working in parallel on the same or self-coordinated tasks (e.g., Moonshot AI's Kimi Agent Swarm running ~100 Kimi agents; UnderCover's 25 claim agents). **Agent fleet** = an organization's entire roster of agents across teams, akin to an HR org chart for agents (e.g., LangSmith Fleet — note LangSmith is LangChain's brand for both observability and runtime/deployment products, formerly "LangGraph platform").
- **LangGraph dispatcher pattern (no LLM)**: When to use: any workload that needs to fan out a list of independent units of work to a bounded worker pool with retries — regardless of whether the work involves an LLM. How: define a state with a list field combined via a reducer (`Annotated[list, operator.add]`); a `list_claims`-style node populates the work list; a conditional-edge function (e.g., `fan_out`) returns one `Send(node_name, item)` per work item; cap concurrency with `max_concurrency=N` on `graph.invoke(...)`; attach a `RetryPolicy(max_attempts=N)` to the worker node to handle transient failures automatically.
- **Nesting LangGraph graphs**: When to use: scaling an existing agent (itself a compiled LangGraph graph from `create_agent`) without rewriting it. How: a dispatcher's worker node is just a plain Python function; that function can call `agent.invoke(...)` on an entirely separate, already-compiled graph — there is no special "graph of graphs" API, because a node is "just a Python function" that can do anything, including invoking another graph.
- **Scaling diagnostics via Langfuse**: When to use: comparing swarm sizes (e.g., 5 vs. 25 concurrent agents) to see real throughput gains vs. noise. How: trace each swarm run as a distinct root observation (toggle "is root observation" to declutter the trace list), use Langfuse's timeline view to visually compare wall-clock behavior, and record run-level scores (`langfuse.create_score`) so swarm-size comparisons appear as columns/scorecards rather than only in notebook output.

## Key Concepts
- **Fan-out**: distributing a list of independent tasks to multiple parallel node executions via a conditional edge returning a list of `Send` objects.
- **`Send` object**: a LangGraph primitive that targets a named node with a specific payload, used inside a fan-out routing function to dynamically spawn N parallel node invocations.
- **`operator.add` reducer**: the standard way to tell LangGraph "combine updates to this list field by concatenation" — used so that many parallel `work_claim` nodes can each append one result to a shared `results` list.
- **`max_concurrency`**: the parameter on `graph.invoke(...)` that caps how many nodes LangGraph will run in parallel at once — the single knob controlling "team of 5" vs. "swarm of 25."
- **`RetryPolicy(max_attempts=N)`**: a LangGraph node-level policy that automatically retries a node if it raises an exception, without any manual try/except orchestration.
- **Rate limits / connection caps**: practical scaling constraints to design around — e.g., provider 429 errors under high concurrency, and Supabase's free-tier cap of 5 concurrent Postgres connections.
- **Weakest-link effect**: in a bounded-concurrency swarm, total wall-clock time is set by the slowest individual item, not the average — adding more workers speeds things up but hits diminishing returns (e.g., 25 agents only ran ~1.5x faster than 5, not 5x, due to Supabase connection/write contention).
- **Thread-ID granularity choice**: running an entire swarm under one thread ID (as in the lab) vs. one thread ID per claim (as in the product) — the latter is needed when you want human-in-the-loop on one claim to not block the rest of the graph's completion.
- **`interrupt()`'s exception-based implementation (a LangGraph internals gotcha)**: `interrupt()` works by raising and catching a special exception internally; broad `try/except` blocks inside your own nodes can accidentally swallow that exception and silently break `interrupt()` — a known framework "gotcha" to watch for.
- **Scoreboard metrics added beyond Day 1**: `slowest_claim` (the weakest-link time) and `seconds_per_claim` (average), used specifically to evaluate swarm performance rather than just decision quality.

## Mental Models
- Think of the dispatcher as proof that LangGraph is a general workflow engine: "it doesn't need to be orchestrating LLM calls — it can orchestrate anything, including this code right here."
- Treat a node as "just a Python function" with no special constraints — it's the mental unlock for nesting one graph (an agent) inside another graph's node (a dispatcher) without any extra API.
- When evaluating a swarm-size change, separate "noise" metrics (accuracy varying slightly run-to-run due to LLM stochasticity) from the metric that's actually meaningful for scaling (wall-clock time) — don't over-read small accuracy deltas between swarm sizes as a real effect.
- Picture concurrency not as fixed batches but as a sliding pool: as soon as one worker finishes, the next queued item starts immediately — it's not "wait for all 5, then start the next 5."
- Use the "would I trust a stranger's code inside my own process" and "does this logic depend only on fixed rules" framing from Day 1 when deciding dispatcher-vs-agent: the dispatcher's fan-out/retry logic is fixed business logic (code), while deciding each individual claim is still the agent's job (LLM judgment).

## Anti-patterns
- **Treating "agent team / swarm / fleet" as having one universally agreed definition**: these are hype-laden, vendor-specific terms — clarify what a stakeholder means before designing around the label.
- **Building fan-out/retry/concurrency-limiting logic as autonomous LLM tool calls** ("make 5 tool calls to kick off 5 workers") instead of fixed code: unreliable, since it depends on token-prediction getting the orchestration right instead of using LangGraph's deterministic `max_concurrency` and `Send`.
- **Using broad exception handling inside LangGraph nodes without awareness of `interrupt()`'s internals**: can silently swallow the exception that `interrupt()` relies on, breaking human-in-the-loop pausing in confusing ways.
- **Assuming linear speedup from more concurrent agents**: shared-resource contention (e.g., a database connection pool) caps real-world gains well below the naive N-times-faster expectation.
- **Anthropomorphizing agent fleets into org-chart-style HR rosters**: risks falling back into Day 1's "human-like agents" trap — frame swarms/fleets by the business problem and measured throughput/outcome, not by giving agents job titles.
- **Giving every agent in a swarm the same thread ID when you need independent human-in-the-loop pausing per item**: blocks the whole graph on one item's approval; use per-item thread IDs when that independence matters.

## Code Examples
```python
class Dispatch(TypedDict):
    claims: list[str]
    results: Annotated[list, operator.add]

def list_claims(state: Dispatch):
    """Populate claims with all 25 items to process."""
    return {"claims": ALL_CLAIM_IDS}

def fan_out(state: Dispatch):
    """Conditional edge: spawn one Send per claim, routed to work_claim."""
    return [Send("work_claim", claim_id) for claim_id in state["claims"]]
```
- **What it demonstrates**: the reducer (`operator.add`) that lets many parallel `work_claim` nodes safely append to a shared `results` list, and the `fan_out` routing function that turns a list into N parallel `Send`-targeted node invocations.

```python
dispatcher.invoke({}, config={"max_concurrency": 5})
# ... later:
dispatcher.invoke({}, config={"max_concurrency": 25})
```
- **What it demonstrates**: the entire mechanism for controlling "5 workers at a time" vs. "25 at once (a swarm)" — a single integer parameter, with LangGraph handling all pool-management internally.

```python
def flaky_stand_in(claim_id):
    tries[claim_id] = tries.get(claim_id, 0) + 1
    if claim_id == "CLM-1207" and tries[claim_id] == 1:
        raise Exception("transient failure")
    time.sleep(1)
    return {"claim_id": claim_id, "start": ..., "end": ...}

builder.add_node("work_claim", flaky_stand_in, retry_policy=RetryPolicy(max_attempts=3))
```
- **What it demonstrates**: attaching a `RetryPolicy` to a node so a transient exception is automatically retried (here, up to 3 attempts) with no manual retry loop in application code.

```python
def work_claim(claim_id: str):
    """Dispatcher node that invokes an entirely separate, already-compiled agent graph."""
    result = payout_agent.invoke({"messages": [{"role": "user", "content": make_user_prompt(claim_id)}]})
    return {"results": [extract_decision(result)]}
```
- **What it demonstrates**: a dispatcher worker node that is itself just a function calling `.invoke()` on another LangGraph graph (the claim agent) — proving there's no special API needed to nest graphs, since a node is "just a Python function."

## Reference Tables
| Concurrency setting | Behavior observed | Wall-clock time for 25 one-second stand-in tasks |
|---|---|---|
| `max_concurrency=1` | Fully sequential | ~25 seconds |
| `max_concurrency=5` | 5 workers, next item starts as soon as any finishes | ~5-7 seconds |
| `max_concurrency=25` | All 25 run essentially at once | ~1 second |

| Swarm size (real claim agents, Luna) | Accuracy / cost-of-mistakes | Time | Notes |
|---|---|---|---|
| 5 at once | ~baseline accuracy, similar cost of mistakes | ~1m33s for 25 claims | — |
| 25 at once | Similar accuracy (differences are noise) | Faster, but not 5x faster | Bottlenecked by Supabase's free-tier 5-connection cap under heavy checkpointing |

| Term | Typical scale | Typical composition |
|---|---|---|
| Agent team | 3-4 agents | Distinct roles, shared task list, one manager |
| Agent swarm | 20-100+ agents | Mostly identical agents, parallel or self-coordinating |
| Agent fleet | Organization-wide | All agents across all teams, "HR roster" framing |

## Worked Example
The Week 2 Day 4 lab (`labs/week2day4.ipynb`) builds in four stages. Part 1 builds a pure-code LangGraph dispatcher with a `Dispatch` state (`claims`, `results` with an `operator.add` reducer), a `list_claims` node, and a `fan_out` conditional edge emitting one `Send("work_claim", ...)` per claim; using a fake one-second `stand_in` function and varying `max_concurrency` from 1 to 5 to 25 demonstrates near-linear scaling (25 seconds → ~5-7 seconds → ~1 second) visualized with a custom `show_timeline` bar chart. Swapping in a `flaky_stand_in` that deliberately throws on the first attempt for one specific claim ID, plus a `RetryPolicy(max_attempts=3)` on the worker node, shows the failed claim automatically succeeding on retry with zero extra application code. Part 2 replaces the stand-in with a real call into the UnderCover claim agent (itself a compiled LangGraph graph with `calculate_payout` and `record_decision` tools) — running a swarm of 5 agents over 25 harder claims and tracing it in Langfuse, including the tree view (`list_claims` → `fan_out` → `work_claim` ×25), a visual timeline, and a graph view. Part 3 extends Day 1's eval suite with `slowest_claim` and `seconds_per_claim`, logs scores to Langfuse via `langfuse.create_score`, and compares a swarm of 5 vs. a swarm of 25 on identical data: accuracy/cost-of-mistakes differences between the two runs are attributed to LLM stochasticity (noise), while wall-clock time is the metric that meaningfully improves — though not linearly, due to Supabase connection contention under heavy parallel checkpoint writes. Part 4 is the full product build: Claude Code (given a `SWARM.md` business brief) adds a dashboard control to pick swarm size N, a "Start Dispatch" button, and a new "Scoreboard" tab supporting evaluation runs against either the existing eval set or a previously untouched held-out set. Running a live swarm of 25 claims completes in about 41 seconds for under 3 cents of model cost, with handler sign-off still required above the $1,000 approval threshold, and the Scoreboard surfaces accuracy, decision accuracy, cost of mistakes ($456), and straight-through rate (84%) directly in the product — again illustrating Day 1's point that explainability and evals belong in the platform itself, not only in Langfuse.

## Key Takeaways
1. LangGraph is a general dependency-graph/workflow engine — it can orchestrate pure Python code with zero LLM calls just as well as it orchestrates agents.
2. Fan-out at scale reduces to one state field with an `operator.add` reducer, one routing function emitting `Send` objects, and a single `max_concurrency` integer — no bespoke worker-pool code required.
3. `RetryPolicy` on a node gives automatic, framework-level retries for transient failures, replacing manual try/except retry loops.
4. Nesting graphs (an agent inside a dispatcher's worker node) requires no special API — a node is just a function, and that function can call `.invoke()` on another compiled graph.
5. Watch for `interrupt()`'s exception-based internals clashing with your own broad exception handling inside nodes.
6. More concurrent workers speeds things up, but shared resources (a free-tier database connection cap, provider rate limits) mean the gain is sub-linear — budget for that in capacity planning.
7. Separate "noise" metrics (small accuracy swings between runs) from metrics that genuinely reflect a scaling change (wall-clock time, slowest-claim).
8. Choose thread-ID granularity deliberately: one thread for a whole batch run vs. one thread per unit of work, depending on whether you need independent human-in-the-loop pausing per item.
9. "Team," "swarm," and "fleet" are useful shared vocabulary but not rigorously defined — confirm what a stakeholder means rather than assuming.

## Connects To
- **Week 2, Day 3**: directly reuses the LangGraph state/node/edge/checkpoint machinery from Day 3's durable-execution lesson, now applied to fan-out/concurrency rather than single-item pause/resume — and the `interrupt()` exception gotcha is an explicit callback to Day 3's human-in-the-loop mechanics.
- **Week 2, Day 1**: extends the eval/scoreboard discipline (confusion matrix, cost of mistakes, kappa) with new throughput-oriented metrics (`slowest_claim`, `seconds_per_claim`) needed specifically to evaluate swarm scaling.
- **Week 2, Day 5**: the held-out dataset introduced on the Scoreboard ("more about that tomorrow") becomes central to Day 5's self-improvement methodology, where held-out data prevents overfitting the improver's changes to the data it can see.
