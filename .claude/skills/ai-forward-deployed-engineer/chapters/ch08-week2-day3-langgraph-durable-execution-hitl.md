# Week 2, Day 3: Durable Execution, LangGraph, and Human in the Loop

## Lessons Covered
- Lesson 62: Durable Execution for AI Agents and an Introduction to LangGraph
- Lesson 63: LangGraph Fundamentals - State, Nodes, Fixed and Conditional Edges
- Lesson 64: LangGraph Reducers, Super-Steps, Checkpoints and Threads
- Lesson 65: Human in the Loop - LangGraph interrupt() and Designing the Humans Step
- Lesson 66: Set Up a Free Supabase Postgres Database and the Supabase MCP Server
- Lesson 67: From LangChain create_agent to Writing Your Own LangGraph Nodes
- Lesson 68: Pause for Approval, Crash the Kernel and Resume a Claim from Supabase
- Lesson 69: Plan the Durable Build with Claude Code and the Supabase MCP Server
- Lesson 70: Run UnderCovers Durable Build - Dashboard Sign-Off, Payments and Traces

## Core Idea
Durable execution — a workflow that can stop (on a crash, a rate limit, or a human taking days to approve something) and resume exactly where it left off, without ever repeating an action it already took — is the production Achilles' heel of agentic AI, and LangGraph solves it with three primitives (state, nodes, edges) plus a Postgres-backed checkpointer and `interrupt()`/`Command(resume=...)` for human approval.

## Frameworks Introduced
- **Durable execution's "four no-brainers"**: ordinary production-software requirements (long-running jobs, failures happen, humans take time, servers go down) that apply doubly to agentic systems because agents are inherently stochastic and unreliable. When to use: any agent workflow headed to production. How: build on a framework (LangGraph) with first-class checkpointing rather than hand-rolling retry/resume logic.
- **LangGraph's three core concepts — State, Nodes, Edges**:
  - When to use: whenever you need fine-grained, auditable control over an agent workflow beyond what `create_agent` wraps for you.
  - How: **State** is an immutable snapshot (a `TypedDict` or Pydantic object) of everything the graph needs to know; a new state replaces, never mutates, the old one. **Nodes** are plain Python functions — `(state) -> state-or-partial-update` — each one a unit of work (can call an LLM, a tool, or nothing AI-related at all). **Edges** connect nodes: fixed edges always fire; conditional edges pick a branch based on a routing function (e.g., `after_agent`, `after_decide`).
- **Reducers and super-steps**: When to use: whenever multiple nodes can update state in the same turn (including in parallel). How: define a reducer (or use LangGraph's default) that combines each node's partial update into one new state object; one full turn of the graph producing a new combined state is a **super-step** (often just called a "step").
- **Checkpoints and threads**: When to use: any workflow where you need to pause/resume, audit history, or run independent concurrent instances (e.g., many claims at once). How: after every super-step, LangGraph saves the state as a checkpoint; a **thread ID** scopes an independent lineage of checkpoints (one thread per user conversation, or — as in UnderCover — one thread per claim). Use the `PostgresSaver` checkpointer (vs. `InMemorySaver`) for production durability.
- **`interrupt()` / `Command(resume=...)` human-in-the-loop pattern**: When to use: any node where a human decision must be awaited, potentially for hours or days, without an agent loop sitting idle. How: call `interrupt(...)` as the first thing in an "approval" node to pause graph execution; later, call `graph.invoke(Command(resume=<data>), config=...)` to resume — the interrupted node re-runs from the top, so put side effects (payments) in a *separate* downstream node, never before the `interrupt()` call.
- **Four factors for designing a human-approval step**: (1) the risk threshold that triggers approval (payment size, low eval score, irreversibility); (2) what information the human needs to decide well (policy excerpts, related claim history); (3) what actions they can take (not just approve/reject — maybe adjust amount, add a note); (4) what gets recorded about their decision. Balance against the downsides: cost, latency, and decision fatigue (too many low-value approvals train humans to rubber-stamp "yes").
- **Claude Code + Supabase MCP server durable build workflow**: When to use: moving a lab prototype into a real product feature. How: write a business-facing spec (`DURABLE.md`), drop it in `docs/`, authenticate Claude Code to a remote (Streamable HTTP) Supabase MCP server via OAuth, have Claude Code produce a phased plan, then implement phase-by-phase with its own code review process — mirroring Week 1's harness-build pattern but now for durable, checkpointed infrastructure.

## Key Concepts
- **State**: an immutable snapshot object (TypedDict/Pydantic) describing everything the graph needs to recreate exactly what's going on at any point; you never mutate a state instance, you produce a new one.
- **Node**: a plain Python function taking a state (or part of it) and returning a new state or partial update; LangGraph's smallest unit of work.
- **Edge**: the dependency/connection between nodes — fixed (always taken) or conditional (chosen by a routing function).
- **Super-step**: one full turn of the graph, in which one or more nodes (possibly in parallel) update state, which gets combined via reducers into the new official state and checkpointed.
- **Checkpoint**: a saved snapshot of state after a super-step, enabling rewind/resume.
- **Thread / thread ID**: a scoping key (e.g., a claim ID) that keeps one invocation's checkpoints and state independent from all others.
- **PostgresSaver / InMemorySaver**: LangGraph checkpointer implementations; Postgres (via Supabase) is production-grade and persists across restarts, InMemory does not.
- **`durability="sync"`**: an option on `graph.invoke` that guarantees a checkpoint is written before the next step runs, closing the window where a crash could lose track of what already happened.
- **Idempotent node design**: writing nodes (like `pay`) so that re-running them (because the graph resumed, or a node re-executes after an `interrupt()`) never double-executes a side effect — enforced here by checking the payments table for an existing record before paying.
- **`CompiledStateGraph`**: the actual type returned by `create_agent` — confirming that `create_agent` is a wrapper that builds and compiles a LangGraph graph behind the scenes, and has been all along since Week 1.
- **Mermaid graph visualization**: calling `.get_graph().draw_mermaid_png()` (or just printing the agent object in Jupyter) to visualize the compiled dependency graph of nodes and edges.
- **Remote MCP server with OAuth**: the Supabase MCP server connects over Streamable HTTP and authenticates via a browser OAuth flow — Claude Code never sees the user's Supabase password, only a scoped, authenticated MCP session (here, read-only on the tables).

## Mental Models
- Think of state like a Google Doc: many "nodes" (collaborators) make concurrent edits, and a reducer-like mechanism merges them into a new consistent document state you can always roll back to a prior version of.
- Treat `create_agent` as a "wizard" (a Windows-wizard analogy) that assembles a LangGraph graph for you — useful for getting started, but Day 3 is about lifting the hood and wiring nodes/edges yourself when you need granular control.
- When writing a node, always ask "what happens if this exact function runs twice?" — design for idempotency rather than assuming a clean single execution.
- Use "is this a side effect?" as the litmus test for whether code belongs before or after an `interrupt()` call — anything irreversible belongs in its own downstream node, never in the node that calls `interrupt()`.
- A dependency graph (LangGraph) is not inherently an "AI" concept — it's general workflow orchestration (like n8n/Zapier or React/Redux state management) that happens to be a great fit for agentic AI's reliability problem.

## Anti-patterns
- **Putting payment/side-effect code before an `interrupt()` call in the same node**: the whole node re-runs on resume, so any side effect placed before the interrupt will fire again.
- **Treating `create_agent`'s return value as a simple "agent object"**: it's actually a `CompiledStateGraph`; not understanding this hides the dependency-graph machinery already running underneath you.
- **Sending every decision to a human "just to be safe"**: causes decision fatigue, where reviewers rubber-stamp approvals and miss the ones that actually matter — calibrate thresholds instead.
- **Using the InMemory checkpointer in anything resembling production**: state is lost on restart; use a durable backend like Postgres/Supabase.
- **Skipping idempotency checks on side-effecting nodes**: without checking a system of record (e.g., "has this claim already been paid?") before acting, a resumed or re-run graph can double-pay or double-send.
- **Mutating a state object's fields directly inside a node**: violates LangGraph's immutability contract and breaks checkpointing/reducer behavior; always return a new state or a partial update instead.

## Code Examples
```python
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": claim_id}}
result = graph.invoke(initial_state, config=config, durability="sync")
# ... later, to resume after a crash or human approval:
result = graph.invoke(Command(resume={"handler": "Priya"}), config=config)
```
- **What it demonstrates**: the near-minimal pattern for durable, resumable, per-claim execution — a Postgres checkpointer, a thread ID scoped to the business entity (the claim), synchronous checkpoint durability, and resuming via `Command(resume=...)`.

```python
class ClaimState(TypedDict):
    messages: list
    claim_id: str
    decision: str
    payout: float
    rationale: str
    approved_by: str

def after_agent(state: ClaimState) -> str:
    """Route after the agent node: to tools if it called one, else to decide."""
    ...

def after_decide(state: ClaimState) -> str:
    """Route after decide: end, approval (needs sign-off), or pay."""
    if state["decision"] != "approve":
        return "end"
    if state["payout"] > APPROVAL_LIMIT:
        return "approval"
    return "pay"
```
- **What it demonstrates**: defining the state schema as a `TypedDict`, and the routing functions used by conditional edges to decide the next node.

```python
def approval(state: ClaimState):
    """Human-in-the-loop node: pause until a handler signs off."""
    decision = interrupt({"claim_id": state["claim_id"], "payout": state["payout"], "rationale": state["rationale"]})
    return {"approved_by": decision["handler"]}

def pay(state: ClaimState):
    """Idempotent payment node: only pays if not already paid for this claim."""
    # INSERT ... WHERE NOT EXISTS (SELECT 1 FROM payments WHERE claim_id = ...)
    ...
```
- **What it demonstrates**: the `interrupt()` pattern for pausing on human approval, and separating the actual payment into its own idempotent node so the approval node re-running on resume never risks a double payment.

## Reference Tables
| LangGraph concept | What it is | Analogy used |
|---|---|---|
| State | Immutable snapshot of all workflow data | A version of a Google Doc |
| Node | Python function: state in, state/update out | A box in a workflow diagram (n8n/Zapier) |
| Edge (fixed) | Always-taken connection between nodes | Solid line |
| Edge (conditional) | Branch chosen by a routing function | Dashed line |
| Super-step | One full turn producing a new combined state | A "turn" in a game |
| Checkpoint | Saved state after a super-step | An undo-able save point |
| Thread | Independent lineage of checkpoints (e.g., per claim) | A separate conversation/session |

| Checkpointer | Persistence | Use case |
|---|---|---|
| InMemorySaver | Lost on process restart | Prototyping only |
| PostgresSaver (Supabase) | Durable, survives crashes/restarts | Production |

## Worked Example
The Week 2 Day 3 lab (`labs/week2day3.ipynb`) has four parts. Part 1 sets up a free Supabase Postgres project and connects via the session-pooler connection string (password must be alphanumeric only — a known gotcha), saved into `.env` as `DATABASE_URL`, plus registers a Supabase MCP server for Claude Code. Part 2 confirms that `create_agent`'s return value is actually a `langgraph.graph.state.CompiledStateGraph` by drawing its Mermaid diagram — proving LangGraph has been running under the hood since Week 1. Part 3 rebuilds the same claims agent by hand: a `ClaimState` TypedDict, five nodes (`agent`, `tools`, `decide`, `approval`, `pay`), fixed and conditional edges, and a `PostgresSaver` checkpointer — then runs it on a $243 claim (auto-approved, no human needed) and a $3,750 wrist-fracture claim (exceeds the $1,000 approval limit, so `interrupt()` fires and the graph pauses; resuming with `Command(resume={"handler": "Dana Whitfield"})` completes the payment). Calling the `pay` node function directly a second time for the same claim confirms it refuses to double-pay (checks the payments table first). Part 4 is "the crash": a claim is left paused mid-approval, the Jupyter kernel is restarted (wiping all in-memory Python state), and simply re-running the setup cells and querying the claim's status shows it still correctly "waiting for approval" — then approving it from the fresh kernel completes the payment exactly as if nothing had happened, with the full checkpoint history (start → agent → tools → agent → decide → approval → pay) visible both via `graph.get_state_history()` and directly in Supabase's `checkpoints` table. The day culminates in a ~2-hour-46-minute Claude Code build (using the Supabase MCP server, authenticated via OAuth) that implements this entire durable, human-in-the-loop workflow into the real UnderCover product: five claims are queued, two require handler sign-off in the dashboard (processed by named handlers Dana and Marcus), and Supabase's `payments` table confirms exactly three payments ($420, $2,240, $1,150) with no duplicates — all traceable both in-product and in Langfuse.

## Key Takeaways
1. Durable execution means a workflow can crash or wait on a slow human for days and resume exactly where it left off — treat this as non-negotiable for production agents, not a nice-to-have.
2. `create_agent` has always been building a LangGraph `CompiledStateGraph` behind the scenes; writing nodes/edges by hand is the same mechanism with full manual control.
3. State is immutable — nodes never mutate a state object, they return new state (or a partial update that a reducer merges in).
4. Scope checkpoints per business entity using thread IDs (one per claim, not one per session) so each unit of work has its own independently resumable history.
5. `interrupt()` re-runs the entire node on resume — always put side effects (payments, external calls) in a separate downstream node, and make that node idempotent by checking the system of record first.
6. Design human-approval thresholds around risk (amount, score, reversibility), and weight that against cost, latency, and decision fatigue from over-escalating.
7. Remote MCP servers with OAuth (like Supabase's) let an agent (or Claude Code) use an authenticated connection without ever handling the user's actual credentials.
8. Use `durability="sync"` when you need a guarantee that the checkpoint is written before the next step executes, closing crash-related gaps.
9. Surfacing traces/decisions directly in the product (not just Langfuse) continues to pay off for explainability, exactly as flagged on Day 1.

## Connects To
- **Week 2, Day 2**: the MCP concepts (host/client/server, stdio vs. Streamable HTTP, OAuth) introduced for agentic RAG/memory are reused directly here for the Supabase MCP server, now over the remote/Streamable-HTTP transport with real authentication.
- **Week 2, Day 4**: the dispatcher pattern for running agent swarms at scale builds on the same LangGraph primitives (nodes, `Send`, `max_concurrency`, `RetryPolicy`) introduced here, applied to fan-out rather than durability.
- **Week 1 harness-build process**: the Claude Code planning/phased-implementation/code-review workflow for the durable build directly mirrors the harness-engineering process established in Week 1, now applied to a much deeper technical spec.
