# Chapter 7: Frameworks for Agentic AI — Part 2

## Core Idea
LangGraph is a low-level, graph-based orchestration framework purpose-built for stateful, long-running, cycle-and-branch-capable agents (completing the four-framework landscape alongside Microsoft Agent Framework, OpenAI Agents SDK, and CrewAI from Ch 6) — and the chapter's direct side-by-side comparison across architecture, memory, orchestration, communication, scalability, and human-in-the-loop support yields one governing rule: there is no single best framework, only the framework whose control-flow model, abstraction level, and ecosystem fit match your project's maturity, team readiness, and scale requirements.

## Frameworks Introduced

### LangGraph — Stateful Graph Architecture
Three components: **Node** (discrete operation/function — process input, query DB, call API), **Edge** (static/linear, cyclical/looping, or dynamic/conditional data-flow control between nodes), **State** (centralized mutable object — schema + reducer functions — that persists and evolves as nodes update it; typically a `TypedDict` or Pydantic model, reflecting a structured-state preference). Unlike linear chains, a graph natively represents cycles and conditional branching — the architectural requirement agentic workflows have that simple chains don't.
- **Memory**: short-term = thread-scoped state via **Checkpointers**; long-term = persistent stores shared across conversational threads — backed by SQLite, PostgreSQL, MongoDB, or cloud storage (S3, GCS, Azure Blob).
- **Tools**: via `ToolNode`, a pre-built component that executes LLM-issued tool-call instructions and feeds results back into graph state.
- **Pause/resume**: the `interrupt` function explicitly pauses a workflow (signals stop, saves checkpoint); the `command` primitive resumes exactly from the interrupted state — purpose-built for human-in-the-loop approval steps (e.g., a discount-percentage decision needing human sign-off).
- When to use: any task needing genuine cyclic/iterative reasoning, fine-grained execution control, and durable pause-resume — the chapter's own framing: "low-level" and "does not abstract prompts or architecture," trading ease-of-use for maximum control.

### LangGraph Multi-Agent Patterns
- **Network**: agents share one scratchpad of messages — maximum transparency, but verbose if only the final answer matters; a rule-based router inspects output (tool call → invoke tool; `FINAL ANSWER` → return to user; else → next agent).
- **Supervisor**: a central supervisor agent routes tasks to specialist sub-agents (each keeping its own private scratchpad) and aggregates only their final outputs onto a global scratchpad (Weather-Team example: supervisor coordinates a Temperature Expert and Precipitation Expert, synthesizing their results) — reduces cognitive load on specialists by hiding the full system from them.
- **Nested**: extends supervisor by letting node-level agents themselves be entire LangGraph graphs — enables multi-layered team hierarchies (a top-level supervisor managing sub-supervisors managing sub-agents).
- **Handoff**: each agent node decides, from its own read of current state, whether to terminate or hand control to another agent via explicit graph-defined pathways — decision-of-when-and-to-whom lives with the active agent, not a central router.
- When to use: Network for maximum transparency on exploratory tasks; Supervisor when specialists shouldn't see each other's internal reasoning; Nested for genuinely hierarchical organizations of agents; Handoff when the active agent itself is best positioned to judge the next specialist needed (mirrors the bank-bot swarm example built with `create_swarm`).

### Four-Framework Comparison (Table 7.1, condensed)
| Dimension | OpenAI Agents SDK | LangGraph | Microsoft Agent Framework | CrewAI |
|---|---|---|---|---|
| Core philosophy | Lightweight, Python-first, minimal abstractions | Low-level graph orchestration, explicit cycles, stateful/durable | Separates intelligence (Agents) from orchestration (Workflows); type-safe, resumable | Role-playing agents, Crews + Flows, mirrors human teams |
| Memory | Context object; no built-in long-term memory | Short-term (thread-scoped Checkpointers) + long-term (persistent stores) | AgentThreads + middleware; vector-store integration (Azure AI Search, Cosmos DB, Redis) | ChromaDB (short-term/entity RAG) + SQLite (long-term); Mem0 |
| Orchestration | Python-native, handoff-based | Graph (nodes/edges), LLM-decided next node | Sequential/Concurrent/Handoff/Group Chat/Magentic | Processes (Sequential/Hierarchical) + event-driven Flows |
| Low-code | Code-first; AgentKit for visual building | Low-level; LangGraph Studio for visual prototyping | Declarative YAML + Azure AI Foundry tooling | Crew Studio (no/low-code conversational interface) |
| Human-in-the-loop | Limited native support (manual/ad-hoc) | Seamless — callback nodes, conditional branching, checkpoints | Built-in request/response nodes, explicit HIL patterns | Task-level `human_input` flag |
- When to use: this table is the chapter's answer to "which framework" — but the real decision factors are elaborated separately below (project maturity, team readiness, ecosystem, scale).

### Framework Selection Criteria
Four decision axes, each with a worked example:
1. **Project-framework alignment** (maturity stage): early prototypes/MVPs → CrewAI or OpenAI Agents SDK (low setup friction, fast time-to-value); production/regulated systems → Microsoft Agent Framework or LangGraph (observability, lifecycle management, structured logging). Example: a travel-itinerary-planner startup picks CrewAI; a large travel platform needing governed fallback-to-human support picks Microsoft Agent Framework or LangGraph.
2. **Team readiness** (skillset fit): Microsoft Agent Framework and CrewAI are friendlier to less experienced teams (linear flows, declarative patterns); OpenAI Agents SDK and LangGraph assume higher technical maturity (async programming, context/memory management). Example: a logistics data-science team with no async-framework experience picks LangGraph for its visual graph flow and gentler learning curve; a fintech cloud-native team picks Microsoft Agent Framework for fraud detection.
3. **Ecosystem compatibility** (existing stack): Azure-first → Microsoft Agent Framework (Azure OpenAI, Functions, Cognitive Search, Teams); OpenAI-API-centric or LangChain-plugin-heavy → OpenAI Agents SDK or LangGraph; plugin-heavy Python-native → CrewAI. Example: a government citizen-services chatbot on Azure with data-residency requirements picks Microsoft Agent Framework; a LangChain-plugin startup picks LangGraph.
4. **Performance/scalability needs**: high-concurrency enterprise deployments → Microsoft Agent Framework (separation of agents/orchestration, event-driven parallel sessions); simpler sequential use cases (contract summarization, email drafting) → CrewAI or OpenAI Agents SDK (sufficient power without hierarchy/threading overhead). Example: a retailer's Black-Friday multi-agent support system picks Microsoft Agent Framework for resumable stateless agents at scale; a legal firm's internal contract summarizer picks CrewAI.
- When to use: run through all four axes together for any real framework decision — picking on maturity alone while ignoring team readiness (or vice versa) is how projects either stall on a too-advanced framework or hit a capability ceiling on a too-simple one.

## Key Concepts
- **Durable execution / checkpointing as the cross-framework convergence point**: every framework in this book's survey (LangGraph's Checkpointers, Microsoft Agent Framework's superstep-boundary CheckpointManager, OpenAI SDK's session state serialization, CrewAI's task-level `human_input` pause) solves the same underlying problem — survive a pause, a failure, or a wait-for-human-approval without losing progress — via different mechanics.
- **Structured vs. unstructured state**: unstructured (dictionary-like) state favors rapid prototyping/flexibility; structured state (Pydantic/TypedDict, as LangGraph and CrewAI's "Structured State" both recommend) adds type-checking and validation — the chapter explicitly frames this as "rapid prototyping vs. production-grade maintainability," a specific instance of the Ch 3 validation principle applied to agent *state* rather than agent *I/O*.
- **A2A for cross-runtime, MCP for tool access** (Microsoft Agent Framework's richest-communication claim): A2A lets agents in *different frameworks/environments* collaborate via structured protocol messaging; MCP lets agents dynamically discover/invoke external tools — two distinct interoperability problems solved by two distinct protocols, both introduced in Ch 3 and now shown operating inside a concrete framework.
- **Low-code studios vs. code-first production**: every framework surveyed (AgentKit, LangGraph Studio, Crew Studio, Azure AI Foundry/DevUI) offers a visual layer for rapid prototyping while keeping code as the production path — a consistent industry pattern, not a one-off framework quirk.

## Mental Models
- When evaluating "is this task right for LangGraph," ask whether it genuinely needs cycles/branching and fine-grained execution control — if the workflow is fundamentally linear, LangGraph's low-level graph machinery is overhead you don't need (reach for CrewAI or OpenAI Agents SDK instead).
- Treat framework selection as a four-axis simultaneous optimization (maturity stage, team skill, ecosystem fit, scale need), not a single "which is most powerful" ranking — the chapter's four worked examples each pick a *different* framework precisely because the axes pulled differently each time.
- The multi-agent pattern choice (network/supervisor/nested/handoff within LangGraph, or the broader framework-level handoff/supervisor/sequential/concurrent patterns across all four frameworks) should follow from how much internal reasoning transparency sub-agents need to share — full transparency (network) trades verbosity for visibility; supervisor trades visibility for focus.
- Checkpointing/HITL support is not a "nice to have" feature row in a comparison table — it's the structural answer to the question "what happens when this long-running agentic task needs to pause for a human or recover from a crash," and every production-grade framework answers it somehow.

## Anti-patterns
- **Picking LangGraph (or any low-level framework) for a simple, linear single-agent task**: the chapter is explicit that CrewAI or OpenAI Agents SDK "may suffice" and are "easier to learn and faster to implement" for sequential/simple cases — reaching for maximal control when you don't need cycles/branching adds complexity without benefit.
- **Matching framework power to project needs while ignoring team readiness**: "investing in high-powered frameworks with limited team readiness can delay delivery" — a technically optimal framework choice that the team can't execute on is not actually the right choice.
- **Ignoring ecosystem fit for a "best in class" framework**: choosing a non-Azure framework inside a strict Azure-first, data-residency-constrained environment (or vice versa) creates integration friction the chapter calls "wasted effort on bridging gaps that could have been avoided" — ecosystem compatibility is "a multiplier for engineering velocity," not a tie-breaker.
- **Treating unstructured state as fine for complex production workflows**: both LangGraph and CrewAI explicitly recommend structured (Pydantic/TypedDict) state for anything beyond simple prototyping — unstructured state's flexibility becomes a maintainability liability as workflow complexity grows.
- **Assuming any one framework "wins" overall**: the chapter's explicit conclusion — "there is no single best framework; suitability is highly dependent on the specific problem domain" — makes framework-ranking without context a category error.

## Code Examples

**LangGraph — stateful graph with Pydantic state and a conversational loop**:
```python
class ConversationState(BaseModel):
    user_input: Optional[str] = None
    reply: Optional[str] = None
    history: List[str] = []

def reducer(state: ConversationState, updates: dict) -> ConversationState:
    return state.copy(update=updates)

builder = StateGraph(ConversationState)
builder.add_node("capture_input", RunnableLambda(capture_input))
builder.add_node("respond", RunnableLambda(respond))
builder.add_node("update_history", RunnableLambda(update_history))
builder.set_entry_point("capture_input")
builder.add_edge("capture_input", "respond")
builder.add_edge("respond", "update_history")
builder.add_edge("update_history", "display")
builder.add_edge("display", "capture_input")  # loop to continue conversation
app = builder.compile()
app.invoke(ConversationState())
```

**LangGraph — ReAct agent with built-in handoff tools (bank-bot swarm)**:
```python
kyc_agent = create_react_agent(
    model=llm,
    tools=[
        update_customer_address,
        create_handoff_tool(agent_name="card_unlock_agent"),
        create_handoff_tool(agent_name="pin_reset_agent"),
    ],
    prompt="You are a KYC agent. ... For requests related to PIN reset, transfer to pin_reset_agent ...",
    name="kyc_agent",
)

checkpoint = InMemorySaver()
supervisor = create_swarm(
    agents=[card_unlock_agent, pin_reset_agent, kyc_agent],
    default_active_agent="card_unlock_agent",
)
app = supervisor.compile(checkpointer=checkpoint)
```

## Reference Tables

**Memory handling across frameworks**: OpenAI Agents SDK — Context object, built-in tracing, no native long-term memory · LangGraph — Checkpointer-backed short-term (thread-scoped) + persistent-store long-term · Microsoft Agent Framework — AgentThreads + middleware + vector-store integration (Azure AI Search/Cosmos DB/Redis/Elasticsearch/PostgreSQL) · CrewAI — ChromaDB (short-term/entity, RAG-based) + SQLite (long-term, cross-session).

**Human-in-the-loop mechanics**: OpenAI Agents SDK — `ToolApprovalItem` interruptions, `result.state.approve()`/`.reject()`, resumable via state serialization · Microsoft Agent Framework — `ApprovalRequiredAIFunction` emits `FunctionApprovalRequestContent`, checkpoint pauses for hours/days · CrewAI — task-level `human_input` flag pauses/resumes execution · LangGraph — callback-function nodes + conditional branching (e.g., a `check_confidence` function routing to a `human_input` node) + checkpoints.

**Orchestration pattern richness**: OpenAI Agents SDK (handoff-based only) < CrewAI (Sequential/Hierarchical Processes + event-driven Flows) < LangGraph (graph-native loops/branching, LLM-decided next node) ≈ Microsoft Agent Framework (Sequential/Concurrent/Handoff/Group Chat/Magentic — the most orchestration patterns of the four).

## Worked Example

**Bank assistant bot, LangGraph swarm (handoff pattern)**: three agents — KYC agent, PIN reset agent, card-unlock agent — each built via `create_react_agent` with its own tools plus `create_handoff_tool` entries pointing at the other two. A `create_swarm` call wires them together with `default_active_agent="card_unlock_agent"` and compiles with an `InMemorySaver` checkpointer. User says "I need to reset my PIN" → the default card-unlock agent recognizes it's out of scope and calls `transfer_to_pin_reset_agent`; the PIN reset agent takes over, collects verification details, and resets the PIN. The same conversation then pivots to "I want to change my address" → PIN reset agent hands to the KYC agent, which collects and applies the new address — all within one continuous session, with the swarm's checkpointer preserving state across the hand-offs. This is the same underlying business requirement implemented in Ch 6 three different ways (Microsoft Agent Framework's `HandoffBuilder`, OpenAI SDK's `handoffs` list) — now shown as a LangGraph graph of agent-nodes that decide their own handoff targets, concretely illustrating the "Handoff" multi-agent pattern described above.

## Key Takeaways
1. LangGraph's Node/Edge/State graph model exists specifically to support cycles and conditional branching that linear chains cannot — the architectural requirement for genuinely iterative, long-running agentic reasoning.
2. LangGraph offers four multi-agent patterns (network, supervisor, nested, handoff) trading transparency for focus and flat structure for hierarchy — pick based on how much internal agent reasoning other agents genuinely need to see.
3. Checkpointing + pause/resume (via `interrupt`/`command` in LangGraph, and analogous mechanisms in every other framework) is the structural prerequisite for human-in-the-loop and fault-tolerant long-running agents — not an optional extra.
4. The four-framework comparison (Table 7.1) confirms no single framework dominates on every axis — OpenAI Agents SDK wins simplicity, LangGraph and Microsoft Agent Framework win control/durability, CrewAI wins role-based human-team mirroring.
5. Framework selection should run through four axes together: project maturity stage, team technical readiness, existing ecosystem/stack compatibility, and performance/scalability requirements — not a single "most powerful" ranking.
6. Structured state (Pydantic/TypedDict) over unstructured (dict-like) state is the production-readiness signal across frameworks, mirroring the Ch 3 principle that validated, schema-bound data beats ad hoc structures as system complexity grows.

## Connects To
- **Ch 6 (Frameworks Part 1)**: the bank-assistant-bot requirements and Microsoft Agent Framework/OpenAI Agents SDK/CrewAI implementations there are directly compared against this chapter's LangGraph implementation and the unified Table 7.1.
- **Ch 3 (Core Technologies)**: A2A and MCP introduced there are now shown operating concretely inside Microsoft Agent Framework's communication layer; CoT/CoD prompting underlies the "LLM-decided next node" mechanism in LangGraph's graph routing.
- **Ch 4 (Agentic AI Architecture)**: the handoff, workflow, and planning patterns introduced architecturally there map directly onto this chapter's LangGraph multi-agent patterns (handoff pattern ≈ LangGraph's Handoff pattern; workflow pattern ≈ Sequential/Hierarchical processes across frameworks).
- **Ch 8 (Value Add with Agentic AI)**: shifts from framework mechanics to industry-specific business-impact use cases (dynamic route optimization, personalized hospital treatment), building on the framework literacy established across Ch 6–7.
