# Chapter 5: Agent Patterns and Design Paradigms

## Core Idea
Agent architectures range from **reactive** (fast, stateless stimulus→response) to **deliberative** (slower, plans ahead, handles complexity) to **hybrid** (routes between the two); beyond a complexity threshold, **multi-agent** networks with specialized roles beat any single monolithic agent, and sufficiently large/complex agent networks exhibit genuine **emergent behavior** — system-level properties not present in, or predictable from, any individual agent.

## Frameworks Introduced

### LangGraph's State + Execution Graph Model
Two core concepts implement any agent workflow:
- **State**: a `TypedDict` holding all data flowing through the graph — shared across nodes, updated (merged, not directly mutated) by the framework after each node runs. This is *the* mechanism by which agents communicate.
- **Execution graph**: nodes = functions that process state (your agents/tasks); edges = triggers between nodes. **Regular edges** always fire; **conditional edges** route based on a logic function over the state. A node has either unconditional edges OR conditional edges emanating from it, never both (preserves determinism). Self-loop edges support retry/refinement loops.
- How parallel execution works: a node with edges to multiple targets triggers them concurrently; each parallel node gets an independent state copy, no shared memory between them, and LangGraph merges results back (deterministic merge order, but don't rely on it — use conditional logic for required sequencing). All parallel paths must converge before continuing.
- Common structures: **Fan-out/Fan-in** (map-reduce: router → [parallel analyses] → aggregator) and **conditional parallel** (some branches fire in parallel only when a condition selects them).
- When to use: any multi-step agentic workflow with decision points — which describes essentially all real agentic systems.

### Reactive vs. Deliberative vs. Hybrid Agent Behavior
- **Reactive**: direct stimulus→response, no internal state/planning. Fast, predictable, easy to debug; cannot handle complexity or remember past interactions. Use for FAQ bots, alert systems, simple classifiers, speed-critical paths.
- **Deliberative**: builds an internal model → plans → executes via reasoning. Handles complex/novel problems, adapts plans to goals, considers future consequences. Slower, resource-heavy, brittle (plans can fail in the real world), less deterministic. Use for strategic planning, complex problem solving, multi-step workflows needing look-ahead.
- **Hybrid**: dynamically shifts between reactive and deliberative based on situation — a router directs each incoming case to the fast or deep path as needed.
- When to use which: reactive when speed is critical and problems are well-defined; deliberative when complex reasoning/multi-factor planning is required; hybrid when the workload mixes both kinds of cases (the realistic default for production systems).

### Multi-Agent Architecture (Role Specialization)
Past a complexity threshold, a single monolithic agent becomes cumbersome; splitting into specialized agents (router, sentiment analysis, knowledge base, order management, technical support, response composer in the book's customer-support example) each doing one job yields better quality, lower cost, and better performance than one "jack of all trades" agent.
- Why it works: specialization + division of labor + debate/critique between agents improves reasoning and enables meta-cognitive oversight (reducing catastrophic single-point failures), mirroring how human societies scale capability through coordination rather than individual intelligence growth.
- How: define a hierarchical/orchestrated topology — a router classifies and dispatches, conditional edges send work to the right specialist(s), a composer agent synthesizes the final output, with escalation branches for human-in-the-loop cases.

### Emergent Behavior in Agentic Networks
Complex systems of interacting elements exhibit behaviors that cannot be predicted from, or explained by, the individual components alone ("more is different" — Anderson, 1972). Four mechanisms produce it:
1. **Stochasticity** — individual agents give different outputs for the same input (LLM hallucination, modulated by temperature).
2. **Nonlinear interactions** — small input changes have disproportionate effects (LLM prompt sensitivity).
3. **Adaptive feedback** — agents change behavior based on system-level outcomes (shared state propagation).
4. **Network scale/topology** — density and connection pattern determine whether coordination emerges at all.
- **Phase transition** is the tell-tale sign: a sudden qualitative behavior change past a critical threshold (traffic flows smoothly below a density, jams above it; networks fragment below a connectivity level, a giant connected component emerges above it).
- When to use this lens: when scaling an agentic network (more agents, more specialties, more feedback loops), expect qualitatively new system-level behavior to appear past some threshold — plan for it rather than being surprised by it.

## Key Concepts
- **Router agent**: classifies/dispatches incoming requests to the correct specialist path.
- **Shared state as communication medium**: nodes never directly mutate state; the framework merges returned updates — this is what makes multi-agent coordination tractable and debuggable.
- **MCP-compliant knowledge bases**: the book notes that in production, mock knowledge-base lookups (as used in the demo) would sit behind MCP servers (Ch 3/6), not ad-hoc dictionaries.

## Mental Models
- Treat "single agent vs. multi-agent" as a complexity-threshold decision, not an ideological one: a single LLM call with good prompting is genuinely sufficient for small task counts and simple business logic — don't over-architect prematurely.
- Diagnose agent-type mismatch by symptom: brittle plans failing unpredictably → you built deliberative where reactive would suffice (too much latency/complexity for the problem); agent missing obvious context/complexity → you built reactive where deliberative was needed.
- When evaluating a multi-agent system's safety, look for all four emergence preconditions (stochasticity, nonlinearity, feedback, scale) — their presence predicts you *will* eventually see emergent behavior, good or bad, as the network grows.

## Anti-patterns
- **Letting an agent hallucinate "factual" answers that should be retrieved**: the book's single-agent demo deliberately lets the LLM hallucinate an order's shipping status — in production this must be real RAG/tool lookup, not model guesswork (fixed in the multi-agent version via a dedicated `order_management_agent`).
- **Relying on parallel-merge ordering**: LangGraph's merge order is deterministic *within a run* but should not be depended upon across runs/logic — use explicit conditional sequencing instead.
- **Ignoring the possibility of collusive/adversarial emergence**: as networks scale, agents can spontaneously develop inter-agent protocols, implicit signaling, or even collusive coordination (e.g., algorithmic price-fixing) that was never explicitly programmed — safety review must anticipate this, not just test programmed behaviors.
- **Treating emergent capability gains as purely positive**: the same mechanisms that enable collective knowledge discovery also enable deception/manipulation as emergent strategies — emergence is a double-edged sword requiring designed safeguards, not optimism alone.

## Reference Tables

**Reactive vs. Deliberative decision guide**

| Factor | Reactive | Deliberative |
|---|---|---|
| Speed | Fast | Slow (planning overhead) |
| Determinism | High (same input→same output) | Lower (plan quality varies) |
| Handles complexity | No | Yes |
| Resource cost | Low | High |
| Best for | FAQ bots, alerts, classifiers, speed-critical | Strategic planning, multi-step workflows, look-ahead |

**Emergence preconditions checklist**

| Mechanism | Present in LLM-based multi-agent systems via |
|---|---|
| Stochasticity | Temperature-driven output variation / hallucination |
| Nonlinear interactions | Prompt sensitivity |
| Adaptive feedback | Shared state propagation between agents |
| Network scale/topology | Number of agents × connection pattern (the one still-open design parameter) |

## Worked Example

**Customer support multi-agent network (the chapter's running implementation):**
- State: `CustomerSupportState` carries `customer_query`, `query_type`, `sentiment`, `order_id`, `technical_issue`, `knowledge_base_results`, `proposed_solution`, `requires_escalation`, `final_response`.
- Graph: `router_agent` (classify type/urgency/order_id via JSON-structured LLM output) → `sentiment_analysis_agent` (emotional tone, frustration 1–10, escalate boolean) → `knowledge_base_agent` (keyword-matched article lookup) → **conditional fan-out** to `order_management_agent` (real lookup in `MOCK_ORDERS`, no hallucination) or `technical_support_agent`, or skip straight to → `response_composer_agent` (synthesizes everything into the final customer-facing reply) → conditional edge to escalate or END.
- Contrast with the single-agent version: the single agent correctly classified query type/urgency/sentiment in one LLM call, but *hallucinated* the order's ship date since it had no real data access. The multi-agent version fixes this by giving a dedicated `order_management_agent` factual access to `MOCK_ORDERS` — a direct, concrete demonstration of why specialization beats a single do-everything agent once accuracy matters.

```python
# Conditional routing logic used to build the graph's branching edges
def should_check_order_or_technical(state: CustomerSupportState) -> Literal["order", "technical", "skip"]:
    if state["order_id"] or state["query_type"] == "order_inquiry":
        return "order"
    elif state["query_type"] == "technical_support":
        return "technical"
    return "skip"

graph.add_conditional_edges(
    "knowledge_base", should_check_order_or_technical,
    {"order": "order_management", "technical": "technical_support", "skip": "response_composer"}
)
```
- **What it demonstrates**: conditional-edge routing is how a shared state field (`query_type`, `order_id`) drives which specialist agent(s) fire next — the mechanical heart of multi-agent orchestration.

## Key Takeaways
1. Don't reach for multi-agent architecture by default — a single well-prompted agent suffices for small task counts and simple logic; add specialization only once complexity/accuracy demands it.
2. Classify every agent-design decision as reactive, deliberative, or hybrid first — it determines the speed/intelligence trade-off you're accepting.
3. Use shared state (not direct inter-agent calls) as the coordination substrate — it's what makes LangGraph-style systems debuggable and mergeable under parallelism.
4. Specialized agents with real data access (not hallucination) are the concrete fix for factual-accuracy problems that single monolithic agents can't solve safely.
5. Expect emergent behavior as agent networks scale — all four preconditions (stochasticity, nonlinearity, feedback, scale) are already present in LLM-based multi-agent systems today.
6. Emergence is double-edged: design explicit safeguards against adversarial/collusive emergent behavior rather than assuming scale only produces beneficial capability gains.

## Connects To
- **Ch 3 (The Agentic Ecosystem)**: LangGraph is one of the MAS frameworks introduced there; MCP-compliant knowledge bases are referenced from Ch 3/6.
- **Ch 4 (LLMs as Cognitive Cores)**: temperature and structured-output techniques from Ch 4 are used directly in the router/sentiment agent implementations here.
- **Ch 6 (Tools and Interfaces)**: the next chapter formalizes tool/function calling, which this chapter's agents use informally via mock lookups.
- **Ch 13 (Security and Safety)**: the emergence risks flagged here (collusion, deception) connect directly to that chapter's safety framework.
