# Chapter 6: Frameworks for Agentic AI — Part 1

## Core Idea
Three production-grade agentic frameworks — Microsoft Agent Framework (enterprise orchestration, separation of agent-intelligence from workflow-control), OpenAI Agents SDK (minimal primitives: Agents/Tools+Guardrails/Handoffs/Sessions), and CrewAI (role-based Crews for collaborative work + event-driven Flows for deterministic control) — solve the same core problem (multi-agent orchestration, tool integration, state/context management) with different design philosophies, demonstrated concretely by building the same bank-assistant chatbot (card lock/unlock, PIN reset, KYC/address update, routed by a triage agent) in all three.

## Frameworks Introduced

### Microsoft Agent Framework — Layered Architecture
Unifies Semantic Kernel's enterprise patterns with AutoGen's dynamic multi-agent orchestration, built on `Microsoft.Extensions.AI` for provider-agnostic type safety. Four layers: user/application APIs → Agent Framework SDK (orchestration engine) → core components (model clients, agent threads, context providers, middleware) → external integrations (**MCP** for tool/data discovery, **A2A** for cross-runtime agent collaboration, **OpenAPI-first** import of any REST API as a tool with no wrapper code).
- Core separation of concerns: **agents** handle intelligence/reasoning (LLM-powered, stateless, reusable across threads); **workflows** handle orchestration/control flow (deterministic business logic: sequential, concurrent, group chat, handoff, magentic patterns).
- **Agent threads**: multi-tenant conversation-state containers — one agent instance serves many isolated threads concurrently; threads are serializable for persistence/resumption across application instances.
- **Tools**: `@ai_function`-decorated Python functions; `ToolMode` controls invocation (`AUTO` = model decides, `REQUIRED_ANY`/`REQUIRED(function_name=...)` = must use a tool, `NONE` = no tools).
- **Workflows**: graph of **Executors** (agents or custom logic) connected by **Edges** (conditional routing, fan-out, fan-in), executing in **supersteps** (all executors with pending input run in parallel each superstep; continues until no pending work remains) — graph-validated at build time (reachability, cycle detection, type alignment). **Checkpointing** persists complete workflow state (executor states, pending messages, shared context) at each superstep boundary for fault-tolerant resume, with pluggable storage (in-memory for dev, file/DB/blob for production).
- When to use: enterprise deployments needing tight Azure integration, multi-protocol interoperability (MCP+A2A+OpenAPI simultaneously), and production-grade observability/checkpointing out of the box.

### OpenAI Agents SDK — Four Core Primitives
A lightweight, minimal-abstraction Python framework (production evolution of the Swarm experiment): **Agents** (LLM + instructions + tools, the reasoning unit), **Tools/Guardrails** (external interfaces plus input/output safety validation), **Handoffs** (one agent delegates full conversational control to another specialist, auto-wrapped as a callable tool), **Sessions/Runner** (conversation-state tracking across turns, in-memory or SQLite-backed).
- **Guardrails**: `@input_guardrail`/`@output_guardrail`-decorated functions that return a tripwire result; a triggered tripwire stops agent execution immediately — this is the SDK's structural answer to the Ch 3 data-validation problem, applied at the agent-input/output boundary specifically.
- **Orchestration styles**: **LLM-led (declarative)** — a planner agent decides the next handoff via its own reasoning, encoding workflow logic in instructions (less code, less predictable); **Code-led (imperative)** — Python `if`/`else`/loops around `Runner.run()` calls explicitly select the next agent (more predictable, easier to debug).
- **Centralized vs. decentralized multi-agent patterns**: centralized — a manager agent wraps specialist agents as callable tools via `.as_tool()`, keeping one agent in control and presenting unified results (translation-manager example); decentralized — peer agents hold mutual `.handoffs` lists and delegate to each other based on their own judgment, suited to tasks without a rigid predetermined flow (conversational/email/Slack-agent example).
- When to use: rapid prototyping and production deployments that want the smallest possible abstraction surface — reach for centralized/manager pattern when one agent should own the final unified response, decentralized/peer pattern when the right next step genuinely depends on runtime content no single coordinator can fully anticipate.

### CrewAI — Crews vs. Flows
Role-based multi-agent framework optimized for speed/cost (minimized token usage) and asynchronous concurrent-crew throughput. Two complementary abstractions, both built on **Agents** (role + goal + backstory + tools), **Tasks** (unit of work with expected output, sequential or parallel, auto dependency-managed), and **Tools** (pre-built — SerperDevTool, ScrapeWebsiteTool, PGSearchTool — or custom, via `BaseTool` subclass or `@tool` decorator):
- **Crews**: teams collaborating toward a goal with open-ended strategy — best for creative/exploratory/research-synthesis tasks where agents autonomously decide how to divide the work.
- **Flows**: event-driven, structured orchestration with explicit execution paths, state management, and conditional branching (`@start`, `@router`, `@listen` decorators) — best for predictable, auditable processes like onboarding or API orchestration. Flows can embed Crews as steps, combining autonomous collaboration with deterministic control.
- Agent-to-agent communication happens through **structured task handoffs**, not free-form messaging: the Crew controller injects one task's output directly into the next task's prompt context — a deterministic pipeline, not an open conversation.
- When to use: Crews for open-ended collaborative problem-solving (market research, content synthesis); Flows for processes that must follow a defined, auditable sequence (customer onboarding, regulated workflows); combine them when a structured pipeline needs a creative/exploratory sub-step.

## Key Concepts
- **Separation of agent intelligence from workflow control** (Microsoft Agent Framework's core principle): agents reason, workflows decide control flow — this makes agents modeless/reusable across different orchestration contexts, a reusability property the other two frameworks achieve differently (OpenAI SDK via stateless Agent objects + Sessions; CrewAI via Agent/Task separation).
- **Handoff vs. tool-wrapped delegation**: a handoff transfers full conversational control and context to another agent (OpenAI SDK, Microsoft Agent Framework); wrapping an agent `.as_tool()` (OpenAI SDK's centralized pattern) keeps the calling agent in control and treats the sub-agent as a callable function — these are architecturally different delegation models with different "who owns the final response" semantics.
- **Supersteps** (Microsoft Agent Framework workflow execution model): all executors with pending messages run in parallel within one superstep; the next superstep begins only if messages remain queued — gives predictable, bounded concurrency rather than fully async chaos.
- **Guardrail tripwire**: a structured pass/fail signal from an input/output validation function that halts execution on failure — distinct from simply raising an exception, because it carries auditable detail about what check ran and why it failed.
- **Triage/coordinator pattern**: all three frameworks implement the same architectural idea (one agent identifies intent and routes to a specialist) with different mechanics — Microsoft Agent Framework's `HandoffBuilder` + `set_coordinator`, OpenAI SDK's triage agent with a `handoffs` list, CrewAI's Flow `@router` dispatching to the right Crew.

## Mental Models
- When comparing frameworks, don't ask "which is best" — ask "which delegation/control model (deterministic workflow graph, LLM-led vs. code-led handoff, role-based Crew vs. event-driven Flow) matches how predictable vs. how adaptive this specific task needs to be."
- Treat guardrails/validation at agent boundaries as non-negotiable infrastructure, not an advanced feature — the OpenAI SDK's input/output guardrails are the same defensive principle as Ch 3's Pydantic validation, applied specifically at the point where an LLM-driven agent hands control to another system or back to a user.
- A framework's "enterprise-readiness" signal is checkpointing + observability + multi-protocol support (Microsoft Agent Framework), not feature count — evaluate framework maturity by what happens when something fails mid-workflow, not just by what it can do when everything succeeds.
- The same bank-assistant-bot requirements (triage → PIN/card/address specialists) implemented three different ways is the chapter's deliberate teaching device: architecture choice is orthogonal to business logic — the "what" (investigate_card, unlock_card, reset_pin tools) stays constant while the "how agents coordinate" changes completely.

## Anti-patterns
- **Choosing a framework by feature list instead of by control-flow fit**: a task needing fully auditable, step-ordered execution (loan approval, KYC) is a poor fit for an LLM-led declarative orchestration style where the model decides the next step — use code-led orchestration or a Flow instead.
- **Skipping guardrails because "the agent usually gets it right"**: guardrails exist for the cases where the agent doesn't — an agent without input/output validation has no structural stop mechanism when it drifts off-topic or produces unsafe output.
- **Using decentralized peer-to-peer handoffs for a process that must happen in a fixed sequence**: decentralized delegation is good for adaptive routing (conversational agent → email or Slack agent, as needed) but wrong for processes requiring strict ordering/auditability — that calls for a workflow graph (Microsoft Agent Framework) or a Flow (CrewAI), not open peer handoffs.
- **Treating Crew (collaborative, open-ended) and Flow (structured, deterministic) as interchangeable in CrewAI**: using a Crew for a regulated, must-follow-this-order process sacrifices auditability; using a Flow for genuinely open-ended research sacrifices the collaborative flexibility a Crew provides.

## Code Examples

**Microsoft Agent Framework — tool-equipped agent + handoff workflow**:
```python
card_agent = chat_client.create_agent(
    instructions="You are a customer support agent that handles card requests...",
    name="card_agent",
    tools=[investigate_card, unlock_card],
    tool_choice=ToolMode.AUTO,
)

workflow = (
    HandoffBuilder(name="multi_tier_support", participants=[triage, pin_agent, card_agent, address_update_agent])
    .set_coordinator(triage)
    .add_handoff(triage, [pin_agent, card_agent, address_update_agent])
    .with_termination_condition(lambda conv: sum(1 for msg in conv if msg.role.value == "user") > 8)
    .build()
)
```

**OpenAI Agents SDK — centralized manager-as-tool pattern**:
```python
manager = Agent(
    name="Manager",
    instructions="Use the tools to translate text as requested.",
    tools=[
        spanish_agent.as_tool(tool_name="to_spanish", tool_description="Translate to Spanish"),
        french_agent.as_tool(tool_name="to_french", tool_description="Translate to French"),
    ]
)
result = Runner.run(manager, "Translate 'hello' to Spanish and French.")
```

**OpenAI Agents SDK — guardrails**:
```python
content_agent = Agent(
    name="Content Creator",
    instructions="Generate marketing content.",
    output_type=ContentOutput,
    guardrails=[
        InputGuardrail("Reject requests over 5000 words"),
        OutputGuardrail("Ensure content is family-friendly"),
    ]
)
```

**CrewAI — agent + task + crew, then Flow routing**:
```python
agent_reset_pin = Agent(
    name="ResetPinAgent", role="pin_resetter",
    goal="Reset the customer's PIN after verifying their identity.",
    backstory="You are a banking agent specialized in resetting customer PINs...",
    llm=llm, allow_delegation=True,
)
task_reset_pin = Task(description="Reset the customer's PIN using tool reset_pin.",
                       expected_output="Response from the PIN reset operation.",
                       agent=agent_reset_pin, tools=[reset_pin])
crew = Crew(agents=[agent_reset_pin], tasks=[task_reset_pin], process=Process.sequential)

@router(start_method)
def second_method(self):
    if self.state.topic == SupportTopic.PINReset:
        return "pin_reset"
    # ... dispatches to the matching @listen-decorated crew handler
```

## Reference Tables

**Framework comparison at a glance**:
| Dimension | Microsoft Agent Framework | OpenAI Agents SDK | CrewAI |
|---|---|---|---|
| Core abstraction | Agents + graph Workflows (Executors/Edges/supersteps) | Agents, Tools/Guardrails, Handoffs, Sessions | Agents + Tasks, under Crews (collaborative) or Flows (structured) |
| Delegation model | `HandoffBuilder` graph with coordinator | Handoffs (control transfer) or `.as_tool()` (centralized) | Structured task handoff via Crew controller |
| Fault tolerance | Built-in checkpointing per superstep | Session persistence (in-memory/SQLite) | Flow state management |
| Protocol integration | MCP + A2A + OpenAPI natively | Primarily function tools + managed tools (web search, code interpreter) | Pre-built + custom tools via `BaseTool`/`@tool` |
| Best fit | Enterprise, Azure-centric, multi-protocol | Lightweight prototyping to production, minimal abstraction | Role-based collaboration, YAML-configurable, cost-optimized |

## Worked Example

**Bank assistant bot, decentralized triage pattern (OpenAI Agents SDK)**: a `triage_agent` holds `handoffs=[pin_agent, card_agent, address_update_agent]`; each specialist's own `handoffs` list points back only to the triage agent (specialists escalate back, not sideways to each other, in this implementation). A user says "I need to reset my PIN" → triage hands off to PIN Agent, which collects customer ID/card number/DOB/email, calls the `reset_pin` tool, and confirms via email. The same user then asks to update their address → PIN Agent hands back to Triage Agent, which routes to Address Update Agent, which collects the new details and calls `update_customer_address`. This demonstrates the full round-trip: triage → specialist → (back to triage) → different specialist, all preserving context without the user re-identifying themselves at each hop — the concrete payoff of the handoff pattern introduced architecturally in Ch 4.

## Key Takeaways
1. Microsoft Agent Framework separates agent intelligence from workflow control via a graph-based Executor/Edge model with superstep execution and built-in checkpointing — the strongest fit for enterprise, multi-protocol (MCP+A2A+OpenAPI), fault-tolerant deployments.
2. OpenAI Agents SDK reduces agentic systems to four primitives (Agents, Tools/Guardrails, Handoffs, Sessions) and offers both LLM-led (declarative) and code-led (imperative) orchestration, plus centralized (manager-as-tool) and decentralized (peer handoff) multi-agent patterns.
3. CrewAI's Crews (collaborative, open-ended) and Flows (structured, event-driven) are complementary, not competing — Flows can embed Crews as steps to combine deterministic control with creative collaboration.
4. Guardrails (OpenAI SDK) and validation/checkpointing (Microsoft Agent Framework) are each framework's structural answer to the same underlying need: a stop/recovery mechanism at agent/workflow boundaries, not an optional add-on.
5. All three frameworks solve triage/routing the same conceptual way (intent-detecting coordinator → specialist handoff) with different mechanics — recognizing this shared pattern transfers across frameworks faster than learning each one's API from scratch.
6. Framework choice should be driven by required control-flow predictability (deterministic graph vs. LLM-led routing vs. role-based collaboration), protocol/ecosystem needs (MCP/A2A/OpenAPI breadth), and deployment context (enterprise Azure integration vs. lightweight prototyping vs. cost-optimized concurrent crews) — not by feature-list comparison alone.

## Connects To
- **Ch 4 (Agentic AI Architecture)**: the handoff pattern, tool use pattern, and workflow pattern introduced architecturally there are given full concrete framework implementations here.
- **Ch 3 (Core Technologies)**: Pydantic-based data validation there is the same defensive principle underlying OpenAI SDK guardrails and structured-output agents here.
- **Ch 7 (Frameworks for Agentic AI — Part 2)**: introduces LangGraph and provides the direct four-way comparison (LangGraph vs. Microsoft Agent Framework vs. OpenAI Agents SDK vs. CrewAI) deferred from this chapter.
- **Ch 11 (Productionizing Agentic AI)**: the checkpointing, observability, and enterprise-deployment features previewed here (especially Microsoft Agent Framework's Azure AI Foundry deployment) get full production treatment there.
