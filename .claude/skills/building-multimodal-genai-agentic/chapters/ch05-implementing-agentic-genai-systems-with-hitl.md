# Chapter 5: Implementing Agentic GenAI Systems with Human-in-the-loop

## Core Idea
Agentic GenAI systems are built from a catalog of 19 reusable multi-agent design patterns (parallel, sequential, loop, router, hierarchical, HITL, etc.), and production-grade trustworthy systems combine a retrieval-generation pipeline with an explicit human-in-the-loop (HITL) approval/retry gate, orchestrated via LangGraph's StateGraph.

## Frameworks Introduced
- **19 Multi-agent design patterns** (the chapter's core framework — see Reference Tables for full catalog): Parallel, Sequential, Loop, Router, Aggregator, Network, Hierarchical, Human-in-the-loop, Shared tools, Database with tools, Memory transformation, Planner-executor, Critic/validator, Negotiator, Multimodal agent, Voting/consensus, Supervisor-subordinate, Watchdog/recovery, Temporal planner.
  - When to use: pick by task shape — independent subtasks → Parallel; staged refinement → Sequential/Loop; conditional dispatch → Router; synthesis from multiple sources → Aggregator; decentralized peer collaboration → Network; delegated authority → Hierarchical/Supervisor-subordinate; safety-critical judgment → HITL; quality gating → Critic/validator; fault tolerance → Watchdog/recovery; scheduling → Temporal planner.
- **HITL (Human-in-the-loop) pattern**: Agent execution halts after generation → human reviews/approves/rejects → execution resumes or retries (bounded, e.g., 3 attempts) → graceful fallback message if still rejected.
  - How: implement `human_approval_required()` returning True/False from user input; wrap generation in a retry loop (max N attempts) that calls this gate after each generation.
- **Multi-agent HITL RAG via LangGraph StateGraph**: Decompose the RAG pipeline into isolated agents (retrieval agent, generation agent, human-feedback agent) connected by a directed `StateGraph` with conditional edges routing back to generation on rejection or to `END` on approval.
- **Agentic AI vs. AI agents**: AI agents = narrow, reactive, single-step task executors (e.g., FAQ bot, scheduler) with bounded autonomy. Agentic AI = multi-agent systems with goal-setting, multi-step planning, orchestration, persistent memory, and adaptive learning (reinforcement/meta-learning) across broader domains.

## Key Concepts
- **Orchestrator agent**: Higher-level controller that decomposes complex goals into subtasks and routes/integrates specialized agents' outputs.
- **Retry loop with bounded attempts**: Standard HITL safety mechanism — e.g., max 3 regeneration attempts before a graceful "rejected after multiple attempts" fallback, preventing infinite loops.
- **StateGraph (LangGraph)**: Directed graph of named nodes (agent functions) and edges (including conditional edges) that encodes workflow control flow, entry points, and termination (`END`).
- **Persistent memory (agentic AI)**: Shared context across agents/sessions enabling agents to build on prior work, track dependencies, and refine strategy over time — a defining trait distinguishing agentic AI from simple AI agents.
- **Source citation tracking**: Attaching `source_documents` to generated answers for traceability/grounding, carried through from retrieval agent to generation agent to final output.

## Mental Models
- Think of the 19 patterns as a vocabulary, not a prescription — real systems compose several patterns together (e.g., this chapter's HITL RAG combines Sequential + HITL + Loop).
- Use "AI agent = tool, agentic AI = system of thinkers" as the fast heuristic for scoping a project: if the system only executes one bounded task reactively, you have an AI agent; if it plans, delegates, and adapts across steps, you're building agentic AI.
- Treat HITL as a safety valve, not a crutch — it should gate specific high-stakes decision points (final answer approval), not every micro-step, and must have a bounded retry/fallback so the system doesn't hang indefinitely.
- The Critic/validator pattern and HITL pattern are structurally identical (produce → review → approve/revise) — the difference is simply whether the reviewer is an AI agent or a human.

## Anti-patterns
- **Unbounded retry loops in HITL systems**: without a maximum attempt count and fallback message, a rejected answer can loop forever; always cap retries (e.g., 3) and provide a graceful exit.
- **Monolithic, centralized logic masquerading as "agentic"**: a single linear chain with no isolated, testable agent boundaries is not a true multi-agent system, even if it uses an LLM — real agentic design requires modular, independently addressable agents connected via an orchestration graph.
- **Treating narrow AI agents as agentic AI**: a single reactive task executor (e.g., one FAQ-answering function) lacks the planning, delegation, and adaptive learning that define agentic AI — mislabeling inflates expected capability.
- **Skipping human validation in high-stakes domains**: deploying autonomous agent output directly in legal, healthcare, or financial contexts without a HITL checkpoint risks acting on hallucinated or ungrounded content.

## Code Examples
```python
def human_approval_required():
    return input("\nApprove the answer? (yes/no): ").strip().lower() != "yes"
```
- **What it demonstrates**: Minimal HITL gate function — returns True (needs retry) unless the user explicitly types "yes".

```python
retries = 3
for attempt in range(retries):
    ...
    if not human_approval_required():
        return result["answer"]
    print("\nRetrying with same question...")
return "Answer rejected after multiple attempts."
```
- **What it demonstrates**: Bounded retry loop pattern wrapping generation with human approval, with a graceful fallback after exhausting attempts.

```python
def retrieval_agent(state):
    return {"documents": retriever.get_relevant_documents(state["question"])}

def generation_agent(state):
    result = rag_chain.invoke({"question": state["question"]})
    return {"answer": result["answer"], "source_documents": result.get("source_documents", [])}

def human_feedback_agent(state):
    approved = not human_approval_required()
    return {"approved": approved}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieval_agent)
workflow.add_node("generate", generation_agent)
workflow.add_node("validate", human_feedback_agent)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "validate")
workflow.add_conditional_edges(
    "validate",
    lambda state: "end" if state.get("approved") else "generate",
    {"end": END, "generate": "generate"}
)
```
- **What it demonstrates**: Full multi-agent HITL RAG workflow as a LangGraph StateGraph — isolated retrieval/generation/validation agents wired with a conditional retry edge back to "generate".

## Reference Tables
| Pattern | Structure | Best for |
|---|---|---|
| Parallel | Agents run concurrently, merge outputs | Independent subtasks, speed via parallelism |
| Sequential | Agent A → B → C pipeline | Layered/staged transformation |
| Loop | Generate → evaluate → feedback → repeat | Iterative refinement until threshold |
| Router | Classify input → dispatch to specialist agent | Conditional task delegation |
| Aggregator | Multiple inputs → merge/normalize → single output | Synthesizing diverse sources |
| Network | Decentralized peer-to-peer agent mesh | Distributed, emergent problem-solving |
| Hierarchical | Planner/supervisor → worker agents → aggregated result | Task decomposition, clear responsibility |
| Human-in-the-loop | Halt → human review → resume | Safety-critical domains (legal, healthcare) |
| Shared tools | Agents share a common toolkit/API/DB | Consistency, reduced duplication |
| Database with tools | Agent + tool enrich/persist data in real time | State tracking, enrichment workflows |
| Memory transformation | Tool/agent updates memory → future decisions adapt | Personalization, learning over sessions |
| Planner-executor | Planner forms plan → executors act → feedback to planner | Multi-step reasoning, traceable execution |
| Critic/validator | Producer output → validator reviews → approve/revise | Quality control, reduced hallucination |
| Negotiator | Agents exchange proposals → agreement/failure | Simulated stakeholder negotiation |
| Multimodal agent | Route by modality → specialized processing → fusion | Multi-sensory reasoning (text/image/audio) |
| Voting/consensus | Agents predict in parallel → aggregator picks best | Robustness via ensemble/majority vote |
| Supervisor-subordinate | Supervisor monitors/corrects autonomous workers | High integrity with autonomous execution |
| Watchdog/recovery | Monitor health → detect failure → rerun/escalate | Production resilience, fault tolerance |
| Temporal planner | Plan includes timestamps/deadlines | Scheduling, long-horizon/time-aware tasks |

| Dimension | AI agents | Agentic AI |
|---|---|---|
| Scope | Narrow, well-defined tasks | Higher-order, multi-step problems |
| Behavior | Reactive, single-step | Proactive, multi-step planning |
| Coordination | Independent | Multi-agent collaboration, orchestration |
| Memory | Minimal/none | Persistent, contextual |
| Learning | Rule-based/supervised | RL, meta-learning, adaptive |

## Worked Example
A fully local, HITL multi-agent RAG system: PDF parsed/chunked with metadata → local embedding model (Nomic) + Chroma vector store → hybrid retriever (BM25 + vector similarity) → `retrieval_agent` fetches relevant chunks → `generation_agent` runs a ReAct-style chain to produce an answer with source citations → `human_feedback_agent` prompts "Approve the answer? (yes/no)" → on rejection, LangGraph's conditional edge routes back to `generate` (up to 3 retries) → on approval or exhausted retries, the workflow reaches `END` and returns the final answer (or a rejection fallback message) to the user. The whole pipeline runs with no external API dependency.

## Key Takeaways
1. The 19 design patterns are composable building blocks — real agentic systems typically combine several (e.g., Sequential + HITL + Loop) rather than using exactly one.
2. HITL requires a bounded retry mechanism and graceful fallback — never leave it open-ended.
3. LangGraph's StateGraph with conditional edges is the standard mechanism for encoding retry/approval logic in multi-agent workflows.
4. Isolating retrieval, generation, and validation into separate agent functions (vs. one monolithic chain) is what makes a system genuinely modular, testable, and extensible.
5. Agentic AI is defined by multi-step planning, orchestration, persistent memory, and adaptive learning — not merely by using an LLM or calling a tool.
6. HITL checkpoints are essential, not optional, for safety-critical domains (legal, healthcare, enterprise compliance).

## Connects To
- **Ch 1**: Builds directly on the agent/tools/reasoning/planning primitives and agentic-RAG-vs-non-agentic-RAG framing introduced there.
- **Ch 2**: Extends the multi-agent orchestration concept first previewed in the multimodal-systems chapter.
- **Ch 6**: The grading/validation mechanism here (critic/validator pattern, HITL approval) is extended into formal grading mechanisms for two/multi-stage GenAI systems.
- **Ch 14-15**: The planner-executor and router patterns here recur directly in agentic text-to-SQL architecture decision-making.
