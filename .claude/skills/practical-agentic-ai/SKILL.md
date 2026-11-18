---
name: practical-agentic-ai
description: "Knowledge base from \"Practical Approach to Agentic AI\" by Shweta Kamath and Vikas Gautam (BPB Publications, 2026). Use when designing agentic AI architecture (seven-layer stack, reflection/tool-use/workflow/planning/handoff patterns), implementing with frameworks (Microsoft Agent Framework, OpenAI Agents SDK, CrewAI, LangGraph), building data pipelines and governance for agentic systems, applying ML/RL/NLP foundations (MDP, value/policy-based methods, dialogue state tracking), using A2A/MCP communication protocols, agentic training and reward design, chain-of-thought/chain-of-draft prompting, productionizing agents (LLM selection, evaluation, security/governance, observability, deployment, scaling), or studying industry case studies (logistics, fraud detection, healthcare, retail) and the TnTMart end-to-end implementation."
---

# Practical Approach to Agentic AI
**Authors**: Shweta Kamath, Vikas Gautam | **Chapters**: 12 | **Generated**: 2026-11-18

## How to Use This Skill

- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about `MCP`, `reward hacking`, `LangGraph`, or another indexed topic; I find and read the relevant chapter
- **With a chapter** — ask for `ch06`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Three-generation AI progression** (Ch 1): traditional (rule-based) → generative (prompted content) → agentic (autonomous, goal-oriented, multi-step). The test for "agentic" is autonomous problem identification + solution proposal + plan execution, not just LLM use. Agentic frameworks must cover five pillars: core architecture, tool integration, task management, communication protocols, observability.

**Sense-think-act cycle + memory** (Ch 2): STM (contextual/working) vs. LTM (explicit semantic/episodic + implicit procedural). Context ≠ state — state is the durable cross-turn structure context feeds into. Multi-agent topologies: centralized, decentralized, hierarchical. Planning via ReAct (interleaved reason+act) or ReWOO (plan-then-execute).

**Core technologies** (Ch 3): ML paradigm selection by feedback signal (supervised/unsupervised/RL, formalized via MDP; value-based vs. policy-based RL methods). NLP pipeline (preprocess → semantic understanding → feature representation → intent ID → dialogue state tracking). Data validation via Pydantic schemas at every agent boundary. **A2A** = agent↔agent coordination; **MCP** = agent↔tool/data access (solves the N×M integration problem). Agentic training ≠ RL (linguistic self-critique + prompt/memory updates vs. scalar-reward gradient descent). CoT (transparency, costs tokens) vs. CoD (speed/cost, ≤5-word steps).

**Seven-layer architecture + five patterns** (Ch 4): perception → decision-making → task decomposition → communication → execution → orchestration → evaluation. Patterns: reflection (metacognitive self-critique), tool use (external grounding), workflow (deterministic multi-step pipelines), planning (non-deterministic dynamic environments), handoff (specialist routing with context preservation). Scale **out** (distributed multi-agent) not **up** (bigger model). Multimodal only when genuinely cross-referenced context is needed.

**Data management** (Ch 5): four challenges — sparse metadata, fragmentation, quality, integration (+ synthetic-data gaps). Countered by data catalogs (metadata/lineage/classification/discovery/access-policy), unified data lakes (shared agent memory), dimensionality reduction, data governance (ownership/quality/compliance/access-control/auditability), semantic layers + contextual metadata, federated/multi-agent access design, continuous policy monitoring.

**Frameworks** (Ch 6, 7): **Microsoft Agent Framework** (separates agent intelligence from workflow orchestration; graph-based Executors/Edges/supersteps; richest orchestration patterns — Sequential/Concurrent/Handoff/Group Chat/Magentic; MCP+A2A+OpenAPI native). **OpenAI Agents SDK** (four primitives: Agents, Tools/Guardrails, Handoffs, Sessions; lightweight, code-first; centralized manager-as-tool vs. decentralized peer handoff). **CrewAI** (role-based Crews for open-ended collaboration + event-driven Flows for deterministic control). **LangGraph** (low-level graph orchestration: Node/Edge/State; explicit cycles/branching; network/supervisor/nested/handoff multi-agent patterns; strongest pause-resume via Checkpointers + `interrupt`/`command`). No framework dominates every axis — choose via project maturity, team readiness, ecosystem fit, and scale need, together.

**Value add** (Ch 8): time/cost/reliability gains across logistics (real-time route re-optimization, predictive maintenance), fraud detection (behavioral analytics, reasoning-model evidence correlation), healthcare (explainable, HITL-gated personalized treatment), retail (demand forecasting, autonomous replenishment). Every case ends in a quantified ROI case: `(Savings − Cost) / Cost`.

**Design thinking → implementation** (Ch 9, 10): decompose any use case by asking where data lives, how the agent accesses it, what states exist, how derived values are computed, how exceptions are handled. TnTMart case study: PostgreSQL schema + generic MCP `execute_query` tool + dedicated `@ai_function` tools for custom logic + layered system prompts encoding business rules as explicit AND/OR conditions. Memory hierarchy: ephemeral/persistent/semantic/procedural, with context compression, relevance scoring, tiered access, shared context registries. AgentOps metrics must match each agent's agency/autonomy risk profile — never apply uniform metrics.

**Productionizing** (Ch 11): LLM selection is multi-dimensional (capability/latency/cost/compliance), resolved via model tiering + multi-model orchestration, not a leaderboard pick. No single architecture pattern suffices — compose them, and design HITL (Approve-or-Reject, Review-and-Edit) in from the start. Evaluation spans five dimensions (performance/reasoning/safety/UX/robustness) × three methodologies (offline/online/hybrid), plus multi-agent coordination metrics (redundancy, deadlocks, handoff efficiency, role fidelity). Security = API security + access control + content filters, verified by red teaming (Attack Success Rate). Observability = logs + metrics + traces (OpenTelemetry) answering "why did the agent choose that path," because agentic failure is *misaligned behavior*, not just downtime. Deployment: Kubernetes for steady high-volume, serverless for bursty/unpredictable. Scale multi-agent systems via async A2A pub-sub, but always pair with scaling *controls* (bounded instances, meaningful-metric gating, priority tiers).

**Future directions** (Ch 12): unresolved challenges — security (indirect prompt chaining), legacy integration, scalability, transparency. Economic impact is dual (displacement + new agent-economy roles); human override mechanisms are a governance requirement, not a feature. AGI blocked by four technological hurdles (computational scaling, transformer saturation, multimodal integration complexity, catastrophic forgetting); expert consensus timeline ~2040, genuinely uncertain. Democratization requires three pillars together: open-source platforms, cloud accessibility, and actually bridging the digital divide.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-introduction-to-agentic-ai.md) | Introduction to Agentic AI | Three-generation AI comparison, LLM decision phases, framework landscape, Urban Mobility Maestro |
| [ch02](chapters/ch02-essential-concepts-of-agentic-ai.md) | Essential Concepts of Agentic AI | Sense-think-act cycle, memory taxonomy, context vs. state, ReAct/ReWOO, multi-agent topologies, HITL modes |
| [ch03](chapters/ch03-core-technologies-enabling-agentic-ai.md) | Core Technologies Enabling Agentic AI | ML/MDP/RL, NLP pipeline, data validation, A2A, MCP, agentic training, CoT/CoD |
| [ch04](chapters/ch04-agentic-ai-architecture.md) | Agentic AI Architecture | Seven-layer architecture, five patterns, scalability/interoperability/performance, multimodal agents |
| [ch05](chapters/ch05-data-management-for-agentic-ai.md) | Data Management for Agentic AI | Data challenges, catalogs, data lakes, governance, semantic layers, federated access |
| [ch06](chapters/ch06-frameworks-for-agentic-ai-part-1.md) | Frameworks for Agentic AI — Part 1 | Microsoft Agent Framework, OpenAI Agents SDK, CrewAI (bank-bot implementations) |
| [ch07](chapters/ch07-frameworks-for-agentic-ai-part-2.md) | Frameworks for Agentic AI — Part 2 | LangGraph, four-framework comparison, framework selection criteria |
| [ch08](chapters/ch08-value-add-with-agentic-ai.md) | Value Add with Agentic AI | Route optimization, fraud detection, personalized treatment, inventory/supply chain, compliance |
| [ch09](chapters/ch09-practical-example-case-study.md) | Practical Example: Case Study | TnTMart design thinking (order tracking, proactive reminders, refunds) |
| [ch10](chapters/ch10-implementation-approach-to-agentic-ai.md) | Implementation Approach to Agentic AI | PostgreSQL schema, MCP server, Microsoft Agent Framework tools, scaling, memory, AgentOps |
| [ch11](chapters/ch11-productionizing-agentic-ai.md) | Productionizing Agentic AI | LLM selection, architecture patterns, evaluation, security/governance, observability, deployment, scaling |
| [ch12](chapters/ch12-future-of-agentic-ai.md) | Future of Agentic AI | Challenges, economic/social implications, AGI, democratization roadmap |

## Topic Index

- **A2A / MCP** → ch03, ch04, ch06, ch07, ch10, ch11, ch12
- **AgentOps / observability** → ch10, ch11
- **AGI** → ch12
- **Architecture patterns (reflection/tool-use/workflow/planning/handoff)** → ch04, ch11
- **Chain of thought / chain of draft** → ch03, ch04, ch10, ch11
- **Compliance (GDPR/HIPAA/DPDP)** → ch08, ch11
- **CrewAI** → ch06, ch07
- **Data catalogs / lakes / governance** → ch05
- **Dialogue state tracking** → ch03
- **Evaluation (offline/online/hybrid, multi-agent coordination)** → ch11
- **Fraud detection** → ch08
- **Framework selection / comparison** → ch06, ch07
- **Healthcare / personalized treatment** → ch08
- **Human-in-the-loop / human override** → ch02, ch04, ch08, ch10, ch11, ch12
- **LangGraph** → ch07
- **LLM selection (open-source vs. proprietary, model tiering)** → ch11
- **Markov Decision Process / RL methods** → ch03, ch04
- **Memory (ephemeral/persistent/semantic/procedural)** → ch02, ch10
- **Microsoft Agent Framework** → ch06, ch10
- **Multi-agent scaling / agent marketplace** → ch11
- **Multimodal agents** → ch01, ch04
- **OpenAI Agents SDK** → ch06
- **Red teaming / security / governance** → ch11, ch12
- **Reward design / reward hacking** → ch03
- **Route optimization (logistics)** → ch08
- **Scaling (infrastructure/database/orchestration/connection)** → ch10, ch11
- **Supply chain / inventory management** → ch08
- **TnTMart case study** → ch09, ch10
- **Kubernetes / serverless deployment** → ch11

## Supporting Files

- [glossary.md](glossary.md) — ~60 key terms with chapter cross-references
- [patterns.md](patterns.md) — 17 concrete architectural/technique patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision tables, formulas, and tells-and-smells quick reference

---

## Scope & Limits

This skill covers the book's 12 chapters in full, synthesized as structure, not verbatim text — frameworks, decision rules, worked examples, and code patterns are captured; the book's Appendices (Glossary, Bibliography, Further Readings) are summarized into this skill's own glossary rather than reproduced, and the author's running Discord-promotion asides are excluded as non-substantive. For hands-on implementation, combine with the book's companion GitHub repository (referenced throughout Ch 6, 7, 9, 10, 11) for runnable source code. This is a separate skill from `agentic-ai-unleashed` (a different book, different authors) — for cross-book synthesis on a topic, check both skills. For agentic-AI topics beyond either book's scope, ask directly.
