---
name: agentic-ai-unleashed
description: "Knowledge base from \"Agentic AI Unleashed: A guide to designing, building, and deploying autonomous AI systems\" by Jayaram Nanduri and Anand Oka (BPB Publications, 2026). Use when designing agent architectures (single/multi-agent, reactive/deliberative/hybrid), implementing tool use (MCP/ACP/A2A/function calling), memory systems (RAG, working/episodic/long-term memory), agent development lifecycle, performance optimization, testing/QA, monitoring/observability, deployment strategies (cloud/edge/hybrid), agentic security (prompt injection, jailbreaking, auth/authz), AI ethics/alignment, or studying enterprise/consumer agentic AI applications."
---

# Agentic AI Unleashed
**Authors**: Jayaram Nanduri, Anand Oka | **Pages**: ~430 | **Chapters**: 18 | **Generated**: 2026-11-18

## How to Use This Skill

- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about `MCP`, `RAG`, `reward hacking`, or another indexed topic; I find and read the relevant chapter
- **With a chapter** — ask for `ch06`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Agency** (Ch 1) = perceive + recollect + plan + decide + act + adapt. An LLM alone is a *cognitive core*, not an agent — agency requires the surrounding perception/memory/action/adaptation loop. Agency is a system-level property that can emerge from networks of collaborating agents, not just single entities.

**Reasoning architectures** (Ch 2): symbolic (rigid, explainable), analogy-based (RAG's lineage), connectionist (transformers — the winning paradigm for cognitive cores, but needs the other two as supporting sub-modules). Decisions should use **MEU** (`Σ P(s|d)·U(s)`), not bare accuracy (**MAP**) — MAP is MEU's special case under symmetric costs.

**Standardization layer** (Ch 3, 6): **MCP** (Anthropic) = agent↔tool/data connectivity. **A2A** (Google) = agent↔agent coordination. **ACP** = agent-hosting/tool-use layer. These are complementary, not competing — build for all three where relevant. **ReAct** (interleaved reason+act) is the near-universal substrate pattern.

**LLM tiers** (Ch 4): base (fast/unaligned) → post-trained (safe, the agentic workhorse) → reasoning (slow/deep, best for complex logic). Assign tier **per agent role**: reasoning model for orchestrators, post-trained for narrow specialists.

**Agent architecture spectrum** (Ch 5): reactive (fast, no planning) ↔ deliberative (slow, handles complexity) ↔ hybrid (routes between them). Multi-agent specialization beats one monolithic agent past a complexity threshold — but don't reach for it below that threshold. Emergent behavior requires stochasticity + nonlinearity + feedback + scale — all four are already present in LLM-based multi-agent systems today, so expect it as systems scale.

**Tool/grounding layer** (Ch 6, 7): function calling (manual, low-level) vs. tool calling (framework-managed, default choice). RAG — structured (SQL-agent) or unstructured (vector/semantic) — fixes LLM hallucination by grounding in real data; always include a **rejection path** (score threshold) so "I don't know" is a reachable, honest answer.

**Memory ontology** (Ch 7): short-term (working/scratchpad, episodic) vs. long-term (declarative/procedural/semantic, implemented via RAG). Use a dedicated scratchpad for working memory, not raw conversation scrollback.

**Development lifecycle** (Ch 8): specify via needs→requirements→constraints→acceptance-criteria as an iterative loop, not a line. Climb the prototyping ladder (hard-coded rules → Wizard-of-Oz → MVA → incremental capability) before building a sophisticated agent. Agentic feedback **never gets sparse** — build a permanent triage system, not a bug tracker.

**Optimization layers** (Ch 9): model-level (quantization/distillation: 2–10x), system-level (caching/batching/parallelization: 5–10x), infrastructure-level (auto-scaling/load-balancing: scale economics). Diagnose which layer actually explains a performance problem before optimizing.

**Testing** (Ch 10): trace/span instrumentation is the foundation — test the reasoning path, not just output. Layer deterministic asserts (schema/safety/routing) beneath probabilistic LLM-as-judge evaluation. Continuous evaluation against an SME-curated gold-standard dataset never stops after launch.

**Observability** (Ch 11): monitoring (alerts, "what's wrong") vs. observability (decision trails, "why"). Use adaptive/behavioral baselines, not static thresholds, for anomaly detection — context-driven variation is legitimate.

**Deployment** (Ch 12): match cognitive function (perception/planning/execution/reflection) to its natural tier (edge/gateway/cloud). Evaluate latency at P95/P99, never average. Version independently across RUNTIME.MODEL.SCHEMA.CAPABILITIES. Design SLO-tiered graceful degradation (target→degraded→minimum) for every capability.

**Security** (Ch 13): the attack surface is the reasoning process itself (prompt injection, jailbreaking, social engineering, polymorphic LLM-generated attacks). No single defense suffices — layer input sanitization, prompt hardening, boundary enforcement, behavioral analysis, output filtering, execution isolation. Least privilege + agency-over-impersonation + strict delegation scoping bound blast radius. Audit the *reasoning*, not just the action.

**Ethics** (Ch 14): alignment failures are specification failures (reward hacking), not intelligence failures — specify implicit constraints, not just the literal goal. Tailor transparency to three audiences (user/developer/auditor). Treat bias mitigation as full-lifecycle and continuous, never a pre-launch checkbox.

**Applications** (Ch 15, 16): the dominant success pattern everywhere is **augmentation over automation** with a human-accountability layer retained for consequential decisions — true across enterprise (finance/healthcare/legal/education/manufacturing) and consumer (personal assistants/tutoring/health/home automation) domains alike.

**Future directions** (Ch 17): near-term (1–3yr) = better multimodal/tool-use/context; medium-term (3–7yr) = embodied agents in structured settings, mature multi-agent systems; long-term (7yr+) = genuinely speculative. Emergent capability is a safety problem specifically because it's unpredictable, not because novelty is inherently bad.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-introduction-to-agentic-systems.md) | Introduction to Agentic Systems | Six-verb agency ontology, agency-as-system-property |
| [ch02](chapters/ch02-theoretical-frameworks.md) | Theoretical Frameworks | Deduction/induction, Bayes' law, MEU, three reasoning tribes |
| [ch03](chapters/ch03-the-agentic-ecosystem.md) | The Agentic Ecosystem | Four-layer stack, ReAct, MCP, A2A |
| [ch04](chapters/ch04-llms-as-cognitive-cores.md) | LLMs as Cognitive Cores | Base/post-trained/reasoning tiers, structured output |
| [ch05](chapters/ch05-agent-patterns-and-design-paradigms.md) | Agent Patterns and Design Paradigms | LangGraph state/graph, reactive/deliberative/hybrid, emergence |
| [ch06](chapters/ch06-tools-and-interfaces.md) | Tools and Interfaces | Function vs. tool calling, MCP/ACP separation of concerns |
| [ch07](chapters/ch07-memory-systems.md) | Memory Systems | Memory ontology, structured/unstructured RAG, reflection |
| [ch08](chapters/ch08-development-lifecycle.md) | Development Lifecycle | Four-element spec, MVA, continuous feedback loop |
| [ch09](chapters/ch09-performance-optimization.md) | Performance Optimization | Quantization, distillation, critical-path, cost-aware execution |
| [ch10](chapters/ch10-testing-and-quality-assurance.md) | Testing and Quality Assurance | Span-level testing, LLM-as-judge, continuous evaluation |
| [ch11](chapters/ch11-monitoring-and-observability.md) | Monitoring and Observability | Logging schema, behavioral baselines, feedback loops |
| [ch12](chapters/ch12-deployment-strategies.md) | Deployment Strategies | Cloud/edge/hybrid, versioning, canary, graceful degradation |
| [ch13](chapters/ch13-security-and-safety.md) | Security and Safety | Prompt injection, jailbreaking, RBAC/ABAC, audit trails |
| [ch14](chapters/ch14-ethical-and-responsible-development.md) | Ethical and Responsible Development | Alignment, transparency, bias, privacy, governance |
| [ch15](chapters/ch15-enterprise-applications.md) | Enterprise Applications | BPA, scientific research, finance/health/legal/education/mfg |
| [ch16](chapters/ch16-consumer-applications.md) | Consumer Applications | Personal assistants, tutoring, entertainment, health, home automation |
| [ch17](chapters/ch17-future-directions.md) | Future Directions | Multimodal, embodiment, collective intelligence, alignment research |
| [ch18](chapters/ch18-example-of-conversational-agent.md) | Example of Conversational Agent | Full clinical-research-agent build (RAG+MCP+LangGraph+HITL) |

## Topic Index

- **A2A / ACP / MCP** → ch03, ch06, ch18
- **Alignment / reward hacking** → ch02, ch14, ch17
- **Audit trails / compliance** → ch13, ch14
- **Bias / fairness** → ch14
- **Canary deployment / rollback** → ch12
- **Causal reasoning / common sense** → ch17
- **Circuit breakers** → ch09, ch12
- **Collective intelligence / swarms** → ch17
- **Compression (quantization/distillation/pruning)** → ch09, ch12
- **Consciousness debates** → ch17
- **Delegation / impersonation vs. agency** → ch13
- **Edge deployment** → ch12
- **Embodied agents / robotics** → ch17
- **Emergent behavior** → ch05, ch17
- **Explainability** → ch13, ch14
- **Feature flags** → ch12
- **Function calling vs. tool calling** → ch06
- **Human-in-the-loop** → ch08, ch12, ch13, ch16, ch18
- **Jailbreaking / prompt injection** → ch13
- **LangGraph (state/graph model)** → ch05, ch07, ch18
- **LLM tiers (base/post-trained/reasoning)** → ch04
- **Memory (working/episodic/long-term)** → ch07
- **Multi-agent architecture** → ch05, ch17, ch18
- **Observability / tracing** → ch10, ch11, ch18
- **Performance optimization (model/system/infra layers)** → ch09
- **Polymorphic attacks** → ch13
- **RAG (structured/unstructured)** → ch07, ch18
- **RBAC / ABAC** → ch13
- **ReAct pattern** → ch03
- **Self-improving systems** → ch17
- **SLOs / graceful degradation** → ch12
- **Testing (span-level, LLM-as-judge, continuous eval)** → ch10
- **Versioning (multidimensional)** → ch12
- **Value learning** → ch14, ch17

## Supporting Files

- [glossary.md](glossary.md) — ~75 key terms with chapter cross-references
- [patterns.md](patterns.md) — 17 concrete architectural/technique patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision tables, formulas, and tells-and-smells quick reference

---

## Scope & Limits

This skill covers the book's 18 chapters in full (Parts I–V: Foundations, Technical Architecture, Engineering/Implementation, Operational Considerations, Applications/Future Directions), synthesized as structure, not verbatim text — frameworks, decision rules, and worked examples are captured; the book's own Index is excluded as non-substantive. For hands-on implementation in your own codebase, combine with project-specific tools and the book's companion GitHub repository (referenced in Ch 18) for runnable source code. For agentic-AI topics beyond this book's scope, check other skills or ask directly.
