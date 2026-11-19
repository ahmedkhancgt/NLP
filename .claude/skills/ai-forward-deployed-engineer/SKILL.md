---
name: ai-forward-deployed-engineer
description: "Knowledge base from \"AI FDE: The Forward Deployed Engineer\" by Ed Donner. Use when building production AI agents — tool-calling/agent loops, harness engineering, agent evals (LLM-as-judge, Cohen's kappa), agentic RAG and MCP, LangGraph durable execution and human-in-the-loop, multi-agent swarms, AWS Bedrock AgentCore deployment, or studying the course."
---

<!-- argument-hint: [topic, framework name, week/day, or lesson number] -->

# AI FDE: The Forward Deployed Engineer
**Author**: Ed Donner | **Chapters**: 15 (one per course day, 3 weeks x 5 days) | **Generated**: 2026-11-19

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about `MCP`, `harness engineering`, `LangGraph`, or another indexed topic; I find and read the relevant day-chapter
- **With a chapter** — ask for `ch08`, or by week/day — "Week 2 Day 3"
- **With a lesson number** — ask for "Lesson 60"; I find which day-chapter covers it (each chapter lists its lessons)
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

### Week 1 — Become an AI Engineer
- **Agent = Model + Tool + Loop.** Before any framework, understand the hand-rolled loop: model call → tool-call branch → append `role:tool` message → repeat until plain text. Frameworks (LangChain `create_agent`, Claude Agent SDK) are thicker wrappers over this same shape — `create_agent`'s return value is literally a LangGraph `CompiledStateGraph`.
- **Use benchmarks (Artificial Analysis Pareto front, METR) to shortlist, never to decide.** Always run your own accuracy/cost/speed experiment on your actual business task — a model "winning" a leaderboard can still lose on your data.
- **Prefer a fixed workflow over an agent when the tool-call sequence never varies.** An agent adds autonomy you don't need, at higher cost and latency, for zero accuracy gain.
- **Structured outputs: put rationale before decision** in the Pydantic schema so "thinking" happens before commitment, not as post-hoc justification.
- **Harness, Hardening, Humans (the Three H's of ADLC-as-coding).** Harness = the coding-agent platform; Hardening = linters/type-checks/tests/pre-commit hooks that constrain what the agent can get away with; Humans = a named accountable reviewer who demands evidence, never rubber-stamps a good demo.
- **Spec-driven development**: write `SPEC.md`/`AGENTS.md`/`TECH_STACK.md` in business language first, with progressive disclosure (short root file, detail loaded on demand) — let the agent plan with success criteria before it builds.
- **Open-weight ≠ free.** Someone always pays (your data, your electricity, your hardware). Quantization is a near-free efficiency win; distillation is a real intelligence cut — don't confuse them.
- **Harness engineering succeeded context engineering as the frontier skill** (prompt engineering → context engineering → harness engineering), and has seven components: context, workflow, state, sub-agents, agent teams, permissions, observability, self-improvement.
- **Self-improvement ≠ recursive self-improvement.** Self-improvement (agent updates its own process docs after a review) is practical today. RSI (improving the improvement process itself) is real but incremental, not an imminent takeoff.

### Week 2 — Become an AI Architect
- **The Three Traps**: (1) "I need an agent for X" — demand the measurable business problem first; (2) anthropomorphizing — don't draw agent org charts before measuring; start with one agent, add sub-agents only when they measurably help; (3) plausible vs. accurate — fluent LLM output is not validated output.
- **Business logic belongs in code, not prompts.** Deterministic math (deductibles, caps) as a Python tool collapsed UnderCover's cost-of-mistakes from $5,300 to $640.
- **Pair accuracy with Cohen's kappa.** A dominant-class baseline ("approve everything") can score 56%+ "accuracy" while being worthless — kappa rescales against chance agreement.
- **Agentic RAG subsumes traditional RAG's cottage industry** (query rewriting, expansion, re-ranking) by letting the agent decide how and how often to search. Use RAG for simple single-pass lookup, agentic RAG for multi-source/iterative needs, fine-tuning only when a generalizable skill (not facts) must be learned.
- **MCP's value is adoption and translation, not technical novelty.** Use stdio for your own locally-run servers, Streamable HTTP for remote OAuth-authenticated ones. Never use MCP for your own tools on your own agent — `@tool` is simpler.
- **Durable execution is non-negotiable for production agents.** LangGraph State (immutable snapshot) + Nodes (plain functions) + Edges (fixed/conditional) + a Postgres checkpointer scoped by thread ID per business entity = a workflow that survives a crash or a week-long human delay and resumes exactly where it left off.
- **`interrupt()` re-runs the whole node on resume.** Always put side effects (payments) in a separate, idempotent downstream node.
- **LangGraph is a general workflow engine, not an LLM-only tool.** A no-LLM dispatcher (state field with `operator.add` reducer + `Send` fan-out + `max_concurrency` + `RetryPolicy`) scales to a swarm with no bespoke worker-pool code; nesting an agent inside a dispatcher node needs no special API — a node is just a function.
- **Self-improvement must be validated on held-out data the improver never saw proposing the change**, or it overfits to the visible eval set's quirks. Gate every autonomous change behind human approval (`HumanInTheLoopMiddleware` or `interrupt()`).

### Week 3 — Become an AI FDE
- **The Five Phases of the ADLC** (agent-building sense — disambiguate from Week 1's coding-ADLC): Discovery → R&D → Development → Implementation → Maintenance, each with a sponsor sign-off and a tollgate.
- **Screen every use case with Three Questions** before RICE-ranking: avoids the Week 2 traps? uses proprietary data/evals? automation vs. acceleration vs. differentiation (the bar has risen past pure cost-cutting)?
- **Three security risks, three separate controls**: runtime (sandboxing — solved by a managed runtime), API scope (constrain tools, never just prompt wording — scope can't be talked around), prompt injection (layer a model-based screener, since keyword checks are trivially reworded).
- **The Lethal Trifecta** (private data + untrusted content + external communication) must be re-audited on every new integration, not just at initial design.
- **Defense in depth**: the same business rule (e.g., $1,000 limit) enforced independently at the model, the gateway Cedar policy, and the application layer — redundancy is the point.
- **A gateway call = authenticate → authorize (policy) → attach outbound credentials.** Roll out new Cedar policies in LOG_ONLY mode before ENFORCE.
- **Keep the context layer, the gateway, memory, and the database as four separate concerns with four separate owners** — resist vendor pitches that collapse them into one "context substrate." Start long-term memory with one strategy (semantic) and namespace it in code (never let the model set its own scope key) — this is the actual leakage-prevention mechanism.
- **Decouple ephemeral session identity from durable thread identity** on a managed runtime: a disposable microVM can die; the LangGraph checkpoint in Postgres is what survives and lets any new session resume via the same thread ID.
- **The handover conversation, not the code, is the deliverable.** Define multi-dimensional success criteria (throughput, accuracy, compliance) at Discovery and literally measure them live, with the client, at the end — never just declare victory.
- **You can never blame the LLM.** As builder/FDE you are accountable for outcomes; this is the line that separates an FDE from an end user of AI.

---

## Chapter Index

| # | Week/Day | Title | Key Frameworks |
|---|----------|-------|-----------------|
| [ch01](chapters/ch01-week1-day1-fde-roadmap-and-first-agent.md) | Week 1, Day 1 | The FDE Roadmap and Your First Agent Harness | Agent Harness (first principles), FDE Role Framework, FDE Definition (Narrow vs Broad) |
| [ch02](chapters/ch02-week1-day2-frontier-llms-and-first-claims-agent.md) | Week 1, Day 2 | Frontier LLMs, Model Selection and the First Claims Agent | Three-Layer AI Product Stack, Six Tricks GPT→ChatGPT, Pareto Front, Structured Outputs, Fixed Workflow vs Agent |
| [ch03](chapters/ch03-week1-day3-coding-agents-and-the-adlc.md) | Week 1, Day 3 | Coding Agents, the ADLC and Building the UnderCover Pilot | Three H's (Harness/Hardening/Humans), Two Meanings of ADLC, Spec-Driven Development, Be the Boss |
| [ch04](chapters/ch04-week1-day4-open-weight-models-and-local-inference.md) | Week 1, Day 4 | Open-Weight Models, Local Inference and Token Economics | Proprietary/Open-Source/Open-Weight taxonomy, Dense vs MoE, Quantization, OpenRouter Provider Pinning |
| [ch05](chapters/ch05-week1-day5-harness-engineering-and-self-improvement.md) | Week 1, Day 5 | Harness Engineering, Self-Improvement and Hardening the Harness | Prompt→Context→Harness Engineering, Seven Components of Harness, Build vs Configure Spectrum, RSI |
| [ch06](chapters/ch06-week2-day1-architecture-evals-observability.md) | Week 2, Day 1 | Agent Architecture Traps, Evals, and Observability with Langfuse | Three Traps, Five Reasons for Sub-Agents, Confusion Matrix/Kappa, Langfuse Experiments |
| [ch07](chapters/ch07-week2-day2-agentic-rag-mcp-memory.md) | Week 2, Day 2 | Agentic RAG, MCP Architecture, and Long-Term Memory | Agentic RAG, MCP host/client/server, RAG vs Agentic RAG vs Fine-Tuning, Long-Term Memory types |
| [ch08](chapters/ch08-week2-day3-langgraph-durable-execution-hitl.md) | Week 2, Day 3 | Durable Execution, LangGraph, and Human in the Loop | State/Nodes/Edges, Checkpoints & Threads, interrupt()/Command(resume), Human-Approval Design |
| [ch09](chapters/ch09-week2-day4-teams-swarms-dispatcher.md) | Week 2, Day 4 | Agent Teams, Swarms, and a No-LLM LangGraph Dispatcher | Teams vs Swarms vs Fleets, Dispatcher Pattern, Send/max_concurrency/RetryPolicy, Nested Graphs |
| [ch10](chapters/ch10-week2-day5-self-improvement-jev-judge.md) | Week 2, Day 5 | Self-Improvement, LLM-as-a-Judge with Jev, and RSI | LLM-as-Judge w/ Reference Answer, Improver's 5-Step Recipe, Held-Out Data Discipline, Keep/Rollback Rule |
| [ch11](chapters/ch11-week3-day1-adlc-governance-and-aws-setup.md) | Week 3, Day 1 | The Agent Development Lifecycle, Governance, and AWS Setup | Five Phases of ADLC, RICE Scoring, Three Ways to Host, Three-Tier Governance, Six Pillars of Responsible AI |
| [ch12](chapters/ch12-week3-day2-security-guardrails-and-agentcore-gateway.md) | Week 3, Day 2 | Security, Guardrails, and the AgentCore Gateway | Three Security Risks, Lethal Trifecta, Four Guardrail Kinds, API/MCP/Model Gateway Triad, Cedar Policy |
| [ch13](chapters/ch13-week3-day3-enterprise-context-layer-and-memory.md) | Week 3, Day 3 | The Enterprise Context Layer, Integrations, and AgentCore Memory | Four Parts of Context Layer, Gartner Lens, Four Memory Strategies, Namespace Partitioning |
| [ch14](chapters/ch14-week3-day4-agentcore-runtime-and-production-deployment.md) | Week 3, Day 4 | Production Deployment to AgentCore Runtime | Five Production Config Concerns, AgentCore CLI Lifecycle, Durable Session Resume, Sign Off vs Send Back |
| [ch15](chapters/ch15-week3-day5-capstone-copilot-and-course-wrap-up.md) | Week 3, Day 5 | The Capstone Copilot, Handover, and Course Wrap-Up | UnderCover Copilot Pattern, Second-Opinion Plan Review, Handover Conversation, Final Teardown Checks |

## Topic Index

- **Accountability / "never blame the LLM"** → Ch11
- **ADLC (two meanings)** → Ch03 (coding), Ch11 (agent-building)
- **AgentCore (Gateway/Policy/Memory/Runtime/Identity)** → Ch12, Ch13, Ch14, Ch15
- **Agent teams/swarms/fleets** → Ch09
- **Agentic RAG** → Ch07
- **Autonomy dial / risk-based approval** → Ch11, Ch14
- **Be the Boss / accountability model** → Ch03
- **Business logic in code vs. prompt** → Ch06
- **Cedar policy language** → Ch12, Ch13
- **Checkpoints, threads, durable execution** → Ch08, Ch14
- **Claude Agent SDK** → Ch05
- **Cohen's kappa** → Ch06
- **Confusion matrix / precision / recall** → Ch06
- **Context engineering** → Ch02, Ch05
- **Context layer (enterprise)** → Ch13
- **Copilot pattern** → Ch15
- **Dense vs. MoE models** → Ch04
- **Dispatcher / fan-out pattern** → Ch09
- **Distillation vs. quantization** → Ch04
- **Evals (three kinds)** → Ch06
- **Governance / three-tier / tollgates** → Ch11
- **Guardrails (four kinds)** → Ch12
- **Handover conversation** → Ch15
- **Harness engineering (seven components)** → Ch05
- **Held-out data discipline** → Ch10
- **Human-in-the-loop (interrupt/Command/HumanInTheLoopMiddleware)** → Ch08, Ch10, Ch14
- **Improver agent / self-improvement** → Ch05, Ch10
- **Jev (decision model)** → Ch02, Ch10
- **Langfuse (traces/datasets/experiments/scores)** → Ch06, Ch09
- **LangGraph (state/nodes/edges)** → Ch08
- **Lethal Trifecta** → Ch12
- **LLM-as-a-judge** → Ch06, Ch10
- **Local inference (Ollama/vLLM/quantization)** → Ch04
- **Long-term memory strategies** → Ch07, Ch13
- **MCP (host/client/server, stdio/HTTP)** → Ch03, Ch07, Ch12, Ch13
- **METR / Pareto front / benchmarks** → Ch02
- **Model selection methodology** → Ch02, Ch04
- **Namespace partitioning (memory)** → Ch13
- **OpenRouter / provider pinning** → Ch01, Ch04
- **Permission modes (coding agents)** → Ch03
- **Prompt injection** → Ch12
- **RICE scoring / use-case screening** → Ch11
- **RSI (recursive self-improvement)** → Ch05, Ch10
- **Security risks (runtime/scope/injection)** → Ch12
- **Sign off / send back** → Ch14
- **Spec-driven development** → Ch03, Ch05
- **Structured outputs** → Ch02
- **Sub-agents (five reasons)** → Ch06
- **Teardown / cost discipline** → Ch11, Ch15
- **Three Traps (agent architecture)** → Ch06
- **Tool calling fundamentals** → Ch01, Ch02

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — architecture/technique patterns with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the course's 15 days (45 lessons' worth of slides + lesson transcripts + lab code), synthesized as structure, not verbatim transcripts. The book's two appendices (Appendix A: The UnderCover Repository, Appendix B: The Bench Repository — full source code for the course's companion projects) are excluded as raw code dumps, not instructional content; the chapters reference them where relevant. For topics beyond this course, check other skills or ask directly.
