# Glossary

Key terms from *Practical Approach to Agentic AI*, with chapter cross-references.

**A2A (Agent-to-Agent Protocol)** — Google's open protocol for agent-to-agent communication: standardized language/message structure, agent identification (Agent Card) and discovery, task management (Task ID/Session ID), and secure authenticated exchange. Ch 3, 4, 6, 7, 11, 12.

**ABAC (Attribute-Based Access Control)** — Access-control model using combined attributes (role, data sensitivity, time, location, device context) for dynamic, fine-grained decisions; contrasted with simpler role-only RBAC. Ch 5, 11.

**Actor-critic method** — Policy-based RL technique where one component (actor) decides actions and another (critic) evaluates decision quality, smoothing learning in complex environments. Ch 3.

**Agent Card** — JSON document describing an agent's capabilities, endpoint, and authentication requirements for A2A discovery. Ch 3.

**Agent marketplace** — Architectural pattern treating agents as modular, discoverable, versioned, reusable services via a central registry, enabling dynamic runtime agent selection and elastic scaling. Ch 11.

**Agent sprawl** — Uncontrolled proliferation of agents producing a fragmented, hard-to-govern system; mitigated by event-driven orchestration and registries. Ch 10, 11.

**Agent thread** (Microsoft Agent Framework) — Multi-tenant conversation-state container; one stateless agent instance serves many isolated threads concurrently, serializable for persistence/resumption. Ch 6, 7.

**AgentOps** — Specialized lifecycle-management discipline for autonomous agents (design/implementation → evaluation → deployment → observability → feedback loop), extending DevOps/MLOps for non-deterministic, self-governing behavior. Ch 10, 11.

**Agentic communication** — Umbrella term for agent-agent (A2A) and agent-data-source (MCP) standardized interaction protocols. Ch 3.

**Agentic training** — Developing autonomous agents' independence/decision-making/adaptability via linguistic self-critique and architectural design, distinct from traditional RL's scalar-reward optimization. Ch 3.

**AGI (Artificial General Intelligence)** — Theoretical human-level general intelligence: cross-domain generalization, creative problem-solving, common-sense understanding, self-awareness; expert-consensus timeline centers ~2040 with significant uncertainty. Ch 12.

**Algorithmic impact assessment (AIA)** — Structured pre-deployment evaluation to identify/evaluate potential harms (especially bias) before an AI system, particularly at AGI scale, goes live. Ch 12.

**Attack Success Rate (ASR)** — Red-teaming output metric: percentage of adversarial attempts that successfully elicited unsafe/non-compliant agent behavior, broken down by risk category and attack complexity. Ch 11.

**Calibrated uncertainty** — Production evaluation metric assessing whether an agent recognizes when it doesn't know an answer and defers appropriately rather than guessing; primary reliability metric in high-stakes domains. Ch 11.

**Catastrophic forgetting** — Neural networks systematically forgetting previously learned tasks when trained on new ones; a core unresolved barrier to AGI-level cumulative learning. Ch 12.

**Centralized training, decentralized execution (CTDE)** — MARL approach: train agents with full environment context, deploy with local decision-making only. Ch 3.

**Chain of draft (CoD)** — Prompting technique emphasizing concise (≤5-word), information-dense reasoning steps instead of CoT's verbosity, optimized for speed/cost. Ch 3, 4.

**Chain of thought (CoT)** — Prompting technique where the model explicitly generates intermediate reasoning steps before a final answer; doubles as a loggable audit trail for consequential decisions in production. Ch 3, 4, 10, 11.

**Checkpointing** — Persisting complete workflow/agent state at defined points (e.g., superstep boundaries) to enable pause-resume, fault-tolerant recovery, and cross-environment migration. Ch 6, 7, 10, 11.

**Chain of thought logging** — Capturing every intermediate reasoning step of an agent's decision as a structured, auditable record — turning CoT from a debugging aid into verifiable documentation. Ch 11.

**Contextual embeddings** — Word/token representations that incorporate surrounding context (vs. static embeddings like Word2Vec), the mechanism behind LLM-era language understanding. Ch 3, 4.

**Crews** (CrewAI) — Teams of role-based AI agents with open-ended collaborative strategy, suited for creative/exploratory tasks; contrasted with structured Flows. Ch 6, 7.

**Curriculum learning** — Training agents on progressively harder tasks (simple → complex) to build foundational skills before tackling full complexity. Ch 3.

**Data fragmentation** — Data scattered across disparate systems (physical or logical), undermining holistic agent reasoning and explainability. Ch 5.

**Data lineage** — Full traceable journey of a dataset (source → transformations → downstream usage), essential for agent decision explainability and compliance. Ch 5.

**Deadlock (multi-agent)** — Failure mode where agents wait on each other indefinitely (e.g., an endless handoff loop), detected via coordination metrics and prevented by arbitration rules. Ch 11.

**Dialogue state tracking (DST)** — Maintaining structured conversational context (intent, slots, history, last action) across multi-turn interactions, updated each turn. Ch 3.

**Dimensionality reduction** — Techniques (PCA, t-SNE/UMAP, autoencoders, attention/SHAP feature selection) simplifying high-dimensional agent input while preserving decision-relevant information. Ch 5.

**Disentangled embeddings** — Representations that separate distinct concepts into independent sub-parts (e.g., β-VAE separating "car" from "red"), improving interpretability. Ch 3, 4.

**Executor** (Microsoft Agent Framework) — A workflow's processing unit (agent or custom logic component), connected to others via Edges within a graph-based Workflow. Ch 6, 7.

**Faithfulness** — Production evaluation metric: whether an agent's output is grounded in provided knowledge/tool results rather than fabricated. Ch 11.

**Flows** (CrewAI) — Event-driven, structured orchestration abstraction with explicit execution paths, state management, and conditional branching; can embed Crews as steps. Ch 6, 7.

**Handoff pattern** — Architectural pattern where one agent dynamically transfers control and context to a more specialized agent, triggered by capability mismatch, rule match, or explicit request. Ch 4, 6, 7, 11.

**High agency, low autonomy** — Design principle where an agent is granted the ability to act consequentially (agency) while deliberately withholding permission to act unsupervised (autonomy) for high-stakes/irreversible actions. Ch 10, 11.

**HITL (Human-in-the-Loop)** — Design pattern requiring human review/approval at defined checkpoints; implemented via Approve-or-Reject and Review-and-Edit patterns for critical/high-risk actions. Ch 4, 8, 10, 11, 12.

**Human override mechanism** — Foundational accountability control ensuring humans retain ultimate authority to intercept/review/reverse autonomous agent actions, embedded as a governance requirement, not an optional feature. Ch 12.

**Indirect prompt chaining** — Attack where an adversary manipulates an agent via a trusted intermediary's output (another agent, tool, or system) rather than direct prompting, exploiting inter-agent trust boundaries. Ch 12.

**Intent recognition** — Identifying the purpose/goal behind user input via rule-based, ML-based, or transformer-LLM-based approaches. Ch 3.

**LLM Cost Per Task (LCPT)** — Total token consumption × model cost for a complete interaction chain; the chapter's standard efficiency metric for comparing agents. Ch 10, 11.

**MARL (Multi-Agent Reinforcement Learning)** — RL subfield where multiple agents learn/interact in a shared environment, facing non-stationarity and credit-assignment challenges absent in single-agent RL. Ch 3, 4.

**MCP (Model Context Protocol)** — Anthropic's open standard for AI systems to access external data/tools via client-server architecture (servers/clients/hosts) and four capabilities (Tools, Resources, Prompts, Sampling), solving the N×M integration problem. Ch 3, 4, 6, 7, 10, 11, 12.

**MDP (Markov Decision Process)** — Formal RL framework: States, Actions, Rewards, Transitions, Discount factor, used to model sequential agent decision-making. Ch 3, 4.

**Model tiering** — Production cost-optimization strategy: smaller/faster models handle routine reasoning, escalating to larger models only for complex cases. Ch 11.

**Multimodal fusion** — Combining information across input modalities (early/feature-level, late/decision-level, or hybrid/intermediate fusion) to enable cross-referenced agent reasoning. Ch 4.

**N×M integration problem** — Without a shared protocol, each of N agents needs a custom connector for each of M data sources/tools; the problem MCP exists to eliminate. Ch 3, 5.

**Nash equilibrium** — Multi-agent stable state where no agent benefits from unilaterally changing strategy given others' strategies; the target outcome of MARL training in competitive/cooperative settings. Ch 4.

**OpenTelemetry (OTel)** — Vendor-neutral industry-standard framework for instrumenting/collecting/exporting telemetry (Metrics, Events, Logs, Traces — MELT), extended for generative AI with retrieval/evaluation/feedback data. Ch 11.

**Policy-as-code (PaC)** — Codifying governance rules as executable, enforceable policy rather than documentation, part of the continuous-assurance security paradigm for agentic systems. Ch 12.

**Reasoning model** — LLM tier (e.g., chain-of-thought-capable models) that decomposes problems into sub-steps and cross-references evidence, vs. a general-purpose LLM's pattern-matching response. Ch 3, 8.

**Red teaming** — Simulated adversarial attacks (prompt injection, policy bypass) to uncover agent vulnerabilities, measured via Attack Success Rate across risk categories. Ch 11.

**Reflection pattern** — Architectural pattern where an agent logs, reviews, critiques, and updates its own past decisions/reasoning (metacognition) to improve over time. Ch 4, 11.

**Reward hacking** — An agent optimizing the literal reward signal in a way that subverts the goal's intended spirit (e.g., CoastRunners boat spinning for points instead of finishing the race). Ch 3.

**Semantic caching** — Caching LLM responses by meaning/intent similarity (vector embeddings + similarity threshold) rather than exact match, reducing inference load. Ch 4.

**Semantic routing** — Directing incoming requests to the right LLM/cache/rejection path based on embedded-query similarity search against known task types. Ch 4.

**Shared Context Registry (SCR)** — Mechanism (via MCP) letting agents publish/subscribe to context summaries rather than redundantly re-fetching the same data, preventing duplication/inconsistency. Ch 10.

**Supersteps** (Microsoft Agent Framework) — Workflow execution model where all executors with pending messages run in parallel per step; the next superstep begins only if messages remain queued. Ch 6, 7.

**Swarm** (LangGraph) — Multi-agent coordination object where agents hand off control to each other based on their own judgment of customer intent, backed by a default active agent and checkpointer. Ch 7.

**Threshold breach** — Governance-violation signal counting when an agent proposes an action (e.g., a refund amount) exceeding a predefined limit, forcing HITL intervention. Ch 10.

**Tool use pattern** — Architectural pattern where an agent invokes external tools/APIs/services to extend capability beyond internal reasoning/memory. Ch 4, 11.

**Transformer performance saturation** — Empirical finding that scaling model size/training tokens beyond a threshold yields diminishing (1-2%) accuracy gains for exponentially higher compute — a named limit on AGI-via-scaling strategies. Ch 12.

**Workflow pattern** — Architectural pattern breaking tasks into a predefined or dynamically generated sequence of steps (static or dynamic), bringing structure/modularity/auditability to multi-step business processes. Ch 4, 11.
