# Patterns

Concrete architectural and technique patterns from *Practical Approach to Agentic AI*, with when-to-use, how, and trade-offs. Cross-references point to chapter files for full detail.

## 1. Five-Layer Architecture Pattern Composition
**When to use**: designing any production agentic system — never select a single pattern in isolation.
**How**: start by analyzing/redesigning the underlying business workflow before choosing a pattern; evaluate task complexity, required autonomy, environment adaptability, specialization need, and latency/cost tolerance; compose patterns (Planning depends on Tool Use for sub-tasks; a failed plan triggers a Reflection loop; an orchestrator supervises specialists each using their own Planning/Reflection/Tool Use).
**Trade-off**: more composed patterns = more capability but more governance/debugging complexity. Always add HITL (Approve-or-Reject, Review-and-Edit) for consequential/irreversible steps regardless of which patterns compose the rest.
**Source**: Ch 4, 11.

## 2. Schema-First Agentic Design
**When to use**: before writing any database-backed agent tool.
**How**: build a well-normalized, richly-described data model first; agents infer relationships (e.g., "adding a product needs both product ID and customer ID") from schema clarity, not from hardcoded logic.
**Trade-off**: upfront schema investment pays off in agent reasoning accuracy and SQL-generation correctness; skipping it pushes complexity into brittle, hand-coded tool logic.
**Source**: Ch 10.

## 3. MCP Generic Query Tool vs. Dedicated `@ai_function` Tools
**When to use**: deciding how an agent should access a backend data store or perform custom logic.
**How**: expose a generic `execute_query`-style MCP tool for varied, agent-composed SQL access against a documented schema; reserve dedicated custom tools (`@ai_function`) for calculations, external API calls, or actions needing validation/side effects beyond "run this SQL."
**Trade-off**: generic MCP access scales better across varied queries but requires excellent schema documentation; dedicated tools are more reliable for fixed, high-stakes logic but proliferate if overused.
**Source**: Ch 10.

## 4. Layered System Prompt (Business-Rule-as-Code) Pattern
**When to use**: any agent governing a conditional, policy-bound process (refund approval, loan eligibility, compliance checks).
**How**: structure instructions in sections — general hygiene → schema understanding → interaction basics → explicit AND-conditions for approval → explicit OR-conditions for rejection → personalization/closing. Never leave approval/rejection logic implicit.
**Trade-off**: verbose prompts cost tokens but eliminate ambiguity in financially/legally consequential decisions — the cost is worth it specifically where errors are expensive or hard to reverse.
**Source**: Ch 10.

## 5. High Agency, Low Autonomy (Escalation-Gated Autonomy)
**When to use**: any agent that can initiate consequential, hard-to-reverse actions (payments, refunds, data deletion).
**How**: grant the agent the technical capability (agency) to act, but require human approval below a defined confidence/risk threshold (autonomy deliberately kept low) via Approve-or-Reject or threshold-breach escalation.
**Trade-off**: slower end-to-end resolution for gated actions, but bounds financial/compliance/reputational risk — the right trade for high-stakes, low-volume decisions; wrong trade for high-volume, low-stakes ones (use full autonomy there instead).
**Source**: Ch 8, 10, 11, 12 (human override mechanism).

## 6. Design-Thinking Decomposition for New Use Cases
**When to use**: scoping any new agentic feature before writing code.
**How**: ask in sequence — where does the data live? how does the agent access it? what are the possible states/stages? how is a derived value (ETA, risk score) computed? how is the exception/delay/ambiguous case handled?
**Trade-off**: slower upfront design time, but prevents building an agent without a tractable data/state foundation to reason over.
**Source**: Ch 9.

## 7. Recompute-Cadence Matching (Batch vs. Real-Time)
**When to use**: any personalization or prediction signal that could be computed in real time but doesn't need to be.
**How**: match compute frequency to how fast the underlying signal actually changes — use a hybrid (e.g., monthly baseline + lightweight daily adjustment) rather than defaulting to either pure real-time or pure static batch.
**Trade-off**: real-time recomputation is more "fresh" but can be wastefully expensive when the underlying pattern changes slowly; pure batch misses sudden shifts. The hybrid balances cost, freshness, and simplicity.
**Source**: Ch 9.

## 8. Escalation as a First-Class Branch
**When to use**: designing any agent that handles customer disputes, claims, or ambiguous requests.
**How**: design the "forward to human" path with the same rigor as the happy path — explicit triggers (unlisted reason, ambiguous evidence, policy mismatch, threshold breach), not a generic catch-all.
**Trade-off**: adds design and engineering effort for a path that (by design) runs less often than the happy path — but it's where trust and accountability actually live.
**Source**: Ch 9, 10, 11.

## 9. Memory Hierarchy with Optimization Strategies
**When to use**: any agent that must "remember" across turns, sessions, or customers at scale.
**How**: separate ephemeral (session, Redis/in-memory), persistent (PostgreSQL/vector DB), semantic (embedding similarity recall), and procedural (encoded workflows) memory tiers; apply context compression (periodic summarization), relevance scoring (prune low-value entries), tiered access (namespace by agent role), and a Shared Context Registry for cross-agent reuse.
**Trade-off**: unmanaged memory grows exponentially and causes context drift/retrieval lag/storage bloat; the optimization strategies cost engineering effort but are mandatory past a certain scale.
**Source**: Ch 2, 10.

## 10. Risk-Profile-Matched AgentOps Metrics
**When to use**: instrumenting any agent for production observability.
**How**: classify the agent by agency (can it act on/modify the environment?) and required autonomy (should it act unsupervised?); apply efficiency metrics (latency, LCPT) for high-volume/low-risk agents, and governance metrics (tool-selection accuracy, HITL escalation rate, threshold breach, policy mismatch) for high-agency/low-autonomy agents.
**Trade-off**: uniform metrics across all agents miss what actually matters for each risk profile — efficiency dashboards don't catch governance failures, and governance-heavy instrumentation is wasted overhead on low-risk agents.
**Source**: Ch 10, 11.

## 11. Multi-Model Orchestration with Dynamic Routing
**When to use**: production systems with varied task complexity across requests.
**How**: route routine/simple tasks to smaller, faster, cheaper models; escalate complex reasoning to larger models; add failover to backup models on primary failure.
**Trade-off**: added routing-logic complexity and potential latency from the routing decision itself, in exchange for materially lower average cost and better resilience to single-provider outages.
**Source**: Ch 11.

## 12. Offline → Online → Hybrid Evaluation Progression
**When to use**: validating any agentic system before and during production exposure.
**How**: start with offline testing against curated datasets/golden answers (low-risk, repeatable regression testing); progress to online A/B/canary testing on a live traffic subset; converge on hybrid continuous evaluation (automated LLM-as-judge scoring + human review of flagged edge cases feeding back into retraining).
**Trade-off**: offline alone misses real-world variance; online alone risks real users on an unvalidated system; hybrid costs ongoing operational investment but is the only approach that scales safely.
**Source**: Ch 11.

## 13. Asynchronous A2A Pub-Sub Scaling
**When to use**: scaling a multi-agent system under variable, bursty load (e.g., a product launch spike in one query category).
**How**: deploy each agent as an independent containerized microservice; route via publish-subscribe over a message bus (Kafka/RabbitMQ/NATS); let Kubernetes HPA spin up additional instances of just the overloaded agent type, which self-register with the bus with zero reconfiguration of other agents.
**Trade-off**: requires upfront investment in decoupled messaging infrastructure and shared-state consistency (Redis/Dapr), but enables elastic, isolated scaling of exactly the bottlenecked component without touching the rest of the system.
**Source**: Ch 11.

## 14. Controlled Scaling (Bounded Elasticity)
**When to use**: any auto-scaling multi-agent deployment, to prevent cost/complexity runaway.
**How**: set explicit min/max instance bounds and resource quotas per agent type; scale on meaningful signals (queue depth, latency, SLA adherence) rather than raw request count; apply priority tiers (critical agents scale aggressively, low-priority agents stay capped); add predictive/scheduled scaling for known-pattern spikes.
**Trade-off**: adds policy-design and monitoring overhead, but prevents elastic scaling from becoming an unbounded cost and governance liability.
**Source**: Ch 11.

## 15. Agent Marketplace (Modular Discoverable Agents)
**When to use**: large organizations with many narrow-domain agents across business units.
**How**: package each agent as a self-contained service with a capability-described interface in a central registry; let orchestrators/meta-agents discover the best-fit agent at runtime by capability/load/historical accuracy; version agents for parallel A/B rollout with attached governance policies.
**Trade-off**: significant platform investment (registry, versioning, governance attachment) but pays off at scale by avoiding hardcoded routing logic and enabling reuse across teams.
**Source**: Ch 11.

## 16. Interpretable-by-Design Observability (Logs + Metrics + Traces)
**When to use**: any production agent, especially where decisions must be auditable.
**How**: instrument all three pillars together — structured logs (full I/O + reasoning chain + trace/session/user IDs), metrics (latency breakdown, error/throttle rates, agentic execution metrics like step count/looping rate, economic metrics like cost-per-task), and hierarchical traces (OpenTelemetry-based, capturing intent/action/confidence/policy-constraint semantic fields, not just HTTP calls).
**Trade-off**: substantial instrumentation investment, but it's the only way to answer "why did the agent choose that path" — infrastructure metrics alone cannot reveal misaligned (vs. merely broken) behavior.
**Source**: Ch 11.

## 17. Federated / Multi-Agent Data Access
**When to use**: agents operating across organizational, geographic, or regulatory boundaries that forbid centralizing all data.
**How**: let agents query data where it lives (APIs, virtual views, data mesh) rather than replicating into one central store; scope agent data access via RBAC+ABAC per role; share outputs between agents via pub-sub or message-queue inter-agent protocols rather than direct data duplication.
**Trade-off**: more query complexity and latency than full centralization, but satisfies compliance (GDPR, data residency) and avoids costly, risky full replication.
**Source**: Ch 5.
