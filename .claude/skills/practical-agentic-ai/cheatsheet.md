# Cheatsheet

## ML Paradigm Selection

| Signal | Paradigm |
|---|---|
| Have labeled examples | Supervised learning |
| Only unlabeled data, looking for structure | Unsupervised learning |
| Sequential decisions, delayed feedback/reward | Reinforcement learning (model via MDP) |

## RL Method Selection

- Need "which action is best here" per state, small/discrete space → value-based (Q-learning).
- Continuous/probabilistic actions, complex environment → policy-based / actor-critic (Policy Gradient).
- Need to simulate ahead before acting, limited data → model-based planning (MCTS, value iteration).
- Simple trial-and-error, no internal environment model needed → model-free planning.

## Intent/Search Technique Selection

| Need | Approach |
|---|---|
| Few fixed choices, high precision | Rule-based |
| Moderate scale, labeled training data available | ML-based (Bag-of-Words/TF-IDF + classifier) |
| Wide scenario coverage, limited labeled data | Transformer LLM-based (state of the art) |
| Keyword search misses synonyms | Agentic RAG / semantic search over vector DB |

## CoT vs. CoD Selection

| Constraint that binds | Use |
|---|---|
| Need maximum transparency/accuracy, tokens are cheap | Chain of Thought (zero-shot / few-shot / Auto-CoT / program-aided / ToT / self-consistency / Reflexion) |
| Need speed/low latency/low cost, accuracy comparable | Chain of Draft |
| Search-heavy planning problem, budget allows | Tree of Thought |
| Need iterative self-correction | Reflexion (CoT + self-critique) |

## Framework Selection (Four-Axis Check)

Run all four before deciding:
1. **Project maturity**: prototype/MVP → CrewAI or OpenAI Agents SDK. Production/regulated → Microsoft Agent Framework or LangGraph.
2. **Team readiness**: less async/agent experience → Microsoft Agent Framework or CrewAI. Higher technical maturity → OpenAI Agents SDK or LangGraph.
3. **Ecosystem fit**: Azure-first → Microsoft Agent Framework. OpenAI-API-centric/LangChain-plugin-heavy → OpenAI Agents SDK or LangGraph. Python-native plugin-heavy → CrewAI.
4. **Scale need**: high-concurrency enterprise → Microsoft Agent Framework. Simple sequential tasks → CrewAI or OpenAI Agents SDK.

## Multi-Agent Pattern Selection (LangGraph / general)

| Need | Pattern |
|---|---|
| Max transparency, all agents see all steps | Network (shared scratchpad) |
| Specialists shouldn't see each other's internals | Supervisor |
| Genuinely hierarchical org of agents | Nested |
| Active agent best judges next specialist | Handoff |

## Architecture Pattern Decision Flow

1. Tasks clearly defined/predictable? → **Workflow** pattern.
2. Needs external functionality (REST API, DB)? → **Tool use** pattern.
3. Multi-step, non-deterministic, dynamic environment? → **Planning** pattern.
4. Needs self-correction after failure? → **Reflection** pattern (often layered on Planning).
5. Task exceeds one agent's scope/expertise? → **Handoff** pattern.
6. Always: add HITL (Approve-or-Reject / Review-and-Edit) for consequential/irreversible steps, regardless of the above.

## Open-Source vs. Proprietary LLM

| Priority | Choose |
|---|---|
| Data sovereignty, offline operation, deep customization | Open-source (LLaMA, Mistral, Falcon) |
| Fastest time-to-market, top-tier out-of-box performance | Proprietary (GPT-4o, Claude, Gemini) |
| Cost-sensitive/compliance-heavy workload within a hybrid stack | Open-source tier |
| High-value reasoning within a hybrid stack | Proprietary tier |

## Deployment Platform Selection

| Traffic shape | Platform |
|---|---|
| Steady, high-volume, predictable | Kubernetes (better TCO via volume discounts) |
| Bursty, unpredictable, intermittent | Serverless/FaaS (scale-to-zero eliminates idle cost) |
| Bursty but latency-critical | Serverless + Provisioned Concurrency / pre-warmed instances |
| Long-running iterative workflow on serverless | Durable orchestration layer (Step Functions / Durable Functions) with checkpointing |

## Scaling Axes Checklist (audit all four independently)

1. Infrastructure (containerized replicas, HPA by session/schedule).
2. Database (read replicas, partitioning, connection pooling).
3. Agent orchestration (event-driven pub-sub to prevent agent sprawl).
4. Connection/MCP layer (multiple instances behind a load balancer, connection caching).

## Scaling Control Rules (prevent runaway)

- Set min/max instance bounds and resource quotas per agent type.
- Scale on queue depth / latency / SLA adherence — never on raw request count alone.
- Apply priority tiers: critical agents scale aggressively, low-priority agents stay capped.
- Use predictive/scheduled scaling for known-pattern spikes (launches, billing cycles).

## AgentOps Metric Selection by Risk Profile

| Agent profile | Primary metrics |
|---|---|
| High-volume, low-risk, read-only (e.g., order tracking) | Average Task Duration, LLM Cost Per Task (LCPT) |
| Moderate autonomy, behavior-triggered (e.g., proactive reminders) | Task Success Rate, Context Utilization Score, LCPT, Redundant Steps, Tool Selection Accuracy, Decision Turn Count, Recovery Rate |
| High agency / deliberately low autonomy (e.g., refunds, payments) | Tool Selection Accuracy, HITL Escalation Rate & Success, Threshold Breach, Policy Mismatch |

Rule: Robustness (high Recovery Rate) directly increases LLM cost via retry/re-planning — only prioritize it where failure consequences are severe.

## Evaluation Dimension Checklist (all five required)

1. Task performance (success rate, error rate, resolution time, efficiency).
2. Reasoning quality (coherence, logical consistency, domain soundness).
3. Safety/compliance (hallucination risk, bias, policy adherence, tool-use appropriateness).
4. User experience (dialogue naturalness, latency, clarity, trust).
5. Operational robustness (degraded-mode handling, failure recovery, workload adaptability).

Plus, for multi-agent systems: redundancy, deadlocks, handoff efficiency (hops per resolution), role fidelity, conflict resolution, systemic outcomes.

## Security Layering (apply all three, not one)

1. API security (auth, authz, rate limiting, input validation, anomaly detection).
2. Access control (RBAC/ABAC scoping exact data/tool/capability access).
3. Content safety filters (runtime output scanning for toxicity/leakage/policy violations).

Then: red team (PyRit-style adversarial scan) to verify enforcement actually holds — measure via Attack Success Rate.

## Compliance Quick Reference

| Aspect | GDPR (EU) | HIPAA (US) | DPDP (India) |
|---|---|---|---|
| Scope | EU residents' personal data | Protected health information | Indian citizens' personal data |
| Consent | Explicit, informed, revocable | Patient consent for PHI | Explicit consent required |
| Erasure | Strong right to be forgotten | Access yes; limited deletion | Right to access/correct/erase |
| Localization | Not mandatory; cross-border restricted | No specific requirement | Some localization obligations |

## Tells & Smells

- Agent "succeeds" but used far more tool calls than expected → check trace for silent retry loops or redundant multi-agent processing (not just output-level success).
- Two agents answer the same query simultaneously with conflicting info → missing ownership/routing logic (redundancy failure).
- An agent handoff loop never resolves → deadlock; needs arbitration rules (one agent has final authority).
- High agent step count + successful completion → task was genuinely complex, not a problem.
- High agent step count + failed completion → inefficient reasoning, context drift, or unnecessary deliberation — a real problem.
- Canary metrics green but users complain the agent feels "off" → automated metrics missed a qualitative regression; add human review sampling.
- Cost spikes unexpectedly → missing cost-aware budget constraints, or a robustness/recovery mechanism silently driving up retries.
- Agent trained in simulation fails immediately in the real world → reality gap; apply domain randomization and real-world fine-tuning before calling training "done."
- Scaling triggered on request count alone, even though queue depth/latency are fine → wasteful scale-up; fix the scaling signal.
- Refund/payment agent acting without any approval gate → missing the "high agency, low autonomy" design — add an Approve-or-Reject HITL checkpoint.

## Quick Formulas

- ROI: `(Annual Savings − Implementation and Operational Costs) ÷ Implementation and Operational Costs`
- LLM Cost Per Task (LCPT): `total token consumption × model cost`, for the full interaction chain.
- Reward function shape (multi-objective example): `reward = success_bonus − (penalty_per_violation × count) − (penalty_per_unit_time × time_taken)`.
- Cost-effectiveness framing: always report cost alongside savings — ROI, not raw savings, is what justifies stakeholder investment.
