# Patterns

## ReAct (Reason + Act) Loop
**When to use**: any multi-step task requiring tool calls interspersed with reasoning.
**How**: alternate explicit "think" and "act" steps; each action's result feeds the next reasoning step.
**Trade-offs**: reduces hallucination vs. pure reasoning/action chains, but adds latency per step.

## Supervisor + Specialist Multi-Agent
**When to use**: task complexity/volume exceeds what one well-prompted agent handles reliably.
**How**: a router/supervisor agent classifies and dispatches to narrow, focused specialist agents via shared state; a composer synthesizes the final output.
**Trade-offs**: better accuracy/cost/latency per task than a monolithic agent, but adds orchestration complexity — don't reach for this below the complexity threshold.

## Score-Threshold RAG Rejection
**When to use**: any RAG system where a confidently wrong answer is worse than "I don't know" (clinical, legal, financial).
**How**: use a thresholdable distance metric (L2, not cosine); filter out all results beyond the threshold; return empty rather than nearest-wrong.
**Trade-offs**: may occasionally reject a borderline-relevant result, but eliminates the silent-wrong-answer failure mode.

## Structured Output as Architecture Signal
**When to use**: whenever an agent's output must drive downstream control flow (routing, review gating), not just inform a user.
**How**: define a typed schema (Pydantic) with fields like `requires_human_review` or `route_to`; force output via `with_structured_output()`; let the graph branch on the field directly.
**Trade-offs**: a non-conforming LLM response raises a validation error instead of silently passing bad data — treat this strictness as a feature, not friction.

## Structural Human-in-the-Loop (interrupt/resume)
**When to use**: any consequential, hard-to-reverse action (financial, medical, data deletion, permission change).
**How**: pause graph execution at a review node (e.g., LangGraph `interrupt()`); require an explicit external resume signal keyed to the same session/thread ID before continuing.
**Trade-offs**: adds latency and requires a working escalation UI, but makes oversight enforced rather than merely documented policy.

## Distributed Review-Trigger Judgment
**When to use**: multi-agent systems where different specialists carry different risk profiles.
**How**: let each specialist agent set its own `requires_review` flag based on domain risk (e.g., always for drug info, conditionally for guidelines, never for greetings) rather than one central policy gate.
**Trade-offs**: more accurate, domain-aware oversight; requires trusting each specialist's own risk assessment.

## Least-Privilege + Delegation Scoping
**When to use**: any agent acting with delegated user authority.
**How**: grant only the minimum permission set needed, time-boxed and revocable; delegated authority must be a strict subset of the delegator's own permissions; prefer agency (`Agent_ID acting for User_ID`) over impersonation.
**Trade-offs**: more setup overhead per capability, but bounds blast radius of a compromised or manipulated agent.

## Defense-in-Depth Prompt Injection Mitigation
**When to use**: any agent processing untrusted user input or retrieved external content.
**How**: layer input sanitization (structural + intent classification) + prompt hardening (delimiters, reinforcement, privilege markers) + semantic boundary detection + behavioral/coherence analysis + output filtering + execution sandboxing.
**Trade-offs**: no single layer is sufficient against polymorphic attacks; the combination is what provides real protection, at the cost of added latency and engineering surface.

## Dual-Agent / Multi-Agent Consensus Verification
**When to use**: high-stakes autonomous actions where a single manipulated agent could cause harm.
**How**: require two or more independently-reasoning agents to agree before a sensitive action executes.
**Trade-offs**: manipulating one agent's framing doesn't guarantee fooling a second reviewing from a different angle; doubles inference cost per gated action.

## Canary Deployment with Domain-Specific Gates
**When to use**: rolling out any model/prompt/agent change to production traffic.
**How**: progress through 0.1%→1%→10%→50%→100% traffic stages, each gated by success criteria that include agent-specific metrics (reasoning coherence, tool hallucination rate, context retention), not just error rate/latency.
**Trade-offs**: slower rollout, but catches agent-specific quality regressions that generic infra metrics miss.

## Multidimensional Versioning (RUNTIME.MODEL.SCHEMA.CAPABILITIES)
**When to use**: any stateful agent system needing independent evolution of runtime, model, state schema, and tool capabilities.
**How**: version each dimension separately so a model upgrade needn't trigger schema migration, and a capability addition needn't touch runtime.
**Trade-offs**: more versioning bookkeeping, but enables safe partial updates instead of all-or-nothing releases.

## Graceful Degradation via SLO Tiers
**When to use**: any production agent system that must stay useful under partial failure.
**How**: define target SLO → degraded mode → minimum acceptable for each capability; implement circuit breakers and tiered capability reduction (5 load levels) that automatically escalate/de-escalate.
**Trade-offs**: more design upfront, but converts binary up/down failure into continuous, bounded degradation.

## Phased, Human-in-the-Loop Deployment (Ethical Risk Lever)
**When to use**: introducing a new agentic capability with uncertain fairness/alignment/privacy properties.
**How**: start with the agent proposing actions and a human approving every one in a controlled context; expand autonomy only as evidence accumulates that safeguards hold.
**Trade-offs**: slower capability rollout, but treats deployment scale as an ethical-risk lever, not just a technical-reliability one.

## Model-Tier Assignment by Agent Role
**When to use**: any multi-agent system with an orchestrator and specialist sub-agents.
**How**: assign a reasoning model to the orchestrator (needs deep planning/intent understanding); assign cheaper post-trained models to narrow specialist sub-agents.
**Trade-offs**: requires profiling which roles actually need deep reasoning; mismatched assignment wastes cost (over-provisioned) or accuracy (under-provisioned).

## Continuous Evaluation Against a Gold-Standard Dataset
**When to use**: any production agent whose prompts, models, tools, or input distribution will change over time (i.e., always).
**How**: build an SME-curated gold-standard dataset; score every relevant change against it via trace-aware automated scoring; surface results as a required CI check.
**Trade-offs**: significant upfront SME investment, but this is explicitly named as the most important component — automated metrics are only as good as what they're measured against.

## Hybrid Tiered Processing (Edge/Gateway/Cloud)
**When to use**: systems with mixed latency requirements across perception, planning, execution, and reflection.
**How**: match each cognitive function to its natural tier — execution stays local (real-time), planning caches locally with cloud fallback for novel cases, deep analysis/reflection goes to the cloud.
**Trade-offs**: more architectural complexity than pure cloud or pure edge, but each tier failure degrades rather than eliminates capability.

## Cost-Aware Budget-Constrained Execution
**When to use**: any agent system operating under a real compute/financial budget (always, in production).
**How**: estimate each task's cost, rank by cost-effectiveness (priority/cost), execute within budget, explicitly skip low-value tasks once exhausted.
**Trade-offs**: requires reasonably accurate cost estimation per task type; prevents runaway spend at the cost of occasionally deferring lower-priority work.
