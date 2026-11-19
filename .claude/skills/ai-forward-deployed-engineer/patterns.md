# Patterns

## Model + Tool + Loop agent pattern
**When**: build intuition before adopting any framework.
**How**: tool as a plain function + JSON schema; call model with `tools=[...]`; if tool call, run it and append a `role:tool` message; loop until plain text. Cap with `max_steps`.
**Trade-offs**: full control, but you own all retries/error handling/history growth.

## Fixed Workflow vs. Agent decision rule
**When**: before reaching for an agent framework.
**How**: if the tool-call sequence never varies, use a single structured-output call, not an agent.
**Trade-offs**: an agent wrapper on a fixed sequence costs more, runs slower, with no accuracy gain (Ch02).

## Business logic in code, not prompts
**When**: deterministic math or a fixed sequence.
**How**: implement as a Python function exposed as a tool (e.g. `calculate_payout`); reserve the LLM for genuine judgment.
**Trade-offs**: small LLM-cost increase, large accuracy/auditability gain ($5,300→$640 cost of mistakes, Ch06).

## Five legitimate reasons to add a sub-agent
**When**: only when a measured outcome improves.
**How**: context isolation, independent check/LLM-as-judge, cost efficiency, speed (fan-out), security (ring-fenced permissions).
**Trade-offs**: never add to mirror a human org chart — each sub-agent adds latency/cost/complexity.

## Structured Outputs, rationale before decision
**When**: need a reliably parseable business decision.
**How**: `rationale: str` before `decision: Literal[...]` in the schema; call `.parse()` with `response_format=`.
**Trade-offs**: minor prompt rigidity for trustworthy, non-post-hoc reasoning.

## LLM-as-judge with reference answer
**When**: you have golden answers and want consistency grading.
**How**: rubric (A=better than reference, B=consistent, C=partial, F=contradicts); feed both candidate and reference; add boolean explainability checks.
**Trade-offs**: stronger than isolated grading, but still token prediction — use concrete anchors and a different model family.

## Cohen's kappa for dominant-class data
**When**: one class dominates the eval set.
**How**: `(observed - expected)/(1-expected)`, expected = chance agreement from each side's class distribution.
**Trade-offs**: catches trivial "always approve" baselines that raw accuracy rewards.

## Cost-of-mistakes matrix
**When**: multi-class decisions with different error costs.
**How**: agree a $ cost table with the business before running evals; sum across cases as the primary metric.
**Trade-offs**: needs an upfront business conversation but prevents optimizing the wrong target.

## Agentic RAG (self-directed retrieval)
**When**: retrieval needs iteration, multiple sources, or query refinement.
**How**: give the agent a "retrieve context" tool; let it decide how often/how to search.
**Trade-offs**: more accurate, but slower, pricier, harder to evaluate than traditional RAG.

## MCP server integration
**When**: consuming tools built by someone else; skip for your own tools on your own agent.
**How**: stdio transport for local servers, Streamable HTTP for remote OAuth-authenticated ones.
**Trade-offs**: adoption is the value, not novelty; vet unfamiliar servers before use.

## LangGraph durable execution with Postgres checkpointer
**When**: a production agent that might crash or wait on a human for hours/days.
**How**: `PostgresSaver` checkpointer, thread ID per business entity, `durability="sync"`.
**Trade-offs**: adds a DB dependency but is the only way to survive a crash with zero lost work.

## interrupt() / Command(resume=...) human-in-the-loop
**When**: a node needs a human decision that may take hours/days.
**How**: call `interrupt(...)` first in an approval node; resume via `Command(resume=data)`. Put side effects in a separate downstream node.
**Trade-offs**: the interrupted node re-runs fully on resume — make side-effecting nodes idempotent.

## LangGraph fan-out dispatcher (no LLM)
**When**: fan out independent work to a bounded worker pool with retries.
**How**: `Annotated[list, operator.add]` reducer; conditional edge returns `[Send(node, item) ...]`; cap with `max_concurrency`; attach `RetryPolicy`.
**Trade-offs**: scales near-linearly until a shared resource (DB pool, rate limit) bottlenecks it.

## Nesting agents inside a dispatcher node
**When**: scaling an existing agent without rewriting it.
**How**: a worker node is just a function calling `.invoke()` on another compiled graph.
**Trade-offs**: none structurally; watch thread-ID granularity for independent human-in-the-loop pausing.

## Self-improvement loop with held-out validation
**When**: automating measure-change-remeasure for a production agent's config.
**How**: Improver's steps — read_results, propose_change (human-approved), run_evaluation (eval set once, held-out set twice), keep/rollback, optionally propose_recipe (human-approved).
**Trade-offs**: without held-out data the improver overfits; without approval gates, changes ship unreviewed.

## Keep-or-rollback decision rule
**When**: deciding if a proposed agent change should ship.
**How**: keep only if a measure improves beyond a noise threshold on held-out data and nothing else degrades past tolerance, within a floor and budget cap.
**Trade-offs**: skipping this risks shipping noise as "improvement."

## Namespace partitioning for agent memory
**When**: multi-tenant long-term memory.
**How**: hierarchical templates (`/customers/{actorId}/facts/`); pass actor/session IDs programmatically, never let the model choose them.
**Trade-offs**: the only reliable way to prevent cross-tenant leakage; a naming convention alone is not enough.

## Defense-in-depth guardrail layering
**When**: any consequential agent action.
**How**: layer a model-based screener, a gateway Cedar policy, and an app-level check — same rule, 2-3 independent layers.
**Trade-offs**: redundancy is the point; keyword-only checks are trivially bypassed.

## Cedar policy rollout: log-only then enforce
**When**: deploying a new gateway policy.
**How**: attach in `LOG_ONLY` first, verify via metrics, then switch to `ENFORCE`.
**Trade-offs**: skipping log-only risks breaking live traffic with an untested rule.

## OpenAPI-to-MCP-tools conversion
**When**: a company already has OpenAPI-documented REST APIs.
**How**: AgentCore Gateway reads the spec and exposes each endpoint as an MCP tool automatically.
**Trade-offs**: simplest path to tool-enabling existing APIs; no bespoke MCP server needed.

## Teams vs. Swarms vs. Fleets sizing
**When**: deciding orchestration scale.
**How**: team (3-4, distinct roles) for collaborative builds; swarm (20-100+ near-identical) for parallel throughput; fleet (org-wide roster) for portfolio tracking.
**Trade-offs**: terms are vendor-specific — confirm what a stakeholder means first.

## Durable session/thread decoupling
**When**: deploying to a managed runtime with disposable sessions.
**How**: decouple ephemeral session ID (microVM) from durable thread ID (Postgres checkpoint lineage); resume via same thread ID + action `resume`.
**Trade-offs**: requires discipline to never rely on in-memory/session-local state.

## Copilot-under-operator-permissions pattern
**When**: giving a human operator an AI assistant inside an existing product.
**How**: reuse existing tools/infra but invoked under the operator's own permissions; give it its own namespaced memory (per handler, not per customer).
**Trade-offs**: convenience must never imply expanded authority — sign-off still needs explicit operator action.
