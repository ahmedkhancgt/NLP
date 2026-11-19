# Cheatsheet

## Decision rules
- "I need an agent for X" → demand the measurable business problem first; try without an agent. (Ch06)
- Tool-call sequence always identical → single structured-output call, not an agent. (Ch02)
- Deterministic math or fixed sequence → code behind a tool, not the prompt. (Ch06)
- Writing sub-agent instructions as a numbered sequence → that's a for-loop; write it in code. (Ch05)
- Agent makes a mistake → engineer the harness so that mistake becomes structurally impossible, don't just re-prompt. (Ch05)
- Agent-built code demos well → assume spaghetti until a human reads the sensitive parts. (Ch03, Ch05)
- Judging a proposed config change → decide only on held-out data the improver never saw. (Ch10)
- Side effect could re-fire on `interrupt()` resume → put it in a separate idempotent downstream node. (Ch08)
- Scoping a memory namespace → set actor/session IDs in code, never let the model choose. (Ch13)
- Benchmark/leaderboard looks good → treat as a shortlist only; test accuracy/cost/speed on your own task. (Ch02, Ch04)
- Evaluating reasoning effort → test empirically per task; higher effort is not reliably more accurate. (Ch02, Ch04)
- New integration added → re-audit for the lethal trifecta; it can appear later even if absent at design. (Ch12)

## Decision trees

**RAG vs. Agentic RAG vs. Fine-tuning**
- Simple, one pass enough, fast/cheap → traditional RAG.
- Need multi-source accuracy, accept cost/latency → agentic RAG.
- Model must learn a generalizable skill (rare) → fine-tuning; otherwise avoid it.

**Adding a sub-agent**
- Boundary not tied to a measured outcome → don't add it.
- Tied to context isolation / independent check / cost / speed / security → add it.

**Team vs. Swarm vs. Fleet**
- 3-4 agents, distinct roles, one coordinator → Team.
- 20-100+ near-identical, same task type, parallel → Swarm.
- Org-wide roster across projects → Fleet (track as software assets, not HR).

**Hosting an agent**
- Simplicity, full control → your own infra (Lambda/ECS/K8s).
- Need identity/memory/permissions fast → managed runtime (AgentCore etc.).
- Only need one lab's models, their harness → the lab's own managed agents (hardest to justify in enterprise).

**New risk needs a guardrail**
- Can constrain via API/tool scope? → do that first.
- Residual phrasing/intent risk? → add a model-based screener (not the task model).
- Need a hard backstop? → add gateway policy AND app-level check — both, not either.

## Trade-off matrices

| Harness framework | Build effort | Model flexibility | Built-in |
|---|---|---|---|
| Raw LLM calls | Max | Any | None |
| `create_agent` | Moderate | Any | Tool/loop boilerplate |
| Deep Agents | Low | Any | To-do list, filesystem, sub-agents |
| Claude Agent SDK | Minimal | Anthropic-optimized | Full Claude Code agent |

| Quantization step | Impact |
|---|---|
| 16→8-bit | ~1% or less |
| 16→4-bit | Noticeable, esp. coding agents |
| Distillation | Real intelligence cut, not free |

| Guardrail | Catches | Bypassed by |
|---|---|---|
| Keyword check | Fixed phrases | Rewording, translation |
| Model screener | Intent/meaning | Novel phrasing |
| Gateway Cedar policy | Structured params | Nothing, if scoped right |
| App-level limit | Business rule | Nothing, if scoped right |

## Thresholds & defaults
- Course API spend: $0 (free models) to ~$10 comfortable, $20+ for full infra work.
- UnderCover approval threshold: payouts >$1,000 need human sign-off.
- AWS budget alerts: 50/80/100%; always exclude Credit/Refund charge types.
- AgentCore Runtime: sessions up to 8h; cold start ~2.5-8.6s, warm call ~0.2s.
- Supabase free tier: 5 concurrent Postgres connections — real swarm bottleneck.
- Capstone targets: straight-through ≥70% (achieved 88%); accuracy ≥60% (achieved 88%); over-limit compliance 100% (achieved 100%).
- Cost-of-mistakes example: wrongful decline $1,500, needless referral $50, wrongful payment = amount paid.
- Use-case screen: Three Questions (real problem? proprietary data/evals? automation/acceleration/differentiation?) before RICE.
- Job market: 95% of 316 surveyed postings mention FDE explicitly; ~89% mention ≥4 of 5 core areas — don't expect every bullet matched.

## Tells & smells
- Slick working demo ≠ good code — read the sensitive parts yourself. (Ch03, Ch05)
- Rigid step-by-step sub-agent prompts are hard-coded workflow dressed as autonomy. (Ch05)
- One successful eval run (even 5/5) proves nothing — stochasticity swings accuracy run to run; average multiple. (Ch05, Ch10)
- Raw accuracy looks great but a trivial baseline nearly matches it → distrust it, check Cohen's kappa. (Ch06)
- Vendor folds memory/permissions/governance into one "context layer" product → marketing, not architecture. (Ch13)
- A "free" open-weight model feels free → someone still pays (your data, electricity, or quality). (Ch02, Ch04)
- A fluent, confident agent explanation → evidence of plausibility, not accuracy. (Ch06)
- About to blame the LLM for a bad outcome → stop; as builder, you are accountable. (Ch11)
