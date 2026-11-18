# Cheatsheet

## Model Tier Selection

| Priority | Best → Worst |
|---|---|
| Complex reasoning | Reasoning > Post-trained > Base |
| Speed | Base > Post-trained > Reasoning |
| Safety | Reasoning ≈ Post-trained > Base |
| Cost | Base > Post-trained > Reasoning |

Rule: orchestrator/planner → reasoning model; narrow specialist sub-agents → post-trained model.

## Agent Architecture Decision Rule

- Single task class, simple logic → single well-prompted agent.
- Multiple specialized functions, accuracy/cost matters → multi-agent (router → specialists → composer).
- Speed-critical, well-defined problem → reactive agent.
- Complex/novel, needs look-ahead → deliberative agent.
- Mixed workload → hybrid (router dispatches to reactive or deliberative path per case).

## Compression Technique Selection

| Technique | Size↓ | Accuracy | Training needed |
|---|---|---|---|
| INT8 quantization | 4x | 95–98% | No |
| INT4 quantization | 8x | 85–92% | Yes (QAT) |
| Distillation | 5–10x | 80–90% | Yes (extensive) |
| Structured pruning | 2–4x | 90–95% | Yes (fine-tune) |
| Combined | 15–30x | 75–85% | Yes (multi-stage) |

Default: try INT8 post-training quantization first; escalate only if size/speed targets still unmet.

## RAG Variant Selection

- Plain need → basic semantic (unstructured) or SQL-agent (structured) RAG.
- Need both keyword precision and semantic recall → Hybrid RAG.
- Fast-changing domain (news, regs, docs) → Temporal RAG.
- Large/organized corpus → Hierarchical RAG.
- Retrieval precision is the bottleneck → add Reranking.

## Edge vs. Cloud Placement

| Signal | Favors Edge | Favors Cloud |
|---|---|---|
| Latency need | <10ms, real-time | 100ms–5s acceptable |
| Workload type | Perception preprocessing, execution | Deep analysis, novel planning |
| Traffic pattern | High-volume, steady | Bursty, unpredictable |
| Compute need | Light | GPU-heavy |

Always evaluate at P95/P99 latency, never average.

## Deployment Rollout Gate (Canary)

| Stage | Traffic | Rollback trigger |
|---|---|---|
| Internal | 0.1% | Any error spike |
| Early | 1% | Error rate >1% |
| Medium | 10% | Task success <90% baseline |
| Large | 50% | Sustained degradation |
| GA | 100% | Major incident |

Agent-specific gates to add: reasoning coherence, tool hallucination rate, context retention, safety-violation rate.

## Security Defense Layering (apply all, not one)

1. Input sanitization (structural + intent classification)
2. Prompt hardening (delimiters, reinforcement, privilege markers)
3. Semantic boundary detection
4. Behavioral/coherence analysis
5. Output filtering (classification, differential, multi-model verification)
6. Execution isolation (container/VM/sandbox)
7. Least-privilege permissions + human approval for high-stakes actions

## Authorization Check Order

RBAC (role permits action type?) → ABAC (attribute conditions hold?) → dynamic risk score → if high risk, require human approval → grant + log.

## Tells & Smells

- Agent gives identical answer regardless of persona/system-prompt change → check temperature=0 and prompt actually wired in.
- Agent "succeeds" but used 3x expected tool calls → check trace for silent retry loops (span-level check, not output-level).
- RAG confidently answers an out-of-corpus question → missing score-threshold rejection.
- Multi-agent system shows a capability nobody programmed → check all 4 emergence preconditions (stochasticity, nonlinearity, feedback, scale) — expect and audit, don't just marvel.
- Canary metrics are green but users complain about "robotic" or "off" responses → automated metrics missed a qualitative regression; add human review sampling.
- Same failure recurs after a fix → the fix addressed the symptom; check whether it became a permanent regression-test case (it should always).
- Cost spikes unexpectedly in production → missing cost-aware budget constraints or runaway speculative/retry execution.

## Quick Formulas

- MEU: `E[U(d)] = Σ P(s|d)·U(s)` — choose `d` maximizing this, not just P(correct).
- MAP = MEU special case when utilities are symmetric (equal penalty both error types).
- Latency budget: `L_total = L_network + L_queue + L_processing + L_serialization` — evaluate at P95.
- Cost-effectiveness ranking: `priority_weight / max(cost, ε)`.
