# Chapter 12: Risk Performance Domain

## Core Idea
Risk management isn't just threat avoidance — it's building project resilience by proactively increasing the probability/impact of positive risks (opportunities) while decreasing the probability/impact of negative risks (threats), so the project can anticipate, prepare for, respond to, and adapt to disruption.

## Frameworks Introduced
- **Risk Classification matrix** (Figure 2-46): Known-Known (facts/requirements — managed as scope, *not a risk*) / Known-Unknown ("classic risk" — probability and impact can be identified) / Unknown-Known ("hidden fact" — knowledge exists in the community but not with the team) / Unknown-Unknown ("emergent risk" — knowledge doesn't exist anywhere within reach).
  - When to use: classify any uncertain item into one of these four quadrants before deciding how to respond — a Known-Known shouldn't be tracked as a risk at all; an Unknown-Unknown needs management reserve, not a specific response plan.
- **Risk response strategy types**: for threats — avoidance, mitigation, transference, acceptance, escalation; for opportunities — exploiting, sharing, enhancing, acceptance, escalation; for overall project risk — strategies applied to the project as a whole rather than a specific event (up to and including cancellation if overall risk is too high).
  - When to use: once a risk is classified and analyzed, pick the response type deliberately rather than defaulting to "mitigate everything."
- **Ambiguity vs. Uncertainty**: ambiguity = not knowing what to expect / unclear situation (from too many options or lack of clarity); uncertainty = lack of understanding of issues/paths/outcomes, dealing with probabilities. Neither automatically escalates into a risk — collaborative problem-solving with subject matter experts can resolve many ambiguous/uncertain situations before they become tracked risks.

## Key Concepts
- **Risk**: an uncertain event/condition that, if it occurs, has a positive (opportunity) or negative (threat) effect on objectives. Often described in "cause, event, consequence" structure.
- **Issue**: a condition that has *already occurred* and may need immediate attention — distinct from a risk (which hasn't happened yet), though issues can arise from poorly managed risks.
- **Overall risk**: the effect of uncertainty on the project as a whole (not just individual risks) — if too high, the organization may cancel the project.
- **Risk appetite**: the degree of uncertainty an organization/individual will accept for a reward; quantified via **risk threshold** (e.g., ±5% around a cost objective = lower risk appetite than ±10%).
- **Risk exposure**: an aggregate measure of the potential impact of all risks at a given point in time.
- **Project resilience**: ability to absorb impacts and recover quickly from setbacks, black swan events, or emergent (unknown-unknown) risks — reserve analysis is closely tied to building resilience.

## Mental Models
- Not every uncertain-feeling situation is a risk — check whether it's actually ambiguity/uncertainty that collaborative problem-solving can resolve before formally tracking it as a risk.
- Use the four-quadrant classification as a gate: Known-Known → manage as scope; Known-Unknown → classic risk response planning; Unknown-Known → seek out the hidden knowledge (someone, somewhere, already knows); Unknown-Unknown → resilience/reserve planning, not a specific response plan.
- Tailor risk-assessment *frequency* to development approach: predictive projects assess at defined points; adaptive projects should assess at the start of every sprint, not just during initial planning.

## Anti-patterns
- **Treating every ambiguous situation as a risk requiring a formal response**: this inflates the risk register with noise and dilutes attention from real threats/opportunities — resolve what collaborative problem-solving/expert input can resolve first.
- **One-and-done risk identification**: initial identification is always incomplete; skipping iterative re-identification as the project evolves misses emergent risks.
- **Confusing contingency reserve usage with failure**: effective contingency use (steps taken proactively to prevent threats) should *limit* contingency use over time — if it's being burned down constantly, treat that as a signal, not routine.

## Reference Tables

**Risk Performance Domain — 5 Processes**

| Process | Key Output | Focus Area |
|---|---|---|
| Plan Risk Management | Risk management plan | Initiating (begins at project conception) |
| Identify Risks | Risk register, risk report | Planning |
| Perform Risk Analysis | Project document updates (qualitative + quantitative) | Planning |
| Plan Risk Responses | Change requests, plan updates across Schedule/Finance/Quality/Resource/Procurement/Scope/Schedule/Cost baselines | Planning |
| Implement Risk Responses | Change requests | Executing |
| Monitor Risks | Work performance information, change requests | Monitoring & Controlling |

**Tailoring Considerations** (2.7.3): project size/complexity (detailed vs. simplified approach); risk appetite/threshold (historical experience, risk-aversion level); holistic view (risk impact/response viewed across schedule/budget/scope/stakeholders); strategic importance (breakthrough opportunities vs. performance blocks change risk level); development approach (predictive/adaptive/hybrid changes assessment cadence); flexibility in implementing responses; complementary techniques (GenAI/data analytics for identification/analysis); resilience planning (scenario planning aligned with business continuity/emergency response plans).

**Table 2-11 — Check Outcomes, Risk Performance Domain (condensed)**

| Outcome | How to check |
|---|---|
| Environmental awareness (technical/social/political/market/economic) | Team incorporates these contexts when evaluating uncertainty/risk/response. |
| Proactively explores and responds to uncertainty | Responses aligned with budget/schedule/performance constraints. |
| Capacity to anticipate threats/opportunities | A well-understood process exists for identifying/assessing/documenting/responding to risks. |
| Minimal negative impact from unknown events | Reserves in place and used; delivery dates met; budget within variance threshold. |
| Opportunities realized | Established mechanisms to identify/leverage/track opportunity realization. |
| Contingency reserves used effectively | Proactive threat prevention limits contingency-reserve burn. |
| Resilience/quick recovery developed | Team aware of org's business continuity/emergency response plan; project continuity plan exists; management reserve available; team can quickly adjust structure/process during crisis. |

## Worked Example

**Agile risk-tailoring (2.7.3.1, Example 2)**: A software project facing rapidly changing market demands tailors risk management by running risk assessments at the **start of every sprint** (not just initial planning), holding risk-review meetings with stakeholders at the **end of every iteration** to incorporate feedback, and maintaining a **risk-adjusted backlog**. This cadence mismatch-correction (frequent, iterative risk management instead of a single up-front pass) is the key tailoring move for any fast-changing adaptive project.

## Key Takeaways
1. Classify uncertain items into the four-quadrant risk matrix before responding — Known-Knowns aren't risks at all, and Unknown-Unknowns need resilience/reserves, not a specific mitigation plan.
2. Distinguish risk (hasn't happened) from issue (already happening) — they need different urgency and different logs.
3. Risk appetite and threshold should be explicit and quantified (e.g., a specific % band), not left as an unstated assumption.
4. Not all ambiguity/uncertainty needs to become a tracked risk — try collaborative problem-solving with subject matter experts first.
5. Match risk-assessment cadence to development approach — adaptive projects need per-sprint risk assessment, not a single initial pass.
6. Falling contingency-reserve usage over time is a *good* sign (proactive prevention working); rising usage is a signal worth investigating.

## Connects To
- **Ch 7 (Scope)**, **Ch 8 (Schedule)**, **Ch 9 (Finance)**, **Ch 10 (Stakeholders)**: Risk is named as closely interrelated with all four — stakeholders are a critical source of risk information; scope/schedule/finance absorb the direct impact of risks materializing.
- **Ch 9 (Finance)**: contingency reserve (known risks) and management reserve (unknown-unknowns) are defined in Finance but directly operationalized here.
- **Ch 3 (Principles)**: project resilience and proactive risk management connect to the Adopt a Holistic View and Focus on Value principles.
