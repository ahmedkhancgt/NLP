# Chapter 5: Building Business Cases

## Core Idea
Business cases compare a decision against not making it, counting only incremental costs and benefits (often at the unit-economics level) to find the breakeven point and the levers that move it.

## Frameworks Introduced
- **Business Case Core Principles**: Decisions (frame as act vs. don't-act), Costs/Benefits/Breakeven (net benefit = benefits − costs; breakeven = net benefit of zero), Incrementality (count only costs/benefits caused by the decision), Unit economics (analyze the average customer; sign of net benefit usually survives scaling).
  - When to use: evaluating any new campaign, model, feature, or data purchase before committing resources.
  - How: enumerate costs/benefits → restrict to incremental ones → compute on a per-unit (average customer) basis → solve/inspect the breakeven inequality.
- **Retention Campaign Breakeven**: `(TP/B) × r − (A/B) × c ≥ 0`, i.e., (true positive rate × ARPU saved) ≥ (acceptance rate × incentive cost).
  - Levers: improve true positive rate (better model), lower incentive cost c, target high-ARPU (r) customers.
- **Fraud/Transaction-Limit Decision Rule**: block whenever `E[net benefit of blocking] ≥ E[net benefit of accepting]`: `p·c_f − t + (1−p)·c_ch ≥ 0`, where p = fraud probability, c_f = fraud cost, c_ch = churn cost from a wrongly blocked legit transaction, t = ticket value.
- **Project Prioritization Inequality**: choose project A over B if `revenue_A − cost_A ≥ revenue_B − cost_B` — a common monetary scale that fixes the 2×2 (Ch 4) ranking problem when one project dominates on value and the other on effort.

## Key Concepts
- **Breakeven**: the point of zero net benefit; the worst-case boundary condition for a decision.
- **Incrementality**: only costs/benefits that wouldn't occur absent the decision count (e.g., your salary isn't incremental to running one particular experiment).
- **Unit economics**: analysis on the average customer; aggregate net-benefit sign usually matches the unit-level sign since scaling affects costs and benefits proportionally.
- **True positive (TP)**: in a retention campaign, customers targeted who were actually going to churn.
- **False positive (business-case sense)**: customers given an incentive who would have stayed anyway — pure cost, no incremental benefit.
- **False negative (business-case sense)**: churners not targeted — lost revenue, usable to trade off precision/recall.
- **KPI-cost threshold for data purchase**: `KPI(augmented dataset) − KPI(original dataset) ≥ c`, where the KPI gain must be convertible to money to compare with purchase cost c.

## Mental Models
- Always frame as "decision vs. no-decision," never in absolute terms.
- Marginal/incremental analysis (economics term) = what this chapter calls incrementality; think "what changes because of this action, and only that."
- A business case is itself a decision problem under uncertainty — true positive/acceptance rates are just sample estimates of expected benefit/cost before the campaign runs.

## Anti-patterns
- **Counting sunk or non-incremental costs**: e.g., including your own salary as a campaign cost when you'd be paid regardless.
- **Ignoring false positives/negatives in ML-driven decisions**: a business case built only on true positives overstates benefit; must net out false-positive incentive costs and false-negative opportunity costs.
- **Using the 2×2 value/effort grid for projects that dominate on different axes**: switch to the monetary-scale inequality instead (see Ch 4 cross-reference).

## Code Examples
```text
Retention breakeven:      (TP/B)·r − (A/B)·c ≥ 0
Fraud block decision:     p·c_f − t + (1−p)·c_ch ≥ 0
Dataset purchase:         KPI(augmented) − KPI(original) ≥ c
Project ranking:          revenue_A − cost_A ≥ revenue_B − cost_B
```
- **What it demonstrates**: breakeven/decision inequalities reduce every business case to a single comparable net-benefit expression.

## Reference Tables
*Table 5-1: Costs and benefits for fraud prevention*

| Action | Outcome | Benefit | Cost |
|---|---|---|---|
| Accept | Fraud | t | c_f |
| Accept | Legit | t | 0 |
| Block | Fraud | 0 | 0 |
| Block | Legit | 0 | c_ch |

## Worked Example
Proactive retention: target base B, A accept the incentive, TP of targeted were true churners, ARPU saved = r, incentive cost = c. Breakeven: `(TP/B)·r ≥ (A/B)·c` — first term is the sample true-positive rate times ARPU, second is acceptance rate times incentive cost. Fraud/transaction-limit: expected net benefit of accepting = `t − p·c_f`; of blocking = `−(1−p)·c_ch`; block when `p·c_f − t + (1−p)·c_ch ≥ 0`. As the block threshold (ticket limit L) rises, p(fraud|L) typically increases (fraudsters chase large quick rewards), creating a tunable trade-off between fraud savings and false-positive churn cost.

## Key Takeaways
1. Every business case reduces to comparing incremental costs vs. incremental benefits around a decision, not its absolute totals.
2. Unit economics (the average customer) is usually sufficient — aggregate sign rarely flips on scaling.
3. Breakeven inequalities expose the specific levers (true positive rate, cost, targeting, threshold) that can flip a negative case positive.
4. False positives and false negatives must be explicitly priced into ML-driven business cases, not just true positives.
5. A common monetary scale (project ranking inequality) resolves ranking problems the 2×2 value/effort grid cannot solve.
6. Business-case thinking applies uniformly: retention, fraud limits, dataset purchases, and even which DS project to start next.

## Connects To
- **Ch 2**: Reuses the true-positive-rate/acceptance-rate submetrics from metrics decomposition as business-case inputs.
- **Ch 4**: Directly resolves the 2×2 value/effort prioritization limitation with a common monetary scale.
- **External concept**: marginal analysis / cost-benefit analysis from microeconomics.
