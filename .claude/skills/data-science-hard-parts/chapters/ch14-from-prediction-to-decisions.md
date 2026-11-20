# Chapter 14: From Prediction to Decisions

## Core Idea
ML's ROI comes from improving decisions under uncertainty, not from prediction accuracy itself; classification models naturally yield threshold-based decision rules that can be optimized against the real costs and benefits of false positives/negatives.

## Frameworks Introduced
- **Decisions-Under-Uncertainty Flow**: Levers (actions) × Uncertainty (unresolved state) → Outcome → Metric. Always map your use case through this chain before reaching for ML.
  - How to find new ML use cases: (1) identify the key decision, its metric and levers, (2) understand the role of uncertainty, (3) build a business case for an ML solution (ties to Ch 5).
- **Threshold Decision Rule**: `Action A if p_i ≥ τ; Action B if p_i < τ` — the generic mechanism turning a classifier's probability score into an actionable decision.
  - When to use: whenever a classification model needs to be turned into a concrete business action (approve/review, call/don't call, block/accept).
- **Confusion-Matrix Threshold Optimization**: choose τ to minimize expected cost or maximize expected profit:
  - `E[Cost(τ)] = P_FP(τ)·c_FP + P_FN(τ)·c_FN`
  - `E[Profit(τ)] = P_TP(τ)·b_TP + P_TN(τ)·b_TN − P_FP(τ)·c_FP − P_FN(τ)·c_FN`
  - When to use: whenever false positives and false negatives carry materially different costs (e.g., fraud blocking) — raising c_FP pushes the optimal threshold up (favor precision); raising c_FN pushes it down (favor recall).
- **Volume-Threshold Rule (lead generation / capacity-constrained case)**: sort leads by score descending, contact only the top V the team can handle; implicitly sets τ by capacity rather than by cost-benefit optimization. Works because precision is (typically) increasing in the probability score, so prioritizing top-V leads maximizes conversion efficiency given a fixed sales-force size.

## Key Concepts
- **Precision**: `TP / (TP + FP)` — of everything predicted positive, what % was truly positive; reflects the cost of a false positive.
- **Recall**: `TP / (TP + FN)` — of everything truly positive, what % was predicted positive; reflects the cost of a false negative.
- **Confusion matrix**: cross-tab of actual vs. predicted label (TN, FP, FN, TP), parameterized by the chosen threshold τ.
- **Decision rule vs. prediction**: a prediction becomes a decision rule only once a threshold (or more general mapping from score to action) is chosen — this choice is itself a business decision with cost/benefit trade-offs.
- **Lead generation funnel decomposition**: `Sales = (Sales/Called) × (Called/Leads) × Leads = Conversion Efficiency(τ) × Call Rate(FTE) × Leads(τ)` — conversion efficiency ≈ precision, call rate depends on sales-team FTE capacity, leads volume depends on τ.

## Mental Models
- "Think decisions first, then ML" — identify the decision, its levers, and the uncertainty it must resolve, before reaching for a predictive algorithm.
- ML adoption without positive ROI is common (per McKinsey/Ransbotham survey data cited: only ~10% of companies report significant financial benefit from AI) because organizations skip straight to modeling without mapping the decision chain.
- A threshold choice is never "free" — it always encodes an implicit cost-benefit trade-off between precision and recall, whether or not you make that trade-off explicit.

## Anti-patterns
- **Optimizing a classifier purely for accuracy/AUC without mapping it to a decision's real costs**: a technically strong model with the wrong threshold can still destroy value.
- **Assuming the volume-threshold rule implies "contact everyone" is best**: it isn't — leads with low scores have expected costs (certain FTE cost) exceeding expected benefits (uncertain, lower-probability sale); the volume rule only optimizes precision given a *fixed* team size, it doesn't argue for infinite team size.
- **Ignoring false-negative costs when only false-positive costs are salient** (or vice versa): most real decisions (e.g., fraud) require weighing both; lead generation is an atypical case where FN costs are often implicitly weighted near zero.
- **Treating threshold optimization as needing absolute costs**: only *relative* costs matter (e.g., FN costs 3× FP) — costs can be normalized to one reference outcome.

## Code Examples
```text
Threshold decision rule:         Action A if p_i ≥ τ, else Action B
Expected cost:                   E[Cost(τ)] = P_FP(τ)·c_FP + P_FN(τ)·c_FN
Expected profit:                 E[Profit(τ)] = P_TP(τ)·b_TP + P_TN(τ)·b_TN − P_FP(τ)·c_FP − P_FN(τ)·c_FN
Precision:                       TP / (TP + FP)
Recall:                          TP / (TP + FN)
Lead-gen funnel:                 Sales = Conv.Eff(τ) × CallRate(FTE) × Leads(τ)
```
- **What it demonstrates**: the core formulas for converting classifier outputs into optimized, cost-aware business decisions.

## Reference Tables
*Table 14-1: Examples of ML decision use cases (abridged)*

| Category | Use case | Decision | Uncertainty | Outcome |
|---|---|---|---|---|
| Service ops | Claims processing | Auto-pay vs. review | Legitimate or not | Lower cost, higher satisfaction |
| Service ops | Staffing | Hire or relocate | Staff size needed | Higher satisfaction, lower idle cost |
| Service ops | Proactive support | Call or not | Will customer have a problem | Higher satisfaction, lower churn |
| Supply chain | Demand forecasting | Manage inventory | Future demand | Higher sales, lower depreciation |
| Fraud | Chargeback prevention | Approve/decline | Legitimate or not | Lower fraud cost, higher satisfaction |
| Marketing | Lead generation | Call or not | Will they buy | Higher sales efficiency |
| ML products | Recommender | Recommend A or B | Will they buy | Higher engagement, lower churn |

*Table 14-2: Confusion matrix structure*

| Actual \ Predicted | Negative (τ) | Positive (τ) |
|---|---|---|
| Negative | TN | FP |
| Positive | FN | TP |

## Worked Example
Lead generation: contact a lead if predicted probability ≥ τ. FN = a lead that would have converted but wasn't sent to telemarketing (cost = forgone revenue); FP = a lead sent to telemarketing that doesn't convert (cost = processing time × hourly rate, e.g., k minutes at $X/hour = $kX/60). Volume rule: sort by score, send the top V leads the team can handle each period — this simultaneously maximizes precision (since higher scores are more likely to convert) and keeps the sales team's capacity fully utilized without idle time. Fraud threshold optimization: with symmetric costs/benefits (all normalized to 1) and a balanced outcome, the optimal threshold is ~0.5; doubling the cost of a false positive shifts the optimal threshold up (favoring precision — fewer legitimate transactions blocked), while doubling the cost of a false negative shifts it down (favoring recall — catching more fraud even at the cost of more false blocks).

## Key Takeaways
1. ML creates ROI through improved decisions, not prediction accuracy in isolation — map the decision/lever/uncertainty/outcome chain before building a model.
2. Classification models naturally produce threshold-based decision rules; the threshold choice is itself a business decision encoding a precision/recall trade-off.
3. Precision reflects false-positive cost sensitivity; recall reflects false-negative cost sensitivity — know which one your use case actually needs to optimize.
4. Confusion-matrix-based cost/profit optimization gives a principled way to pick a threshold when both FP and FN costs matter.
5. A volume-threshold rule is appropriate when capacity (not cost-benefit) is the binding constraint, such as a fixed-size sales team.
6. Only relative costs matter for threshold optimization — you don't need precise absolute dollar figures, just ratios.
7. Most organizations fail to realize AI ROI because they skip the decision-mapping step and jump straight into modeling.

## Connects To
- **Ch 1**: Directly extends "own M(A(X))" — decisions (A) are the explicit mechanism by which predictions (X) move metrics (M).
- **Ch 2**: The lead-generation funnel decomposition reuses Chapter 2's funnel-analytics technique.
- **Ch 5**: Threshold cost/benefit optimization is a direct application of business-case breakeven logic.
- **Ch 6**: Precision/recall by decile connects to the lift/TPR analysis from Chapter 6.
