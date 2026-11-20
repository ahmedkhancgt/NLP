# Chapter 2: Metrics Design

## Core Idea
Great data scientists are great at metrics design — finding and decomposing metrics so they are measurable, actionable, relevant, and timely enough to drive decisions.

## Frameworks Introduced
- **Four Desirable Properties of Metrics**: Measurable, Actionable, Relevant, Timely (MART-like, unnamed acronym in book).
  - When to use: whenever proposing or auditing any KPI.
  - How: reverse-engineer from "metrics exist to drive action" to derive necessary conditions.
- **Funnel Decomposition**: for a sequence of stages E → s1 → s2 → s3 → M, decompose M by multiplying/dividing by each intermediate stage's metric, working right to left, finishing by multiplying by the entry metric E, so all terms cancel back to M.
  - Formula: `M = (M/s_last) × (s_last/s_prev) × ... × (s1/E) × E` — each fraction is a stage conversion rate.
  - When to use: any metric that is the end-result of a multi-step process (sales funnel, growth funnel, marketplace funnel).
- **Stock-Flow Decomposition**: `Stock_t = Stock_(t-1) + Inflow_t − Outflow_t` (e.g., MAU_t = MAU_(t-1) + Incoming Users_t − Churned Users_t).
  - When to use: any accumulating ("stock") metric like MAU, account balances.
  - How: split inflow further (e.g., New Users vs. Resurrected Users) to find more levers.
- **P×Q Decomposition**: `Revenue = (Revenue/Units Sold) × Units Sold = Price × Quantity`.
  - When to use: revenue-type metrics; note price and quantity move in opposite directions, making it useful for price optimization.
- **Marketplace / two-sided funnel decomposition**: `Purchases = (P/V) × (V/B) × B` combined with seller-side `L/S` to get checkout efficiency × buyer engagement × seller engagement × buyer/seller volume terms.

## Key Concepts
- **Proxy metric**: a measurable stand-in correlated with an unmeasurable true construct (e.g., time-to-first-use as a proxy for purchase "intentionality").
- **Habit (growth term)**: recurrence of usage; proxied by number of interactions in the first X days.
- **Relevance**: a metric is only "good" relative to a specific business question, not in the abstract.
- **Timeliness trade-off**: lengthening a churn-detection window (1→3 months) reduces false positives but delays actionability.
- **Conversion rate**: each funnel-decomposition fraction, interpreted as % of prior-stage units advancing to the current stage.
- **Fact vs. dimension tables**: facts = verbs/metrics; dimensions = adverbs describing the metric — grammar analogy for data warehouse decomposition.

## Mental Models
- Metrics design is iterative: start imperfect, keep refining as you learn the business.
- "If not us, who else?": data scientists are the natural owners of metrics quality because they calculate/report/optimize metrics constantly.
- Decomposition as lever discovery: every decomposition's terms are candidate levers owned by different teams (DS, sales, product, pricing).

## Anti-patterns
- **Optimizing top-line metrics directly**: top-line metrics (e.g., revenue) usually aren't directly actionable — decompose first.
- **Ignoring proxy limitations**: treating a proxy metric as if it perfectly measures the true (unmeasurable) construct.
- **One-size-fits-all timeliness**: using the same measurement window regardless of how fast action needs to be taken.

## Code Examples
```sql
SELECT DATE_TRUNC('MONTH', ft.fact_timestamp) AS month,
       dt.dimension1 AS dim_values,
       SUM(ft.my_metric) AS monthly_metric
FROM my_fact_table ft
LEFT JOIN my_dim_table dt ON ft.primary_key = dt.primary_key
GROUP BY 1,2
ORDER BY 1,2
```
- **What it demonstrates**: standard pattern for building an aggregate table split by a dimension, the input needed for growth decompositions (Ch 3).

## Reference Tables
| Funnel Stage | Metric |
|---|---|
| Lead generation | L (number of leads) |
| First contact | C1 |
| Second contact | C2 |
| Offer | O |
| Close | S (sales) |

Decomposition: `S = (S/O) × (O/C2) × (C2/C1) × (C1/L) × L`

## Worked Example
Sales funnel: Lead generation (L) → First contact (C1) → Second contact (C2) → Offer (O) → Sale (S). Decompose S/L into conversion rates at each stage: lead quality (DS lever), first-contact rate (sales staffing), offer-to-sale rate (pricing/negotiation). Marketplace example: Sellers → Listed Items → Views → Purchases, expanded with Buyers to get `Purchases = (P/V) × (V/B) × B` plus seller-side listing engagement `(L/S)`, yielding five levers: checkout efficiency, buyer engagement, seller engagement, buyer volume, seller volume.

## Key Takeaways
1. Good metrics must be measurable, actionable, relevant, and timely — audit every KPI against these four properties.
2. Decomposition (funnel, stock-flow, P×Q) is the primary tool for turning a non-actionable top-line metric into actionable submetrics.
3. The choice of decomposition metric (what to multiply/divide by) is non-obvious and requires deep business knowledge.
4. Treat metrics design as a continuous, iterative practice, not a one-time exercise.
5. Every decomposition reveals levers that different teams across the org can own and pull.

## Connects To
- **Ch 3**: Growth decompositions reuse these same multiplicative/additive structures to explain *changes* in metrics over time.
- **Ch 4**: The p×q decomposition is reused as the basis for 2×2 designs (e.g., price vs. transactionality quadrants).
- **Ch 5**: Business cases often rely on decomposed unit-economics terms (true positive rate, acceptance rate) derived here.
