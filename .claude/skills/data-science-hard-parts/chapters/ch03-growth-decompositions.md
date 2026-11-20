# Chapter 3: Growth Decompositions: Understanding Tailwinds and Headwinds

## Core Idea
When a metric changes over time, additive, multiplicative, and mix-rate decompositions let you quantify which underlying inputs (segments, drivers) contributed to the change — giving hints, not proof, about root causes.

## Frameworks Introduced
- **Additive Decomposition** (for `y_t = y1_t + y2_t`): growth in output is the weighted average of input growth rates, weights = each input's prior-period share.
  - Formula: `g_y,t = ω1,t-1 × g_1,t + ω2,t-1 × g_2,t`, with `ω1,t-1 + ω2,t-1 = 1`.
  - Simplified view: `Growth in Output = SUM(inputs' contributions to growth)`, contribution = lagged weight × growth rate.
  - When to use: output is a sum of parts (e.g., total sales = sum of sales by region).
- **Multiplicative Decomposition** (for `y_t = y1_t × y2_t`): growth in output = sum of input growth rates + a small combined/cross effect.
  - Simplified view: `Growth in Output = SUM(growth in inputs) + combined effect`.
  - When to use: output is a product of parts (e.g., Revenue = ARPU × MAU).
- **Mix-Rate Decomposition** (for `y_t = w_t · x_t`, weighted average/inner product): `Δy_t = Δy^x + Δy^w + Δw·Δx`.
  - Δy^x = change holding weights fixed (the "rate" effect); Δy^w = change holding inputs fixed (the "mix" effect); Δw·Δx = combined/interaction term.
  - Simplified view: `Growth in Metric = SUM(partialled-out effects) + combined effect`.
  - When to use: output is a weighted average across segments (e.g., ARPU across customer segments, weighted by each segment's user share).

## Key Concepts
- **Tailwinds/headwinds**: factors positively/negatively affecting a metric's change, the business-jargon framing this chapter formalizes.
- **Growth rate**: `g_t = Δx_t / x_(t-1)`.
- **First difference**: `Δx_t := x_t − x_(t-1)`.
- **Mix effect**: the counterfactual change in the metric if only the weights (segment shares) had changed, inputs fixed.
- **Rate effect**: the counterfactual change if only the per-segment inputs had changed, weights fixed.
- **Combined/cross effect**: the residual term from simultaneous changes in both weights and inputs (or both multiplicative factors); often small enough to drop.
- **Waterfall plot**: standard visualization for additive/mix-rate decomposition contributions (see Ch 8).

## Mental Models
- Growth-decomposition output is "hints, not root causes" — statistical decomposition narrows the search; business knowledge (pricing changes, competitor action, local economy) supplies the actual explanation.
- Partialling-out intuition ("what if only X had changed, Y fixed") is the natural language for presenting mix-rate results to stakeholders — it's inherently a storytelling device (see Ch 7).
- Pipeline workflow: (1) build a periodically updated aggregate table across dimensions, (2) script the decomposition per dimension, (3) loop over all dimensions, (4) assemble into one table of all variation sources — then apply business judgment.

## Anti-patterns
- **Treating decomposition output as causal**: a decomposition shows contribution, not root cause; always ask "why" behind the contributing segment.
- **Keeping combined effects when they dominate**: if the combined/cross term is large, don't drop it — it signals meaningful interaction, not noise.
- **Using growth rates when differences would simplify notation**: the author recommends working in first differences (Δ) rather than growth rates for mix-rate math, as it considerably simplifies derivation and code.

## Code Examples
```text
Additive:        g_y,t = ω1,t-1·g_1,t + ω2,t-1·g_2,t
Multiplicative:   g_y,t = g_1,t + g_2,t + (combined effect)
Mix-rate:         Δy_t = Δy^x + Δy^w + Δw·Δx
```
- **What it demonstrates**: the three canonical growth-decomposition identities; all reduce algebraically back to the observed growth/change (sanity check when coding them).

## Reference Tables
| Decomposition | Output structure | Formula |
|---|---|---|
| Additive | y = y1 + y2 | g_y = ω1·g1 + ω2·g2 |
| Multiplicative | y = y1 × y2 | g_y = g1 + g2 + combined |
| Mix-rate | y = w·x (weighted avg) | Δy = Δy^x + Δy^w + Δw·Δx |

## Worked Example
Regional sales additive decomposition: national sales fell 4.6% (simulated example), explained mainly by a strong deceleration in the Northwest region (−5.8pp contribution), with Southwest and West also decelerating and South having a strong quarter — visualized as a waterfall chart of regional contributions. ARPU mix-rate decomposition across 3 segments: had shares (mix) stayed fixed, ARPU would have risen $3.2 (pure rate effect); had per-segment ARPU (rate) stayed fixed, ARPU would have fallen $1.6 (pure mix effect) — the sum of these plus the combined effect equals the actual ARPU change.

## Key Takeaways
1. Root-cause analysis for metric changes is inherently hard; decompositions only generate hints, exploiting extra sources of variation.
2. Choose the decomposition that matches your metric's mathematical structure: additive (sums), multiplicative (products), mix-rate (weighted averages).
3. Mix-rate's partialled-out ("what if only mix/only rate changed") framing is especially persuasive for stakeholder storytelling.
4. Combined/cross-effect terms are mathematically necessary but can often be dropped if negligible — check first.
5. Build a repeatable pipeline: aggregate table → per-dimension decomposition script → loop across all dimensions → consolidated hint table.

## Connects To
- **Ch 2**: Reuses the same additive, multiplicative, and p×q metric structures as the base for decomposing growth instead of levels.
- **Ch 8**: Waterfall charts are the primary datavis tool for presenting these decomposition results.
- **Ch 10**: Explaining variation in an output from variation in inputs foreshadows the regression/variation discussion there.
