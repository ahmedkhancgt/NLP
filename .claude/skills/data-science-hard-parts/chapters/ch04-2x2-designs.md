# Chapter 4: 2×2 Designs

## Core Idea
Deliberately simplifying a complex business down to two key factors on a 2×2 grid — in either an experimental or non-experimental setting — is a powerful tool for communication, hypothesis testing, and decision prioritization.

## Frameworks Introduced
- **2×2 Design**: plot two factors on horizontal/vertical axes, threshold each into high/low, producing 4 quadrants (A: high/high, B: low factor1/high factor2, C: low/low, D: high factor1/low factor2).
  - When to use: (1) simultaneously testing a model's predictive power and a lever's effectiveness; (2) understanding/segmenting customer behavior (e.g., LTV drivers, product-market fit); (3) decisions under correlated features (e.g., credit risk vs. acceptance); (4) prioritizing a project backlog (value vs. effort).
  - How: pick exactly two factors relevant to the decision, set meaningful thresholds, and (in experimental settings) randomize to hold everything else constant on average.
- **Model + Lever joint test**: horizontal axis = model probability score, vertical axis = lever on/off. Test two hypothesis families:
  - *Monotonicity*: CR_A > CR_B and CR_D > CR_C (higher score ⇒ higher conversion).
  - *Effectiveness*: the new lever outperforms baseline within matched score groups.

## Key Concepts
- **Quadrant-based labeling for ML**: label users in quadrant A as 1, everyone else as 0, then train a classifier to understand what drives membership (profitable fit, high-LTV users, etc.) — feeds into Ch 13 interpretability tools.
- **Ex ante equal on average**: the property guaranteed by proper randomization — treatment and control don't differ systematically before the test.
- **Adverse selection**: in credit/insurance 2×2s, users most likely to accept an offer are also most likely to be risky (positive correlation between acceptance and default).
- **Profitable fit**: combination of high engagement (product-market fit proxy) and high willingness-to-pay (revenue proxy).
- **LTV 2×2**: `LTV = Σ_t r_t × s_t / (1+d)^t` plotted as survival probability (x) vs. revenue (y) using an aggregation method (average/median) or an arbitrary fixed period.

## Mental Models
- "Everything as simple as possible, but not simpler" (Einstein) — simplification is a skill, not a shortcut; the hard part is choosing which two dimensions matter.
- Partialling out one factor at a time (holding others fixed) is the generic scientific principle behind 2×2 designs, echoed again in Ch 10's regression discussion.
- 2×2 is a stepping stone to more sophisticated modeling (label quadrant membership and go open the black box), not an end state.

## Anti-patterns
- **Comparing projects that dominate on different axes**: 2×2 project-prioritization (value vs. effort) can't rank projects where one wins on value and loses on effort — needs a common-scale approach (Ch 5's business case math).
- **Assuming non-experimental 2×2s control for confounders**: without randomization (e.g., customer segmentation by price × transactionality), you cannot claim "everything else held constant" — only randomized designs guarantee that.
- **Ignoring statistical power when combining model + lever tests**: joint 2×2 tests often lack sample size for both dimensions; the book recommends settling for a well-powered lever test and accepting casual (not rigorous) evidence on the model.

## Code Examples
```text
LTV = Σ_t [ r_t × s_t / (1+d)^t ]
```
- **What it demonstrates**: customer lifetime value as an inner product of discounted revenue and survival-probability streams — the metric collapsed onto 2×2 axes (survival proxy vs. revenue) for segmentation.

## Reference Tables
| Quadrant | Factor 1 | Factor 2 | Typical use |
|---|---|---|---|
| A | High | High | Target segment / best responders |
| B | Low | High | Safer segment in adverse-selection cases |
| C | Low | Low | Baseline / least interesting |
| D | High | Low | Organic converters (no lever needed) |

## Worked Example
Cross-sell campaign test: x-axis = classifier probability score, y-axis = communication lever on/off. Expect CR_D > CR_C (organic purchases happen even without a nudge) and CR_A > CR_B (monotonic score). Credit origination: adverse selection makes acceptance-probability and default-probability positively correlated (quadrant A = high accept + high default); safest strategy targets quadrant B (likely to accept, unlikely to default), or thresholds are adjusted for more volume at acceptable risk.

## Key Takeaways
1. Simplification to two dimensions is a deliberate design choice that aids both analysis and stakeholder communication.
2. In experimental settings, 2×2 designs let you jointly test a model's predictiveness and a lever's effectiveness via monotonicity and effectiveness hypotheses.
3. In non-experimental settings, 2×2s still help focus attention, even without a "constant everything else" guarantee.
4. Quadrant-based labeling (1 for target quadrant, 0 otherwise) turns a 2×2 insight into an ML classification problem for deeper investigation.
5. 2×2 project prioritization (value × effort) breaks down when projects dominate on different axes — see Ch 5 for the fix.
6. Adverse selection creates correlated-feature decision problems; 2×2s simplify picking a risk threshold.

## Connects To
- **Ch 2**: The p×q decomposition (price × transactionality) is a common source of the two factors used in 2×2s.
- **Ch 5**: Business case math (common monetary scale) resolves the project-ranking limitation of the value/effort 2×2.
- **Ch 10**: Partialling out other factors via randomization previews the "holding everything else constant" logic in regression.
- **Ch 13**: Quadrant-membership classification models are opened up using storytelling/interpretability techniques from Ch 13.
