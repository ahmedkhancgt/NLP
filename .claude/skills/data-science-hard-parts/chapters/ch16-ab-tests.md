# Chapter 16: A/B Tests

## Core Idea
A/B tests use randomization to compare a baseline (A) against an alternative (B) on a chosen metric; rigorous design requires balancing statistical significance (false positives) against statistical power (false negatives) via the minimum detectable effect, and statistical significance must be distinguished from business significance.

## Frameworks Introduced
- **A/B Test Ingredients**: Metric (Y, chosen via Ch 2 metrics-design techniques), Levers/alternatives (A=baseline/control, B=contender/treatment — choose the lever first, from the metric, never reverse-engineer a metric from an arbitrary UI tweak), Randomized selection (breaks confounding/selection bias per Ch 15).
  - When to use: always start from the metric, then find levers that directly affect it — never start from "let's test a button color" and search for a metric afterward.
- **Decision Criterion (two-sided t-test)**: `H0: θ=0` vs `H1: θ≠0`, where `θ = Ȳ_B − Ȳ_A`. Reject H0 (keep lever B) if `|t-stat| falls in the rejection region at significance level α` (commonly 5% or 1%).
  - Equivalent via regression: `Y = α + θD + ε`; θ̂_OLS's p-value gives the same test, with the bonus of being able to add control variables for tighter confidence intervals.
- **Minimum Detectable Effect (MDE)**: the smallest true effect size you can reliably detect given significance level α, power (1−β), sample size N, treatment fraction P, and outcome variance.
  - Formula: `MDE = (t_α + t_(1−β)) × sqrt(Var(Y) / (N·P·(1−P)))`.
  - When to use: before launching any experiment — determines whether your planned sample size can detect a business-meaningful effect, or inversely, solve for minimum sample size given a target MDE.
- **False Positive / False Negative framing for A/B tests**: a false positive = concluding there's an effect when there isn't (controlled by α, the significance level); a false negative = concluding there's no effect when there is (controlled by β, i.e., 1−power).
  - When to use: as the lens for choosing α and β/power jointly, trading off the two error types for a fixed MDE.
- **Statistical vs. Business Significance**: a statistically detectable effect (e.g., MDE = 3.3e-5 with 1B customers) may be commercially meaningless; always set the MDE based on what matters to the business, not just what's statistically feasible.

## Key Concepts
- **Treatment/Control groups**: units receiving B (treatment) vs. A (control), using Ch 15's causal-inference language.
- **Significance level (α)**: probability of a false positive under the null; conventionally 5% or 1%.
- **Statistical power (1−β)**: probability of correctly detecting a true effect; conventionally targeted around 85% (β=15%).
- **Fraction of treated units (P)**: proportion assigned to treatment; MDE is minimized at P=0.5 (equal-sized groups), the standard default.
- **Outcome variance estimation**: estimate from existing data (e.g., sample variance of ARPU), from Bernoulli variance `q(1−q)` for binary/conversion-rate outcomes using a historical conversion rate as q, or from a prior A/A test (both groups get the same baseline A, used purely to estimate noise).
- **A/A test**: a test where both groups receive the baseline, used solely to empirically estimate the outcome's variance (and validate the test infrastructure/randomization) before running a real A/B test.
- **Hypotheses backlog**: a ranked, shared list of experiment hypotheses with their target metric and causal rationale — the organizational discipline for running a culture of experimentation.
- **Good hypothesis structure**: states both the *how* (direction of effect) and the *why* (causal mechanism); quantified hypotheses ("conversion rate increases by 1.2pp") are more useful for ranking than purely directional ones.
- **Experimentation governance objectives**: Accountability (clear owner), Business safety (guardrails/kill-switches on KPI thresholds), Customer/human centricity (ethical standards), Global vs. local effectiveness (avoid overlapping tests, manage quarantine/resting periods), Knowledge incrementality (build a repository of results, positive and negative), Replicability/reproducibility (document code/design), Security (data privacy compliance), Transparency/monitoring (timely, widely available results).

## Mental Models
- A/B tests are "local optimization" for an organization's key metrics — small, randomized, causally clean experiments that incrementally improve decisions.
- MDE first, sample size second: fix the smallest effect that matters to the business, then solve backward for the sample size needed to detect it — don't just run whatever sample size is convenient and hope for significance.
- Metric-first, lever-second discipline avoids the common failure mode of "we have an idea for a UI change, what metric should we measure?" — which almost always produces wasted effort and inconclusive results.
- Underpowered tests don't prove "no effect" — they prove "we couldn't detect an effect of this size with this sample," a crucial distinction for interpreting negative results.

## Anti-patterns
- **Reverse-engineering a metric from a chosen UI tweak**: nearly always wastes time and yields inconclusive, hard-to-interpret results — always start from the metric and work toward the lever.
- **Running underpowered experiments and concluding "no effect"**: a true effect smaller than the MDE will appear statistically insignificant purely due to insufficient sample size — not because it doesn't exist.
- **Conflating statistical significance with business significance**: a detectable 0.003pp lift with a billion-user sample is statistically real but often commercially irrelevant; set your MDE target from business impact, not from "whatever N we happen to have."
- **Using top-line KPIs as A/B test metrics**: top-line metrics are usually too many causal steps removed from the lever (low "actionability/relevance" per Ch 2), weakening the test's power and interpretability — decompose to a closer, more directly movable metric.
- **Launching experiments without a documented causal hypothesis (how + why)**: makes it impossible to rank hypotheses by expected value or learn systematically from results.
- **Running overlapping tests without governance**: concurrent experiments with overlapping treatment/control populations can contaminate each other's estimates.

## Code Examples
```python
def compute_mde(sample_size, var_outcome, size=0.05, power=0.85):
    dof = sample_size - 1
    t_alpha = stats.t.ppf(1 - size, dof)
    t_ombeta = stats.t.ppf(power, dof)
    p = 0.5
    den = sample_size * p * (1 - p)
    MDE = (t_alpha + t_ombeta) * np.sqrt(var_outcome / den)
    return MDE
```
- **What it demonstrates**: computing the Minimum Detectable Effect from sample size, outcome variance, significance level, and power.

```python
def compute_sample_size(mde, var_outcome, data_size, size=0.05, power=0.85):
    dof = data_size - 1
    t_alpha = stats.t.ppf(1 - size, dof)
    t_ombeta = stats.t.ppf(power, dof)
    sum_t = t_alpha + t_ombeta
    p = 0.5
    sample_size = var_outcome / (p * (1 - p)) * (sum_t**2 / mde**2)
    return sample_size
```
- **What it demonstrates**: inverting the MDE formula to solve for the minimum sample size needed to detect a target effect — the practical planning tool for experiment design.

## Reference Tables
| Concept | Controls | Typical value |
|---|---|---|
| Significance level (α) | False positive probability | 5% or 1% |
| Power (1−β) | True positive probability (1 − false negative rate) | 85% (β = 15%) |
| Fraction treated (P) | MDE minimized at this value | 0.5 |

| MDE driver | Effect on MDE |
|---|---|
| ↑ Sample size (N) | ↓ MDE (easier to detect smaller effects) |
| ↑ Outcome variance | ↑ MDE (noisier data needs bigger samples for same MDE) |

## Worked Example
10-unit toy example: `θ = 0.84 − 1.05 = −0.21`, t-stat = −0.44, p-value = 0.67 → cannot reject H0, keep default A (too few units, too much noise to draw a conclusion). Simulation with true effect θ=0.5, residual variance σ²=3, N=500: minimum sample size for MDE=0.5 is N*=346; sampling at sizes from 50%–150% of N* shows true-positive rate rising and false-negative rate falling with sample size, both crossing the target power threshold (β=15%) right around N*, validating the MDE formula's practical accuracy. Simulation with θ=0 (no true effect): roughly 5% of resamples incorrectly flag a significant effect, matching the chosen α=5% — a direct demonstration of what the significance level actually controls. Conversion-rate email test (baseline 4%): with N=1,000, MDE=3.3pp (can't detect even a strong 5.5% conversion rate — don't run the test); with N=1,000,000, MDE=0.1pp (detects anything above 4.1%); with N=1,000,000,000, MDE=0.0033pp (detects anything above 4.003%, likely too fine-grained to matter commercially) — illustrating how to set a business-sensible MDE/sample-size target jointly with stakeholders.

## Key Takeaways
1. Always design A/B tests metric-first, lever-second — never reverse-engineer a metric from an arbitrary UI idea.
2. The MDE formula ties together significance level, power, sample size, and outcome variance; use it to plan experiments, not just analyze them after the fact.
3. An underpowered experiment's "no significant effect" result may be a false negative, not evidence of no true effect — insufficient sample size, not absence of effect, could be the cause.
4. Distinguish statistical significance from business significance: set your target MDE based on what incremental change would actually matter to the business.
5. Estimate outcome variance from historical data, the Bernoulli formula for binary outcomes, or an A/A test when nothing else is available.
6. A hypotheses backlog (ranked, with metric + causal how/why rationale) is a core practice for building an organizational culture of experimentation.
7. Mature experimentation programs need governance: accountability, business-safety guardrails, ethical standards, non-overlapping tests, a knowledge repository, reproducibility, security, and transparency.

## Connects To
- **Ch 2**: Reinforces that A/B test metrics should be close to the lever (actionable/relevant), per the metrics-design properties.
- **Ch 14**: Reuses the false-positive/false-negative threshold-decision framing from classification decision rules.
- **Ch 15**: Directly builds on the randomization/unconfoundedness material — A/B tests are the practical implementation of that chapter's causal-inference ideal.
- **Ch 9**: Bootstrapping/Monte Carlo simulation techniques are reused to validate the MDE formula empirically via simulation.
