# Chapter 3: Statistical Experiments and Significance Testing

## Core Idea
Classical hypothesis testing (p-values, t-tests, chi-square, ANOVA) exists to protect against the human tendency to see meaningful patterns in random chance, but for data science purposes resampling methods (permutation tests, bootstrap) answer the same questions more flexibly and transparently, and multi-arm bandits often replace formal A/B testing entirely when the goal is optimization rather than publication-grade proof.

## Frameworks Introduced
- **A/B Testing**: randomized two-group (or more) comparison of treatments against a control, with a single pre-specified test statistic.
  - When to use: comparing two (or more) concrete treatments (prices, web designs, drugs) where subjects can be randomly assigned.
  - How: randomize subjects to treatment/control, pick one metric in advance, collect data, compare using a hypothesis test or resampling.
- **Permutation Test**: combine all groups' data, repeatedly shuffle and redeal into group-sized resamples, and compare the observed statistic to the resulting null distribution.
  - When to use: general-purpose, assumption-free hypothesis testing for any statistic and any group-size combination.
  - How: combine → shuffle → split into original-size groups → compute statistic → repeat R times → p-value = proportion of resamples ≥ observed difference.
- **t-test**: standardized comparison of two group means using Student's t-distribution as a formula-based shortcut for what a permutation test would show.
  - When to use: numeric two-group comparison, especially when resampling infrastructure isn't available or as a cross-check.
  - How: `t.test()` in R / `scipy.stats.ttest_ind()` in Python with `alternative` specifying one- vs two-tailed.
- **ANOVA (Analysis of Variance)**: tests whether differences among 3+ group means exceed what chance variation would produce, via the F-statistic (ratio of between-group to within-group variance).
  - When to use: comparing numeric outcomes across 3+ treatment groups.
  - How: decompose variance into grand average + treatment effect + residual; compute F = MS(treatment)/MS(error); compare to F-distribution, or run as a permutation test (`aovp` in R's `lmPerm`).
- **Chi-Square Test**: tests whether observed category counts depart from an expected/independence distribution, via Pearson residuals summed and squared.
  - When to use: r×c contingency tables, multi-treatment count comparisons (e.g., 3+ headlines' click rates), feature selection by class-count association.
  - How: compute expected counts under the null, Pearson residual = (observed − expected)/√expected per cell, Χ² = Σresidual², compare to chi-square distribution (df = (r−1)×(c−1)) or via resampling.
- **Fisher's Exact Test**: enumerates all possible rearrangements exactly rather than approximating, used when cell counts are very low (≤5).
  - When to use: small/sparse contingency tables where the chi-square approximation is unreliable.
- **Multi-Arm Bandit (epsilon-greedy, Thompson sampling)**: adaptive allocation across 2+ treatments that shifts traffic toward better-performing arms during the experiment instead of waiting for a fixed-sample conclusion.
  - When to use: optimizing conversion/click-through across multiple treatments in production (web testing), where speed/efficiency matters more than formal proof.
  - How (epsilon-greedy): with probability epsilon, pick a random arm (explore); otherwise, pick the current best-performing arm (exploit).
- **Power and Sample Size calculation**: four interdependent quantities — sample size, effect size, significance level (alpha), power — specify any three to solve for the fourth.
  - When to use: planning how much data/traffic an experiment needs before running it.
  - How: use `pwr.2p.test` (R) or `statsmodels.stats.power` (Python), or simulate via bootstrap-generated hypothetical samples.

## Key Concepts
- **Null hypothesis / alternative hypothesis**: null = "observed difference is due to chance"; alternative = the effect you hope to demonstrate.
- **p-value**: the probability, under the null (chance) model, of seeing a result as extreme as what was observed — NOT the probability the hypothesis is true.
- **Alpha**: the pre-specified threshold (commonly 0.05) of "unusualness" required to call a result statistically significant.
- **Type 1 error**: concluding an effect is real when it's actually chance. **Type 2 error**: concluding no effect when one truly exists (often really a sample-size problem).
- **One-way vs. two-way test**: one-tailed counts extremity in only one direction (appropriate when you'll only switch from the default if clearly better); two-tailed counts both directions.
- **Multiple testing / alpha inflation / vast search effect**: running many significance tests on the same data inflates the chance of a false "discovery"; mitigated by adjustment (Bonferroni, Tukey's HSD) or, in ML, by holdout/cross-validation.
- **Degrees of freedom**: the number of values free to vary in a statistic's calculation; used to select the correct reference distribution (t, F, chi-square).
- **Decomposition of variance**: any observed value = grand average + treatment effect + residual error — the conceptual basis of ANOVA and later regression.
- **False discovery rate**: across many tests (or many classified "1" predictions), the rate at which "discoveries" are actually false positives.

## Mental Models
- Think of every formula-based classical test (t-test, chi-square, F-test) as a fast mathematical approximation to what a permutation/resampling test would show directly — useful to know because software outputs them, but not more "correct" than resampling.
- Use the "torture the data" heuristic: if you search hard enough across many variables, subgroups, or models, you will eventually find something that looks significant purely by chance — guard with holdout sets, pre-registered hypotheses, or target shuffling.
- Treat statistical significance and practical significance as separate axes: a tiny, meaningless effect can be "significant" with enough data, and a "non-significant" result with a cheap intervention may still be worth pursuing.
- Picture the multi-arm bandit as continuously reallocating slot-machine pulls toward the arm that currently looks best, while never fully abandoning the others — a middle path between a rigid fixed-sample A/B test and a greedy "go with whatever looks best right now."

## Anti-patterns
- **Treating the p-value as "probability the hypothesis is true"**: the ASA's 2016 statement explicitly warns against this; a p-value only measures incompatibility between data and the null model.
- **Skipping the control group**: without one, you cannot separate treatment effect from other confounding changes over time.
- **Choosing the test statistic after seeing the data**: opens the door to researcher bias / data snooping.
- **Running many pairwise comparisons without adjustment**: inflates the overall Type 1 error rate (alpha inflation) — use ANOVA/chi-square as an omnibus test or apply Bonferroni/Tukey's HSD.
- **Applying generic A/B test "rules of thumb"** (e.g., "run for a week, 1,000 visitors") without a proper power/sample-size calculation based on actual effect size and baseline rate.
- **Using the standard deviation on highly skewed binary-outcome revenue data**: produces nonsensical (e.g., implying negative) summaries; mean absolute deviation is more honest here.

## Code Examples
```r
perm_fun <- function(x, n_a) {
  n <- length(x)
  x_resampled <- sample(x)
  x_a <- x_resampled[1:n_a]
  x_b <- x_resampled[(n_a + 1):n]
  return(mean(x_b) - mean(x_a))
}
perm_diffs <- replicate(1000, perm_fun(session_times$Time, 21))
mean(perm_diffs > (mean_b - mean_a))   # permutation p-value
```
- **What it demonstrates**: the general permutation-test pattern used throughout the chapter for A/B comparisons.

```r
summary(aov(Time ~ Page, data = four_sessions))
#             Df Sum Sq Mean Sq F value Pr(>F)
# Page         3  831.4   277.1    2.74 0.0776
# Residuals   16 1618.4   101.2
```
- **What it demonstrates**: one-way ANOVA F-test output for comparing stickiness across 4 web pages.

```r
chisq.test(clicks, simulate.p.value = TRUE)
# X-squared = 1.6659, df = NA, p-value = 0.4853
```
- **What it demonstrates**: resampled chi-square test for independence across 3 headline click rates (Python: `scipy.stats.chi2_contingency`).

```r
library(pwr)
effect_size <- ES.h(p1 = 0.0121, p2 = 0.011)
pwr.2p.test(h = effect_size, sig.level = 0.05, power = 0.8, alternative = "greater")
# n = 116601.7
```
- **What it demonstrates**: solving for required sample size given effect size, alpha, and desired power (Python: `statsmodels.stats.power.TTestIndPower`).

## Reference Tables
| Test | Data type | Groups | R/Python function |
|---|---|---|---|
| Permutation test | numeric or binary | 2+ | custom `perm_fun` / `np.random.permutation` |
| t-test | numeric | 2 | `t.test()` / `scipy.stats.ttest_ind` |
| ANOVA / F-test | numeric | 3+ | `aov()` / `statsmodels.formula.api.ols` + `anova_lm` |
| Chi-square test | counts | r×c table | `chisq.test()` / `scipy.stats.chi2_contingency` |
| Fisher's exact test | counts (small/sparse) | r×c table | `fisher.test()` (no simple Python equivalent) |

| Known 3 of 4 → solve for | Formula role |
|---|---|
| Sample size, effect size, alpha | → Power |
| Sample size, power, alpha | → minimum detectable effect size |
| Effect size, power, alpha | → required Sample size (most common use) |

## Worked Example
An ecommerce price test (Price A vs. Price B) produces a 2×2 table: Price A converts 200/23,739 (0.8425%), Price B converts 182/22,588 (0.8057%) — a 0.0368-point difference. A permutation test (shuffling a combined box of 382 "1"s / 45,945 "0"s into groups of 23,739 and 22,588) yields p ≈ 0.313: the observed difference is well within chance variation, confirmed by the `prop.test` normal approximation (p ≈ 0.3498). Separately, a web-stickiness A/B test (21 sessions on Page A, 15 on Page B, mean difference 35.67 seconds) is tested the same way (p ≈ 0.12–0.14 across permutation and t-test), again not significant. A third example extends this to 4 pages via ANOVA (p ≈ 0.078–0.093, borderline) and to 3 headlines via chi-square (34 total clicks across A/B/C, p ≈ 0.43–0.49, not significant) — illustrating that the same resampling logic (combine → shuffle → compare) underlies 2-group, multi-group, and count-based significance questions alike.

## Key Takeaways
1. Resampling (permutation, bootstrap) is a more intuitive, assumption-free, "one-size-fits-all" alternative to formula-based tests (t, F, chi-square) and should be the default mental model for understanding what any p-value means.
2. A p-value measures compatibility with the null model, not the probability a hypothesis is true or the importance of an effect — never treat passing/failing alpha = 0.05 as the sole basis for a decision.
3. Always pick your test statistic and significance threshold (alpha) before looking at results, and guard against the vast search / multiple-testing effect with holdout data or formal adjustment.
4. For production web/product optimization, multi-arm bandits typically beat traditional fixed-sample A/B tests because they reduce exposure to inferior treatments while still learning.
5. Run a power/sample-size calculation before collecting data to avoid gathering an inconclusive sample; specify 3 of {sample size, effect size, alpha, power} to get the 4th.
6. Statistical significance ≠ practical significance — check whether an effect's size actually matters for the business decision at hand.

## Connects To
- **Ch 2 (Data and Sampling Distributions)**: the bootstrap and permutation machinery introduced there is applied directly here to hypothesis testing.
- **Ch 4 (Regression and Prediction)**: ANOVA's decomposition of variance and the F-statistic are direct precursors to regression's overall model significance test.
- **External concept**: the 2016 American Statistical Association statement on p-values is the standard reference for the chapter's critique of p-value misuse.
