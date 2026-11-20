# Chapter 7: Hypothesis and Inference

## Core Idea
Statistical hypothesis testing turns a question ("is this coin fair?", "which ad converts better?") into a decision rule built from a null hypothesis, a normal approximation, and a significance level, with p-values, confidence intervals, and Bayesian inference offered as three different but related framings of the same underlying inference problem.

## Frameworks Introduced
- **Classical null/alternative hypothesis testing**: define H0 (default, e.g. "coin is fair, p=0.5") vs. H1, pick a significance level (type 1 error rate, commonly 5%), compute the rejection region from the normal approximation to the test statistic's distribution, and reject H0 if the observed statistic falls in that region.
  - When to use: deciding whether an observed effect is likely real or explainable by chance under a default assumption.
  - How: `normal_approximation_to_binomial(n, p)` gives `(mu, sigma)`; use `normal_two_sided_bounds`/`normal_upper_bound`/`normal_lower_bound` to get critical values at a chosen significance.
- **p-values**: the probability, assuming H0 true, of observing a value at least as extreme as the one actually observed; reject H0 if p-value < significance level. Equivalent to the bounds-based test but framed around a continuous "how surprising is this" score.
  - How: `two_sided_p_value(x, mu, sigma)`; apply a continuity correction (e.g., use 529.5 instead of 530) when approximating a discrete binomial count with the continuous normal.
- **Confidence intervals**: construct an interval around the parameter estimate (e.g., `p_hat = heads/n`) such that it would contain the true parameter 95% of the time across repeated experiments.
  - When to use: reporting an estimate's uncertainty directly, rather than testing a specific null value.
- **A/B testing via a two-proportion z-statistic**: `z = (p_B - p_A) / sqrt(sigma_A^2 + sigma_B^2)`, approximately standard normal under the null that `p_A = p_B`.
  - When to use: comparing two variants (ads, UI treatments) by click/conversion rate.
- **Bayesian inference with a Beta prior**: start with a Beta(alpha, beta) prior on an unknown probability p, observe h heads/t tails, and the posterior is exactly Beta(alpha+h, beta+t) (the Beta distribution is the conjugate prior to the Binomial).
  - When to use: when you want probability statements about the *parameter* itself ("5% chance p is between 49% and 51%") rather than about a test statistic under a null.

## Key Concepts
- **Null hypothesis (H0) / alternative hypothesis (H1)**: the default assumption to be tested vs. the competing claim.
- **Type 1 error (false positive)**: rejecting H0 when it's actually true; its rate is the chosen significance level.
- **Type 2 error (false negative)**: failing to reject H0 when it's actually false; `power = 1 - P(type 2 error)`.
- **Statistical power**: the probability a test correctly rejects a false H0; depends on what the true alternative value actually is.
- **Continuity correction**: adjusting a discrete count (e.g., 530) by 0.5 when approximating with a continuous normal distribution, for a more accurate probability estimate.
- **p-hacking**: running enough hypothesis tests (or massaging data/outliers) until one appears "significant" purely by chance — a 5%-significance test will, by construction, falsely reject a true null about 5% of the time.
- **Conjugate prior**: a prior distribution family that, after Bayesian updating with the relevant likelihood, yields a posterior in the same family (Beta is conjugate to Binomial).

## Mental Models
- **One-sided vs. two-sided tests**: choose based on what the alternative hypothesis actually claims — "p ≠ 0.5" needs a two-sided test; "p > 0.5" needs a one-sided test, which is more powerful against that specific alternative because it doesn't "waste" significance on the opposite tail.
- **p-values and confidence intervals are two views of the same test**: rejecting H0 at 5% significance is equivalent to the null value falling outside the 95% confidence interval around the estimate.
- **Frequentist vs. Bayesian framing**: frequentist methods make probability statements about the test/data ("X% chance of seeing data this extreme if H0 true"); Bayesian methods make probability statements about the parameter itself ("X% chance p is in this range") — philosophically different, and the book favors frequentist methods but flags Bayesian as a legitimate alternative.
- **More data overwhelms the prior**: as sample size grows, the Bayesian posterior converges to (nearly) the same place regardless of the starting prior — subjectivity in prior choice matters less and less with more evidence.

## Anti-patterns
- **p-hacking**: testing many hypotheses, or selectively removing outliers, until p < 0.05 — a near-guaranteed way to manufacture "significant" findings that aren't real; the book explicitly warns this is common in bad data science and references "The Earth Is Round" by Jacob Cohen.
- **Setting hypotheses after looking at the data**: determine hypotheses and clean data *before* examining results tied to those hypotheses, or you're effectively p-hacking.
- **Using `normal_probability_above` on non-normal data**: computing a "one in a million" probability by assuming normality when the data isn't normally distributed is explicitly called out as a common, serious mistake — check normality (plot the data at minimum) first.
- **Over-literal reading of a confidence interval**: a 95% CI is a statement about the interval-generating procedure over repeated experiments, not a 95% probability that the true parameter lies in *this particular* interval (frequentist framing).
- **Treating estimated standard deviations as known**: using the normal approximation (vs. a t-distribution) when sigma is estimated from data is "sort of cheating," acceptable only because sample sizes are large enough that the difference is negligible.

## Code Examples
```python
def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    if x >= mu:
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)
```
- **What it demonstrates**: approximating a binomial test statistic as normal, then computing a two-sided p-value from it.

```python
def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)
```
- **What it demonstrates**: a from-scratch two-proportion z-test for comparing two ad variants' click rates.

```python
def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
```
- **What it demonstrates**: the Beta distribution PDF used as a Bayesian prior/posterior over an unknown probability p.

## Reference Tables
| Concept | Formula |
|---|---|
| Type 1 error rate | significance level (e.g. 5%) |
| Power | `1 - P(type 2 error)` |
| Two-sided p-value | `2 * P(X >= \|observed - mu\|)` under H0 |
| 95% CI for p_hat | `normal_two_sided_bounds(0.95, p_hat, sqrt(p_hat(1-p_hat)/n))` |
| A/B z-statistic | `(p_B - p_A) / sqrt(sigma_A^2 + sigma_B^2)` |
| Beta posterior | prior Beta(alpha, beta) + h heads, t tails -> Beta(alpha+h, beta+t) |

## Worked Example
Testing a coin's fairness with n=1000 flips: under H0 (p=0.5), `mu=500, sigma=15.8`. A 95% two-sided rejection region is roughly (469, 531). Observing 530 heads gives a two-sided p-value of 0.062 (don't reject); observing 532 heads gives p=0.0463 (reject). The A/B test: ad A gets 200/1000 clicks, ad B gets 180/1000 — z=-1.14, p=0.254 (no real difference detected); if B had only gotten 150/1000 clicks, z=-2.94, p=0.003 (statistically significant difference). The Bayesian coin example: starting from a uniform Beta(1,1) prior and observing 3 heads in 10 flips yields posterior Beta(4,8) centered at 0.33; starting from a skeptical Beta(20,20) prior yields Beta(23,27) centered at 0.46 — different priors converge as more data arrives.

## Key Takeaways
1. A hypothesis test is a decision rule: pick a null hypothesis, a significance level, derive the rejection region via the normal approximation, then compare the observed statistic.
2. p-values and confidence intervals are two equivalent framings of the same classical test — know both, since different audiences expect different language.
3. One-sided tests are more powerful than two-sided tests against a one-directional alternative — match test shape to the actual alternative hypothesis.
4. Avoid p-hacking: fix hypotheses and data-cleaning rules before looking at results, and remember that a 5%-significance test will, by design, false-positive 5% of the time.
5. Bayesian inference reframes the problem as updating a prior distribution over the parameter itself, using a conjugate prior (Beta-Binomial here) for mathematical tractability.
6. Always sanity-check normality assumptions (plot the data) before applying normal-approximation-based p-values.

## Connects To
- **Ch6**: directly reuses `normal_cdf`, `inverse_normal_cdf`, and the binomial-to-normal CLT approximation built there.
- **Ch5**: the A/B test statistic and confidence intervals extend Ch5's mean/standard-deviation machinery to proportions.
- **Ch16 (Logistic Regression)**: hypothesis-testing intuition (significance, confidence) recurs when evaluating coefficient significance in regression models.
- **t-distribution / scipy.stats**: the book flags that a rigorous treatment of small-sample inference should use the t-distribution rather than the normal approximation used throughout this chapter.
