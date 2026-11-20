# Chapter 6: Probability

## Core Idea
Probability quantifies uncertainty over outcomes, and the chapter builds from dependence/independence and conditional probability through Bayes's theorem, random variables, and continuous distributions (uniform, normal), ending with the central limit theorem as the reason normal distributions are so broadly useful.

## Frameworks Introduced
- **Conditional probability**: `P(E|F) = P(E,F) / P(F)` — the probability E happens given F happens; independence means `P(E|F) = P(E)`.
  - When to use: whenever one event's probability depends on knowledge of another.
- **Bayes's theorem**: `P(E|F) = P(F|E) P(E) / P(F)`, with `P(F) = P(F|E)P(E) + P(F|¬E)P(¬E)` — "reverses" a conditional probability you know into one you want.
  - When to use: classic diagnostic-test problem — given test accuracy and disease prevalence, find P(disease | positive test). Low base-rate diseases yield surprisingly low P(D|T) even with accurate tests.
- **Central Limit Theorem (CLT)**: the average (or sum) of many independent, identically distributed random variables is approximately normally distributed, regardless of the underlying distribution.
  - When to use: justifies approximating a Binomial(n,p) with a Normal(np, sqrt(np(1-p))) for large n — simplifies otherwise hard combinatorial probability computations, and underlies hypothesis testing in Ch7.
- **Inverse CDF via binary search**: since `normal_cdf` has no closed-form inverse, `inverse_normal_cdf` bisects the z-range `[-10, 10]` until `normal_cdf(mid_z)` is within `tolerance` of the target probability `p`.
  - When to use: any monotonic CDF without a closed-form inverse; binary search is the general from-scratch fallback.

## Key Concepts
- **Dependence/independence**: E, F independent iff `P(E,F) = P(E)P(F)`; otherwise dependent.
- **Random variable**: a variable whose possible values have an associated probability distribution; its **expected value** is the probability-weighted average of its values.
- **Discrete vs. continuous distribution**: discrete assigns positive probability to individual outcomes (coin flip); continuous assigns zero probability to any single point and is described by a probability density function (PDF) whose integral over an interval gives probability.
- **PDF vs. CDF**: PDF gives relative likelihood density (`P(x < X < x+h) ≈ h*f(x)` for small h); CDF gives `P(X <= x)`, the integral of the PDF up to x.
- **Normal distribution**: bell curve fully specified by mean `mu` and standard deviation `sigma`; `mu=0, sigma=1` is the "standard normal."
- **Bernoulli/Binomial**: `Bernoulli(p)` is 1 with probability p, 0 otherwise; `Binomial(n, p)` is the sum of n independent Bernoulli(p) trials, with mean `np` and standard deviation `sqrt(np(1-p))`.

## Mental Models
- **The "either girl" vs. "older girl" conditioning puzzle**: conditioning on "at least one girl" (1/3 chance both are girls) gives a different answer than conditioning on "the older child is a girl" (1/2 chance both are girls) — a canonical illustration that *how* you condition matters, not just *that* you condition.
- **Base-rate neglect**: Bayes's theorem reveals that even a 99%-accurate test for a rare (1-in-10,000) disease yields under 1% P(disease|positive) — a reminder to always weigh test accuracy against prior prevalence, not just accuracy alone.
- **CLT as a simplification tool**: rather than computing an exact discrete distribution (e.g., exact Binomial CDF), approximate with the much easier-to-compute Normal CDF when n is large — this trade-off (exactness for tractability) recurs in hypothesis testing (Ch7).

## Anti-patterns
- **Assuming independence without justification**: the "two unknown children" puzzle shows how easy it is to misjudge conditional probabilities when the conditioning event is stated imprecisely ("older child is a girl" vs. "at least one child is a girl").
- **Ignoring base rates when interpreting a positive test**: a naive reading of "99% accurate" as "99% likely you have the disease if positive" is wrong by two orders of magnitude once the 1-in-10,000 base rate is accounted for.
- **Computing an exact discrete distribution when an approximation suffices**: for large n, use the CLT's normal approximation rather than the exact (and expensive) Binomial CDF.

## Code Examples
```python
def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma)

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
```
- **What it demonstrates**: from-scratch normal PDF/CDF, the latter via Python's built-in `math.erf`.

```python
def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z, hi_z = -10.0, 10.0
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p: low_z = mid_z
        else: hi_z = mid_z
    return mid_z
```
- **What it demonstrates**: binary search used to invert a monotonic function with no closed-form inverse.

```python
def bernoulli_trial(p: float) -> int:
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    return sum(bernoulli_trial(p) for _ in range(n))
```
- **What it demonstrates**: building a Binomial(n,p) sampler from repeated Bernoulli trials, later compared visually to its normal approximation.

## Reference Tables
| Distribution | Parameters | Mean | Std Dev |
|---|---|---|---|
| Bernoulli(p) | p | p | sqrt(p(1-p)) |
| Binomial(n,p) | n, p | np | sqrt(np(1-p)) |
| Uniform(0,1) | none | 0.5 | ~0.289 |
| Normal(mu, sigma) | mu, sigma | mu | sigma |

| Bayes's theorem term | Meaning |
|---|---|
| P(T\|D) | test positive given disease (sensitivity) |
| P(D) | prior/base rate of disease |
| P(T\|¬D) | false positive rate |
| P(D\|T) | posterior probability of disease given positive test |

## Worked Example
Disease affects 1 in 10,000 people; test is 99% accurate both ways. Imagining 1,000,000 people: ~100 have the disease (99 test positive), ~999,900 don't (9,999 still test positive). So P(D|T) = 99/(99+9999) ≈ 0.98%, under 1% — despite the "99% accurate" test. Separately, the two-children puzzle is verified by simulating 10,000 families with `random_kid()`: P(both girls | older is girl) ≈ 0.514 (~1/2), while P(both girls | at least one girl) ≈ 0.342 (~1/3).

## Key Takeaways
1. Conditional probability is sensitive to *exactly* how you condition — "older child is a girl" and "at least one child is a girl" give different answers to the same-seeming question.
2. Bayes's theorem is essential for correctly interpreting diagnostic tests: always weigh test accuracy against the base rate (prevalence) of the condition.
3. The normal distribution's ubiquity comes from the Central Limit Theorem: averages/sums of many independent variables converge to normal, regardless of the underlying distribution.
4. Binary search is a general-purpose way to invert any continuous, monotonic function (like `normal_cdf`) that lacks a closed form.
5. Use the CLT's normal approximation to sidestep expensive exact discrete-distribution computations when n is large.

## Connects To
- **Ch5**: reuses `mean`/`standard_deviation` concepts, now cast as properties of random variables/distributions.
- **Ch7 (Hypothesis and Inference)**: directly builds on `normal_cdf`/`inverse_normal_cdf` and the CLT's binomial-to-normal approximation for significance testing.
- **scipy.stats**: external library providing PDF/CDF functions for essentially every distribution, recommended for production use over this chapter's from-scratch versions.
