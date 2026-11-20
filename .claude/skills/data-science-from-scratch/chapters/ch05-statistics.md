# Chapter 5: Statistics

## Core Idea
Statistics is the toolkit for distilling and communicating a large dataset's relevant features, covering central tendency (mean, median, mode, quantile), dispersion (range, variance, standard deviation, interquartile range), and the relationship between two variables (covariance, correlation) — with explicit warnings about outliers, Simpson's paradox, and correlation vs. causation.

## Frameworks Introduced
- **Central tendency toolkit**: `mean` (sum/n, sensitive to every point and to outliers), `median` (middle value(s), robust to outliers but needs sorting), `mode` (most common value(s)), `quantile(xs, p)` (value below which fraction p of data lies).
  - When to use: mean for smooth/calculus-friendly summaries; median/quantile when outliers may distort the mean.
- **Dispersion toolkit**: `data_range` (max - min), `variance` (avg squared deviation from mean, divided by n-1 for unbiased sample estimate), `standard_deviation` (sqrt of variance, same units as data), `interquartile_range` (75th - 25th percentile, outlier-robust).
  - When to use: range/stdev for a quick global spread; IQR when outliers may be present.
- **Covariance and correlation**: `covariance(xs, ys) = dot(de_mean(xs), de_mean(ys)) / (n-1)` measures how two variables move together; `correlation = covariance / (stdev_x * stdev_y)` normalizes covariance into [-1, 1], unitless.
  - When to use: quantifying linear relationship strength between two numeric variables; always inspect the scatterplot first since correlation is sensitive to outliers and blind to nonlinear relationships.

## Key Concepts
- **Mean**: `sum(xs) / len(xs)`; shifts by `e/n` if any point increases by `e`; distorted by outliers.
- **Median**: the middle value (odd n) or average of the two middle values (even n); robust to extreme values, requires sorting.
- **Variance**: `sum_of_squares(de_mean(xs)) / (n - 1)`; divides by `n-1` (not `n`) for an unbiased sample estimate (Bessel's correction).
- **Standard deviation**: `sqrt(variance)`; same units as the original data, unlike variance.
- **Interquartile range (IQR)**: `quantile(xs, 0.75) - quantile(xs, 0.25)`; a robust dispersion measure unaffected by a small number of outliers.
- **Covariance**: paired analogue of variance; units are the product of the two variables' units, making magnitude hard to interpret directly.
- **Correlation**: covariance normalized by both standard deviations; unitless, in [-1, 1].
- **Simpson's paradox**: a correlation can reverse direction (or appear/disappear) once a confounding variable is controlled for.

## Mental Models
- **Mean vs. median trade-off**: mean uses all data and varies smoothly (good for calculus/optimization) but is outlier-sensitive; median is robust but discontinuous and needs full sort.
- **"All else being equal" assumption**: correlation implicitly assumes no confounders; Simpson's paradox is what happens when that assumption silently fails.
- **Correlation measures linear co-movement only**: zero correlation does not mean no relationship (e.g., `y = |x|`); nonzero correlation also says nothing about how large or meaningful the relationship is.
- **Correlation != causation, but randomized trials can establish causal direction**: splitting users into comparable random groups and giving one a different treatment is the gold-standard way to move from correlation toward causal confidence.

## Anti-patterns
- **Reporting the mean without checking for outliers**: one 200-friend outlier changes the mean from 7.33 but leaves the median at 6, and inflates standard deviation by 60%+.
- **Trusting correlation without plotting the data**: an outlier user (100 friends, 1 minute/day) drags correlation from ~0.57 down to ~0.25 — always visualize before trusting a correlation number.
- **Bucketing data coarsely and ignoring confounders**: the East Coast/West Coast friendliness comparison reverses once PhD status is controlled for (Simpson's paradox).
- **Treating zero correlation as "no relationship"**: correlation only detects linear co-movement; a perfect nonlinear relationship (e.g., `y = |x|`) can show zero correlation.
- **Assuming correlation implies a specific causal direction**: `x` causing `y`, `y` causing `x`, a common third cause, or pure coincidence are all consistent with the same correlation value.

## Code Examples
```python
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def _median_odd(xs): return sorted(xs)[len(xs) // 2]
def _median_even(xs):
    sorted_xs = sorted(xs)
    hi = len(xs) // 2
    return (sorted_xs[hi - 1] + sorted_xs[hi]) / 2
def median(v): return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)
```
- **What it demonstrates**: from-scratch mean/median with the odd/even case split out into private helper functions.

```python
def variance(xs: List[float]) -> float:
    assert len(xs) >= 2, "variance requires at least two elements"
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))
```
- **What it demonstrates**: variance built on Ch4's `sum_of_squares`, dividing by `n-1` for an unbiased sample estimate.

```python
def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have same number of elements"
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

def correlation(xs: List[float], ys: List[float]) -> float:
    stdev_x, stdev_y = standard_deviation(xs), standard_deviation(ys)
    return covariance(xs, ys) / stdev_x / stdev_y if stdev_x > 0 and stdev_y > 0 else 0
```
- **What it demonstrates**: covariance/correlation reusing Ch4's `dot` on de-meaned vectors.

## Reference Tables
| Statistic | Formula | Outlier sensitivity |
|---|---|---|
| Mean | `sum(xs)/n` | high |
| Median | middle value(s) | low |
| Mode | most common value(s) | low |
| Range | `max - min` | high |
| Variance | `sum_of_squares(de_mean(xs))/(n-1)` | high |
| Standard deviation | `sqrt(variance)` | high |
| IQR | `quantile(0.75) - quantile(0.25)` | low |
| Covariance | `dot(de_mean(xs), de_mean(ys))/(n-1)` | high |
| Correlation | `covariance / (stdev_x * stdev_y)` | high |

## Worked Example
`num_friends` (204 DataSciencester members) has mean 7.33, median 6, mode {1, 6}, range 99, variance ~81.54, standard deviation ~9.03, IQR 6. Pairing it with `daily_minutes` (time spent on site) gives covariance ~22.42 and correlation ~0.245 — a weak positive correlation that jumps to ~0.57 once a single outlier (a 100-friend internal test account that spent only 1 minute/day) is removed. The Simpson's paradox example: West Coast data scientists average 8.2 friends vs. East Coast's 6.5 (West looks friendlier) — but split by PhD status, East Coast beats West Coast in *both* the PhD (3.2 vs 3.1) and non-PhD (13.4 vs 10.9) subgroups, because East Coast skews more heavily PhD.

## Key Takeaways
1. Always inspect the data (histogram/scatterplot) before trusting a summary statistic — mean, standard deviation, and correlation are all outlier-sensitive.
2. Prefer median/IQR over mean/standard-deviation/range when outliers are likely or suspected.
3. Correlation only captures linear co-movement, is unitless, and says nothing about causation or relationship magnitude.
4. Watch for Simpson's paradox: a correlation can reverse once you control for a confounding variable — always ask what's being bucketed away.
5. Causal claims need more than correlation — randomized trials are the most reliable way to establish causal direction.

## Connects To
- **Ch4**: `variance`, `covariance`, and `correlation` are built directly on Ch4's `sum_of_squares` and `dot`.
- **Ch1**: reuses the `num_friends`/`daily_minutes` DataSciencester dataset first introduced informally there.
- **Ch14 (Simple Linear Regression)**: correlation here is a precursor to the regression coefficients computed there.
- **SciPy / pandas / StatsModels**: external libraries offering the same (and many more) statistical functions for production use.
