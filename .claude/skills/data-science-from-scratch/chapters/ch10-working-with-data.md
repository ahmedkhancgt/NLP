# Chapter 10: Working with Data

## Core Idea
Working with data is "both an art and a science": this chapter covers exploring data (1D histograms, 2D scatterplots, many-D correlation matrices), choosing better data representations (NamedTuple over dict), cleaning/parsing with graceful failure, manipulating/aggregating data with comprehensions, rescaling for distance-sensitive techniques, tracking long computations with tqdm, and reducing dimensionality from scratch via PCA built on gradient descent.

## Frameworks Introduced
- **Exploratory data analysis ladder**: 1D (summary stats + histogram via `bucketize`/`Counter`/`plt.bar`) -> 2D (scatterplot + `correlation`) -> many-D (`correlation_matrix` + scatterplot matrix via `plt.subplots`).
  - When to use: always explore before modeling; summary statistics alone can hide very different distributions (same mean/stdev, different shape).
- **NamedTuple / dataclass for structured records**: `class StockPrice(NamedTuple): symbol: str; date: datetime.date; closing_price: float` replaces error-prone dict-based records with typed, typo-safe, (optionally) immutable objects that can carry methods.
  - When to use: any "record" of heterogeneous typed fields (replacing a dict-as-struct pattern); prefer `NamedTuple` (immutable, editor-friendly) over `dataclass` (mutable, but reintroduces dict-like typo risk) unless mutability is specifically needed.
- **Fail-soft parsing (`try_parse_row` returning `Optional`)**: validate/parse each field, returning `None` on any failure instead of raising, so a caller can skip bad rows and log them rather than crashing.
  - When to use: ingesting real-world delimited data with inconsistent quality — decide per-dataset whether to drop, fix-at-source, or ignore bad rows.
- **Rescaling (standardization)**: transform each dimension to mean 0, standard deviation 1 (`rescale`), eliminating unit-dependence (e.g., inches vs. centimeters) before computing distances.
  - When to use: any distance-based technique (k-NN, clustering) on dimensions with incomparable units/scales.
- **PCA (Principal Component Analysis) from scratch**: de-mean the data, then use gradient descent to find the unit direction `w` maximizing `directional_variance` (the first principal component); repeat on the residual (after `remove_projection`) for further components; `transform` projects data onto the found components.
  - When to use: reducing dataset dimensionality by finding directions of maximal variance, denoising/consolidating correlated dimensions, or enabling techniques that struggle with high dimensionality — at the cost of interpretability.

## Key Concepts
- **Histogram via bucketize**: `bucketize(point, bucket_size) = bucket_size * floor(point/bucket_size)`, then `Counter` + `plt.bar` for a quick 1D distribution view.
- **Correlation matrix**: an `n x n` matrix whose `(i,j)` entry is `correlation(data[i], data[j])`, summarizing pairwise linear relationships across many dimensions.
- **NamedTuple vs. dataclass**: NamedTuples are immutable, memory-compact, type-annotatable tuples with named fields and methods; dataclasses are mutable regular classes with auto-generated boilerplate — mutability reopens the "typo creates a new silent field" risk dicts had.
- **de_mean**: recenters data so every dimension has mean 0 — required before PCA, or the technique would find the mean itself rather than the actual variation.
- **direction(w)**: rescales any nonzero vector to unit magnitude (`w_i / magnitude(w)`), used to represent a "pure direction" independent of scale.
- **directional_variance**: the variance of the data when projected onto a given unit direction; PCA's first component maximizes this via gradient ascent.
- **tqdm**: a progress-bar library wrapping any iterable (`tqdm.tqdm(range(100))`) or range (`tqdm.trange`), with `.set_description()` for live status updates during slow loops.

## Mental Models
- **Same summary stats, very different distributions**: `uniform` and `normal` datasets can share nearly identical mean and standard deviation while looking completely different in a histogram — always visualize, don't just summarize numerically.
- **Units are not neutral**: switching height from inches to centimeters can flip which point is "nearest" under Euclidean distance — any distance-based method needs rescaled/standardized dimensions when units aren't naturally comparable.
- **Dicts-as-records vs. typed records trade-off**: dicts are flexible but let a typo (`'cosing_price'`) silently create a wrong key instead of erroring, and resist meaningful type annotation; NamedTuple/dataclass trade some flexibility for safety and editor support.
- **Dimensionality reduction trades interpretability for tractability**: PCA components are linear combinations of original features, so they make some downstream techniques easier/better but make the result ("every 0.1 increase in the third principal component...") much harder to explain than a raw feature's coefficient.

## Anti-patterns
- **Using a plain dict for a fixed-schema record**: `stock_price['cosing_price'] = 103.06` runs without error and silently fails — use `NamedTuple`/`dataclass` so a typo is a clear attribute error instead.
- **Trusting summary statistics alone**: two datasets with nearly identical mean/stdev can have wildly different shapes (uniform vs. normal) — always histogram/plot before concluding anything.
- **Computing distances on mismatched-unit dimensions without rescaling**: changing height's unit from inches to cm changes which points are "nearest," an artifact of scale rather than real structure.
- **Letting one bad row crash an entire data-loading pipeline**: parse defensively (`try_parse_row` -> `Optional[StockPrice]`), decide explicitly whether to skip, fix-at-source, or ignore invalid rows based on how much of the data is affected.
- **Nesting tqdm progress bars or wrapping generators**: can cause screen-redraw glitches or hangs — the book flags this as a known flakiness source.
- **Running PCA on a dataset not de-meaned first**: without de-meaning, PCA finds the mean itself rather than the true directions of variation.

## Code Examples
```python
class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']
```
- **What it demonstrates**: a typed, immutable, method-bearing record replacing a dict-based "stock price" structure.

```python
def rescale(data: List[Vector]) -> List[Vector]:
    dim = len(data[0])
    means, stdevs = scale(data)
    rescaled = [v[:] for v in data]
    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]
    return rescaled
```
- **What it demonstrates**: standardizing every dimension to mean 0, standard deviation 1 to remove unit-dependence before distance-based analysis.

```python
def directional_variance(data: List[Vector], w: Vector) -> float:
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)

def first_principal_component(data, n=100, step_size=0.1) -> Vector:
    guess = [1.0 for _ in data[0]]
    for _ in range(n):
        dv = directional_variance(data, guess)
        gradient = directional_variance_gradient(data, guess)
        guess = gradient_step(guess, gradient, step_size)
    return direction(guess)
```
- **What it demonstrates**: finding PCA's first principal component via gradient ascent on directional variance (reusing Ch8's `gradient_step`).

## Reference Tables
| Dimensionality | Exploration technique |
|---|---|
| 1D | summary stats + histogram (`bucketize` + `Counter` + `plt.bar`) |
| 2D | scatterplot + `correlation` |
| Many-D | `correlation_matrix` + scatterplot matrix (`plt.subplots`) |

| Record representation | Mutable | Typo-safe | Type-annotatable |
|---|---|---|---|
| `dict` | yes | no | poorly (uniform value types only) |
| `namedtuple` | no | yes | no |
| `NamedTuple` (typed) | no | yes | yes |
| `@dataclass` | yes | no (regains dict-like risk) | yes |

## Worked Example
A `StockPrice` dataset (symbol, date, closing_price) is grouped by symbol, sorted by date, and converted to day-over-day `DailyChange` records via `pct_change = today/yesterday - 1`, using `zip(prices, prices[1:])` to pair consecutive days. The maximum one-day change is AAPL on 1997-08-06 (+33%), the minimum is AAPL on 2000-09-29 (-51.5%); grouping all changes by calendar month and averaging reveals October as historically the best month to hold tech stocks. Separately, a height/weight PCA example shows that the "wrong axes" dataset's true variation lies along direction `[0.924, 0.383]` after de-meaning — found via gradient ascent on `directional_variance`, then removed via `remove_projection` to reveal the orthogonal second component.

## Key Takeaways
1. Always explore data visually (histograms, scatterplots, scatterplot matrices) before modeling — identical summary statistics can hide very different distributions.
2. Prefer typed `NamedTuple` records over dicts for fixed-schema data: safer against typos, documents the schema, and supports methods.
3. Parse defensively: wrap row-parsing in a function returning `Optional[Record]`, decide explicitly how to handle invalid rows (skip, fix, ignore) based on their prevalence.
4. Rescale/standardize dimensions (mean 0, stdev 1) before any distance-based technique when units aren't naturally comparable.
5. PCA (de-mean, maximize directional variance via gradient ascent, repeat on residuals after removing each found component) reduces dimensionality but trades away interpretability.
6. Use `tqdm` to track long-running loops, accepting occasional flakiness as the cost of visibility.

## Connects To
- **Ch4**: PCA and rescaling build directly on `Vector`, `dot`, `magnitude`, `subtract`, `vector_mean`.
- **Ch5**: `standard_deviation` and `correlation` reused for exploration and rescaling.
- **Ch8**: `first_principal_component` is literally a gradient-descent application (`gradient_step`) to a new objective (directional variance).
- **Ch12 (k-Nearest Neighbors)** and **Ch20 (Clustering)**: both are distance-based techniques that depend on the rescaling discussed here.
- **pandas / scikit-learn**: external libraries that implement all of this chapter's data manipulation and PCA far more efficiently for production use.
