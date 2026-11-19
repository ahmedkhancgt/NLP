# Chapter 1: Exploratory Data Analysis

## Core Idea
The first and most important step in any data-driven project is to look at the data — summarize and visualize it (Tukey's EDA tradition) before modeling, and modern AI assistants (LLMs) can accelerate both building data dictionaries and interpreting visualizations, but their output must always be verified by a human expert.

## Frameworks Introduced
- **Exploratory Data Analysis (EDA)**: Tukey's discipline of using plots (boxplot, scatterplot) and summary statistics (mean, median, quantiles) to understand data before formal inference.
  - When to use: at the start of every data science project.
  - How: compute location/variability estimates, then visualize distributions (univariate), relationships (bivariate), and conditioned views (multivariate via faceting).
- **Robust estimation**: using estimators insensitive to outliers (median, trimmed mean, MAD, IQR) instead of (or alongside) mean/variance/standard deviation.
  - When to use: data has outliers, skew, or small sample sizes where a few extreme points distort classical estimates.
  - How: trim p% from each tail for trimmed mean; take median of absolute deviations from the median for MAD.
- **AI-assisted data dictionary generation**: prompting an LLM (via langchain `ChatOpenAI` in Python, or `ellmer`/`chat_openai` in R) with an existing data dictionary plus a CSV sample to produce an improved, structured (YAML) dictionary.
  - When to use: documenting a new/undocumented dataset, or improving a sparse one.
  - How: construct a `PromptTemplate`, chain it to the LLM object (`|` operator), invoke with `{data_dictionary, data_sample}` context; iterate and have a human verify claims.
- **AI-assisted visualization interpretation**: feeding a chart image to a multimodal LLM (e.g., ChatGPT) and asking for a one-paragraph interpretation, then follow-up prompts to probe specific features (e.g., "how about the two clusters above the main cluster?").
  - When to use: complex multi-variable plots where patterns aren't immediately obvious.
  - How: paste image + prompt; iteratively refine with clarifying questions; finish with a "write a concise summary" prompt.

## Key Concepts
- **Rectangular data / data frame**: the two-dimensional matrix of rows (records) and columns (features) that is the standard input to statistical/ML models.
- **Estimate vs. metric**: statisticians "estimate" (acknowledging uncertainty about a population); data scientists "measure" a metric from data at hand.
- **Outlier**: a value very different from the rest of the data; not necessarily erroneous, but worth investigating.
- **Degrees of freedom (n − 1)**: dividing by n−1 instead of n in the variance formula removes bias in estimating population variance from a sample.
- **Percentile / quantile**: value below which P percent of data falls; IQR = 75th − 25th percentile.
- **Correlation coefficient (Pearson's r)**: standardized (−1 to +1) measure of linear association between two numeric variables; sensitive to outliers and only captures linear relationships.
- **Contingency table**: cross-tabulation of counts for two categorical variables.
- **Data dictionary / data catalog**: document(s) describing a dataset's variables, source, and meaning (dictionary = single dataset; catalog = organization-wide inventory with governance info).

## Mental Models
- Think of the mean, median, and trimmed mean as a spectrum from "uses all data, sensitive to outliers" (mean) to "ignores extremes entirely" (median), with the trimmed mean as a tunable compromise.
- Use a boxplot when you want a fast five-number summary (quartiles + whiskers + outlier points) across many groups; use a violin plot when you need to see the shape/nuances of the distribution (e.g., bimodality) that a boxplot hides.
- Treat histograms (numeric x-axis, bars touching, order matters) and bar charts (categorical x-axis, bars separated, order doesn't matter) as visually similar but conceptually distinct — never confuse them.
- When a single scatterplot becomes an unreadable "cloud" at scale, think of hexagonal binning/contour plots as 2D histograms — the same move from raw points to density that a 1D histogram makes from raw values.

## Anti-patterns
- **Reporting only the mean**: hides skew and outlier influence (e.g., Bill Gates living in Medina distorts a neighborhood's mean income but not its median).
- **Using correlation coefficient on nonlinear relationships**: r can be near zero even when there's a strong nonlinear association (e.g., the tax-rate-vs-revenue Laffer-curve relationship).
- **Trusting LLM-generated data dictionaries or chart interpretations uncritically**: LLM output is stochastic and can state unverifiable claims (e.g., "these are the standard eBay options") that must be checked by a human expert.
- **Using pie charts for serious analysis**: statisticians and visualization experts generally consider them less informative than bar charts.
- **Ignoring empty bins in histograms/frequency tables**: dropping them loses real information about distribution gaps.

## Code Examples
```r
# Weighted mean and weighted median accounting for population
library(matrixStats)
weighted.mean(state[["Murder.Rate"]], w = state[["Population"]])
weightedMedian(state[["Murder.Rate"]], w = state[["Population"]])
```
- **What it demonstrates**: computing population-weighted summary stats instead of naive per-state averages.

```python
# LLM-assisted data dictionary improvement (langchain)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

chain = (
  PromptTemplate.from_template(PROMPT) |
  ChatOpenAI(model="gpt-5")
)
result = chain.invoke({"data_dictionary": data_dictionary, "data_sample": data_sample})
print(result.content)
```
- **What it demonstrates**: the langchain pattern of piping a prompt template into a chat model to generate structured output (YAML data dictionary).

```r
# Boxplot with ggplot2
ggplot(state, aes(y = Population / 1000000)) +
  geom_boxplot(staplewidth = 0.25) +
  labs(y = "Population (millions)")
```
- **What it demonstrates**: five-number-summary visualization; whiskers extend to 1.5×IQR by default.

```r
# Faceted hexbin plot conditioning on a third variable (zip code)
ggplot(kc_subset, aes(x = SqFtTotLiving, y = TaxAssessedValue)) +
  geom_hex(color = "white") +
  facet_wrap("ZipCode")
```
- **What it demonstrates**: multivariate EDA via conditioning/faceting to reveal that an apparent cluster is actually a location effect.

## Reference Tables
| Estimator | Robust to outliers? | Formula basis |
|---|---|---|
| Mean | No | Sum / n |
| Trimmed mean | Partially | Mean after dropping p extreme values each end |
| Median | Yes | Middle value of sorted data |
| Weighted mean/median | Depends on weights | Mean/median with per-value weights |
| Variance / SD | No | Squared deviations from mean |
| Mean absolute deviation | Partially | Mean of \|deviation\| from mean |
| MAD (median absolute deviation) | Yes | Median of \|deviation\| from median |
| IQR | Yes | 75th percentile − 25th percentile |

| Plot type | Variable types | Purpose |
|---|---|---|
| Boxplot / violin plot | 1 numeric (± 1 categorical) | distribution shape & outliers |
| Histogram / density plot | 1 numeric | distribution shape |
| Bar chart | 1 categorical | frequency per category |
| Scatterplot | 2 numeric (small n) | bivariate relationship |
| Hexbin / contour plot | 2 numeric (large n) | bivariate density |
| Contingency table | 2+ categorical | joint counts |
| Facet plot | 2 numeric + 1+ conditioning var | multivariate, stratified relationships |

## Worked Example
King County, WA housing tax data (`kc_tax`): filtering to `TaxAssessedValue < 750000`, `100 < SqFtTotLiving < 3500` yields 432,693 records. A hexbin plot of square footage vs. tax-assessed value shows a clear positive relationship plus two faint secondary bands of homes valued higher than the main band at the same size. Faceting by zip code (98105, 98108, 98126, 98188) resolves this: 98105 and 98126 show systematically higher assessed values for the same square footage than 98108 and 98188 — the "extra bands" were a location effect hidden in the pooled plot. The authors reproduce this insight via a multimodal LLM (ChatGPT) dialogue: first asking for a one-paragraph interpretation of the pooled hexbin plot, then explicitly asking about the secondary clusters, then interpreting the faceted version, and finally asking for a concise summary — demonstrating an iterative "ask → probe → summarize" workflow for AI-assisted visual analysis.

## Key Takeaways
1. Always compute both a classical estimate (mean, SD) and a robust one (median, MAD/IQR) — large divergence signals skew or outliers worth investigating.
2. Pick the chart type by variable type and scale: scatterplots degrade at scale; switch to hexbin/contour/density for large n.
3. A single aggregate view of two numeric variables can hide a confounding categorical structure — facet/condition on candidate variables before concluding a relationship is "noisy."
4. LLMs are useful accelerators for writing data dictionaries and drafting chart interpretations, but every factual claim they produce needs independent verification.
5. Correlation coefficients only capture linear association — always look at the scatterplot, not just the number.

## Connects To
- **Ch 4 (Regression and Prediction)**: correlation and scatterplots here are the visual precursor to regression modeling.
- **Ch 7 (Unsupervised Learning)**: contingency tables and clustering-by-location previews later unsupervised segmentation methods.
- **External concept**: Tukey's 1977 *Exploratory Data Analysis* and the "grammar of graphics" (ggplot2) underpin the visualization approach throughout the book.
