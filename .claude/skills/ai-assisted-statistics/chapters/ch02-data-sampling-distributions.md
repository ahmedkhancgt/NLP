# Chapter 2: Data and Sampling Distributions

## Core Idea
Random sampling and resampling (especially the bootstrap) remain essential even in the age of big data, because data quality and representativeness matter more than raw volume, and the bootstrap offers a distribution-free way to quantify the uncertainty of almost any sample statistic.

## Frameworks Introduced
- **Random sampling / stratified sampling**: drawing a representative subset from a population, with or without replacement; stratified sampling draws separately from homogeneous subgroups (strata) to guarantee adequate representation.
  - When to use: whenever building an estimate or model from a subset of a larger population, especially when subgroups of interest are rare.
  - How: define the accessible population precisely, then sample randomly within it (or within each stratum).
- **The Bootstrap**: resampling with replacement from the observed sample (not the population) to estimate the sampling distribution of any statistic.
  - When to use: need a standard error or confidence interval for a statistic with no known/simple formula (e.g., median, correlation, model coefficients).
  - How: draw a sample of size n with replacement, compute the statistic, repeat R times; use the spread of the R results as the sampling distribution.
- **Bootstrap confidence interval**: trim [(100−x)/2]% from each end of the R bootstrap results to get an x% confidence interval.
  - When to use: need an interval estimate instead of a point estimate, without relying on normality assumptions.
  - How: run the bootstrap algorithm, then take the appropriate percentiles of the resample distribution.
- **Central Limit Theorem (CLT)**: sample means (and other statistics) tend toward a normal distribution as sample size grows, even if the underlying population is not normal.
  - When to use: justifies classical formula-based confidence intervals/hypothesis tests (t-distribution, normal approximation); less central to data science practice since the bootstrap sidesteps it.
- **Distribution fitting for event processes**: Binomial (successes in n trials), Poisson (events per unit time/space with rate λ), Exponential (time between events, rate λ), Weibull (time-to-failure with changing rate, shape β).
  - When to use: match the process type — fixed-probability binary trials → binomial; constant-rate counts → Poisson; time between constant-rate events → exponential; time-to-failure with rate that changes over time → Weibull.

## Key Concepts
- **Population vs. sample**: population = the full (often theoretical) dataset; sample = the observed subset; x̄ (sample mean) vs. μ (population mean).
- **Sample bias**: a sample that differs from the population in a systematic, nonrandom, and persistent way (classic case: 1936 Literary Digest poll oversampling high-socioeconomic-status subscribers).
- **Selection bias / vast search effect / data snooping**: bias from the process of choosing what data or model to look at, including repeatedly testing hypotheses on the same dataset until something "significant" appears.
- **Regression to the mean**: extreme observations (often a mix of skill + luck) tend to be followed by more average ones once luck normalizes (e.g., "rookie of the year, sophomore slump").
- **Standard error**: the standard deviation of a sample statistic's sampling distribution (SE = s/√n) — distinct from standard deviation of raw data.
- **z-score / standardization**: (x − mean) / standard deviation; puts data on the standard normal scale without making it normal.
- **QQ-plot**: visualizes how closely a sample's distribution matches a theoretical distribution (e.g., normal) by plotting ordered z-scores against theoretical quantiles.
- **Long-tailed distribution**: data that is "normal in the middle" but has far more extreme values in the tails than a normal distribution predicts (e.g., stock returns) — related to Taleb's "black swan" concept.

## Mental Models
- Think of the bootstrap as creating a vast hypothetical population by replicating your sample infinitely (via sampling with replacement) — it reveals how variable your statistic would be across new samples without needing new data or distributional assumptions.
- Use the "quantity vs. quality" frame: more data does not fix a biased sampling process (Literary Digest's 10 million respondents lost to Gallup's 2,000 well-chosen ones); spend saved effort on data quality and exploration instead.
- Distinguish standard deviation (variability of individual data points) from standard error (variability of a sample statistic across repeated samples) — a frequent source of confusion.
- When evaluating a surprising pattern found by "mining" a large dataset, ask "would this still look interesting if I used a holdout set or shuffled the target (permutation/target shuffling)?" before trusting it.

## Anti-patterns
- **Mistaking bigger data for better data**: massive data does not eliminate the need for random sampling or fix systematic bias; it only helps for genuinely big+sparse problems (e.g., Google-scale rare-query matching).
- **Vast search effect**: running many models/questions on the same dataset and reporting only the "interesting" hit — looks like discovery but is often statistical noise (cf. the stadium coin-flip thought experiment).
- **Treating regression to the mean as a real effect to be acted on**: e.g., assuming a "sophomore slump" requires a special intervention rather than recognizing it as the statistical fading of a lucky outlier.
- **Assuming most raw data is normally distributed**: the normal distribution is mainly useful for sampling distributions (via CLT) and errors, not for most real raw variables (income, stock returns, etc. are skewed/long-tailed).
- **Misinterpreting a confidence interval as "P(true value in interval)"**: the correct frequentist interpretation is about the long-run behavior of the interval-construction procedure, not a probability statement about one fixed interval.

## Code Examples
```r
library(boot)
stat_fun <- function(x, idx) median(x[idx])
boot_obj <- boot(loans_income, R = 1000, statistic = stat_fun)
boot_obj
# original bias std. error
# 62000 -70.5595  209.1515
```
- **What it demonstrates**: bootstrapping the median of loan applicant incomes to get bias and standard error estimates without a closed-form formula.

```python
res = stats.bootstrap([loans_income], np.median, n_resamples=1000, method="basic")
print(f"std. error: {res.standard_error}")
print(f"confidence interval: {res.confidence_interval}")
```
- **What it demonstrates**: the scipy equivalent bootstrap workflow in Python, producing a confidence interval directly.

```r
dbinom(x = 2, size = 5, prob = 0.1)   # exact probability of x successes
pbinom(q = 2, size = 5, prob = 0.1)   # cumulative probability of <= x successes
```
- **What it demonstrates**: computing binomial point and cumulative probabilities (Python equivalents: `stats.binom.pmf`, `stats.binom.cdf`).

```r
norm_samp <- rnorm(100)
ggplot(tibble(y = norm_samp), aes(sample = y)) +
  geom_qq() + geom_qq_line()
```
- **What it demonstrates**: building a QQ-plot to visually test normality (Python: `scipy.stats.probplot`).

## Reference Tables
| Distribution | Models | Key parameter(s) |
|---|---|---|
| Binomial | # successes in n trials | n, p |
| Poisson | # events per unit time/space | λ (rate) |
| Exponential | time/distance between events | λ (rate) |
| Weibull | time-to-failure, changing rate | β (shape), η (scale/characteristic life) |
| Normal / t-distribution | sampling distributions of statistics (CLT) | mean, SD (or df for t) |
| Chi-square | departure of category counts from a null/independence model | degrees of freedom |
| F-distribution | ratio of between-group to within-group variability (ANOVA) | df for numerator/denominator |

## Worked Example
LendingClub loan applicant annual income (`loans_income.csv`) is used throughout: a histogram of 1,000 raw income values is skewed and spread out; a histogram of 1,000 means of samples of 5 is more compact and bell-shaped; a histogram of 1,000 means of samples of 20 is even more so — a direct visual demonstration of the Central Limit Theorem. The same dataset is then bootstrapped: a sample of 20 loan incomes has a mean of $55,734; 1,000 bootstrap resamples of this 20-record sample produce a bootstrap mean of $55,836 and support a 90% confidence interval, illustrating that the bootstrap distribution (not the original 20-value mean) is the basis for the interval. The median bootstrap on the full `loans_income` dataset gives original median $62,000, bias ≈ −$70.56, standard error ≈ $209.15.

## Key Takeaways
1. Random/stratified sampling is still necessary with big data — volume cannot fix a biased collection process.
2. Use the bootstrap as a general-purpose, distribution-free tool for standard errors and confidence intervals, especially when no formula exists for your statistic.
3. Guard against the vast search effect with holdout sets or permutation (target-shuffling) tests before trusting a pattern found by extensive data mining.
4. Standard error shrinks with √n (the "square root of n rule") — quadrupling sample size halves standard error.
5. Most raw data is not normal; check with a QQ-plot rather than assuming normality, and remember long-tailed data underestimates the frequency of extreme events if treated as normal.
6. Match event/process data to the right generating distribution (binomial/Poisson/exponential/Weibull) based on what varies (counts vs. time, constant vs. changing rate).

## Connects To
- **Ch 3 (Statistical Experiments and Significance Testing)**: the bootstrap, permutation tests, and the chi-square/F distributions introduced here are used directly for hypothesis testing.
- **Ch 6 (Statistical Machine Learning)**: bagging/random forests apply the bootstrap idea (bootstrap aggregating) to model training.
- **External concept**: Nassim Taleb's "black swan" theory connects long-tailed distributions to risk management practice outside statistics.
