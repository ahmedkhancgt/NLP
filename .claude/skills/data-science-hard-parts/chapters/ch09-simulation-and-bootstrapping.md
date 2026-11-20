# Chapter 9: Simulation and Bootstrapping

## Core Idea
Monte Carlo simulation (generating data from a known data-generating process) and bootstrapping (resampling from your actual dataset) are complementary tools for understanding algorithm behavior and quantifying estimate precision without relying purely on theory.

## Frameworks Introduced
- **Monte Carlo (MC) Simulation Pipeline**: (1) fix parameters/seed/sample size N, (2) define the metric of interest (e.g., bias = E[θ̂ − θ]), (3) fix number of simulations M, simulate+estimate+save each run, then aggregate (mean, 95% CI from 2.5/97.5 percentiles across runs).
  - When to use: testing whether/how an algorithm's performance depends on properties of the data (signal-to-noise ratio, correlated features, outliers).
- **Inverse Transform Sampling**: given uniform draws `u ~ U(0,1)` and a known invertible CDF F, compute `x = F⁻¹(u)` to generate draws from any target distribution.
  - When to use: building custom pseudorandom generators when a package lacks the distribution you need.
- **Bootstrapping Pipeline**: (1) fix B (number of bootstrap samples), (2) for each b, resample N rows with replacement and recompute the estimate, (3) use the B estimates' spread to get variance/CI (`SD(θ̂) = sqrt(Σ(θ̂_b − mean(θ̂_b))² / (B−1))`).
  - When to use: quantifying precision of an estimate without distributional assumptions, or when the analytic variance is hard/costly to derive — e.g., TPR-by-decile confidence intervals, PDP confidence bands.
- **Partial Dependence Plot (PDP) construction**: (1) train the model, (2) compute sample means of all K features, (3) build a linear grid for feature j, (4) construct a "means-grid" matrix (grid values for j, means elsewhere), (5) predict on that matrix → PDP(x_j) = f(means-grid).
  - When to use: interpreting nonlinear models (GBR, RFR) analogously to linear regression's marginal-effect coefficients.

## Key Concepts
- **Data generating process (DGP)**: the full specification of relationships among inputs, noise, and outputs needed to simulate a dataset.
- **Seed**: initial value of a pseudorandom sequence; fix it for reproducibility.
- **Signal-to-noise ratio (SNR)**: amount of explained signal relative to unexplained noise; low SNR → unbiased but imprecise (wide CI) estimates, and can hide true incremental effects entirely.
- **Omitted variable bias**: excluding a feature correlated with an included feature biases that feature's coefficient; `Bias ∝ α2 × Cov(x1,x2)`, zero when features are uncorrelated, and both OLS and gradient boosting show the same directional bias under correlated omitted features — algorithms cannot fix a data/specification problem.
- **Latent variable model**: a binomial classification DGP where an unobserved continuous variable z determines the observed binary outcome y via its sign; true parameters are not identifiable, only parameters normalized by the error scale (α/σ) can be estimated.
- **Marginal effect (classification)**: `∂Prob(y=1)/∂x_k = f(x'α)·α_k`, nonlinear because it requires evaluating the PDF at the linear index, unlike linear regression's constant coefficient.
- **max_features / max_depth (tree ensembles)**: metaparameters that control whether a lower-variance feature gets a "fair chance" to be selected at splits; restricting max_features to fewer than all features can reduce bias toward dominant-variance features.

## Mental Models
- No algorithm works universally well — simulate to stress-test assumptions (SNR, correlation, outliers) before trusting an algorithm's output on real data.
- Algorithms cannot fix data problems: omitted variable bias persists under GBR exactly as under OLS when features are correlated.
- PDPs extend the "marginal effect" interpretability of linear regression to nonlinear models; useful as the Ch 13 interpretability on-ramp.

## Anti-patterns
- **Trusting unbiasedness without checking precision**: low SNR scenarios remain unbiased on average but can be too imprecise to detect real effects (e.g., a true incremental revenue effect invisible against noisier legacy-product revenue).
- **Interpreting raw logistic-regression coefficients as "true" DGP parameters**: they are only identifiable up to scale (normalized by σ); compare only to normalized true parameters.
- **Expecting a predictive algorithm to correct omitted-variable bias**: simulation shows GBR has the same directional bias as OLS under correlated omitted features.
- **Skipping bootstrapped CIs at fine granularity**: plotting TPR at very fine buckets (e.g., ventiles) without CIs can create false "non-monotonicity" alarms that are just sampling noise.

## Code Examples
```python
def logistic_cdf_inverse(y, mu, sigma):
    """
    Return the inverse of the CDF of a logistic random variable
    Inputs:
        y: float: number between 0 and 1
        mu: location parameter
        sigma: scale parameter
    Returns:
        x: the inverse of F(y;mu,sigma)
    """
    inverse_cdf = mu + sigma*np.log(y/(1-y))
    return inverse_cdf
```
- **What it demonstrates**: inverse transform sampling implemented for the logistic distribution — the core trick behind building a custom pseudorandom generator from a uniform draw.

```text
Bias(θ̂, θ) = E[θ̂ − θ]
Omitted variable bias sign: Sgn(Bias) = −Sgn(Cov(x1, x2))   (given true α2 = −5)
Bootstrap SD: SD(θ̂) = sqrt( Σ_b (θ̂_b − mean(θ̂_b))² / (B−1) )
```
- **What it demonstrates**: the core bias and bootstrap-variance formulas used throughout the chapter's MC experiments.

## Reference Tables
| Method | Data source | Primary use |
|---|---|---|
| Monte Carlo simulation | Synthetic, from specified DGP | Test algorithm sensitivity/bias under controlled conditions |
| Bootstrapping | Resampled from real dataset | Quantify estimate variance without distributional assumptions |
| Latent variable simulation | Synthetic, DGP with unobserved z | Simulate binary/classification outcomes |

## Worked Example
Linear model DGP with two features (x1 ~ N(0,3), x2 ~ N(0,10), noise ~ N(0,1)): under baseline SNR, OLS recovers true parameters with tight CIs (300 MC runs); inflating residual variance 1000× keeps OLS unbiased but massively widens CIs — illustrating a real incrementality project where a genuine effect (new product B's revenue) was undetectable purely due to low SNR relative to legacy product A's revenue variance. Omitted-variable-bias simulation: excluding x2 (true α2 = −5) from the regression produces bias that is zero at correlation 0, grows negative as Cov(x1,x2) rises, and equals the full −5 at correlation 1 — identical directional bias shown for GBR via PDP-slope comparison. TPR bootstrap: bucketing a churn model's scores into quintiles/deciles shows clean monotonicity, but 20-tiles (ventiles) show apparent non-monotonic dips at buckets 11/15/19 that bootstrapped 95% CIs reveal as mere sampling noise, not real model problems.

## Key Takeaways
1. No algorithm performs universally well — use simulation to probe sensitivity to SNR, correlation structure, and other DGP properties before trusting it in production.
2. Algorithms cannot fix fundamental data problems like omitted-variable bias; this holds for both OLS and modern ensembles (GBR).
3. Low signal-to-noise ratio can hide genuinely real incremental effects — check SNR before concluding "no effect."
4. PDPs extend interpretable marginal-effect thinking from linear regression to nonlinear tree ensembles.
5. Classification model parameters (e.g., logistic regression) are identifiable only up to a normalizing scale; compare to normalized true values.
6. Bootstrapping is the right tool when you want nonparametric variance estimates or when the analytic variance is hard/costly to derive.
7. Always add confidence intervals to fine-grained metric breakdowns (e.g., ventile TPRs) to avoid chasing noise as if it were signal.

## Connects To
- **Ch 6**: TPR-by-decile lift analysis is directly extended here with bootstrapped confidence intervals.
- **Ch 10**: Omitted variable bias and SNR concepts are foundational to the "going back to basics" linear regression discussion.
- **Ch 13**: PDPs introduced here as a simulation tool reappear as a core interpretability/storytelling technique for production ML models.
- **Ch 15**: SNR limitations directly explain why some incrementality estimates come back statistically insignificant despite a true effect.
