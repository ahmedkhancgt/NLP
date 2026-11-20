# Chapter 13: Storytelling in Machine Learning

## Core Idea
Storytelling in ML has two personas: a scientist (ex ante/interim — hypotheses drive feature engineering and model development) and a salesperson (ex post — interpreting and selling the finished model). A good ML story starts before training, not after.

## Frameworks Introduced
- **Ex Ante Storytelling Flow**: Problem definition → Create hypotheses → Feature engineering → Train model, with feedback loops throughout.
  - When to use: at the start of, and iteratively during, every ML project.
  - How: always ask "do I even need ML for this?", and answer "how is this used / what levers exist / how does it improve decisions?" before proceeding (ties to Ch 1).
- **Lever-Behavior-Metrics Flow**: most prediction problems fall into Metrics from Human Behavior (clicks, churn, purchases), Metrics from Systems Behavior (page load time, cooling optimization), or Downstream/Aggregate Metrics (revenue, FP&A).
  - When to use: classify your prediction target first — it determines where to look for predictive hypotheses.
- **Feature Engineering Stages**: (1) Create ideal features (what you'd measure if omniscient), (2) Approximate with realistic/proxy features (correlated but imperfect substitutes), (3) Transform features (scaling, binarizing, one-hot encoding, imputation, interactions) — algorithm-dependent (e.g., trees handle outliers/interactions natively).
  - When to use: as a disciplined 3-stage pipeline from hypothesis to usable feature, always starting from the ideal/unmeasurable construct.
- **Interpretability Methods for Ex Post Storytelling**: Standardized coefficients (linear benchmark), Feature Importance (impurity-based and permutation), Heatmaps (bivariate decile correlation), Partial Dependence Plots (PDP, two computation methods), Individual Conditional Expectation (ICE, local version of PDP), Accumulated Local Effects (ALE, corrects PDP's correlated-feature bias).
  - When to use: after confirming the model has adequate predictive performance and no leakage — never open the black box on a bad model.

## Key Concepts
- **Local vs. global interpretability**: local = explaining one specific prediction (e.g., why this customer was flagged); global = general pattern of how features affect the outcome across the whole model.
- **Interpretability-performance trade-off**: linear/tree-based simple models are interpretable but lower-performing; deep nets, ensembles, SVMs are higher-performing "black box" learners requiring explicit interpretability tooling.
- **Standardized-feature importance**: compare coefficients after standardizing all features (mean 0, unit variance) to make magnitudes comparable across differently-scaled features — raw coefficient magnitude comparison is an apples-to-oranges mistake.
- **Impurity-based feature importance**: (tree-based only) a feature's importance = aggregate improvement in node-splitting criterion attributable to that feature across all splits/trees.
- **Permutation importance**: (any algorithm) shuffle a feature's values and measure the resulting performance drop; larger drop = more important feature.
- **PDP Method 1 (mean-then-predict)**: fix all other features at their sample means, build a grid for the feature of interest, predict once per grid point — fast (one evaluation per grid value) but statistically incorrect since `E[f(X)] ≠ f(E[X])` for nonlinear f.
- **PDP Method 2 (predict-then-average)**: for each grid value g, predict for every row with feature j set to g, then average across all N rows — the statistically correct, scikit-learn-standard method; N evaluations per grid point.
- **ICE (Individual Conditional Expectation)**: visualizes PDP Method 2's per-unit predictions without averaging — a local-interpretability companion to the PDP's global view.
- **ALE (Accumulated Local Effects)**: fixes PDP's correlated-feature bias by (1) restricting each grid point's slope calculation to a local neighborhood of real observations, (2) computing local slopes, (3) accumulating them into a global effect curve — more computationally efficient and less biased under correlated features than PDPs.

## Mental Models
- "People do what they want and can do" — when predicting human behavior, think about tastes (preferences) and resource constraints (ability) as the two universal hypothesis generators.
- For downstream/aggregate metrics, if direct hypotheses feel too diffuse, lean on autoregressive/spatial structure (time and space correlation) as a fallback source of predictive signal.
- Start hypothesis-building with the average customer/unit, not edge cases — first-order effects dominate predictive performance; corner cases matter less statistically even if interesting.
- A feature decomposition trick (from Ch 2) sharpens hypotheses: e.g., `Sales_g = TAM_g × Prob(sale | TAM_g)` splits one hard story into two more tractable ones (market sizing + conversion probability).

## Anti-patterns
- **Opening the black box before confirming good performance and no leakage**: interpretability work is wasted/misleading on a bad or leaky model.
- **Comparing raw (unstandardized) regression coefficients to rank feature "importance"**: an apples-to-oranges error since features are in different units.
- **Trusting heatmaps for multivariate relationships**: heatmaps only show bivariate correlation with the outcome; a feature with a true positive effect can appear non-monotonic in a heatmap if it's correlated with another feature that dominates the ranking (e.g., a higher-variance correlated feature).
- **Using PDPs uncritically with correlated features**: PDPs impose unrealistic combinations of feature values (e.g., low x1 paired with high x2 despite their real-world positive correlation), biasing the estimated relationship.
- **Chasing ML "because it's sexy"**: always have a value-creation business case (Ch 1, Ch 5) before committing to deploy/monitor/maintain costs.
- **Skipping stakeholder involvement in problem definition**: late stakeholder buy-in is a leading cause of ML project failure.

## Code Examples
```text
PDP Method 1 (mean-then-predict, 1 evaluation per grid point):
  PDP(x_j=g) = f(mean(x_1),...,g,...,mean(x_K))

PDP Method 2 (predict-then-average, N evaluations per grid point — scikit-learn standard):
  PDP(x_j=g) = (1/N) Σ_i f(x_1,i,...,x_{j-1,i}, g, x_{j+1,i},...,x_K,i)
```
- **What it demonstrates**: the two competing PDP computation strategies — the first fast but statistically biased for nonlinear models, the second correct but N times more expensive.

```text
Standardization trick for comparable "importance":
  revenue = β0 + β1·(x1 − mean(x1))/sd(x1) + β2·(x2 − mean(x2))/sd(x2) + ε
  "x is more important than z if |β_standardized,x| > |β_standardized,z|"
```
- **What it demonstrates**: converting differently-scaled features into comparable standard-deviation units before ranking importance.

## Reference Tables
| Method | Works on | Local or Global | Handles correlated features well? |
|---|---|---|---|
| Standardized coefficients | Linear/logistic models | Global (+ local via sign/magnitude) | No |
| Impurity-based importance | Tree-based only | Global | Partially |
| Permutation importance | Any algorithm | Global | No (can split importance across correlated features) |
| Heatmap | Any algorithm | Global (bivariate only) | No |
| PDP | Any algorithm | Global | No — biased |
| ICE | Any algorithm | Local | No |
| ALE | Any algorithm | Global | Yes — designed to fix this |

## Worked Example
Sales-by-geography model: decompose `Sales_g = TAM_g × Prob(sale in g)`. For TAM, hypothesize data-scientist density is driven by company size (big enough to justify the DS hire), industry mix (capital- vs. labor-intensive), and population/age distribution. For sale probability, hypothesize demand (preferences proxied by historical sales or search-trend data, household income) and supply (company/competitor presence by location). Simulated nonlinear-model PDP comparison: with a Gamma-distributed x1 (outlier-prone) and Normal x2, both PDP methods roughly recover the true curve for typical values but diverge at large x1 due to Method 1's sensitivity to the influence of outliers on the mean; introducing correlation (ρ=0.9) between x1 and x2 clearly biases both PDP methods, while ALE with bootstrapped 90% CIs correctly recovers x2's true effect and substantially reduces (though doesn't eliminate) bias in x1's estimated effect.

## Key Takeaways
1. Treat ML storytelling as holistic: a scientist persona (ex ante/interim hypothesis generation) feeding into a salesperson persona (ex post interpretation and delivery).
2. Classify your prediction target as human-behavior, systems-behavior, or downstream/aggregate — each has distinct hypothesis-generation strategies.
3. Feature engineering should move explicitly: ideal feature → proxy feature → transformed feature, never skipping the "what would I measure if I could" step.
4. Never open the black box on a model with inadequate performance or unaddressed leakage — confirm those first.
5. Standardize features before comparing coefficient magnitudes for "importance"; raw coefficients in different units are not comparable.
6. PDPs are biased under correlated features; prefer ALE when features are meaningfully correlated.
7. Match interpretability method to need: local (ICE, individual coefficient signs) vs. global (PDP, ALE, feature importance, heatmaps).
8. Regulatory/ethical contexts (e.g., US Equal Opportunity Act, EU GDPR) may legally require explainability, not just business preference.

## Connects To
- **Ch 1 & Ch 7**: Reuses the "so what / levers" framing and the Chapter 7 storytelling principles, specialized for ML.
- **Ch 2**: The TAM × conversion-probability decomposition trick is directly reused from metrics decomposition.
- **Ch 9**: PDP computation and bootstrapped confidence intervals build directly on the simulation/bootstrapping chapter.
- **Ch 11**: Interpretability is called out as a key leakage-detection tool (unexpectedly dominant or nonsensical top features).
- **Ch 4**: The quadrant-membership classification use case from 2×2 designs is the natural application for these interpretability tools.
