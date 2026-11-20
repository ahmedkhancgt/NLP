# Chapter 15: Incrementality: The Holy Grail of Data Science?

## Core Idea
Incrementality is causal inference applied to business decisions: estimating whether a lever truly improved an outcome, net of confounders and selection bias, is the "holy grail" because it directly identifies which decisions create real value.

## Frameworks Introduced
- **Causal Effect via Counterfactuals**: the causal effect of an action is the difference between the observed outcome and the unobserved counterfactual outcome (what would have happened under the other action), everything else held fixed.
  - When to use: anytime you want to claim a lever (price discount, new feature, new product) caused a metric change, not merely correlates with it.
- **DAGs (Directed Acyclic Graphs)**: nodes = variables, directed edges = causal relationships, acyclic = no feedback loops. Used to formally reason about which variables to control for (Judea Pearl's framework).
  - **Confounder**: common cause `c` of both treatment x and outcome y (x ← c → y); must condition on it (close the back door) to avoid spurious correlation.
  - **Collider**: common effect `c` of both x and y (x → c ← y); must NOT condition on it — doing so opens a back door and creates spurious correlation.
  - When to use: before any causal estimation, draw the DAG for your outcome to decide what to control for (back-door criterion) — control for confounders, never for colliders, and watch for variables that are both.
- **Potential Outcomes / Missing Data Framework**: each unit has `Y1` (outcome if treated) and `Y0` (outcome if not); only one is ever observed — causal inference is fundamentally a missing-data problem.
  - **Decomposition**: `E[Y|D=1] − E[Y|D=0] = E[Y1−Y0|D=1] + {E[Y0|D=1] − E[Y0|D=0]}` = ATT (true causal effect on the treated) + Selection Bias.
  - When to use: as the formal basis for understanding why naive before/after or treated/untreated comparisons can mislead.
- **Unconfoundedness Assumption** (a.k.a. ignorability, conditional exchangeability, selection on observables, conditional independence): potential outcomes are independent of treatment assignment, conditional on observed covariates X: `(Y0,Y1) ⊥ D | X`.
  - When to use: the core identifying assumption behind every non-experimental causal method (matching, regression adjustment, DML); untestable from data alone — relies on your DAG being right and all relevant confounders being observed.
- **Randomization (A/B Testing / RCT)**: assign treatment via a pseudorandom draw (`D_i=1 if u_i≥p, else 0`) — guarantees unconfoundedness by construction since assignment is independent of potential outcomes.
  - When to use: whenever feasible — it's the gold-standard way to break selection bias. Must satisfy SUTVA (Stable Unit Treatment Value Assumption): (i) same treatment for everyone who gets it, (ii) no interference between units' potential outcomes. When (ii) is violated (e.g., marketplace supply/demand externalities), use block/cluster randomization instead of unit-level randomization.
- **Matching / Propensity Score Matching (PSM)**: for each treated unit, find the m most similar untreated units (by feature distance, or by the Propensity Score Theorem using `p(X)=Pr(D=1|X)` as a sufficient single-dimensional summary of X), compute the control-group average outcome, then the individual and average treatment effects.
  - When to use: when randomization isn't feasible (too costly, e.g., installing cell towers) but you believe unconfoundedness holds given observed covariates X.
- **Double Machine Learning (DML)**: partially linear causal model `Y = θD + g(X) + u; D = h(X) + v`, where g, h can be nonlinear ML functions. Uses orthogonalization (partial out X from both Y and D via flexible learners) + sample splitting (train on one half, estimate on the other, to avoid overfitting bias) to estimate θ, the causal effect.
  - When to use: when confounders have complex nonlinear relationships to the treatment/outcome and you want ML's flexibility without sacrificing valid causal inference.

## Key Concepts
- **Treatment**: the lever/action under study (borrowed from clinical-trial terminology: treatment vs. control/placebo group).
- **ATT (Average Treatment Effect on the Treated)**: the average causal effect among those who actually received treatment; distinct from ATE (average treatment effect over the whole population) and ATU (on the untreated).
- **Positive selection**: treatment given preferentially to units more likely to succeed anyway → causal effect overestimated.
- **Negative selection**: treatment given preferentially to units less likely to succeed anyway (e.g., incentivizing at-risk customers) → causal effect underestimated.
- **Proxy confounders**: using a correlated-but-imperfect stand-in for an unobserved confounder reduces but does not eliminate bias — bias severity scales with the proxy's correlation to the true confounder.
- **Cannibalization**: a new product/feature/location reduces sales of an existing one (iPhone → iPod; Netflix streaming → DVD rental; new Starbucks store → nearby store sales) — a key incrementality question when launching anything new.
- **SUTVA violations in marketplaces**: treating the demand side (e.g., a price discount) can reduce available supply for the control group, contaminating its potential outcomes — a classic reason to use cluster/block randomization in two-sided markets (Uber, Lyft, Airbnb).
- **Uplift modeling**: segments customers by predicted purchase probability into high (A, likely to buy organically — don't waste incentive), low (C, unlikely regardless — incentive too costly to move them), and middle (B, the best incremental-targeting candidates).
- **Pretreatment variables**: variables measured before treatment assignment, used to check for selection bias (compare means across treatment/control) and as controls for matching.

## Mental Models
- "Correlation is not causation" (Ch 10) returns here as the central practical risk: observed treated-vs-control differences conflate true causal effect with selection bias.
- Causal reasoning is valuable even for pure prediction: feature engineering implicitly encodes causal hypotheses (Ch 13), and omitted/bad controls bias predictive models too (Ch 9, 10, 11).
- A DAG is a stated assumption, not a fact derivable from data — document it and get stakeholder input, since business context often reveals the true selection mechanism.
- Randomization "closes all back doors" automatically; everything else (matching, DML, proxies) is a second-best attempt to approximate that closure using observed data.

## Anti-patterns
- **Comparing treated vs. untreated means without checking for confounders/selection**: this is the single most common incrementality mistake — always ask "why did this unit get treated, and is that reason related to the outcome?"
- **Controlling for a collider**: including a common-effect variable as a "control" introduces bias where none existed before.
- **Trusting a proxy confounder to fully fix bias**: it only partially reduces bias, proportional to its correlation with the true confounder.
- **Using unit-level randomization in a marketplace with supply/demand interference**: violates SUTVA and biases estimates; use block/cluster randomization instead.
- **Assuming matching/PSM "solves" causality**: it only works if unconfoundedness holds — an untestable assumption that depends entirely on your DAG being complete and correct.
- **Targeting the highest-probability-score segment (group A) in uplift campaigns**: these customers convert organically; incentive spend there is largely wasted — target the mid-probability segment (B) instead.

## Code Examples
```python
def randomize_sample(n_total, frac_treated, seed):
    "Function to find a randomized sample"
    np.random.seed(seed)
    unif_draw = np.random.rand(n_total)
    bool_treat = unif_draw >= frac_treated
    return bool_treat
```
- **What it demonstrates**: the canonical pseudorandom treatment-assignment mechanism that guarantees unconfoundedness by construction in an A/B test.

```text
Observed difference in means = ATT (causal effect) + Selection Bias
E[Y|D=1] − E[Y|D=0] = E[Y1−Y0|D=1] + {E[Y0|D=1] − E[Y0|D=0]}
Propensity score:  p(X) = Pr(D=1 | X)
Euclidean matching distance:  d_ij = sqrt(Σ_k (x_ik − x_jk)^2)
```
- **What it demonstrates**: the formal decomposition explaining why naive comparisons are biased, and the two core matching-distance concepts (raw feature distance vs. the propensity-score summary).

```text
DML partially linear model:
  Y = θ·D + g(X) + u
  D = h(X) + v
Algorithm: (1) split sample in half, (2) train g(),h() on one half,
           (3) compute residuals ỹ=Y−ĝ(X), D̃=D−ĥ(X) on the other half,
           (4) θ̂ = [Σ D̃·D]⁻¹ Σ D̃·ỹ  (orthogonalized estimator), (5) average over both splits.
```
- **What it demonstrates**: how ML flexibility (nonlinear g, h) combines with FWL-style orthogonalization and sample splitting to yield a valid causal estimate.

## Reference Tables
*Table 15-1: Potential outcomes and missing values (illustrative)*

| Unit | Y | Y0 | Y1 | D |
|---|---|---|---|---|
| 1 | 6.28 | 6.28 | NaN | 0 |
| 2 | 8.05 | 8.05 | NaN | 0 |
| 18 | 8.70 | NaN | 8.70 | 1 |
| 7 | 8.90 | NaN | 8.90 | 1 |

| Concept | Confounder | Collider |
|---|---|---|
| Structure | c → x, c → y | x → c, y → c |
| Effect of ignoring c | Spurious correlation appears | No bias (correctly insignificant) |
| Effect of controlling for c | Removes spurious correlation (correct) | Creates spurious correlation (incorrect!) |

## Worked Example
GitHub code-repo incrementality: does offering a code repo (D=1) vs. not (D=0) increase book-sale probability (Y)? Simulated MC results show: without controlling for a true confounder, a spurious statistically significant x-y relationship appears; controlling for it correctly removes the spurious effect. With a true causal effect present and a confounder proxy of varying correlation strength, bias shrinks as the proxy's correlation with the true confounder rises, but never fully disappears except at perfect correlation. Positive selection example: giving the repo preferentially to visitors already more likely to buy overestimates incrementality (A≥B in the chapter's notation); negative selection (targeting those least likely to buy organically) underestimates it. Matching/PSM simulation with a true treatment effect of 2: plain matching and logistic-regression-based PSM both correctly estimate θ≈2 across different control-group sizes (m); gradient-boosting-based PSM slightly underestimates it (though still within the 95% CI), and widening m narrows GBC's confidence intervals but is roughly neutral for the other two methods.

## Key Takeaways
1. Incrementality = causal inference for business decisions; naive treated-vs-control comparisons conflate the true effect (ATT) with selection bias.
2. Always draw (even mentally) a DAG before estimating a causal effect: control for confounders, never control for colliders — getting this backward flips your conclusion.
3. Unconfoundedness (selection on observables) is the untestable linchpin assumption behind every non-experimental causal method; randomization is the only approach that guarantees it by construction.
4. A/B tests (RCTs) remain the gold standard whenever feasible; respect SUTVA, and use block/cluster randomization in interference-prone settings like two-sided marketplaces.
5. When randomization is infeasible, matching and propensity score matching can estimate causal effects, but only as good as your DAG and your observed covariates allow.
6. Uplift modeling formalizes "who to target": avoid organic converters (group A) and lost causes (group C); focus incentives on the movable middle (group B).
7. Double machine learning extends causal estimation to nonlinear confounding relationships by combining orthogonalization with sample splitting, borrowing both FWL (Ch 10) and the bootstrapping/ML tooling (Ch 9) already covered.
8. Even pure predictive modeling benefits from causal thinking — feature engineering implicitly assumes causal direction, and bad controls/confounders bias predictions too.

## Connects To
- **Ch 4**: Adverse selection (credit/loan 2×2) is explicitly identified as a canonical self-selection/selection-bias example.
- **Ch 6**: Lift tables are the simpler cousin of the pretreatment-variable selection-bias check described here (difference-in-means/t-test preferred for statistical reasons).
- **Ch 9**: SNR limitations explain why some true incremental effects come back statistically insignificant (as foreshadowed there).
- **Ch 10**: Confounders, bad controls, and the Frisch-Waugh-Lovell orthogonalization are directly reused and extended into the causal-inference setting.
- **Ch 16**: A/B testing is the practical, hands-on continuation of the randomization material introduced here.
