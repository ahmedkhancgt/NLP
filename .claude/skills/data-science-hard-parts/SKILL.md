---
name: data-science-hard-parts
description: "Knowledge base from \"Data Science: The Hard Parts\" by Daniel Vaughan. Use when applying practical data science techniques beyond modeling — metrics design, growth decomposition, 2x2 experiment designs, building business cases, narratives and storytelling with data, simulation/bootstrapping, data leakage, productionizing models, incrementality, A/B testing, or using LLMs in data science practice."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Science: The Hard Parts
**Author**: Daniel Vaughan | **Pages**: ~257 | **Chapters**: 17 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch09`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**Value creation ladder** (Ch 1): data → decisions → value. Value is never a direct function of data or models; it comes from the *actions* (A) that use a model's output (X) to move a metric (M): `Impact = M(A(X)) − M(A(baseline))`. Own the whole chain, not just the model.

**What / So What / Now What** (Ch 1, 7): the book's recurring structure for both project kickoff and delivery. What = understand the business; So What = find the actionable, often quantified implication (Aha! moments); Now What = drive the decision through the org end to end.

**Metrics design** (Ch 2): good metrics are Measurable, Actionable, Relevant, Timely. Decompose non-actionable top-line metrics via **Funnel** (sequential stages), **Stock-Flow** (`Stock_t = Stock_(t-1) + Inflow − Outflow`), or **P×Q** (`Revenue = Price × Quantity`) decompositions to surface real levers.

**Growth decompositions** (Ch 3): explain *changes* in a metric using **Additive** (sums — weighted-avg of input growth rates), **Multiplicative** (products — sum of input growth rates + combined effect), or **Mix-Rate** (weighted averages — `Δy = Δy^x(rate) + Δy^w(mix) + Δw·Δx(combined)`) decompositions. These give hints, not proof, of root causes.

**2×2 Designs** (Ch 4): deliberately simplify to two key factors on a grid (4 quadrants) for communication, joint model+lever testing, customer segmentation, or correlated-feature decisions (adverse selection). Can't rank items that dominate on different axes — use a business-case monetary scale instead.

**Business cases** (Ch 5): compare decision vs. no-decision; count only *incremental* costs/benefits; analyze at unit-economics level; solve the breakeven inequality (`Net Benefit ≥ 0`); net out false-positive and false-negative costs explicitly.

**Lifts** (Ch 6): `Lift(A,B) = Metric_agg(A)/Metric_agg(B)`. Fast diagnostic for classifier performance (decile TPR) and for self-selection/survivorship bias (feature-by-feature lift tables).

**Narratives** (Ch 7): good DS narratives are Clear, Credible, Memorable, Actionable. Build them iteratively (hypotheses from day one), not by fitting a story onto finished work. Structure delivery as What/So What/Now What, not the classic narrative arc.

**Datavis** (Ch 8): every visualization choice should be judged by "does this convey my message?" Bars for categories, lines for continuous/time series; use ALE over PDP for correlated features' effect plots; maximize data-ink ratio but allow deliberate exceptions for attention-directing annotation.

**Simulation & Bootstrapping** (Ch 9): Monte Carlo simulation stress-tests algorithm behavior against a known DGP (SNR, correlation, omitted variables); bootstrapping estimates precision nonparametrically by resampling your actual data. No algorithm fixes a data problem (e.g., omitted-variable bias hits OLS and GBR alike).

**Linear regression fundamentals** (Ch 10): the Frisch-Waugh-Lovell theorem shows every regression coefficient is a *net* effect after partialling out other covariates — the conceptual root of "control for confounders, never for colliders." Variance in inputs is required to learn anything, for any algorithm.

**Data leakage** (Ch 11): leakage = using data/metadata unavailable at prediction time, correlated with the outcome, inflating test performance. Defend with the **windowing methodology**: define observation window `[t_p−O, t_p]` and left-open prediction window `(t_p, t_p+P]`, and make training mirror scoring exactly.

**Productionizing models** (Ch 12): "scoring reigns" — a model only creates value once deployed, monitored, maintained. Watch for data drift (distribution changes) vs. model drift (DGP changes); the Zillow Offers case ($500M+ loss) is the canonical cautionary tale of skipping this.

**Storytelling in ML** (Ch 13): two personas — scientist (ex ante: hypotheses → feature engineering → training) and salesperson (ex post: interpretability). Interpretability toolkit: standardized coefficients, impurity/permutation importance, heatmaps (bivariate only), PDP (biased under correlation), ALE (corrects that bias).

**Prediction to decisions** (Ch 14): ML's ROI comes from better decisions, not accuracy alone. Classification models yield threshold decision rules (`Action A if p≥τ`); optimize τ via confusion-matrix cost/profit when FP and FN costs differ; use a volume rule when capacity (not cost) binds.

**Incrementality** (Ch 15): causal inference for business decisions. Use DAGs + the back-door criterion (control for confounders, never colliders); the unconfoundedness assumption is the untestable linchpin of every non-experimental method. Randomization (A/B testing) is the only approach that guarantees it by construction; matching/PSM and Double ML are second-best alternatives for observational data.

**A/B Tests** (Ch 16): start from the metric, then find levers (never reverse). Decide via a two-sided t-test; design using the **Minimum Detectable Effect (MDE)** formula to avoid underpowered false negatives; distinguish statistical significance from business significance.

**LLMs and data science** (Ch 17): programming-heavy tasks are most exposed to LLM automation; business knowledge, causal/analytical reasoning, and hypothesis generation remain comparatively durable human advantages — directionally, not definitively.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| 1 | So What? Creating Value with Data Science | What/So What/Now What; M(A(X)) value formula |
| 2 | Metrics Design | Measurable/Actionable/Relevant/Timely; Funnel, Stock-Flow, P×Q decomposition |
| 3 | Growth Decompositions | Additive, Multiplicative, Mix-Rate decomposition |
| 4 | 2×2 Designs | 2×2 grid; model+lever testing; adverse selection |
| 5 | Building Business Cases | Breakeven; incrementality; unit economics |
| 6 | What's in a Lift? | Lift ratio; self-selection/survivorship bias detection |
| 7 | Narratives | Clear/Credible/Memorable/Actionable; What/So What/Now What delivery |
| 8 | Datavis | Bar vs. line; slopegraph; waterfall; data-ink ratio |
| 9 | Simulation and Bootstrapping | Monte Carlo pipeline; bootstrapping; PDP; omitted variable bias |
| 10 | Linear Regression: Going Back to Basics | Frisch-Waugh-Lovell theorem; dummy variable trap; variance-identification |
| 11 | Data Leakage | Windowing methodology; complete/quasi-complete separation |
| 12 | Productionizing Models | Production pipeline stages; data vs. model drift |
| 13 | Storytelling in Machine Learning | Ex ante/ex post storytelling; feature importance; PDP/ICE/ALE |
| 14 | From Prediction to Decisions | Threshold decision rule; confusion matrix optimization |
| 15 | Incrementality: The Holy Grail of Data Science? | DAGs; unconfoundedness; matching/PSM; Double ML |
| 16 | A/B Tests | Decision criterion; Minimum Detectable Effect; hypotheses backlog |
| 17 | LLMs and the Practice of Data Science | Task-exposure scoring framework |

## Topic Index
- **2×2 designs** → Ch 4
- **A/B testing** → Ch 15, 16
- **ALE (Accumulated Local Effects)** → Ch 13
- **Bias (omitted variable)** → Ch 9, 10
- **Bootstrapping** → Ch 9
- **Business cases** → Ch 5
- **Causal inference / DAGs** → Ch 15
- **Collider / Confounder** → Ch 10, 15
- **Confusion matrix** → Ch 14
- **Data leakage** → Ch 11
- **Data visualization** → Ch 8
- **Decision rules / thresholding** → Ch 14
- **Double Machine Learning** → Ch 15
- **Feature importance** → Ch 13
- **Frisch-Waugh-Lovell theorem** → Ch 10
- **Funnel decomposition** → Ch 2
- **Growth decomposition** → Ch 3
- **Interpretability (PDP, ICE, ALE, heatmaps)** → Ch 13
- **LLMs and AI exposure** → Ch 17
- **Lift** → Ch 6
- **Matching / Propensity Score Matching** → Ch 15
- **Metrics design** → Ch 2
- **Minimum Detectable Effect (MDE)** → Ch 16
- **Mix-rate decomposition** → Ch 3
- **Monte Carlo simulation** → Ch 9
- **Narratives / storytelling** → Ch 7, 13
- **Productionizing / MLOps** → Ch 12
- **Selection bias** → Ch 6, 15
- **Simulation** → Ch 9
- **Stock-flow decomposition** → Ch 2
- **Unconfoundedness** → Ch 15
- **Windowing methodology** → Ch 11

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — analytical techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 17 chapters, synthesized as structure, not verbatim text.
