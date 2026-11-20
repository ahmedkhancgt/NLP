# Chapter 12: Productionizing Models

## Core Idea
A model is "production ready" only once deployed, monitored, and maintained for consumption by a human or system; since scoring is where all value is created, data and model drift must be actively monitored and pipelines kept as modular and simple as possible.

## Frameworks Introduced
- **Production Ready = Deployed + Monitored + Maintained** (Chip Huyen's definition), practically: the model is set for consumption (offline or online, by a human or a system).
  - When to use: as the litmus test for whether a model has actually created value, not just scored well on a test set.
- **Generic Production Pipeline Stages**: `get_data()` → `transform_data()` → `validate_data()` → [training: `train_model()`; scoring: `score_data()`] → `validate_model()/validate_scores()` → `deploy_model()/deploy_scores()`.
  - When to use: as a minimal reference architecture for any ML pipeline; training and scoring pipelines share most stages but diverge at train vs. score.
  - How: parameterize `get_data()` by the Ch 11 windowing methodology so training/scoring pull from mirrored observation/prediction windows.
- **Data/Model Drift Monitoring**: save predefined distribution statistics (e.g., deciles) for outcome and features at training and scoring time into a metadata table; test for drift via (a) a time-trend regression per metric/decile (`d_{m,t} = α + β·t + ε`; reject null of no drift if p-value for β is below threshold) or (b) nonparametric quantile-interval checks (compare new observation against historical q2.5%/q97.5%) or (c) Kolmogorov-Smirnov tests.
  - When to use: continuously, as a standing pipeline stage — not a one-off audit.

## Key Concepts
- **Data drift**: the joint distribution of the data (outcome and/or features) changes over time, while the underlying DGP is unchanged.
- **Model drift**: the underlying data generating process itself changes (e.g., gravity differs on Mars vs. Earth; a celebrity endorsement changes a purchase-probability function).
- **Real-time model objects**: online models are stored as serialized objects (model store: S3, MLflow, SageMaker) rather than as score tables; the online feature vector must exactly match the training-time feature vector, often combining real-time and batch data (lambda/kappa architecture).
- **Function as a Service (FaaS)**: typical mechanism for serving model objects and consuming live data on the fly, usually via microservices.
- **ML scores as part of the data warehouse**: designing fact/dimension tables for scores enables simple SQL joins for human or downstream-system consumption (offline, latency-tolerant use cases).
- **Modularization trade-off**: separating `get_data()`/`transform_data()` aids code review, governance, and faster deploy cycles but can add computational overhead vs. pushing transformations into the query engine (which is typically better resourced for large-data operations).

## Mental Models
- "Scoring reigns" (carried over from Ch 11): productionizing is the top priority because only a model actually serving predictions creates value.
- Keep it simple: deployment pipelines compound complexity fast; a dashboard with simple drift alerts usually beats an elaborate statistical-testing framework.
- Training and scoring pipelines are mirror images sharing the same core methods, parameterized differently (by dataset and by window alignment).

## Anti-patterns
- **Treating "good test performance" as the finish line**: a model isn't valuable until deployed, monitored, and maintained in production.
- **Skipping drift monitoring and periodic retraining**: data/model drift silently decays predictive performance; without monitoring, you find out only when it's very costly (see Zillow Offers case).
- **Over-engineering the drift test**: elaborate statistical tests aren't always better than a simple dashboard with threshold alerts — the recommendation is to keep it simple.
- **Mismatched online feature vectors**: if the real-time feature vector doesn't exactly match the training-time construction, predictions will be systematically wrong.

## Code Examples
```text
transform_data(get_data(Data)) ⇒ y, X                 # training pipeline output
train_model(transform_data(get_data(Data))) ⇒ f       # trained model object
score_data(transform_data(get_data(Data)), f()) ⇒ s   # scoring pipeline output
```
- **What it demonstrates**: the shared function-composition structure of training vs. scoring pipelines — same stages, different final step and dataset.

```text
Drift test (parametric):  d_{m,t} = α + β·t + ε
   reject "no drift" if p-value(β) < threshold (10%, 5%, 1%)
Drift test (nonparametric): check if new observation falls within [q_2.5%, q_97.5%] of historical distribution
```
- **What it demonstrates**: two alternative statistical approaches to flagging drift from the stored decile metadata table.

## Reference Tables
*Table 12-2: Example metadata table for monitoring (deciles)*

| Model | Stage | Metric | Decile | Value | Timestamp |
|---|---|---|---|---|---|
| churn | training | outcome | d1 | 100 | 2022-10-01 |
| churn | training | outcome | d10 | 1850 | 2022-10-01 |
| churn | training | feature1 | d1 | -0.5 | 2022-10-01 |
| churn | training | feature1 | d10 | 1.9 | 2022-10-01 |

## Worked Example
**Zillow Offers (2021)**: a real-estate pricing model predicted whether property prices would rise, enabling a "buy cheap, sell expensive" strategy at scale. Initial success led to scaling the product; the underlying DGP drifted (market conditions changed) and the pricing model's predictions degraded without being caught — Zillow ended up owning properties sellable only at a loss, losing over $500 million and laying off roughly a quarter of staff. Had drift monitoring and periodic retraining been in place, the company could have detected the DGP shift and adjusted its buying decisions before scaling the loss.

## Key Takeaways
1. A model only creates value once productionized — deployed, monitored, and maintained for actual consumption.
2. Distinguish data drift (distribution changes, same DGP) from model drift (DGP itself changes) — both decay predictive performance if unmonitored.
3. Build a minimal, modular production pipeline (get/transform/validate/train-or-score/validate/deploy) shared between training and scoring, parameterized by the Ch 11 windowing methodology.
4. Store distribution metadata (e.g., deciles) at each stage to power drift alerts — keep the testing method simple (a dashboard often beats a complex statistical framework).
5. Real-time/online scoring introduces architecture complexity (lambda/kappa, FaaS) and requires the online feature vector to exactly match the training-time vector.
6. Periodic retraining is the primary defense against drift-driven performance decay — the Zillow Offers case is a costly cautionary tale of skipping this.
7. Keep every production stage as simple as possible; unnecessary complexity compounds and makes debugging much harder when problems inevitably arise.

## Connects To
- **Ch 11**: Directly extends the windowing methodology and "scoring reigns" framing into full production pipeline design.
- **Ch 9**: Drift-testing techniques reuse bootstrapping/quantile-interval logic from the simulation chapter.
- **Ch 15**: Monitoring and retraining connect to incrementality — a drifted model can silently distort incrementality/A-B conclusions if not retrained.
