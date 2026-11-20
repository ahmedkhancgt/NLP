# Chapter 13: Deploying a Machine Learning API

## Core Idea
Follow the CRISP-DM process to train a scikit-learn gradient boosting model (predicting fantasy football waiver bid ranges), convert it to the cross-platform ONNX format, and deploy it as a real-time inference FastAPI endpoint — turning a data scientist into an API *producer* of ML predictions, mirroring Part I's architecture but swapping a database for a model-serving backend.

## Frameworks Introduced
- **CRISP-DM (Cross-Industry Standard Process for Data Mining)**: a six-stage framework for organizing ML projects.
  - When to use: structuring any end-to-end machine learning project, from problem framing to production deployment.
  - How: proceed through Business Understanding (objectives/tools) → Data Understanding (collect, explore, verify quality) → Data Preparation (select/format/merge data) → Modeling (pick technique, build model) → Evaluation (assess fitness for production) → Deployment (serve to consumers, then monitor/maintain). This chapter emphasizes Deployment while touching the others lightly.
- **Batch vs. Real-Time Inference**: choosing how predictions are served based on latency needs.
  - When to use: deciding whether an ML model needs an API at all.
  - How: use batch inference (a scheduled script/job processing a whole group of predictions at once) when timing isn't urgent (e.g., a week's worth of predictions); use real-time inference (an API call returning a single prediction immediately) when predictions must reflect up-to-the-minute inputs — this is the scenario that justifies deploying a model as an API.
- **ONNX Conversion Workflow for Cross-Platform Model Deployment**: decouple model training framework from model serving runtime.
  - When to use: deploying any ML model where interoperability, acceleration, or framework independence matters.
  - How: train in a framework-native library (scikit-learn here) → convert the fitted model to ONNX format with a framework-specific converter (`sklearn-onnx`'s `to_onnx()`) → serve inferences using the generic `onnxruntime` library (`InferenceSession`), independent of the original training framework.
- **Multi-Model Quantile Regression for Prediction Ranges**: produce a range instead of a single point estimate.
  - When to use: when a single predicted value is less useful than a low/median/high range (e.g., bidding strategy under uncertainty).
  - How: train separate models with the same algorithm/features but different `loss="quantile"` targets (e.g., `alpha=0.1`, `0.5`, `0.9`), then combine their three outputs into one API response.

## Key Concepts
- **Classification vs. Regression**: classification predicts a category (classifier); regression predicts a continuous numeric value (regressor) — this chapter's dollar-amount prediction is a regression problem.
- **Gradient boosting**: an ensemble ML technique combining multiple decision trees so the combined model outperforms any individual tree; implemented here via scikit-learn's `GradientBoostingRegressor`.
- **Supervised learning**: training a model on historical data with known/labeled outcomes (e.g., past waiver bids with their actual winning dollar amounts) so it can predict outcomes for new, unlabeled inputs.
- **Training vs. inference**: training uses historical input/output pairs to build a model that recognizes patterns; inference applies the trained model's learned patterns to new input data to predict unknown outputs.
- **Fitting**: the act of applying a general algorithm to specific training data to produce a specialized, usable model (`model.fit(X_train, y_train)`).
- **Train/test split**: dividing data (commonly 80/20) into `X_train`/`y_train` (used to fit the model) and `X_test`/`y_test` (held out to evaluate it) — conventional variable names `X` (uppercase, inputs) and `y` (lowercase, outputs) are used throughout the data science community.
- **ONNX (Open Neural Network Exchange)**: an open standard model format enabling models trained in different languages/frameworks to be deployed and run consistently via a common runtime (ONNX Runtime), which also offers inference acceleration.
- **Inference session**: the ONNX Runtime object (`rt.InferenceSession`) loaded from a `.onnx` file, used to run predictions by feeding a NumPy array matching the model's expected input shape/names.
- **Model card / system card**: emerging documentation standards for ML transparency — model cards (proposed by Google) document a single model's purpose/risks/biases; system cards (proposed by Meta) document a broader AI system that may include multiple models holistically.

## Mental Models
- **"Always in motion is the future"** (Yoda epigraph): frames the entire chapter's purpose — fantasy managers (like all ML consumers) want predictions about an uncertain future, and this is precisely what ML models exist to approximate.
- **Mirrored architecture from Part I**: the API's shape (FastAPI controller + Pydantic schemas for I/O validation) is structurally identical to the database-backed API from Chapters 3–4; only the backend data source changes — from SQLAlchemy/SQLite to ONNX Runtime/trained models. Recognizing this parallel makes the ML-serving pattern easy to generalize.
- **Theory-driven feature selection, not blind trial-and-error**: each input feature (budget remaining, weeks remaining, value tier) is included because of an explicit stated theory about why it should predict the output, not just because it's available — mirroring Chapter 9's custom-metric design discipline.
- **Uncertainty communicated as a range, not a false-precision point estimate**: rather than returning one number, the API gives three quantile predictions (10th/50th/90th percentile) so the consumer (a fantasy manager bidding strategically) can reason about risk and uncertainty directly.

## Anti-patterns
- **Deploying a model without any evaluation**: the chapter explicitly notes this is a "first draft" model and that a full project would iteratively evaluate accuracy/fairness before production use — skipping evaluation risks deploying an unreliable model.
- **Selecting model features without a stated rationale**: picking variables just because they're in the dataset, rather than articulating why each one should plausibly predict the output, risks spurious or uninterpretable models.
- **Returning a single point prediction when uncertainty matters**: for a decision like bidding strategy, a single number hides the risk of overbidding/underbidding that a percentile range makes visible.
- **Coupling model-serving code tightly to the training framework**: skipping the ONNX conversion step would lock the deployed API to scikit-learn-specific runtime dependencies, losing the cross-framework portability and acceleration ONNX provides.
- **Treating ML models as self-explanatory**: because models aren't explicitly programmed, their internal decision logic isn't transparent by default — documentation (model/system cards) is necessary, not optional, especially for production use.

## Code Examples
```python
# Training three quantile-regression models (notebook)
model_10th_percentile = GradientBoostingRegressor(loss="quantile", alpha=0.1)
model_50th_percentile = GradientBoostingRegressor(loss="quantile", alpha=0.5)
model_90th_percentile = GradientBoostingRegressor(loss="quantile", alpha=0.9)

model_10th_percentile.fit(X_train, y_train)
model_50th_percentile.fit(X_train, y_train)
model_90th_percentile.fit(X_train, y_train)
```
- **What it demonstrates**: training three separate regressors targeting different quantiles of the same output distribution, using the same features, to produce a low/median/high prediction range.

```python
# Converting to ONNX and saving
X_array = np.column_stack((X['waiver_value_tier'],
                            X['fantasy_regular_season_weeks_remaining'],
                            X['league_budget_pct_remaining']))

onx = to_onnx(model_10th_percentile, X_array[:1])
with open("acquisition_model_10.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```
- **What it demonstrates**: combining features into a 2D NumPy array, converting a fitted scikit-learn model to ONNX using a sample row to infer input/output shapes, and serializing the result to a `.onnx` file.

```python
# main.py — loading ONNX models and defining FastAPI endpoints
sess_10 = rt.InferenceSession("acquisition_model_10.onnx", providers=["CPUExecutionProvider"])
input_name_10 = sess_10.get_inputs()[0].name
label_name_10 = sess_10.get_outputs()[0].name

app = FastAPI(description=api_description, title="Fantasy acquisition API", version="0.1")

@app.get("/", tags=["analytics"])
def root():
    return {"message": "API health check successful"}

@app.post("/predict/", response_model=PredictionOutput, tags=["prediction"])
def predict(features: FantasyAcquisitionFeatures):
    input_data = np.array([[features.waiver_value_tier,
                             features.fantasy_regular_season_weeks_remaining,
                             features.league_budget_pct_remaining]], dtype=np.int64)
    pred_onx_10 = sess_10.run([label_name_10], {input_name_10: input_data})[0]
    pred_onx_50 = sess_50.run([label_name_50], {input_name_50: input_data})[0]
    pred_onx_90 = sess_90.run([label_name_90], {input_name_90: input_data})[0]
    return PredictionOutput(
        winning_bid_10th_percentile=round(float(pred_onx_10[0]), 2),
        winning_bid_50th_percentile=round(float(pred_onx_50[0]), 2),
        winning_bid_90th_percentile=round(float(pred_onx_90[0]), 2))
```
- **What it demonstrates**: loading ONNX inference sessions once at module load (startup), a GET health-check endpoint (recap pattern from Part I), and a POST `/predict/` endpoint converting validated Pydantic input into a NumPy array, running three ONNX inferences, and returning a combined Pydantic response.

```python
# schemas.py
class FantasyAcquisitionFeatures(BaseModel):
    waiver_value_tier: int
    fantasy_regular_season_weeks_remaining: int
    league_budget_pct_remaining: int

class PredictionOutput(BaseModel):
    winning_bid_10th_percentile: float
    winning_bid_50th_percentile: float
    winning_bid_90th_percentile: float
```
- **What it demonstrates**: Pydantic schemas defining both the API's request body (POST input) and response shape, reused by FastAPI for validation and OAS generation, exactly as in Part I.

## Reference Tables
**Table 13-1. New software used in this chapter**

| Software | Purpose |
|---|---|
| ONNX Runtime | Cross-platform runtime for models from multiple frameworks |
| scikit-learn | ML training framework (`GradientBoostingRegressor` used here) |
| sklearn-onnx | Converts scikit-learn models to ONNX format |

**Table 13-2. Conventional variable names for training models**

| Variable | Purpose | Columns | Data |
|---|---|---|---|
| `X` | Input columns, full data | Input | All |
| `y` | Output column, full data | Output | All |
| `X_train` | Input columns, training | Input | 80% |
| `y_train` | Output column, training | Output | 80% |
| `X_test` | Input columns, test | Input | 20% |
| `y_test` | Output column, test | Output | 20% |

**CRISP-DM stages**

| Stage | Focus |
|---|---|
| Business Understanding | Objectives, tools/techniques |
| Data Understanding | Collect, explore, verify quality |
| Data Preparation | Select, format, merge |
| Modeling | Select technique, build model |
| Evaluation | Fitness for production |
| Deployment | Serve, monitor, maintain |

## Worked Example
Business problem: predict the range of waiver-bid dollar costs (10th/50th/90th percentile) a fantasy manager should expect for a player, given `waiver_value_tier`, `fantasy_regular_season_weeks_remaining`, and `league_budget_pct_remaining`. Data: `player_training_data_full.csv` with historical winning bids. In a Jupyter Notebook, the data is loaded, split 80/20 into train/test sets, and three `GradientBoostingRegressor` models (quantile loss at alpha 0.1/0.5/0.9) are fit on the same three features, then each converted to ONNX and saved as separate `.onnx` files. In `main.py`, three `onnxruntime.InferenceSession` objects are loaded at startup; a FastAPI app exposes a `GET /` health check and a `POST /predict/` endpoint that accepts a `FantasyAcquisitionFeatures` Pydantic object, converts it to a NumPy array, runs all three ONNX sessions, and returns a `PredictionOutput` with three dollar-value predictions. Tested via Swagger UI with `{"waiver_value_tier": 1, "fantasy_regular_season_weeks_remaining": 12, "league_budget_pct_remaining": 88}`, returning a 200 response with the three predicted bid amounts.

## Key Takeaways
1. Use CRISP-DM to structure ML projects end-to-end, even when (as here) the focus is primarily on the deployment stage.
2. Choose real-time inference (API) over batch inference specifically when predictions must reflect immediate, current inputs rather than being computed on a schedule.
3. Select model features based on an explicit theory of why each should matter, not just availability — and choose an algorithm (gradient boosting here) that fits both the output type (continuous → regression) and the feature complexity (mixed linear/categorical/nonlinear → decision-tree-based).
4. Train multiple quantile-regression models to produce a prediction range (not just a point estimate) when uncertainty communication matters to the consumer's decision.
5. Convert models to ONNX to decouple training framework from serving runtime, gaining cross-platform portability and inference acceleration.
6. Reuse the same FastAPI + Pydantic architectural pattern from Part I for ML-serving APIs — only the backend (ONNX Runtime vs. database) changes.
7. Document ML models' purpose, risks, and limitations via model cards or system cards, since model behavior isn't self-explanatory the way database-backed API logic is.
8. Treat any first-pass deployed model as a draft requiring formal evaluation (accuracy, fairness) before real production use.

## Connects To
- **Ch 3/Ch 4**: The FastAPI + Pydantic architecture here directly mirrors the Part I API pattern, substituting ONNX Runtime for SQLAlchemy/SQLite as the backend data source.
- **Ch 9**: Reuses the theory-driven feature/metric design discipline established for the Shark League Score.
- **Ch 12**: Implements the "inference endpoint" / real-time ML-as-API pattern previewed there.
- **Ch 14**: This deployed ML-serving API (or similar) could itself become a tool an LLM agent calls, continuing Part III's AI/API integration theme.
- **External**: *Designing Machine Learning Systems* (Chip Huyen); *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron); *Practical Data Science with Python* (Nathan George); Google's model card proposal; Meta's system card description.
