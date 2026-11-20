# Chapter 10: Pipelines and MLOps

## Core Idea
SageMaker Pipelines ties feature engineering, training, evaluation, conditional registration, and deployment into a single, automated, lineage-tracked pipeline that can be triggered by GitOps code commits, new S3 data, time schedules, or statistical drift — realizing the MLOps v3.0 vision from Chapter 1, with human-in-the-loop workflows (Amazon A2I, SageMaker Ground Truth) improving accuracy along the way.

## Frameworks Introduced
- **SageMaker Pipelines**: the standard, full-featured SageMaker-native pipeline orchestrator.
  - When to use: any production ML workflow needing deterministic repeatability plus built-in SageMaker Experiments/Lineage integration.
  - How: wrap Processing Jobs in `ProcessingStep`, Training Jobs in `TrainingStep`, evaluation in another `ProcessingStep` with a `PropertyFile`, gate registration with `ConditionStep` (`ConditionGreaterThanOrEqualTo`), register with `RegisterModel`, and wrap it all in `Pipeline(steps=[...])`.
- **SageMaker Model Registry**: central place to register, version, and approve models for staging/production.
  - When to use: controlling which trained model version is promoted, with separate approval steps per environment.
  - How: `RegisterModel` step + `sm.update_model_package(ModelApprovalStatus="Approved")`.
- **SageMaker Projects + GitOps automation**: pre-built CloudFormation MLOps templates with separate `modelbuild` and `modeldeploy` components.
  - When to use: triggering a pipeline run on every code commit, with separated responsibility between data science (build) and DevOps (deploy) teams.
- **EventBridge + CloudTrail S3 triggers**: start a pipeline automatically when new data lands in S3.
  - How: enable CloudTrail data-event logging on the bucket, create an EventBridge rule matching `PutObject`/`CompleteMultipartUpload`/`CopyObject`, target a Lambda function that calls `sm.start_pipeline_execution()`.
- **EventBridge time-based (cron/rate) triggers**: scheduled pipeline runs (e.g., hourly) via `events.put_rule(ScheduleExpression='rate(1 hour)')`.
- **Statistical drift triggers**: SageMaker Model Monitor + Clarify detect data-quality, model-quality, bias, or feature-attribution drift against a baseline and can trigger pipeline retraining.
- **Alternative orchestrators**: AWS Step Functions Data Science SDK, Kubeflow Pipelines (on Amazon EKS), Apache Airflow (Amazon MWAA), MLflow, TensorFlow Extended (TFX, built on Apache Beam) — each with different trade-offs in managed-infra vs. flexibility.
- **Amazon A2I (Augmented AI)**: human review workflows triggered by low model-prediction confidence.
  - When to use: sensitive/regulated predictions, or low-confidence predictions that could hurt the business if wrong.
  - How: define a Worker Task Template + Flow Definition; call `a2i.start_human_loop()` when `confidence_score < CONFIDENCE_SCORE_THRESHOLD`.
- **SageMaker Ground Truth (active learning)**: combines human labeling workforces with a continuously retrained auto-labeling model.
  - When to use: scaling data labeling — the model takes over high-confidence labels, freeing humans for hard cases.

## Key Concepts
- **MLOps v1.0 → v2.0 → v3.0**: manual build/train/deploy → manual pipeline orchestration → fully automated pipelines triggered by deterministic (GitOps, new data) or statistical (drift/bias) events — the organizing framework for this whole chapter.
- **Pipeline step caching**: SageMaker Pipelines skips re-executing a step if its input artifacts/parameters haven't changed, reusing prior results (`CacheConfig(enable_caching=True, expire_after=...)`).
- **ConditionStep / minimum accuracy gate**: a pipeline can conditionally register/deploy a model only if it clears an evaluation threshold (e.g., 95% accuracy), otherwise the pipeline ends without promoting a bad model.
- **modelbuild vs. modeldeploy separation**: splitting the GitOps MLOps template into a build phase (data science owns, pushes to staging) and a deploy phase (DevOps owns, pushes to production with manual approval) — enforces separation of responsibility and access control.
- **Worker Task Template / Flow Definition / Human Loop**: the three A2I abstractions — UI template for workers, the review workflow definition (workforce + instructions), and the actual triggered review instance.
- **Active learning**: starts with human labeling, then progressively shifts routine/high-confidence cases to an auto-labeling model, reserving human effort for hard/low-confidence cases — implemented by SageMaker Ground Truth.

## Mental Models
- **ML pipelines as "CI/CD for models"**: just as software CI/CD automated build-test-deploy for code, ML pipelines automate ingest-validate-train-evaluate-deploy for models — the same cultural shift (reducing "over the wall" handoffs) applies.
- **Data quality as pipeline "step 0"**: Deequ-based data-quality checks belong at the very start of the pipeline — catching bad data before it wastes a training job's time and money, and before it reaches production.
- **Don't over-engineer early**: in the exploratory research phase, a rigid pipeline can be premature; pipelines pay off once you're ready for regular, repeated training — but even a lightweight pipeline provides data-quality checking and lineage benefits early.
- **Effective pipeline checklist**: data ingestion/versioning → data analysis/validation → feature engineering (+ feature store) → model training/tuning → model evaluation (+ bias check) → model versioning/deployment → model feedback/skew detection — a canonical shape to evaluate any pipeline against.

## Anti-patterns
- **Manually handing a trained model "over the wall" to DevOps/app teams**: the high-friction handoff pattern that CI/CD eliminated for software and that ML pipelines should eliminate for models.
- **Skipping data-quality checks before training**: the single most common cause of bad ML pipelines, per the authors' stated experience — catch bad data before training, not after deployment.
- **Building a rigid pipeline too early in exploratory research**: locks in assumptions before the modeling approach has stabilized; stay in the "research lab" until ready to productionize regularly.
- **Starting a full pipeline run on every single S3 file upload indiscriminately**: may be wasteful depending on use case — consider whether incremental/batched triggering is more appropriate.
- **Running Spot Instance training within a pipeline without checkpointing**: loses progress on interruption, same risk as in Ch. 7/8, now compounded across a scheduled, repeated pipeline.

## Code Examples
```python
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo

minimum_accuracy_condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(step=evaluation_step, property_file=evaluation_report,
                 json_path="metrics.accuracy.value"),
    right=0.95)
minimum_accuracy_condition_step = ConditionStep(
    name="AccuracyCondition",
    conditions=[minimum_accuracy_condition],
    if_steps=[register_step, create_step],
    else_steps=[])
```
- **What it demonstrates**: gating model registration/deployment on a 95% accuracy threshold computed by a prior evaluation step — the pipeline silently stops if the model isn't good enough.

```python
pattern = {
  "source": ["aws.s3"], "detail-type": ["AWS API Call via CloudTrail"],
  "detail": {"eventSource": ["s3.amazonaws.com"],
             "eventName": ["PutObject", "CompleteMultipartUpload", "CopyObject"],
             "requestParameters": {"bucketName": ["{}".format(watched_bucket)]}}}
response = events.put_rule(Name='S3-Trigger', EventPattern=json.dumps(pattern), State='ENABLED')
```
- **What it demonstrates**: an EventBridge rule that fires whenever new data lands in a watched S3 bucket, feeding a Lambda shim that calls `sm.start_pipeline_execution()`.

```python
CONFIDENCE_SCORE_THRESHOLD = 0.70
if confidence_score < CONFIDENCE_SCORE_THRESHOLD:
    start_loop_response = a2i.start_human_loop(
        HumanLoopName=str(uuid.uuid4()), FlowDefinitionArn=flowDefinitionArn,
        HumanLoopInput={'InputContent': json.dumps(inputContent)})
```
- **What it demonstrates**: routing only low-confidence predictions (below 70%) to human review via Amazon A2I, while high-confidence predictions pass straight through.

```python
from sagemaker.workflow.steps import CacheConfig
cache_config_prepare = CacheConfig(enable_caching=True, expire_after=<EXPIRE_TIME>)
processing_step = ProcessingStep(name='Processing', ..., cache_config=cache_config_prepare)
```
- **What it demonstrates**: enabling step caching so an unchanged feature-engineering step is skipped on re-run, saving time and cost.

## Reference Tables
| Orchestrator | Best for | Trade-off |
|---|---|---|
| SageMaker Pipelines | Native SageMaker integration, full MLOps | AWS-specific |
| AWS Step Functions | Complex workflows, no infra to manage | Not ML-specific, needs Data Science SDK |
| Kubeflow Pipelines | Kubernetes-native teams | Must build/maintain EKS cluster |
| Apache Airflow (MWAA) | Mature ETL-style DAG orchestration | Not ML-native |
| MLflow | Lightweight experiment tracking + simple workflows | Must manage own EC2/EKS, limited workflow support |
| TFX | TensorFlow-centric, full pipeline structure | Needs Apache Beam for distributed scale |

| Trigger type | Example |
|---|---|
| GitOps (event) | New commit to `train.py` in repo |
| S3 (event) | New file lands via PutObject |
| Time-based (schedule) | `rate(1 hour)` cron-like EventBridge rule |
| Statistical drift | Model Monitor/Clarify bias or data-quality threshold breach |

## Worked Example
A SageMaker Pipeline is built for the BERT review classifier: a `ProcessingStep` runs `preprocess-scikit-text-to-bert-feature-store.py` to produce BERT train/validation/test TFRecords (and publish to Feature Store); a `TrainingStep` trains the TensorFlow/BERT model using those outputs; an `evaluation_step` (another ProcessingStep) computes test accuracy into `evaluation.json`; a `ConditionStep` checks that accuracy ≥ 95% before proceeding to `RegisterModel` (with `ModelMetrics` attached) and `CreateModelStep`. The whole pipeline is parameterized (`InputData`, `MaxSeqLength`, `LearningRate`) and started via `pipeline.start(...)` or the SageMaker Studio UI. Once registered, the model requires manual approval to move to staging, then a second approval to move to production. The pipeline's full artifact and experiment lineage (feature engineering → training → evaluation → approval → deployment) is inspectable via `LineageTableVisualizer` and `ExperimentAnalytics`. Separately, Amazon A2I routes any review-sentiment prediction below 70% confidence to a human workforce, whose labels can feed back into a continuous retraining pipeline — echoing SageMaker Ground Truth's active-learning pattern.

## Key Takeaways
1. Treat data-quality checks as "step 0" of any pipeline — this prevents the most common cause of bad ML pipelines.
2. Gate model promotion on an evaluation-metric threshold (`ConditionStep`) so only models that clear a bar get registered/deployed.
3. Separate `modelbuild` (data science) from `modeldeploy` (DevOps) responsibilities in GitOps automation for clean access control.
4. Combine deterministic triggers (GitOps, S3 arrival, schedule) with statistical triggers (Model Monitor/Clarify drift) to fully realize MLOps v3.0.
5. Use step caching to avoid re-running unchanged pipeline steps, and Spot Instances with checkpointing for training steps to cut cost.
6. Route only low-confidence predictions to human review (Amazon A2I) and use active learning (SageMaker Ground Truth) to scale labeling efficiently over time.
7. Keep full lineage (SageMaker Experiments + ML Lineage Tracking) across every pipeline step for auditability and reproducibility.

## Connects To
- **Ch 1**: Directly implements the MLOps v3.0 vision and pipeline-orchestration options introduced there.
- **Ch 5/9**: Deequ data-quality checks and SageMaker Model Monitor/Clarify drift detection (introduced there) become automated pipeline triggers here.
- **Ch 6/7/8/9**: Each prior chapter's manual step (feature engineering, training, tuning, deployment) becomes one wrapped pipeline step here.
- **Ch 11**: Extends this automation story to continuous streaming data pipelines next.
