# Chapter 9: Deploy Models to Production

## Core Idea
Production model deployment requires choosing real-time (SageMaker Endpoints) vs. batch (Batch Transform) serving, safely rolling out new models (canary, blue/green, A/B, multi-armed bandit), auto-scaling to traffic, and continuously monitoring for data/model/bias/feature-attribution drift via SageMaker Model Monitor and Clarify.

## Frameworks Introduced
- **SageMaker Endpoints**: distributed, multi-AZ REST API containers for real-time, low-latency predictions.
  - When to use: latency-sensitive, single-prediction-request applications.
  - How: `TensorFlowModel(...).deploy(endpoint_name=..., initial_instance_count=1, instance_type=...)`; custom `inference.py` provides `input_handler()`/`output_handler()`.
- **SageMaker Batch Transform**: offline, high-throughput batch predictions over S3 data without a standing endpoint.
  - When to use: throughput-over-latency workloads tolerant of staleness (e.g., nightly recommendation refresh).
  - How: configure `strategy='MultiRecord'`, `compression_type`, `accept_type`, `assemble_with`/`split_type`, enable TensorFlow Serving batching env vars (`SAGEMAKER_TFS_ENABLE_BATCHING`, `SAGEMAKER_TFS_MAX_BATCH_SIZE`), then `batch_predictor.transform(data=..., join_source='Input')`.
- **SageMaker Inference Pipelines**: chain multiple containers (request handler → model → response handler, or model ensembles) behind one endpoint.
  - How: `PipelineModel(models=[request_handler, model, response_handler]).deploy(...)`.
- **Production Variants + traffic-shifting deployment strategies**: canary rollout (small % traffic to new model), blue/green (full traffic cutover, old cluster kept idle as rollback), A/B testing (long-running comparison, often 50/50), multi-armed bandit (MAB) testing (dynamic reinforcement-learning-based traffic allocation).
  - When to use: canary for risk mitigation on upgrades; blue/green for safe full cutover with rollback; A/B for statistically rigorous long-running comparison; MAB when you want to minimize "regret" by dynamically shifting traffic faster than A/B allows.
  - How: `create_endpoint_config()` with multiple `ProductionVariants` (weights), then `update_endpoint_weights_and_capacities()` to shift traffic.
- **SageMaker Auto Scaling (CloudWatch-based)**: target tracking, simple, or step scaling policies for endpoint instance count.
  - When to use: handling variable prediction traffic without manual intervention.
  - How: `register_scalable_target()` + `put_scaling_policy()` with `TargetTrackingScalingPolicyConfiguration` (e.g., `SageMakerVariantInvocationsPerInstance`) or a custom CloudWatch metric; tune `ScaleOutCooldown`/`ScaleInCooldown`.
- **SageMaker Model Monitor**: automatic data capture + scheduled comparison against a baseline to detect data-quality, model-quality, bias, and feature-attribution drift.
  - When to use: any production model, to catch covariate/concept drift before it silently degrades predictions.
  - How: enable `DataCaptureConfig` at deploy time; `DefaultModelMonitor.suggest_baseline()` (data quality, via Deequ) or `ModelQualityMonitor.suggest_baseline()` (model quality, needs ground truth); `create_monitoring_schedule()` with a cron expression; inspect `constraint_violations.json`.
- **SageMaker Neo + TensorFlow Lite**: hardware-specific model compilation (quantization, graph pruning, layer fusing) for up to 2x inference speedup.
- **TorchServe / AWS Deep Java Library**: alternative model-serving runtimes for PyTorch and Java-based inference, natively integrated with SageMaker.
- **AWS Lambda + API Gateway (serverless inference)**: event-driven, serverless model serving with provisioned concurrency to reduce cold-start latency.

## Key Concepts
- **Scaling in/out vs. scaling up/down**: in/out = adding/removing instances of the same type; up/down = changing to a bigger/smaller instance type.
- **Covariate shift vs. concept shift (operationalized)**: data-quality monitoring (Deequ-based) detects covariate shift in model inputs; model-quality monitoring (comparing predictions to human-labeled ground truth) detects concept shift.
- **Action probability vs. sample probability (MAB)**: action probability = likelihood a variant is the best choice given current rewards; sample probability = likelihood the bandit actually selects that variant given the exploration policy — together they determine real traffic allocation.
- **Exploit vs. explore**: MAB's core tension — exploiting the current best-performing model variant vs. continuing to explore other variants in case they're actually better; controlled by the exploration policy (epsilon greedy, Thompson sampling, bagging, online cover).
- **Regret**: the cost of not using the best-performing model during an experiment; MABs minimize regret by shifting traffic dynamically, unlike static A/B tests.
- **Join_source/InputFilter/OutputFilter (Batch Transform)**: mechanisms to join predictions back to original input rows and filter which fields pass through, reducing output size and overhead.
- **Elastic Inference Accelerator (EIA)**: a low-cost, dynamically attached GPU add-on for CPU-based instances, avoiding the cost of oversized standalone GPU instances for small-batch inference.

## Mental Models
- **Real-time vs. batch decision rule**: choose real-time endpoints for latency-sensitive, single-request serving; choose Batch Transform for high-throughput, latency-tolerant, large-S3-dataset scoring.
- **Canary vs. blue/green vs. A/B vs. MAB — different purposes**: canary = risk mitigation for a rollout (small % traffic, short-lived); blue/green = safe full cutover with an idle rollback cluster; A/B = long-running statistical comparison of two static variants; MAB = dynamic, continuously-adapting comparison that can add/remove variants mid-experiment.
- **Homogeneous instance types in a cluster**: mixing instance types in one endpoint cluster complicates auto-scaling policy tuning — keep clusters homogeneous and compare different instance types via separate clusters.
- **Drift detection closes the MLOps loop**: Model Monitor baselines (from training data) + scheduled comparison jobs + CloudWatch alerts → triggered retraining — this is the automated feedback loop that distinguishes MLOps v3.0 (Ch. 1) from manual model management.

## Anti-patterns
- **Leaving unused or overprovisioned endpoints running**: SageMaker Endpoints are long-running resources; forgetting to delete/scale-in after a blue/green deployment wastes money continuously.
- **Stopping an A/B test early**: destroys statistical significance; if early stopping is needed, switch to MAB testing instead, which is designed for that.
- **Running a partial deployment (some instances old, some new) without blue/green discipline**: creates a hard-to-debug mixed state; blue/green avoids this by cutting over atomically.
- **Mixing instance types within one auto-scaled cluster**: makes scaling policies inconsistent and hard to tune.
- **Ignoring Model Monitor violations**: e.g., a `baseline_drift_check` violation (covariate shift in `review_body`, like a sudden shift to emoji-heavy reviews) signals the model needs retraining — ignoring it lets predictions silently degrade.

## Code Examples
```python
updated_endpoint_config=[
    {'VariantName': 'VariantA', 'ModelName': 'ModelA', 'InstanceType':'ml.m5.large',
     'InitialInstanceCount': 20, 'InitialVariantWeight': 95},
    {'VariantName': 'VariantB', 'ModelName': 'ModelB', 'InstanceType':'ml.m5.large',
     'InitialInstanceCount': 1, 'InitialVariantWeight': 5}]
sm.update_endpoint(EndpointName='my_endpoint_name', EndpointConfigName='my_endpoint_config_name')
```
- **What it demonstrates**: a canary rollout sending 5% of traffic to a new model variant while 95% stays on the proven variant.

```python
autoscale.put_scaling_policy(
    PolicyName='bert-reviews-autoscale-policy', ServiceNamespace='sagemaker',
    ResourceId="endpoint/" + tensorflow_endpoint_name + "/variant/AllTraffic",
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 1000.0,
        'PredefinedMetricSpecification': {'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'},
        'ScaleOutCooldown': 60, 'ScaleInCooldown': 300})
```
- **What it demonstrates**: target-tracking auto-scaling keeping invocations-per-instance near 1000, scaling out fast (60s cooldown) but scaling in slowly (300s cooldown) to absorb traffic dips safely.

```python
from sagemaker.model_monitor import DefaultModelMonitor, CronExpressionGenerator
my_default_monitor.create_monitoring_schedule(
  monitor_schedule_name='my-model-monitor-schedule',
  endpoint_input=predictor.endpoint,
  output_s3_uri=s3_report_path,
  statistics=my_default_monitor.baseline_statistics(),
  constraints=my_default_monitor.suggested_constraints(),
  schedule_cron_expression=CronExpressionGenerator.hourly(),
  enable_cloudwatch_metrics=True)
```
- **What it demonstrates**: scheduling hourly data-quality drift checks against a Deequ-generated baseline, emitting violation reports and CloudWatch metrics.

```python
converter = tf.lite.TocoConverter.from_saved_model('./tensorflow/')
converter.post_training_quantize = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```
- **What it demonstrates**: post-training 16-bit quantization via TensorFlow Lite for an order-of-magnitude inference speedup with minimal accuracy loss.

## Reference Tables
| Deployment strategy | Purpose | Traffic pattern |
|---|---|---|
| Canary rollout | Risk mitigation for upgrades | Small % to new variant, short-lived |
| Blue/green | Safe full cutover with rollback | 0→100% cutover, old cluster idle as fallback |
| A/B test | Statistical comparison | Fixed split (e.g., 50/50), long-running, static |
| MAB test | Dynamic optimization, minimize regret | Continuously shifting based on reward/exploration policy |

| Auto-scaling policy | Configuration effort | Control |
|---|---|---|
| Target tracking | Least | Single metric target |
| Simple | Moderate | Fixed scale amount at one threshold |
| Step scaling | Most | Different scale amounts at multiple thresholds |

| Model Monitor type | Detects | Needs |
|---|---|---|
| Data quality | Covariate shift in inputs | Deequ baseline from training data |
| Model quality | Concept shift (accuracy decay) | Ground truth labels (human-in-the-loop) |
| Bias drift | Shift in post-training bias metrics | Clarify bias baseline |
| Feature attribution drift | Shift in SHAP feature importances | Clarify explainability baseline |

## Worked Example
The BERT review classifier (Ch. 7/8) is deployed as a SageMaker Endpoint with a custom `inference.py` that tokenizes raw review text (`input_handler`) and converts predictions back to `{"predicted_label": N}` JSON (`output_handler`); sample predictions correctly classify "This is great!" as 5 and "This is terrible." as 1. Two BERT variants (93%/50% and lower-accuracy ~40% validation accuracy) are then compared using a Vowpal Wabbit-based MAB with the online-cover exploration policy: the bandit initially explores both, exploits the early winner, periodically re-explores, and converges on favoring Model 1 (higher mean action probability 0.743 vs. 0.696, sample probability 0.499 vs. 0.477) — demonstrating exploit/explore dynamics absent from static A/B tests. In parallel, SageMaker Model Monitor is configured with hourly data-quality checks against a Deequ baseline; a detected `baseline_drift_check` violation on `review_body` (e.g., from a sudden shift to emoji-heavy reviews) signals the need to retrain and redeploy — closing the drift-detection loop described in Ch. 1's MLOps v3.0 vision.

## Key Takeaways
1. Choose real-time Endpoints for latency-sensitive single predictions, Batch Transform for high-throughput offline scoring.
2. Use canary rollouts for low-risk gradual upgrades and blue/green for safe, reversible full cutovers — never leave a cluster in a mixed partial-deployment state.
3. Prefer MAB testing over static A/B tests when you need to minimize regret from a poor-performing variant or want to dynamically add/remove variants mid-experiment.
4. Configure auto-scaling with asymmetric cooldowns (fast scale-out, slow scale-in) to absorb traffic spikes without flapping.
5. Enable SageMaker Model Monitor data capture and scheduled baseline comparisons from day one — drift detection is what makes MLOps v3.0 possible.
6. Use SageMaker Neo/TensorFlow Lite quantization and Elastic Inference Accelerators to cut inference cost/latency without retraining.
7. Delete unused endpoints and scale in underutilized clusters promptly — they're a common, easily overlooked cost leak.

## Connects To
- **Ch 5**: Builds directly on the drift/shift concepts (covariate, label, concept shift) and SageMaker Clarify bias metrics introduced there.
- **Ch 7/8**: Deploys the BERT classifier trained and tuned in those chapters.
- **Ch 3/10**: Human-in-the-loop ground truth labeling (Amazon A2I, SageMaker Ground Truth) referenced here for model-quality monitoring is detailed in Ch. 10.
- **Ch 10**: The deployment step here becomes one stage of the end-to-end automated pipeline built next.
- **Ch 12**: VPC/security configuration for endpoints is covered in depth there.
