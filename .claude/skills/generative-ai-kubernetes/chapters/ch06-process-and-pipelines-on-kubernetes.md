# Chapter 6: Process and Pipelines on Kubernetes

## Core Idea
Model evaluation (MLflow built-in and custom metrics) and model deployment (Seldon Core on Kubernetes, using the Open Inference Protocol V2 and native HPA integration) complete the LLMOps pipeline begun in Ch5, with namespace strategy as an explicit, early design decision rather than an afterthought.

## Frameworks Introduced
- **MLflow default text-generation metrics**: toxicity (hate/abusive speech detection), ARI (Automated Readability Index, chars/words & words/sentences ratio), Flesch-Kincaid grade level (syllables/words ratio); ROUGE for summarization tasks.
  - When to use: as a first-pass automated signal on generated text quality/safety before human review.
  - How: pass `evaluators=["default"]` and `model_type="text"` to `mlflow.evaluate()`.
- **Custom MLflow metrics**: define a scoring function returning an `MetricValue`, wrap with `make_metric()`, pass via `extra_metrics=[...]` to `mlflow.evaluate()`.
  - When to use: when built-in metrics don't capture a domain-specific signal (e.g. keyword-based positivity scoring).
  - How: write `eval_fn(predictions, metrics) -> MetricValue(scores=..., aggregate_results={...})`, wrap with `make_metric(eval_fn=..., greater_is_better=True, name=...)`.
- **Seldon Core deployment pattern**: `SeldonDeployment` CRD wraps model artifact URI + server implementation (e.g. `HUGGINGFACE_SERVER`) + Istio-based ingress; integrates natively with HPA for autoscaling model servers independently from the training/tracking pipeline.
  - When to use: production model serving where you need standard REST/gRPC APIs (Open Inference Protocol V2), framework portability, and native K8s scaling.
  - How: install a service mesh (Istio) → install Seldon Core operator via Helm → define `SeldonDeployment` referencing the model's object-storage URI and a secret for storage credentials → apply.
- **Namespace strategy as explicit design**: decide early whether to use per-team/per-project namespaces (RBAC, quotas, resource-name collision avoidance) vs. dedicated clusters vs. no segregation, rather than defaulting into one.
  - When to use: as organizations mature beyond single-team experimentation into multi-team production.
  - How: weigh team size, governance/least-privilege needs, and blast-radius tolerance; combine with RBAC and ResourceQuotas (Ch2) for enforcement.

## Key Concepts
- **Open Inference Protocol V2**: a standard inference API adopted by Seldon Core, KServe, and NVIDIA Triton, enabling model-server portability.
- **Model promotion (MLflow registry)**: copying a registered model version to a named placeholder (e.g. "Staging", "Production") to track deployment-stage lineage.
- **Service Orchestrator (Seldon)**: routes requests across multiple models, aggregates responses, transforms input/output — also key to tracing/monitoring.
- **KServe**: an independent (formerly Kubeflow KFServing) OSS model-serving platform, also V2-protocol compliant, broadly vendor-supported.
- **LLM-as-judge**: using an LLM (via OpenAI or a deployed MLflow endpoint) to evaluate another model's outputs when numeric metrics are insufficient.

## Mental Models
- Think of **evaluation metrics as multi-dimensional, not a single score** — toxicity, readability, and custom domain metrics each surface different risks; compare models/configs across all of them together (e.g. via MLflow's parallel-coordinates charts).
- Use **"decouple storage from compute/deployment"** as the guiding principle — MinIO/object storage as an intermediate artifact store lets the model-serving layer (Seldon) scale independently from the training/tracking layer (MLflow).
- Treat **namespace strategy like disk partitioning upfront** — retrofitting namespace separation after the fact is far more painful than deciding at design time.

## Anti-patterns
- **Relying on a single numeric metric to judge model quality**: language quality is nuanced; combine toxicity + readability + custom metrics and still expect to review raw outputs.
- **Deferring namespace strategy indefinitely**: "we'll add namespaces later" accumulates resource-naming collisions and governance debt — the book explicitly calls out that leaving this as an afterthought causes headaches.
- **Coupling model-serving scale to the whole MLOps stack**: deploying everything (MLflow + model server) as one scaling unit wastes resources — Seldon's separate SeldonDeployment lets the model server scale independently.
- **Ignoring licensing changes in OSS MLOps tools**: Seldon Core moved parts of its licensing to Business Source License (BSL) in Jan 2024 — always verify current licensing terms before committing to production use.

## Code Examples
```python
def evaluate_positive_keywords(predictions, metrics):
    scores = []
    positive_words = ["good", "excellent", "great", "five", "recommend", "amazing", "awesome", "best"]
    for a_prediction in predictions:
        normalized_prediction = a_prediction.lower()
        words = normalized_prediction.split()
        positive_word_count = sum(words.count(word) for word in positive_words)
        scores.append(positive_word_count)
    return MetricValue(
        scores=list(scores),
        aggregate_results={"mean": np.mean(scores), "sum": np.sum(scores)},
    )

keywordMetric = make_metric(eval_fn=evaluate_positive_keywords, greater_is_better=True, name="keywordMetric")

results = mlflow.evaluate(
    model=model_info.model_uri,
    evaluators=["default"],
    model_type="text",
    model_config=parameters,
    data=eval_data,
    extra_metrics=[keywordMetric]
)
```
- **What it demonstrates**: defining and wiring a custom MLflow evaluation metric alongside the built-in default evaluators.

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sd-model-depl
spec:
  protocol: v2
  predictors:
  - graph:
      name: transformer
      implementation: HUGGINGFACE_SERVER
      modelUri: s3://mlflow/<path-to-artifact>
      parameters:
      - name: task
        type: STRING
        value: text-generation
      envSecretRefName: seldon-init-container-secret
    name: default
    replicas: 1
```
- **What it demonstrates**: deploying an MLflow-tracked model artifact (from MinIO/S3-compatible storage) as a standard Seldon Core inference server.

```yaml
apiVersion: machinelearning.seldon.io/v1alpha2
kind: SeldonDeployment
metadata:
  name: sd-model-depl
spec:
  protocol: v2
  predictors:
  - componentSpecs:
    - hpaSpec:
        maxReplicas: 3
        metrics:
        - resource:
            name: cpu
            targetAverageUtilization: 75
          type: Resource
        minReplicas: 1
      spec:
        containers:
        - image: seldonio/mlserver:1.3.5-huggingface
          resources:
            requests:
              cpu: '0.5'
    graph:
      name: lamini-flan-t5
      implementation: HUGGINGFACE_SERVER
      modelUri: s3://mlflow/<path>
    name: scale-example
```
- **What it demonstrates**: `SeldonDeployment` with native HPA integration — scaling model server replicas by CPU utilization, independent of the rest of the pipeline.

## Reference Tables
| Model Server | Protocol | Notable Strengths |
|---|---|---|
| Seldon Core | Open Inference Protocol V2 | Prepackaged servers (MLflow, SKLearn, TF, XGBoost); language wrappers; CRD-native; Prometheus/Jaeger integration |
| KServe | Open Inference Protocol V2 | Broad vendor support (AMD, AWS, NVIDIA); native K8s scaling/rollout; wide framework support |
| Google Vertex AI | Managed | AutoML, SDK/IaC integration, Google's own LLMs (Gemini/Gemma/Imagen) |
| AWS Bedrock | Managed | Wide foundational model catalog; integrates with SageMaker/CloudWatch |
| Azure AI | Managed | OpenAI model integration (GPT 3.5/4, DALL-E); Prompt Flow |

## Worked Example
**Metrics comparison + Seldon deployment walkthrough:** Recreate/reuse the `mlflow` minikube profile → install MLflow via Helm with extended timeout env vars → define `model_names = ["gpt2", "MBZUAI/LaMini-Flan-T5-77M"]` with 4 parameter-set combinations (max_length 20/50 × temperature 0.1/0.9) → run `model-metrics-jobs.yaml` as a K8s Job → compare runs in MLflow UI using parallel-coordinates/contour charts (finds an outlier: LaMini-Flan-T5-77M at max_length=50/temp=0.1 scores ~4x average toxicity) → promote a chosen model version to a "Staging" registry entry → set up MinIO user/secret for Seldon, install Istio + Seldon Core via Helm → copy the chosen run's MLflow artifact path, rewrite `mlflow-artifacts:/...` to `s3://mlflow/...`, deploy as a `SeldonDeployment` with `HUGGINGFACE_SERVER` → deploy a small client app that calls the Seldon V2 inference endpoint and prints the model's response.

## Key Takeaways
1. Combine MLflow's default metrics (toxicity, ARI, Flesch-Kincaid, ROUGE) with custom metrics for a fuller picture of text-generation quality — no single number suffices.
2. Seldon Core (and KServe as an alternative) standardizes model serving via the Open Inference Protocol V2, decoupling serving scale from the rest of the MLOps pipeline.
3. SeldonDeployment integrates natively with Kubernetes HPA — model-server autoscaling uses the same primitives as any other workload (Ch2).
4. Decide your namespace strategy deliberately and early — retrofitting is costly.
5. Always verify current OSS licensing (e.g. Seldon's BSL shift) before committing to a tool for production.

## Connects To
- **Ch2**: SeldonDeployment's HPA integration directly reuses the autoscaling mechanisms introduced there.
- **Ch5**: continues the MLflow pipeline setup, now adding evaluation depth and the deployment stage.
- **Ch7**: the MinIO/object-storage pattern used here as Seldon's model source is explored in depth as part of the broader data-lake strategy.
