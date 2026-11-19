# Chapter 5: Process and Pipelines

## Core Idea
GenAI projects need an MLOps/LLMOps pipeline — tracking (version control for data/models), build-and-test (data curation, model tuning), deploy, and monitor-and-feedback — adapted from DevOps but accounting for the non-determinism and transfer-learning capability unique to LLMs; MLflow is a practical, easy-to-adopt tool for the tracking/registry stage on Kubernetes.

## Frameworks Introduced
- **MLOps pipeline stages**: Tracking (version control) → Build and test (data curation + model tuning) → Deploy → Monitor and feedback — mirrored across code, data, and model pipelines that feed each other.
  - When to use: as the organizing structure for any production GenAI delivery process, not just one-off experiments.
  - How: establish version control/lineage first, then iterate build→test loops before deploying, then close the loop with monitoring feeding back into tracking.
- **MLOps vs LLMOps distinction**: traditional ML models are deterministic and narrow (same input → same output, task-specific); LLMs are non-deterministic (same input → varying output) but have transfer learning (generalize beyond training task) and require alignment-aware evaluation.
  - When to use: when deciding how to adapt a standard MLOps pipeline for an LLM/GenAI project — metrics become approximations, not discrete adjustments.
  - How: budget for repeated/sampled evaluation rather than single-pass metric checks; account for alignment/safety evaluation alongside performance.
- **MLflow components as pipeline building blocks**: model registry + tracking server (parameters/metrics/artifacts), deployment server (serves OSS models and API-backed models like OpenAI/Anthropic), prompt-engineering UI.
  - When to use: as a single consolidated tool to avoid mixing many point solutions when starting out.
  - How: deploy via Helm (Bitnami chart) on Kubernetes, which also provisions MinIO object storage as the artifact backend.

## Key Concepts
- **Exploratory Data Analysis (EDA)**: understanding data quality, distributions, and anomalies before model work begins.
- **Data/Model Version Control**: specialized tooling (Git LFS, LakeFS, Delta Lake) needed because Git alone poorly handles large binary data/model files.
- **Model Registry**: tracks model versions, lineage, and associated metadata/metrics (MLflow's core registry function).
- **ConfigMap**: Kubernetes object storing environment-specific key-value configuration independent of application code.
- **Kubernetes Job**: a one-off, run-to-completion workload (vs. long-running Deployments), ideal for batch comparison/evaluation runs.
- **Experiment (MLflow)**: a named grouping of related runs (e.g. different model/parameter combinations) for comparison.

## Mental Models
- Think of the **data/model/code pipelines as three parallel, interlocking DevOps-like pipelines** that feed each other (data pipeline → model pipeline → application pipeline) — maturity in one benefits the others.
- Use **"baseline first, then iterate"** — always establish a measurable baseline (prompt/parameter combination) before comparing fine-tuning or new models, or comparisons become meaningless.
- Non-determinism means **metrics are approximations, not pass/fail gates** — design evaluation loops (multiple runs, sampling) rather than single deterministic test assertions.

## Anti-patterns
- **Using plain Git for large model/data files**: Git is built for source code diffs, not multi-GB binary artifacts — leads to bloated repos and poor lineage tracking; use Git LFS, LakeFS, or Delta Lake instead.
- **Skipping a registered baseline before tuning**: without a common starting point, you can't meaningfully compare fine-tuning progress or new model candidates over time.
- **Hardcoding server names/ports in application code**: instead inject via Kubernetes ConfigMaps so the same image/code works across dev/test/prod.
- **Treating LLM evaluation like deterministic ML evaluation**: expecting exact-match correctness metrics for LLM outputs ignores their inherent non-determinism; use approximate/sampled evaluation strategies instead.

## Code Examples
```bash
# Create a dedicated minikube profile with no memory cap for MLflow work
minikube start --cpus 4 --memory no-limit -p mlflow
minikube profile mlflow

# Install Helm, then MLflow (+ MinIO) via Bitnami chart
helm install my-mlflow oci://registry-1.docker.io/bitnamicharts/mlflow \
  --set tracking.auth.enabled=false,tracking.service.type=ClusterIP,tracking.service.ports.http=5000
```
- **What it demonstrates**: standing up a full MLflow tracking + MinIO artifact-storage stack on Kubernetes with a single Helm install.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gpt2-runs
spec:
  template:
    spec:
      containers:
      - name: gpt2-runs
        image: gpt2-runs:latest
        imagePullPolicy: IfNotPresent
        command: ["python3", "gpt2-runs.py"]
        env:
        - name: MLFLOW_SERVER_IP
          valueFrom:
            configMapKeyRef:
              name: mlflow-config
              key: mlflow-server.ip
        - name: MLFLOW_SERVER_PORT
          valueFrom:
            configMapKeyRef:
              name: mlflow-config
              key: mlflow-server.port
      restartPolicy: Never
      backoffLimit: 4
```
- **What it demonstrates**: a Kubernetes `Job` injecting MLflow server connection details via `ConfigMap` references, enabling environment-agnostic batch evaluation code.

```python
parameters_sets = [
    {"max_length": 128, "do_sample": True, "temperature": 0.1},
    {"max_length": 128, "do_sample": True, "temperature": 0.4},
    {"max_length": 128, "do_sample": True, "temperature": 0.9}
]

for pipeline_name, pipeline_object in pipelines.items():
    for parameters in parameters_sets:
        signature = mlflow.models.infer_signature(
            input_example,
            mlflow.transformers.generate_signature_output(pipeline_object, input_example),
            parameters,
        )
        run_name = pipeline_name + '(temp:' + str(parameters.get('temperature', '--')) + ')'
        with mlflow.start_run(run_name=run_name):
            for key, val in parameters.items():
                mlflow.log_param(key, val)
            model_info = mlflow.transformers.log_model(
                transformers_model=pipeline_object,
                artifact_path="text_generation",
                input_example=input_example,
                signature=signature,
            )
            results = mlflow.evaluate(
                model=model_info.model_uri,
                model_type='text',
                data=eval_data,
            )
```
- **What it demonstrates**: systematic grid comparison of a model across multiple parameter sets, logging each as a separate MLflow run for later comparison.

## Reference Tables
| Data Version Control Tool | Key Capability |
|---|---|
| Git LFS | Decouples large file storage from Git repo (pointer-based) |
| LakeFS | Git-like versioning for data; schema/format validation |
| Delta Lake | Most comprehensive: versioning, schema validation, cloud object storage integration |

## Worked Example
**End-to-end MLflow model comparison on Kubernetes:** Start a dedicated `mlflow` minikube profile → install Helm → deploy MLflow+MinIO via Bitnami Helm chart → create a `mlflow-config` ConfigMap with the tracking server's service name/port → port-forward the MLflow UI (`:5000`) → build the `gpt2-runs` container image (loads GPT-2 pipeline, three temperature settings, logs params/model/evaluation to MLflow for each) → submit as a Kubernetes `Job` → `kubectl wait --for=condition=complete job/gpt2-runs` → inspect results in the MLflow UI under "Transformers Model Comparison" experiment, comparing three runs by temperature and their evaluation outputs on shared eval prompts.

## Key Takeaways
1. GenAI pipelines need three interlocking tracks — code, data, model — each with its own tooling needs (data/model registries differ from source control).
2. MLflow (registry + tracking + deployment server) is a practical single-tool starting point for GenAI pipeline maturity, easy to stand up via Helm on Kubernetes.
3. Kubernetes `Jobs` + `ConfigMaps` are the natural primitives for running repeatable, parameterized batch evaluation workloads.
4. LLM/GenAI pipelines (LLMOps) differ from classic MLOps mainly in non-determinism, transfer learning, and heavier compute needs — design evaluation accordingly.
5. Establishing a baseline before tuning is essential for meaningful before/after comparison.

## Connects To
- **Ch4**: the parameter-comparison patterns here formalize the manual prompt/parameter experimentation done in Chapter 4 into a repeatable pipeline.
- **Ch6**: directly continues this chapter's MLflow setup into metrics-based evaluation and Seldon Core model deployment.
- **Ch7**: expands on MinIO (introduced here as MLflow's artifact store) into a full data-lake/storage discussion.
