# Chapter 10: Getting Ready for MLOps with Vertex AI

## Core Idea
Production ML requires converting notebook prototypes into pure-Python, version-controlled, single-entry-point pipelines orchestrated through the Vertex AI SDK — enabling automated retraining, predefined custom data splits, model-agnostic swapping (custom Keras vs. AutoML vs. XGBoost), Vizier-based hyperparameter tuning done cheaply on partial data/epochs, and built-in Shapley-based explainability without any model code changes.

## Frameworks Introduced
- **Notebook-to-pipeline conversion**: extract all model code from a Jupyter Notebook into a plain `model.py` + a separate `train_on_vertexai.py` orchestrator.
  - When to use: moving any prototype model toward production.
  - How: `model.py` exposes a `train_and_evaluate()` function and reads data paths from Vertex AI's injected environment variables (`AIP_MODEL_DIR`, `AIP_TRAINING_DATA_URI`, `AIP_VALIDATION_DATA_URI`, `AIP_TEST_DATA_URI`), falling back to sensible defaults when run standalone for development; keep using the notebook only as a thin wrapper (`import model; model.read_dataset(...)`, with `%autoreload`) for ad hoc exploration, never as the production artifact.
- **Vertex AI training pipeline (5 steps)**: load managed dataset → set up training infra → train via `model.py` → find/create endpoint → deploy.
  - When to use: any production training+deployment flow that needs to be automatable (CI/CD, data-arrival triggers).
  - How: `aiplatform.TabularDataset.create(gcs_source=...)` → `aiplatform.CustomTrainingJob(script_path="model.py", container_uri=train_image, model_serving_container_image_uri=deploy_image)` → `job.run(dataset=..., predefined_split_column_name='data_split', machine_type=..., accelerator_type=..., accelerator_count=...)` → look up or create an `aiplatform.Endpoint` by display name → `model.deploy(endpoint=..., traffic_split={"0": 100}, ...)`.
- **Predefined (custom) data split for a managed dataset**: replicate the date-based train/validate/test split inside Vertex AI's managed-dataset abstraction.
  - When to use: whenever rows are correlated (same recurring issue since Ch. 5) and Vertex AI's default random 80/10/10 split would leak information.
  - How: add a `data_split` column (`TRAIN`/`VALIDATE`/`TEST`) to the exported dataset, derived from the `is_train_day` split further subdivided via `FARM_FINGERPRINT`; pass `predefined_split_column_name='data_split'` to `job.run()`.
- **Cheap hyperparameter tuning pipeline**: tune hyperparameters on a fraction of data/epochs, then run only the winning configuration to completion.
  - When to use: tuning model architecture/training hyperparameters (batch size, bucket count, hidden layer sizes) without paying full-training cost per trial.
  - How: parameterize every hardcoded value in `model.py` as a CLI argument with a sensible default; add `--skip_full_eval` and a reduced `--num_examples`/epoch count for trial runs; use "virtual epochs" (`steps_per_epoch = NUM_EXAMPLES // batch_size`) so metrics stay comparable across different batch sizes; report per-epoch metrics via the `cloudml-hypertune` package's `HyperTune().report_hyperparameter_tuning_metric(...)` inside a Keras callback so Vizier can early-stop unpromising trials; define the search with `aiplatform.HyperparameterTuningJob(custom_job=..., metric_spec={'val_rmse': 'minimize'}, parameter_spec={...}, max_trial_count=N, parallel_trial_count=M)`; after the job completes, extract the best trial's parameters and rerun `train_custom_model(..., extra_args=best_params)` to full completion.
- **Post-hoc explainability (Shapley) without model-code changes**: add feature-attribution explanations to an already-trained model purely at deployment time.
  - When to use: any tabular model deployed to Vertex AI where you want per-prediction feature attribution.
  - How: inspect the SavedModel's serving signature via `saved_model_cli show --tag_set serve --signature_def serving_default --dir $model_dir`; build an `explanation-metadata.json` mapping each input/output tensor name; pass `--explanation-method=sampled-shapley --explanation-path-count=10 --explanation-metadata-file=...` to `gcloud ai models upload`; call the deployed endpoint's `:explain` method instead of `:predict` to get per-feature attribution values alongside the prediction.

## Key Concepts
- **MLOps requirements**: (1) model code under version control as text files, not notebooks; (2) the entire dataset→train→deploy process driven by code with a single entry point; (3) retraining triggerable by code changes or new data arrival; (4) easy monitoring/continuous evaluation with the ability to trigger retraining on metric or feature-distribution drift.
- **Virtual epochs**: measuring training progress by number of examples processed (not literal epochs), via `steps_per_epoch = NUM_EXAMPLES // batch_size`, so hyperparameter trials with different batch sizes remain comparable and so partial/full training runs can share the same evaluation cadence.
- **Vizier**: Vertex AI's built-in Bayesian-optimization hyperparameter tuner (default algorithm; can switch to grid search), supporting integer (linear or log scale), categorical, and other parameter spec types.
- **Global importance vs. local explanation**: AutoML's feature-importance chart (global, aggregate) found airport latitude/longitude unimportant, yet Shapley attribution on a specific GST→JNU flight (local, instance-level) found airport longitude the single most important feature — a feature can be unimportant on average but decisive for specific instances.
- **Why hyperparameter tuning trials should be short and on a held-out validation set, not the test set**: (1) expensive — you only need to rank trials, not finish each one; (2) wasteful — if training curves don't cross, a better config stays better throughout, so stop early and run more trials instead of longer ones; (3) wrong — evaluating tuning trials on the test set would contaminate it for final, unbiased evaluation.

## Mental Models
- **"Throw away the notebook" discipline**: notebooks are for prototyping and future ad hoc work only; the moment code needs to be productionized, it becomes the canonical `.py` file, and the notebook becomes a thin importer of that module (`%autoreload` keeps it in sync during continued iteration).
- **Model-framework-agnostic MLOps**: switching from a custom Keras `CustomTrainingJob` to `AutoMLTabularTrainingJob` (or XGBoost/PyTorch/sklearn) requires changing only the job-creation step — the dataset loading, endpoint lookup, and deployment code stay identical, so the ops team's automation barely changes regardless of ML framework choice.
- **Narrow the hyperparameter search iteratively**: if an optimal value lands at the edge of its searched range (e.g., `nbuckets=10` when the range was 5–10), re-run the search with a shifted range (e.g., 10–15) rather than assuming the boundary value is truly optimal.
- **Diminishing returns on tuning once the baseline guess was reasonable**: hyperparameter tuning here only moved RMSE from 0.196 to 0.195 — the lesson is that a well-reasoned initial guess (informed by domain knowledge, like FAA's ~36 air corridor grid cells suggesting ~6 buckets) can already be close to optimal.

## Anti-patterns
- **Deploying a Jupyter Notebook directly to production** (even though Vertex AI technically allows it): breaks version control (binary+text mixed `.ipynb` files don't diff/merge well in git) and can't be cleanly automated/retriggered.
- **Hardcoding any value in `model.py`**: every value that might ever need tuning (embedding dimension, bucket count, hidden layer sizes, batch size) should be a parameterized CLI argument with a sane default.
- **Running hyperparameter tuning trials to full completion on the full dataset**: wastes budget that should go toward more trials; use partial data/epochs instead, since better configurations should stay better throughout training if training curves don't cross.
- **Evaluating hyperparameter tuning trials on the test set**: contaminates the test set, making final evaluation no longer independent — always use a validation split for tuning decisions.
- **Trusting only a model's global feature-importance ranking**: AutoML's aggregate importance said airport location barely mattered, but Shapley local explanation on a specific flight showed it was the dominant feature for that instance — always consider both global and per-instance explanations.

## Code Examples
```python
OUTPUT_MODEL_DIR = os.getenv("AIP_MODEL_DIR")
TRAIN_DATA_PATTERN = os.getenv("AIP_TRAINING_DATA_URI")
EVAL_DATA_PATTERN = os.getenv("AIP_VALIDATION_DATA_URI")
TEST_DATA_PATTERN = os.getenv("AIP_TEST_DATA_URI")
if not TRAIN_DATA_PATTERN:
    TRAIN_DATA_PATTERN = 'gs://{}/ch9/data/train*'.format(BUCKET)
```
- **What it demonstrates**: the environment-variable contract between `model.py` and Vertex AI's managed training environment, with graceful fallback for local development.

```python
job = aiplatform.CustomTrainingJob(
    display_name='train-{}'.format(model_display_name), script_path="model.py",
    container_uri=train_image, model_serving_container_image_uri=deploy_image)
model = job.run(dataset=data_set, predefined_split_column_name='data_split',
    model_display_name=model_display_name, machine_type='n1-standard-4',
    accelerator_type=aip.AcceleratorType.NVIDIA_TESLA_T4.name, accelerator_count=1)
model.deploy(endpoint=endpoint, traffic_split={"0": 100}, machine_type='n1-standard-2')
```
- **What it demonstrates**: the complete, automatable Vertex AI train-and-deploy pipeline in pure Python, with a custom (not random) data split and GPU-accelerated training.

```python
hparam_job = aiplatform.HyperparameterTuningJob(
    display_name='hparam-{}'.format(model_display_name), custom_job=trial_job,
    metric_spec={'val_rmse': 'minimize'},
    parameter_spec={
        "train_batch_size": hpt.IntegerParameterSpec(min=16, max=256, scale='log'),
        "nbuckets": hpt.IntegerParameterSpec(min=5, max=10, scale='linear'),
        "dnn_hidden_units": hpt.CategoricalParameterSpec(values=["64,16", "64,16,4", "64,64,64,8", "256,64,16"])
    },
    max_trial_count=10, parallel_trial_count=2, search_algorithm=None)  # Bayesian
```
- **What it demonstrates**: specifying a mixed-type hyperparameter search space (log-scaled integer, linear integer, categorical) for Vizier's Bayesian optimization.

```bash
gcloud ai models upload ... \
     --explanation-method=sampled-shapley \
     --explanation-path-count=10 \
     --explanation-metadata-file=explanation-metadata.json
# then POST to .../endpoints/${ENDPOINT_ID}:explain instead of :predict
```
- **What it demonstrates**: enabling Shapley-based per-prediction feature attribution purely via deployment configuration, with zero changes to the trained model itself.

## Reference Tables
| Approach | RMSE |
|---|---|
| Custom wide-and-deep Keras model (Ch. 9, hand-picked hyperparameters) | 0.196 |
| AutoML Tables (same features, custom split, RMSE computed via BigQuery export) | 0.199 |
| Custom model after Vizier hyperparameter tuning | 0.195 |

| Hyperparameter tuning pitfall | Why it matters |
|---|---|
| Training trials to completion | Expensive — only ranking is needed, not a finished model |
| Full dataset per trial | Wasteful — use more trials, not longer ones |
| Evaluating on test set | Wrong — contaminates the independent final-evaluation set |

## Worked Example
Converts the Chapter 9 wide-and-deep model into a production MLOps pipeline: (1) extract notebook code into `model.py` (reading data patterns from `AIP_*` environment variables, with local-dev fallbacks) and verify it still trains correctly standalone; (2) write `train_on_vertexai.py` to load a `TabularDataset` from GCS, create a `CustomTrainingJob` pointing at `model.py`, run it on a GPU-accelerated machine with a predefined `data_split` column (TRAIN/VALIDATE/TEST derived via `FARM_FINGERPRINT` on `FL_DATE`), then look up or create a stable `Endpoint` and deploy the resulting model with 100% traffic; (3) swap in `AutoMLTabularTrainingJob` as an alternative to the custom Keras job (same dataset-loading/endpoint code unchanged), export its evaluated predictions to BigQuery to compute RMSE manually (0.199) since AutoML doesn't natively report RMSE, and compare against the custom model's 0.196; (4) parameterize every hardcoded value in `model.py` (embedding dimension, bucket count, batch size, hidden units) as CLI args with sensible defaults; (5) add `--skip_full_eval` and reduced `--num_examples`/epochs for cheap trial runs, and a `HpCallback` using `cloudml-hypertune` to report `val_rmse` per epoch so Vizier can early-stop bad trials; (6) define and run an `aiplatform.HyperparameterTuningJob` searching `train_batch_size` (log-scale 16–256), `nbuckets` (linear 5–10, informed by the FAA's ~36 air-corridor grid cells), and `dnn_hidden_units` (4 categorical architectures) across 10 trials with Bayesian search; (7) extract the best trial's parameters and rerun training to full completion on the full dataset, landing on RMSE 0.195 — a modest improvement confirming the original hand-picked hyperparameters were already close to optimal; (8) add explainability by extracting the model's serving signature, building `explanation-metadata.json`, redeploying with `--explanation-method=sampled-shapley`, and querying the `:explain` endpoint — discovering that for a specific intra-Alaska GST→JNU flight, airport longitude (not departure delay) was the dominant feature, despite AutoML's global importance ranking it as unimportant.

## Key Takeaways
1. Convert prototype notebook code into plain `.py` files with a single orchestrating entry point before considering a model production-ready — notebooks stay for ad hoc exploration only.
2. Read training/eval/test data paths and output directories from Vertex AI's injected `AIP_*` environment variables, with local-dev fallback defaults, so the same code runs identically in both environments.
3. Replicate any custom (non-random) train/validate/test split inside Vertex AI's managed dataset via a `predefined_split_column_name`, just as with BigQuery ML's custom split.
4. Switching ML frameworks (custom Keras ↔ AutoML ↔ XGBoost/PyTorch/sklearn) should require minimal changes to the surrounding MLOps pipeline — only the job-creation step differs.
5. Run hyperparameter tuning trials cheaply (partial data, short training, validation set only) and only train the winning configuration to completion.
6. Report per-epoch metrics during training (via `cloudml-hypertune`) so Vizier can early-stop unpromising trials and save budget.
7. Add explainability (Shapley) at deployment time with zero model-code changes, and always check both global (aggregate) and local (per-instance) feature importance — they can disagree sharply.

## Connects To
- **Ch 8 & 9**: Directly extends the BigQuery ML and TensorFlow models built there into a fully automated, production-ready pipeline; reuses and extends the custom date-based data split methodology from Ch. 5/8/9.
- **Ch 5**: The predefined-split logic (`FARM_FINGERPRINT`-based TRAIN/VALIDATE/TEST) directly extends the `trainday` methodology first introduced there.
- **Ch 11**: Explicitly hands off the unresolved time-windowed aggregate feature problem (average taxi-out time, moving averages) to the next chapter's real-time stream-processing solution.
- **Machine Learning Design Patterns** (Lakshmanan, Robinson, Munn): referenced for the Checkpoints/Virtual Epochs pattern underlying the hyperparameter-tuning shortcuts in this chapter.
