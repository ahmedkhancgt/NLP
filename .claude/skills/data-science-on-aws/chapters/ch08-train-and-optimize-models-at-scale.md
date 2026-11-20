# Chapter 8: Train and Optimize Models at Scale

## Core Idea
SageMaker Hyper-Parameter Tuning (Bayesian or random search) automates finding the best hyper-parameters, warm start reuses prior tuning results to go further for less cost, and SageMaker Distributed Training (data/model parallelism, parameter servers vs. all-reduce) plus I/O optimizations (ShardedByS3Key, Pipe mode, FSx for Lustre) scale training to large clusters efficiently.

## Frameworks Introduced
- **SageMaker Hyper-Parameter Tuning (HPT/HPO)**: automates exploring hyper-parameter space using Bayesian or random search (grid search is not supported — deemed too inefficient).
  - When to use: finding the best hyper-parameter combination for a given algorithm/dataset beyond manual guessing.
  - How: define `hyperparameter_ranges` (Categorical/Continuous/Integer, with Linear/Logarithmic/ReverseLogarithmic/Auto scaling), an `objective_metric_name`, then `HyperparameterTuner(estimator=..., strategy='Bayesian', max_jobs=100, max_parallel_jobs=10, early_stopping_type='Auto')`.
- **Warm Start HPT**: reuses a prior tuning job's results as a prior for a new tuning job.
  - When to use: narrowing a hyper-parameter range after an initial tuning job, or adapting to a changed dataset/algorithm version.
  - How: `WarmStartConfig(warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM or TRANSFER_LEARNING, parents={tuning_job_name})`.
- **SageMaker Distributed Training**: native support for data parallelism (shard dataset across instances) and model parallelism (shard model across instances), using parameter-server or all-reduce (Horovod-compatible) communication strategies.
  - When to use: data parallelism when the dataset doesn't fit on one instance but the model does (the common case, used here for BERT); model parallelism when the model itself is too large for one instance's memory.
  - How: set `instance_count>1` and `distribution={'smdistributed':{'dataparallel':{'enabled': True}}}` on the estimator.
- **Amazon FSx for Lustre / Amazon EFS**: POSIX-compatible, S3-backed distributed file systems for training clusters.
  - When to use: FSx for Lustre for high-throughput, cached, iterative access to large training datasets across a cluster (preferred here); EFS for broader POSIX/NFS compatibility and SageMaker Studio's shared notebook storage.
  - How: `FileSystemInput(file_system_id=..., file_system_type='FSxLustre'|'EFS', directory_path=..., file_system_access_mode='ro')` passed to `estimator.fit()`.
- **ShardedByS3Key**: input distribution strategy that spreads dataset part-files evenly across cluster instances instead of copying the full dataset to every instance.
- **Pipe Mode**: streams training data from S3 directly into the training process (vs. File mode's full upfront copy), using Unix FIFO files; best for 10GB–1TB+ datasets.

## Key Concepts
- **Random search vs. Bayesian optimization vs. grid search**: random search is fast/parallelizable but can miss optimal combos; Bayesian optimization uses a surrogate model + acquisition function to make an informed, partly-sequential search (more efficient but less parallelizable); grid search is exhaustive and too slow — SageMaker HPT does not support it.
- **Data parallelism vs. model parallelism**: data parallelism shards the dataset (model fits on one instance); model parallelism shards the model itself across instances (model doesn't fit on one instance).
- **Parameter server vs. all-reduce**: parameter servers store/share learned parameters (run on every instance for availability/failover); all-reduce uses a ring-like MPI communication pattern (Horovod-compatible) that scales better for very large clusters by avoiding parameter-server bottlenecks.
- **Early stopping (HPT-level)**: `early_stopping_type='Auto'` stops a tuning job's individual training runs early when they're not improving the objective metric, saving cost.
- **File mode vs. Pipe mode**: File mode copies the entire dataset to every instance before training starts (simple, fine for small data); Pipe mode streams data on demand (faster start, lower disk usage, better for huge datasets).

## Mental Models
- **"If you can't measure it, you can't improve it" (Drucker)**: frames the whole chapter — SageMaker Experiments/HPT exist to turn hyper-parameter selection from guesswork into a measured, iterative optimization process.
- **Sequential-vs-parallel trade-off in Bayesian tuning**: running fewer jobs in parallel (e.g., 10 at a time out of 100 total) lets Bayesian optimization learn from completed runs before launching more — full parallelism (100 at once) would forfeit this benefit.
- **Logarithmic-first, then linear-narrow**: when no prior knowledge exists for a hyper-parameter's useful range, explore logarithmically first to find the right order of magnitude, then narrow with a linear range (and reuse that narrowing via warm start).
- **I/O optimization ladder**: default File mode copies everything → ShardedByS3Key spreads files across instances → Pipe mode streams on demand → FSx for Lustre caches S3 reads across repeated epochs — pick the right rung based on dataset size and iteration count.

## Anti-patterns
- **Running grid search**: SageMaker HPT does not support it, and in general it's far slower than random/Bayesian search for large hyper-parameter spaces — avoid emulating it manually.
- **Running Bayesian optimization with full parallelism (no sequential batches)**: defeats the purpose of Bayesian search, since it can't use prior-run knowledge to inform subsequent choices.
- **Copying the entire dataset to every cluster instance (File mode) for very large datasets**: wastes startup time and disk space; use ShardedByS3Key and/or Pipe mode instead.
- **Adding more cluster instances to a non-parallelizable algorithm**: adds communication overhead without any training speedup.
- **Ignoring warm start when iterating on hyper-parameter ranges**: restarting tuning from scratch wastes the prior job's learned information and money.

## Code Examples
```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter
hyperparameter_ranges = {'learning_rate': ContinuousParameter(0.00001, 0.00005, scaling_type='Linear')}
objective_metric_name = 'validation:accuracy'
tuner = HyperparameterTuner(
    estimator=estimator, objective_type='Maximize',
    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=metrics_definitions,
    max_jobs=100, max_parallel_jobs=10,
    strategy='Bayesian', early_stopping_type='Auto')
tuner.fit(inputs={'train': s3_input_train_data, 'validation': s3_input_validation_data, 'test': s3_input_test_data})
```
- **What it demonstrates**: a full Bayesian HPT job exploring `learning_rate`, running 100 total jobs with 10 in parallel, auto-stopping unpromising runs.

```python
from sagemaker.tuner import WarmStartConfig, WarmStartTypes
warm_start_config = WarmStartConfig(
        warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
        parents={tuning_job_name})
hyperparameter_ranges = {'learning_rate': ContinuousParameter(0.00015, 0.00020, scaling_type='Linear')}
tuner = HyperparameterTuner(..., max_jobs=50, max_parallel_jobs=5,
                             strategy='Bayesian', warm_start_config=warm_start_config)
```
- **What it demonstrates**: narrowing the learning_rate search range using warm start, reusing the parent tuning job's learned prior.

```python
estimator = TensorFlow(entry_point='tf_bert_reviews.py', source_dir='src',
                        instance_count=3, instance_type='ml.p4d.24xlarge',
                        distribution={'smdistributed':{'dataparallel':{'enabled': True}}})
```
- **What it demonstrates**: enabling SageMaker's native all-reduce data-parallel distributed training across 3 multi-GPU instances with one config flag.

```python
estimator = TensorFlow(entry_point='tf_bert_reviews.py', source_dir='src', ..., input_mode='Pipe')
# In the training script:
if input_mode == 'Pipe':
    from sagemaker_tensorflow import PipeModeDataset
    dataset = PipeModeDataset(channel=channel, record_format='TFRecord')
```
- **What it demonstrates**: switching from File mode to Pipe mode to stream large TFRecord datasets directly from S3.

## Reference Tables
| HPT strategy | Speed | Parallelizable | Notes |
|---|---|---|---|
| Random search | Fast | Fully | May miss best combo |
| Bayesian optimization | Efficient | Partially (needs some sequential batches) | Uses surrogate model + prior runs |
| Grid search | Slow | Fully | Not supported by SageMaker HPT |

| Distributed strategy | Use when |
|---|---|
| Data parallelism | Dataset too big for one instance, model fits |
| Model parallelism | Model too big for one instance's memory |
| Parameter server | Need per-instance state/failover resilience |
| All-reduce (Horovod-compatible) | Large clusters where parameter-server overhead dominates |

| Warm start type | Use when |
|---|---|
| IDENTICAL_DATA_AND_ALGORITHM | Same data/algorithm, changing hyper-parameter ranges |
| TRANSFER_LEARNING | Updated dataset or newer algorithm version |

## Worked Example
The Ch. 7 BERT review classifier is tuned with a Bayesian HPT job (100 jobs, 10 parallel) exploring `learning_rate` in [0.00001, 0.00005] linearly, holding other hyper-parameters fixed; the best candidate reaches 0.9416 training accuracy (vs. 0.9394 manually chosen in Ch. 7) with `learning_rate=0.000017`, `train_batch_size=128`, `freeze_bert_layer=False`. A second warm-start tuning job (`IDENTICAL_DATA_AND_ALGORITHM`, 50 jobs/5 parallel) narrows the learning rate to [0.00015, 0.00020] but only achieves 0.9216 — worse than the parent — so the original parent's best candidate is kept. The training job is then scaled out to 3× `ml.p4d.24xlarge` instances using SageMaker's all-reduce data-parallel distributed training, with the ~45 TFRecord part-files sharded 15-per-instance via `ShardedByS3Key`, and Pipe mode streaming data directly from S3 to avoid long upfront data-copy delays.

## Key Takeaways
1. Use Bayesian optimization (not grid search, which SageMaker doesn't support) for efficient hyper-parameter tuning, running training jobs in small parallel batches so the Bayesian model can learn from prior runs.
2. Use warm start to reuse a previous tuning job's knowledge when narrowing ranges or updating data/algorithm — but verify the new result actually beats the parent before adopting it.
3. Choose data parallelism when the dataset (not the model) is the scaling bottleneck; reserve model parallelism for models too large for one instance.
4. Use ShardedByS3Key and Pipe mode (and FSx for Lustre for repeated-epoch caching) to avoid I/O bottlenecks at scale — File mode's full dataset copy doesn't scale to large datasets.
5. Use SageMaker's built-in all-reduce distributed training (Horovod-compatible) with a single `distribution` config flag rather than hand-rolling distributed communication.
6. Enable ENA/EFA-capable instance types and consider running in a single VPC for latency-sensitive, large-cluster distributed training.

## Connects To
- **Ch 7**: Directly tunes the BERT classifier trained there, extending its SageMaker Experiments lineage.
- **Ch 3**: SageMaker Autopilot's hyper-parameter selection (mentioned as a baseline source) is covered there.
- **Ch 9**: The hyper-parameter-tuned model is deployed, monitored, and auto-scaled next.
- **Ch 12**: VPC, subnet, and security group configuration referenced here for FSx/EFS is covered in depth.
