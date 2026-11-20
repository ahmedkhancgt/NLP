# Chapter 6: Bayesian Classifier with Apache Spark on Cloud Dataproc

## Core Idea
Cloud Dataproc provides managed, ephemeral Hadoop/Spark clusters that separate compute from storage (data stays on Cloud Storage, not HDFS), letting a two-variable Naive/direct Bayesian classifier (distance + departure delay) be developed interactively in JupyterLab, scaled up on demand with preemptible workers, and productionized as a serverless Spark batch job.

## Frameworks Introduced
- **Ephemeral Dataproc clusters (jobs, not clusters)**: spin up a cluster, run the job, delete it — never keep a Hadoop cluster always-on in the cloud.
  - When to use: any Hadoop/Spark/Pig/Hive workload on GCP.
  - How: `gcloud dataproc clusters create` with `--enable-component-gateway` (for secure web UIs to HDFS NameNode/JupyterLab) and `--optional-components JUPYTER`; keep data on Cloud Storage (not HDFS) so the cluster can be deleted and recreated freely; resize with `gcloud dataproc clusters update --num-workers --num-secondary-workers` (preemptible) instead of fixed capacity; delete with `gcloud dataproc clusters delete`.
- **Histogram equalization for adaptive quantization**: borrow an image-processing technique to bin continuous variables so each bin has roughly equal population.
  - When to use: discretizing a skewed continuous variable (e.g., departure delay has a long tail) for a bin-based classifier.
  - How: use `approxQuantile()` (Spark) or `APPROX_QUANTILES` (BigQuery, Ch. 5) to find bin edges at evenly spaced percentiles (e.g., `np.arange(0, 1.0, 0.2)` for 5 bins); for a variable with a dominant mode (departure delay, >75% ≤0), hand-pick fine-grained thresholds in the interesting range instead (10–20 minutes in 1-minute steps) rather than relying on automatic quantile bins.
- **Serverless Spark batch jobs**: submit a Spark script without managing any cluster at all.
  - When to use: scheduled/production Spark jobs where you want zero infrastructure management, the Dataproc analogue of what Dataflow is for Beam.
  - How: `gsutil cp script.py gs://bucket/` then `gcloud beta dataproc batches submit pyspark gs://bucket/script.py -- --args`; Dataproc provisions, runs, and tears down infrastructure automatically.
- **Dataproc Workflow Templates**: declaratively define an ephemeral managed cluster + job sequence that is created, run, and deleted as one unit.
  - When to use: orchestrated multi-step Hadoop/Spark pipelines that need a cluster only for their duration.
  - How: `gcloud dataproc workflow-templates create` → `set-managed-cluster` (with initialization actions, preemptible workers) → `add-job` (per step) → `instantiate` (runs everything and tears the cluster down).

## Key Concepts
- **MapReduce**: a two-phase programming model (map: emit intermediate key-value pairs; reduce: merge values sharing a key) for distributed processing on a cluster, originally designed for a low-bisection-bandwidth environment where data is presharded onto the same nodes that process it.
- **Apache Hadoop ecosystem**: HDFS (distributed storage) + YARN (scheduler) + MapReduce engine, extended by higher-level tools — Pig (Pig Latin scripting → MapReduce), Hive (SQL over distributed storage), Spark (in-memory, DAG-optimized, much faster for iterative/aggregation workloads).
- **Cloud Dataproc**: GCP's managed Hadoop/Spark/Pig/Hive/Presto service; offers serverless execution specifically for Spark (as of this writing); shares backend services with Dataflow.
- **Preemptible/secondary workers**: discounted (~20% of standard price at time of writing) Compute Engine instances that can be reclaimed with short notice; well-suited to fault-tolerant Hadoop/Spark workloads for large, cheap, temporary capacity boosts.
- **Bayesian classification (direct conditional-probability binning)**: estimate P(on-time | distance bin, delay bin) directly by counting outcomes within each bin of a quantized 2D grid — simple but requires large enough populations per bin; contrasted with Naive Bayes, which factors variables as independent and multiplies per-variable conditional probabilities, scaling better to many variables.
- **Autoscaling policy (Dataproc)**: a YAML-defined policy (`workerConfig`/`secondaryWorkerConfig` min/max instances, `yarnConfig` scale-up/down factors and cooldown) registered via `gcloud dataproc autoscaling-policies import` and attached to a cluster at creation.

## Mental Models
- **Compute/storage separation changes "best practice" Hadoop operation**: the original MapReduce rationale (preshard data onto compute nodes because network is slow) no longer applies inside a GCP data center's petabit-per-second network — so keep data on Cloud Storage, not HDFS, and treat clusters as disposable.
- **"If you can do serverless, do serverless"**: among submitting to a running cluster, workflow templates, Cloud Composer/Airflow orchestration, and serverless Spark batches, prefer the option requiring the least infrastructure fiddling.
- **Occam's razor applied to model complexity**: the two-variable Bayes model performed essentially the same as the single-variable threshold from Chapter 5 — cites Netflix's $1M Prize-winning algorithm never being put into production as the canonical real-world example of this trade-off.
- **Independence check before separate binning**: verify two candidate predictor variables are roughly independent (via a hexbin joint-distribution plot and low Pearson correlation, e.g., 0.07) before choosing their quantization thresholds separately — otherwise joint effects would be missed.

## Anti-patterns
- **Keeping a Hadoop cluster always-on in the cloud**: wastes money since cluster creation takes under two minutes; the on-premises "always-on because data lives on HDFS" justification doesn't apply once data lives on Cloud Storage.
- **SSH-ing into cluster nodes to install software or debug**: fragile and insecure (requires exposing an external IP); use `--enable-component-gateway` web interfaces and initialization-action scripts instead.
- **Using BigQuery/Pandas-sized sampling assumptions on Spark dataframes unnecessarily**: a Spark dataframe doesn't need to fit in memory, but plotting libraries (seaborn) require a Pandas dataframe, so sampling (`df.sample(...).toPandas()`) is still needed only at the visualization boundary, not throughout the pipeline.
- **Equalizing bins uniformly across a variable with a dominant mode**: naive histogram equalization on departure delay put >75% of flights (all on-time/early) into bins with no interesting signal, hiding the one bin that mattered — fixed by hand-selecting fine-grained thresholds (10–20 min) in the known interesting range instead.
- **Developing and testing at full data scale before validating correctness**: the chapter explicitly develops on 1/25th (one JSON shard) of the data on a small cluster first, only scaling to the full dataset and a 20-worker cluster once the code is verified.

## Code Examples
```bash
gcloud dataproc clusters create ch6cluster \
  --enable-component-gateway --region us-central1 --zone us-central1-a \
  --master-machine-type n1-standard-4 --num-workers 2 \
  --worker-machine-type n1-standard-4 --image-version 2.0 \
  --optional-components JUPYTER --scopes https://www.googleapis.com/auth/cloud-platform
```
- **What it demonstrates**: a single command stands up a fully configured, Jupyter-enabled Dataproc cluster with secure web-interface access — no manual Hadoop/YARN setup.

```python
distthresh = flights.approxQuantile('distance', list(np.arange(0, 1.0, 0.2)), 0.02)
distthresh[-1] = float('inf')
delaythresh = range(10, 20)  # hand-picked fine-grained range instead of equalized bins
```
- **What it demonstrates**: combining automatic quantile-based equalization (for a well-behaved variable, distance) with manually chosen fine-grained thresholds (for a skewed variable, departure delay) in the same model.

```python
bdf = flights[(flights['distance'] >= distthresh[m]) & (flights['distance'] < distthresh[m+1])
             & (flights['dep_delay'] >= delaythresh[n]) & (flights['dep_delay'] < delaythresh[n+1])]
ontime_frac = (bdf.agg(F.sum('ontime')).collect()[0][0] /
               bdf.agg(F.count('ontime')).collect()[0][0])
```
- **What it demonstrates**: computing the direct conditional on-time probability per (distance, delay) bin — the core of the simple Bayesian classifier before converting it into a per-distance-bin delay-threshold lookup table.

```bash
gsutil cp bayes_on_spark.py gs://$BUCKET/
gcloud beta dataproc batches submit pyspark \
   --project=$(gcloud config get-value project) --region=$REGION \
   gs://${BUCKET}/bayes_on_spark.py -- --bucket ${BUCKET} --debug
```
- **What it demonstrates**: fully serverless Spark execution — no cluster to create, resize, or delete; Dataproc manages all infrastructure for the duration of the job.

## Reference Tables
| Distance bin (miles) | Delay threshold (min) — full dataset |
|---|---|
| [31, 328] | 14 |
| [328, 541] | 15 |
| [541, 802] | 15 |
| [802, ∞) | 17 |

| Spark job deployment option | Infrastructure management |
|---|---|
| Submit to already-running cluster | Must manage cluster capacity/contention yourself |
| Workflow template (managed cluster) | Cluster created/deleted automatically per template run |
| Cloud Composer (Airflow) orchestration | Managed Airflow triggers workflow templates as part of a larger DAG |
| Serverless Spark batch | No cluster at all — least infrastructure fiddling |

| Dataflow vs. Dataproc | Choose when |
|---|---|
| Dataproc | You need Hadoop-ecosystem tools (Spark/Pig/Hive) or require some on-prem/infra control |
| Dataflow | Pure Beam pipelines, especially streaming (much stronger streaming support) |

## Worked Example
Builds a two-variable Bayesian classifier for the flight-delay decision: (1) spin up a 3-node Dataproc cluster with Jupyter enabled; (2) in JupyterLab, verify distance and departure delay are roughly independent via a BigQuery-then-Spark-SQL hexbin joint plot (Pearson r ≈ 0.07); (3) develop on 1/25 of the data (one JSON shard) using `spark.read.json` + Spark SQL joined against the `trainday` table to exclude test days; (4) quantize distance into 5 histogram-equalized bins via `approxQuantile`, but hand-pick departure-delay thresholds in the 10–20 minute range since automatic equalization hid the interesting tail; (5) compute the on-time fraction per (distance, delay) bin, then collapse this into a lookup table of one delay threshold per distance bin by finding, for each distance bin, the delay value whose on-time fraction is closest to 0.70; (6) resize the cluster to 20 workers (5 standard + 15 preemptible) to rerun on the full dataset, producing a refined lookup table ([31,328]→14min, [328,541]→15min, [541,802]→15min, [802,∞)→17min); (7) evaluate against held-out test days in BigQuery, finding the two-variable model cancels about the same number of meetings (275k vs. 270k) as the Chapter 5 single-variable model, while hitting the 30% accuracy target with finer-grained, distance-aware thresholds; (8) productionize by converting the notebook to a standalone Python script (`bayes_on_spark.py`) and running it via `gcloud beta dataproc batches submit pyspark` for fully serverless execution.

## Key Takeaways
1. Treat Dataproc clusters as ephemeral — create for a job, delete when done; keep data on Cloud Storage, not HDFS, to make this painless.
2. Use histogram equalization / quantile-based binning for well-behaved variables, but hand-pick fine-grained thresholds for skewed variables where automatic equalization would hide the interesting range.
3. Verify candidate predictor variables' independence (joint distribution plot, correlation coefficient) before choosing their quantization thresholds separately.
4. Develop and debug on a small data sample and small cluster first; scale cluster size and data volume together only once the logic is verified.
5. Use preemptible/secondary workers for large, cheap, temporary capacity during big Spark jobs — fault-tolerant Hadoop workloads handle their reclamation gracefully.
6. Prefer serverless Spark batch submission for production jobs when available; it requires the least infrastructure management of all the orchestration options.
7. Apply Occam's razor: when a more complex multi-variable model performs no better than a simpler one on the primary metric, prefer the simpler model unless a secondary benefit (finer-grained decisions, interpretability) justifies the complexity.

## Connects To
- **Ch 2**: Reapplies the scaling-up/scaling-out/data-in-place framework from Chapter 2 to justify keeping data on Cloud Storage rather than HDFS.
- **Ch 5**: Builds directly on the `dsongcp.trainday` split and the single-variable empirical-threshold model, comparing results against it.
- **Ch 7**: Explicitly sets up Chapter 7's move to a more sophisticated Spark ML (logistic regression) approach after this simple binning model shows no improvement.
- **Ch 2 (Cloud Run/Scheduler)**: The serverless Spark batch job can be scheduled via the same Cloud Run + Cloud Scheduler pattern established in Chapter 2.
