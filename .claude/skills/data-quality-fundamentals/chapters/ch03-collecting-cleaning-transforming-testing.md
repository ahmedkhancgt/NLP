# Chapter 3: Collecting, Cleaning, Transforming, and Testing Data

## Core Idea
Data quality must be managed at each discrete stage of the pipeline — collection (entrypoint), cleaning, transformation (normalization/ETL), and testing — using stage-appropriate tools, because each stage has distinct failure modes and a different latency/throughput profile.

## Frameworks Introduced
- **Entrypoint**: the initial point of contact where outside-world data (application logs, API responses, sensor data) enters your pipeline, analogous to Docker's ENTRYPOINT or a program's main method.
  - When to use: as the first place to apply data quality thinking — data here is at its most raw/heterogeneous.
  - How: classify incoming data into application logs, API responses, or sensor data, each with distinct structure/timestamp/failure characteristics.
- **Circuit Breaker pattern (for data pipelines)**: pipelines stop running ("circuit open") when data doesn't meet quality thresholds, preventing bad data from mixing with good data downstream.
  - When to use: for the most critical data incidents where propagating bad data would be costly (freshness, volume, distribution thresholds).
  - How: requires (per Sandeep Uttamchandi) data lineage, data profiling across the pipeline, and automated triggering from profiling results; implement via Airflow's `catchup=False`, `LatestOnlyOperator`, or custom Python breaks.
- **Data Testing (known-unknowns coverage)**: validating organizational assumptions about data (uniqueness, not-null, volume, distribution, known invariants) before/during production — the authors estimate traditional testing covers only ~20% of possible data quality issues.
  - When to use: for predictable, nameable risks (known unknowns); not a substitute for monitoring unknown unknowns (see Ch 4).
  - How: load transformed data into staging, run threshold tests, alert + halt pipeline on failure.

## Key Concepts
- **Application logs**: timestamped, leveled (INFO/WARN/ERROR) records of software actions, used for diagnostics or auditing.
- **API responses**: semi-structured data (often JSON) carrying response codes (e.g., HTTP 200/404/500) whose meaning is use-case dependent.
- **Sensor data**: often noisy, unable to self-report failure (fails silently rather than erroring), prioritizes throughput for ML uses and latency for real-time inference.
- **Syntactic ambiguity**: the same data presented differently (different field names/types for the same metric) — a surface-level friction.
- **Semantic ambiguity**: disagreement about what a data field actually means/represents — more dangerous, can misrepresent business metrics.
- **ETL vs. ELT**: ETL transforms data in staging before loading to production (safer, slower); ELT loads first then transforms in the target system (faster, riskier without strong testing).
- **Schema checking / type coercion**: validating structure matches expectations, and explicitly/implicitly converting values between types (risk of silent precision loss, e.g., integer truncation).

## Mental Models
- Think of the data pipeline in four sequential quality gates: collect → clean → transform → test, each needing distinct techniques.
- Batch processing optimizes throughput (gather lots, tolerate delay); streaming optimizes latency (real-time, tolerate lossiness) — data quality is inherently harder in streaming because testing-before-ingest doesn't work the same way.
- "Pipeline stopping" (circuit breakers / SQL check operators) should be reserved for high-stakes incidents — overuse blocks unrelated, perfectly fine jobs.

## Anti-patterns
- **Treating dbt/testing as a substitute for end-to-end observability**: a failing dbt test may indicate an upstream problem (e.g., a misconfigured Glue Lambda) that the test itself can't pinpoint — "limited visibility."
- **Test fatigue**: adding ill-conceived tests that developers later delete just to unblock CI, which erodes the signal value of testing entirely.
- **Ignoring time zone normalization**: skipping UTC conversion at collection time makes cross-event comparison permanently impossible (cites Y2K-style bugs).
- **Blind trust in ELT speed**: ELT's load-first approach increases throughput but lowers data quality guarantees if testing/monitoring isn't layered on top.
- **Overusing circuit breakers**: stopping entire pipelines for non-critical issues blocks unrelated high-quality jobs from flowing downstream.

## Code Examples
```sql
-- dbt singular test: fail if any order's total payment amount is negative
select order_id, sum(amount) as total_amount
from {{ ref('fct_payments') }}
group by 1
having not (total_amount >= 0)
```
```scala
// Deequ (Scala) — unit tests as a VerificationSuite
val verificationResult = VerificationSuite()
  .onData(data)
  .addCheck(
    Check(CheckLevel.Error, "unit testing my data")
      .hasSize(_ == 5)
      .isComplete("id")
      .isUnique("id")
      .isContainedIn("priority", Array("high", "low"))
      .hasApproxQuantile("numViews", 0.5, _ <= 10))
  .run()
```
```python
# Apache Airflow: SLA + sla_miss_callback
@task(sla=datetime.timedelta(seconds=10))
def sleep_20():
    time.sleep(20)
```
- **What it demonstrates**: three layers of testable guardrails — SQL-based model assertions (dbt), programmatic batch validation with built-in anomaly checks (Deequ), and orchestration-layer SLA enforcement (Airflow) that can halt downstream jobs.

## Reference Tables
| Tool | Language | Best For | Key Limitation |
|---|---|---|---|
| dbt tests | SQL (+ Jinja) | Per-model unit/integration tests tied to ELT | Technical debt; limited visibility into upstream breaks |
| Great Expectations | Python | Flexible, general-purpose tests + "Data Docs" reports, Slack alerts | Python-only; separate from orchestration/transform tooling |
| Deequ / PyDeequ | Scala (+ Python wrapper) | AWS/Spark-native, stateful metrics, built-in anomaly detection | Steep Scala learning curve; not true integration testing; no UI |

| Streaming Tool | Strength | Trade-off |
|---|---|---|
| AWS Kinesis | On-demand scaling, cost-efficient, deep AWS integration, multi-language SDK | Lower max throughput (thousands of records/sec), fixed 7-day retention |
| Apache Kafka | Very high throughput (~30,000 records/sec), highly configurable, open source community | Higher learning curve, Java-centric |

## Worked Example
A four-step data-quality test workflow using Apache Airflow circuit breakers: after a metrics update completes, integrity tests run before downstream jobs execute (Figure 3-3's "circuit closed/open" states). If upstream data is inaccurate, the circuit opens and the pipeline halts — preventing false-positive notifications from reaching analysts and avoiding costly backfills. Concretely, Airflow users set an SLA via `datetime.timedelta` on a Task/Operator; if the task overruns, Airflow surfaces an "SLA missed" event in the UI or fires a custom `sla_miss_callback` with `dag`, `task_list`, `blocking_task_list`, `slas`, and `blocking_tis` parameters, which can notify Slack/Teams/email or trigger a Python-coded pipeline halt.

## Key Takeaways
1. Classify entrypoint data (logs, API responses, sensor data) by structure, timestamp discipline, and failure mode before designing collection-stage quality checks.
2. Normalize aggressively at the first transformation layer: resolve time zones to UTC, coerce types explicitly, and distinguish syntactic from semantic ambiguity through documentation.
3. Choose ETL over ELT when you need pre-production validation; choose ELT only if you compensate with strong monitoring/testing layered on top.
4. Traditional testing (dbt, Great Expectations, Deequ) catches known unknowns (~20% of issues per the authors) — budget separately for monitoring unknown unknowns.
5. Reserve circuit breakers and SQL check operators for genuinely critical data incidents; overuse erodes trust and blocks healthy pipelines.
6. For streaming (Kinesis/Kafka), apply lower-latency checks (schema comparison, byte-volume tracking) rather than throughput-heavy checks like full data drift analysis.

## Connects To
- **Ch 2**: Builds directly on warehouse/lake metadata and ETL/data integration concepts introduced there.
- **Ch 4**: Explicitly hands off "unknown unknowns" to monitoring and anomaly detection, since testing alone covers only known unknowns.
- **Ch 5**: SLAs introduced here for Airflow tasks are generalized into full SLA/SLO/SLI frameworks for data reliability architecture.
- **CI/CD (software engineering)**: circuit breakers and SQL check operators are explicitly modeled on CI/CD deployment-gating practices.
