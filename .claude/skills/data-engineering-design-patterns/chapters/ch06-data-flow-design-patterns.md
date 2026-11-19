# Chapter 6: Data Flow Design Patterns

## Core Idea
Generating business value (Ch 5) is only locally useful until it's coordinated into a broader flow; data flow patterns organize how tasks and pipelines sequence, merge (fan-in), branch (fan-out), and run concurrently — at both the data orchestration layer (cross-team, cross-pipeline) and the data processing layer (within a single job) — primarily to optimize restart/backfill boundaries and readability.

## Frameworks Introduced
- **Local Sequencer**: chain dependent tasks sequentially within one pipeline/job.
  - When to use: decomposing a monolithic job into smaller steps for readability and cheaper restarts.
  - How: order tasks by data dependency (B after A if B needs A's output); decide orchestration-layer vs. processing-layer boundary using separation of concerns, maintainability (cost of recomputing expensive steps on retry), and implementation effort (leveraging orchestrator-native task types).
- **Isolated Sequencer**: coordinate physically separate pipelines (often across teams).
  - When to use: producer and consumer pipelines are owned by different teams/responsibilities and shouldn't be merged.
  - How: define boundaries by producer/consumer or team ownership; trigger via data-based dependency (Readiness Marker — loosely coupled, consumer can swap datasets freely) or task-based dependency (direct trigger — tightly coupled, e.g., Airflow `ExternalTaskMarker`/`ExternalTaskSensor`, which also auto-propagates backfills).
- **Aligned Fan-In**: merge multiple parent branches only when all succeed.
  - When to use: partial-branch failures should block the merge (e.g., daily aggregate needs all 24 hourly partials).
  - How: define parallel branches (e.g., per-hour) that merge into a common task; combine results via UNION (vertical, same columns, more rows) or JOIN (horizontal, more columns); gains faster feedback loops and cheaper targeted backfills at the cost of infrastructure spikes and scheduling skew (child waits for the slowest parent).
- **Unaligned Fan-In**: merge branches even when some parents fail.
  - When to use: partial results are acceptable/preferred over blocking (e.g., release a partial daily cube rather than none).
  - How: relax the trigger rule (Airflow `TriggerRule.ALL_DONE`, AWS Step Functions `Catch`) so the child runs regardless of parent outcome; mark output completeness explicitly (a companion completeness table, metadata tags, or an `is_approximate`/`is_partial` flag) so consumers aren't misled.
- **Parallel Split**: fan out one parent task into multiple independent downstream branches.
  - When to use: the same upstream dataset feeds multiple independent consumers/outputs (e.g., dual-writing during a migration).
  - How: at orchestration layer, declare multiple downstream tasks from one parent; at processing layer, materialize/cache the shared intermediate dataset once (`.persist()` in Spark, a temp table in SQL) so it isn't recomputed per branch; guard against duplicate writes on retry with native write-dedup (Delta Lake `txnVersion`/`txnAppId`) or an idempotency pattern (Ch 4).
- **Exclusive Choice**: route execution down exactly one of several branches based on a condition.
  - When to use: mutually exclusive logic paths (e.g., switch job version on a cutover date, branch by schema shape).
  - How: add a condition-evaluator task before branching (Airflow `BranchPythonOperator`, Azure Data Factory "if condition" activity) or use if-else/switch in the processing layer directly; prefer fast metadata-based conditions over data-based ones to avoid extra processing cost.
- **Single Runner**: guarantee at most one concurrent execution of a pipeline.
  - When to use: the pipeline is stateful/incremental and concurrent runs would corrupt results (e.g., sequential sessionization).
  - How: set orchestrator concurrency to 1 (Airflow `max_active_runs=1` + `depends_on_past=True`); accept that backfills become slow/sequential by necessity.
- **Concurrent Runner**: allow multiple pipeline instances to run in parallel.
  - When to use: pipeline instances are independent (e.g., per-dataset or per-partition ingestion) and sequential execution causes unacceptable latency.
  - How: raise orchestrator concurrency above 1 (Airflow `max_active_runs=5`+); apply workload management/compute quotas to prevent one team's concurrency from starving others; watch for shared-state hazards (e.g., concurrent writes to a Dynamic Late Data Integrator state table).

## Key Concepts
- **Restart boundary**: the unit of work that should be able to fail and retry independently, without recomputing unrelated expensive steps — the main design driver for Local/Isolated Sequencer splits.
- **Data-based vs. task-based dependency**: loosely-coupled (Readiness Marker driven) vs. tightly-coupled (direct trigger) cross-pipeline dependency styles, trading evolution freedom for coordination simplicity.
- **Scheduling skew**: a Fan-In child task's start time is bound by its slowest parent branch.
- **Partial dataset / completeness flag**: explicit signaling (percentage, tag, `is_approximate`) that an Unaligned Fan-In output is incomplete.
- **Workload management**: allocating dedicated compute capacity per team/group so Concurrent Runner usage by one team doesn't starve others.
- **Data lineage**: tooling (e.g., OpenLineage) for tracking which datasets/pipelines consume a given pipeline's output — relevant to Isolated Sequencer's implicit data-based dependencies.

## Mental Models
- Decide sequencing granularity by asking "what should be able to restart/backfill independently?" and "is this a transactional unit?" (if two steps must always succeed/fail together, keep them as one task).
- Fan-In is a dial between strictness (Aligned: all-or-nothing, consistent but brittle) and availability (Unaligned: partial-but-flagged, available but requires completeness tracking).
- Fan-Out is a dial between simultaneity (Parallel Split: all branches run) and exclusivity (Exclusive Choice: exactly one branch runs) — the same "common parent" shape serves opposite purposes.
- Single vs. Concurrent Runner maps directly to whether pipeline instances are stateful/order-dependent (Single) or independent (Concurrent) — never loosen concurrency on a stateful pipeline just to reduce latency.

## Anti-patterns
- **Merging unrelated, high-latency operations into one orchestration task**: forces costly, unnecessary recomputation on every retry.
- **Using a task-based (trigger) dependency when pipelines' schedules diverge**: forces awkward skip-logic on one side; prefer data-based dependency when coupling should stay loose.
- **Running Aligned Fan-In with highly unbalanced parent branch durations**: creates scheduling skew where the merge waits on the single slowest branch.
- **Silently shipping partial datasets from Unaligned Fan-In without a completeness signal**: misleads consumers into treating partial data as final.
- **Re-reading/recomputing a shared intermediate dataset separately for each Parallel Split branch**: wastes compute — always persist/cache it once.
- **Stacking many Exclusive Choice branches without pruning**: produces a "complexity factory" pipeline nobody can explain (use the rubber-duck test).
- **Relaxing Single Runner to Concurrent Runner on a stateful/incremental pipeline**: causes race conditions on shared state (e.g., duplicate or missed backfill triggers in Dynamic Late Data Integrator).

## Code Examples
```python
# Isolated Sequencer: task-based (trigger) dependency with automatic backfill propagation
success_execution_marker = ExternalTaskMarker(
 task_id='trigger_downstream_consumers',
 external_dag_id='devices_aggregator',
 external_task_id='downstream_trigger_sensor',
)
parent_dag_sensor = ExternalTaskSensor(
 task_id='downstream_trigger_sensor',
 external_dag_id='devices_loader',
 external_task_id='trigger_downstream_consumers',
 allowed_states=['success'], failed_states=['failed', 'skipped']
)
```
- **What it demonstrates**: tightly-coupled cross-pipeline triggering where a backfill on the producer DAG automatically propagates to the consumer.

```python
# Aligned Fan-In: dynamically generated hourly branches merging into one task
for hour_to_load in [f"{hour:02d}" for hour in range(24)]:
 file_sensor = FileSensor(task_id=f'wait_for_{hour_to_load}', filepath=...)
 visits_loader = PostgresOperator(task_id=f'load_hourly_visits_{hour_to_load}', params={'hour': hour_to_load})
 clear_context >> file_sensor >> visits_loader >> generate_trends
```
- **What it demonstrates**: 24 independent hourly branches that all must succeed before the common `generate_trends` task runs.

```python
# Parallel Split: caching the shared input so it's read only once across branches
input_dataset = spark_session.read.schema(...).format('json').load(...)
input_dataset.persist(StorageLevel.MEMORY_ONLY)
input_dataset.write...  # branch 1
input_dataset.write...  # branch 2
```
- **What it demonstrates**: avoiding duplicate reads when one dataset feeds multiple independent write branches.

```python
# Exclusive Choice: routing to one of two branches based on execution date
def get_output_format_route(**context):
 migration_date = pendulum.datetime(2024, 2, 3)
 if context['execution_date'] >= migration_date:
  return 'load_job_trigger_delta'
 else:
  return 'load_job_trigger_csv'
format_router = BranchPythonOperator(task_id='format_router', python_callable=get_output_format_route)
```
- **What it demonstrates**: mutually-exclusive branch routing driven by a metadata condition (execution date), keeping old and new job versions in one pipeline's history.

## Reference Tables
| Pattern | Scope | Default behavior |
|---|---|---|
| Local Sequencer | Single pipeline/job | Strict sequence, dependency-ordered |
| Isolated Sequencer | Cross-pipeline/team | Data-based (loose) or task-based (tight) trigger |
| Aligned Fan-In | Merge point | All parents must succeed |
| Unaligned Fan-In | Merge point | Runs on any/all-done outcome; needs completeness flag |
| Parallel Split | Branch point | All branches run from one parent |
| Exclusive Choice | Branch point | Exactly one branch runs, by condition |
| Single Runner | Pipeline concurrency | Max 1 concurrent run |
| Concurrent Runner | Pipeline concurrency | >1 concurrent runs allowed |

| Isolated Sequencer trigger style | Coupling | Who can change freely |
|---|---|---|
| Data-based (Readiness Marker) | Loose | Consumer (can switch datasets) |
| Task-based (direct trigger) | Tight | Neither (rename/removal breaks the chain) |

## Worked Example
Blog analytics platform: a `devices_loader` pipeline (Local Sequencer: `input_data_sensor >> load_data_to_table >> expose_new_table`) separates the Readiness Marker check from the Full Loader step so each can restart independently. Its output feeds a separately-owned `devices_aggregator` pipeline via the Isolated Sequencer pattern — first implemented loosely via a `FileSensor` watching for the loader's output file (data-based dependency), then switched to `ExternalTaskMarker`/`ExternalTaskSensor` so that backfills on `devices_loader` automatically re-trigger `devices_aggregator` (task-based dependency). A separate hourly visits-aggregation pipeline fans in 24 per-hour branches into a daily `generate_trends` task using Aligned Fan-In; when occasional hourly failures start blocking the whole day's output, the team relaxes it to Unaligned Fan-In with `TriggerRule.ALL_DONE` and adds an `is_approximate` flag computed from how many of the 24 hours actually succeeded. During a framework migration, the loader pipeline temporarily dual-writes to both legacy CSV and new Delta Lake outputs from one cached intermediate dataset (Parallel Split with `.persist()`), then later uses Exclusive Choice (`BranchPythonOperator` keyed on `execution_date`) to cut over to Delta Lake only after a fixed migration date while preserving the ability to backfill pre-cutover dates with the legacy path. Finally, the stateful sessionization pipeline from Ch 5 runs under Single Runner (`max_active_runs=1`, `depends_on_past=True`) because its logic depends on prior runs, while the independent per-dataset device ingestion pipelines run under Concurrent Runner (`max_active_runs=5`) since they don't share state.

## Key Takeaways
1. Choose sequencing/fan boundaries based on restart cost and transactional coupling, not just code organization — ask "what should fail and retry independently?"
2. Prefer data-based (Readiness Marker) dependencies between teams for loose coupling; use task-based triggers only when tight coupling and automatic backfill propagation are actually wanted.
3. Aligned Fan-In trades latency/infra spikes for strict consistency; Unaligned Fan-In trades consistency for availability — always pair Unaligned Fan-In with an explicit completeness signal.
4. Parallel Split must cache/materialize shared intermediate data once; never let independent branches trigger redundant reads or duplicate writes.
5. Prefer metadata-based conditions over data-based ones for Exclusive Choice routing to keep branching cheap.
6. Single Runner is mandatory for stateful/incremental pipelines; Concurrent Runner is safe only when pipeline instances share no mutable state.

## Connects To
- **Ch 2 (Data Ingestion)**: Isolated Sequencer's data-based dependency directly reuses the Readiness Marker pattern.
- **Ch 4 (Idempotency)**: Parallel Split's duplicate-write risk on retry is mitigated by idempotency patterns or native write-dedup (Delta Lake `txnVersion`/`txnAppId`).
- **Ch 5 (Data Value)**: Single Runner is required for the Incremental/Stateful Sessionizer patterns; Aligned Fan-In's hourly-to-daily aggregation mirrors Distributed Aggregator use cases.
- **Ch 10 (Data Observability)**: data lineage tooling (OpenLineage) helps track the implicit dependencies that Isolated Sequencer's data-based approach creates.
- **Apache Airflow, AWS Step Functions, Azure Data Factory, AWS EMR, Apache Spark, Delta Lake**: primary technologies implementing these patterns.
