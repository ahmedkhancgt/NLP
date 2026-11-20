# Chapter 10: Using APIs in Data Pipelines

## Core Idea
Move from ad hoc notebook queries to a production-style, orchestrated data pipeline: use Apache Airflow DAGs with HTTP and Python operators to perform an initial bulk load and then recurring incremental (delta) updates from an API into a local analytics database, with upsert logic to handle both new and changed records.

## Frameworks Introduced
- **Full Load + Delta Update Pipeline Pattern**: the standard two-phase approach to keeping a local copy of API-sourced data current.
  - When to use: whenever an API doesn't support full bulk export directly, or ongoing sync needs to be efficient.
  - How: perform one-time initial load from a bulk file (or paginated full API pull) into a local database; thereafter run a recurring job that queries only records changed since the last run (via `minimum_last_changed_date`) and upserts them — cheaper, faster, and uses fewer resources than repeated full reloads.
- **DAG (Directed Acyclic Graph) Orchestration Pattern**: Airflow's model for defining ordered, dependency-aware pipeline steps.
  - When to use: any multi-step pipeline needing scheduling, error handling, retries, and observability beyond what a plain script provides.
  - How: define one Python file per DAG using the `@dag` decorator; break work into discrete `task`s, each backed by an `Operator` (a reusable template — `HttpOperator` for API calls, `PythonOperator` for custom logic); declare task order with the bitshift operator (`task_a >> task_b >> task_c`); pass data between tasks via `XCom`.
- **Upsert Pattern for Idempotent Loads**: insert-or-update logic so repeated pipeline runs don't duplicate or fail on existing records.
  - When to use: any recurring load where the same source record may appear in multiple runs (new or changed).
  - How: use a parameterized SQL statement with `INSERT ... ON CONFLICT(primary_key) DO UPDATE SET ...`, binding values via `?` placeholders (never string-interpolated) to prevent SQL injection.
- **Connections-as-Configuration**: externalize data source/target details from pipeline code.
  - When to use: any pipeline that must run differently across dev/test/prod environments.
  - How: register named connections (`http_conn_id`, database connection) in the orchestrator's admin UI (Airflow's Admin > Connections) with type-specific fields (Host for HTTP, Schema/path for SQLite), then reference them by ID in code (`BaseHook.get_connection(...)`) instead of hardcoding URLs/paths.

## Key Concepts
- **Data pipeline / ETL / ELT**: a structured, typically scheduled process that moves data from a source (API, bulk file, stream, queue, database) through transformation to a destination, as opposed to one-off notebook queries.
- **Delta (change) records**: records that are new or modified since the last pipeline run; filtering by `minimum_last_changed_date` lets a pipeline process only deltas instead of reloading everything.
- **DAG (directed acyclic graph)**: a graph-theory structure (nodes = processes, edges = flow) with steps proceeding in one direction with no loops — the top-level Airflow pipeline definition.
- **Task**: the basic unit of execution within a DAG, shown as one box in the DAG's graph view.
- **Operator**: a predefined task template (e.g., `HttpOperator` to call an API, `PythonOperator` to run custom Python); Airflow ships many built-ins (databases, S3, etc.) plus community-contributed ones.
- **XCom (cross-communication)**: Airflow's mechanism for passing small amounts of data between tasks (e.g., `context["ti"].xcom_pull(task_ids="api_player_query")`).
- **Airflow connection**: a named, UI-managed record of a data source's access details (type, host, schema), decoupling pipeline code from environment-specific configuration.
- **Docker volumes (Airflow setup)**: host directories (`./dags`, `./logs`, `./plugins`, `./config`) mapped into the Airflow containers' filesystem (e.g., `/opt/airflow/dags`), so DAG files edited locally are picked up by the running containers.
- **docker-compose.override.yaml**: a secondary compose file layered on top of the official `docker-compose.yaml`, isolating local customizations (hiding example DAGs, enabling the proxy fix for Codespaces, tuning file-processing interval) so the base file stays unmodified and easy to upgrade.
- **Response check (HttpOperator)**: a callback function (`response_check`) that validates an HTTP response beyond status code — here, confirming the API's health-check JSON body before letting downstream tasks run.
- **Parameterized SQL / prepared statements (recap)**: binding values via placeholders instead of string formatting, preventing SQL injection when inserting API-sourced data.

## Mental Models
- **"Airflow is a spider in a web"** (de Ruiter & Harenslak epigraph): Airflow sits centrally, coordinating distributed, heterogeneous work (API calls, database writes) rather than performing the work itself.
- **Notebooks for exploration, pipelines for production**: Chapter 9's notebook queries are fine for one-off/exploratory analysis, but production analytics needs scheduled, monitored, restartable pipelines — this chapter is the production-maturity step up.
- **Pipeline topology mirrors Figure 10-1/10-7**: two parallel ETL paths (bulk file → initial load; API → recurring delta update) both converge on the same local analytics database, which downstream analytics/report consumers query — sources can differ, but the destination and its upsert contract stay unified.
- **Shared logic lives outside the DAG files**: because both DAGs (bulk load and recurring update) need the same upsert behavior, that logic is factored into a separate `shared_functions.py` imported by both — standard code-reuse discipline applied to pipeline code.

## Anti-patterns
- **Re-querying the full dataset on every scheduled run**: wastes API calls, pipeline runtime, and money compared to delta-only updates.
- **Hardcoding API URLs/database paths directly in DAG code**: makes moving between dev/test/prod environments error-prone; Airflow connections solve this by externalizing config.
- **String-interpolating API data into SQL**: exposes the pipeline to SQL injection from malicious or malformed upstream data; always use parameterized queries with `?` placeholders.
- **Top-level imports of frequently-changing modules in DAG files**: Airflow re-parses DAG files often, so some imports (per the chapter) are deliberately placed inside functions rather than at the top of the file to avoid stale/duplicated reloads.
- **No health check before dependent calls**: the pipeline explicitly checks API health first (`api_health_check_task`) before querying player data, avoiding wasted work against a down API.

## Code Examples
```python
# recurring_player_api_insert_update_dag.py — DAG definition with HttpOperator + PythonOperator
@dag(schedule_interval=None)
def recurring_player_api_insert_update_dag():
    api_health_check_task = HttpOperator(
        task_id="check_api_health_check_endpoint",
        http_conn_id="sportsworldcentral_url",
        endpoint="/",
        method="GET",
        response_check=health_check_response,
    )

    api_player_query_task = HttpOperator(
        task_id="api_player_query",
        http_conn_id="sportsworldcentral_url",
        endpoint=f"/v0/players/?skip=0&limit=100000&minimum_last_changed_date={temp_min_last_change_date}",
        method="GET",
    )

    player_sqlite_upsert_task = PythonOperator(
        task_id="player_sqlite_upsert",
        python_callable=insert_update_player_data,
        provide_context=True,
    )

    api_health_check_task >> api_player_query_task >> player_sqlite_upsert_task

dag_instance = recurring_player_api_insert_update_dag()
```
- **What it demonstrates**: a three-task DAG (health check → delta API query → database upsert) with explicit sequential dependency via `>>`, using the delta-query pattern (`minimum_last_changed_date`) from Chapter 1/3.

```python
# shared_functions.py — upsert with parameterized SQL
cursor.execute("""
    INSERT INTO player (player_id, gsis_id, first_name, last_name, position, last_changed_date)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(player_id) DO UPDATE SET
        gsis_id = excluded.gsis_id,
        first_name = excluded.first_name,
        last_name = excluded.last_name,
        position = excluded.position,
        last_changed_date = excluded.last_changed_date
""", (player['player_id'], player['gsis_id'], player['first_name'],
      player['last_name'], player['position'], player['last_changed_date']))
```
- **What it demonstrates**: SQLite's `ON CONFLICT ... DO UPDATE` upsert syntax with fully parameterized values, shared between both the bulk-load and recurring-update DAGs.

```python
# XCom data passing between tasks
def insert_update_player_data(**context):
    player_json = context["ti"].xcom_pull(task_ids="api_player_query")
    if player_json:
        upsert_player_data(player_json)
    else:
        logging.warning("No player data found.")
```
- **What it demonstrates**: pulling the previous task's HTTP response payload via XCom and passing it into the shared upsert function.

```bash
# Airflow setup (Docker)
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.10.0/docker-compose.yaml'
mkdir -p ./dags ./logs ./plugins ./config
echo -e "AIRFLOW_UID=$(id -u)" > .env
docker compose up airflow-init
docker compose up -d
```
- **What it demonstrates**: the standard bootstrap sequence for running Airflow locally via Docker Compose, including volume directories and the one-time init step.

## Reference Tables
**Data pipeline source types**

| Source type | Characteristics | Example |
|---|---|---|
| REST APIs | Good for incremental updates; full loads costly over many calls | This chapter's focus |
| Bulk files | Efficient for very large datasets; CSV/Parquet common | Initial load DAG |
| Streaming/message queues | Near-real-time continuous feeds | Kafka, Kinesis, RabbitMQ, SQS |
| Direct database connections | Original format access; typically internal-only | N/A here |

**Airflow terminology glossary**

| Term | Meaning |
|---|---|
| DAG | Directed acyclic graph — top-level pipeline, steps flow one direction, no loops |
| Task | A single unit of execution (one box in the DAG graph) |
| Operator | A predefined task template (HttpOperator, PythonOperator, etc.) |
| XCom | Cross-communication mechanism to pass data between tasks |
| Connection | Named, UI-stored record of a data source/target's access config |

**Airflow Connections used in this chapter**

| Connection ID | Type | Key field |
|---|---|---|
| `analytics_database` | Sqlite | Schema: `/opt/airflow/dags/analytics_database.db` |
| `sportsworldcentral_url` | HTTP | Host: API base URL running in Codespaces |

## Worked Example
The chapter installs Apache Airflow via Docker Compose in a Codespace (4-core machine, custom `docker-compose.override.yaml` hiding example DAGs and enabling the Codespaces proxy fix), creates a local `analytics_database.db` SQLite file with a `player` table, and registers two Airflow connections (`analytics_database`, `sportsworldcentral_url`). It then builds `recurring_player_api_insert_update_dag.py`: task 1 (`HttpOperator`) hits the API's health check and validates the JSON body via `response_check`; task 2 (`HttpOperator`) queries `/v0/players/` filtered by a (here hardcoded, production-template-worthy) `minimum_last_changed_date`; task 3 (`PythonOperator`) pulls the player JSON via XCom and calls a shared `upsert_player_data()` function (in `shared_functions.py`) that performs a parameterized SQLite upsert. Triggering the DAG in the Airflow UI shows all three tasks turn green, and a subsequent `sqlite3` query confirms 1,018 player records loaded — matching the dataset size verified throughout Part I. A companion `bulk_player_file_load_dag.py` (not walked through in text, but included in the completed repo) handles the initial bulk-file load feeding the same database.

## Key Takeaways
1. Production analytics needs orchestrated, scheduled pipelines (not repeated manual notebook queries) — Airflow DAGs provide scheduling, error handling, and dependency management.
2. Combine a one-time full/bulk load with a recurring delta-only update (via `minimum_last_changed_date`) to keep a local database current efficiently.
3. Model pipeline steps as DAG tasks backed by reusable operators (`HttpOperator`, `PythonOperator`), with explicit dependencies declared via `>>`.
4. Externalize environment-specific configuration (API URLs, database paths) into Airflow connections rather than hardcoding them in DAG code.
5. Use upsert (`INSERT ... ON CONFLICT DO UPDATE`) with fully parameterized SQL for idempotent, injection-safe loads from API data.
6. Share common logic (like the upsert function) across multiple DAGs in a separate importable module rather than duplicating it.
7. Validate upstream health/status (health-check task) before running dependent, more expensive pipeline steps.

## Connects To
- **Ch 1**: Directly implements the "query by last-changed date" delta-pipeline design recommendation from the original API design tips.
- **Ch 3**: The delta-query/last_changed_date mechanism and SQL upsert technique both build on the Chapter 3 database/CRUD foundations.
- **Ch 8**: Responsible-consumption practices (health checks before heavier calls, parameterized queries against injection) are applied concretely here.
- **Ch 9**: Contrasts notebook-based exploratory querying with this chapter's production-pipeline approach to the same underlying API.
- **External**: *Data Pipelines Pocket Reference* (James Densmore); *Data Pipelines with Apache Airflow* (de Ruiter & Harenslak); Astronomer's Airflow glossary and "Manage task and task group dependencies in Airflow"; Airflow Operators and Hooks Reference.
