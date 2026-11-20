# Chapter 4: Streaming Data: Publication and Ingest with Pub/Sub and Dataflow

## Core Idea
Real-time, location-aware dashboards require a streaming architecture — Pub/Sub for durable asynchronous ingestion, Apache Beam/Cloud Dataflow for unified batch-and-stream transformation and windowed aggregation, and BigQuery as the queryable sink — and Beam's identical code for batch and streaming means the time-correction ETL pipeline built for historical data doubles as the real-time pipeline's foundation.

## Frameworks Introduced
- **GCP streaming reference architecture**: Pub/Sub → Dataflow → BigQuery → Data Studio.
  - When to use: any real-time ingest-and-analyze pipeline on GCP.
  - How: publish events to Pub/Sub topics (durable, at-least-once, scales to millions of msgs/sec); consume and transform/aggregate in a Dataflow (Apache Beam) pipeline; write results to BigQuery; visualize via a Data Studio dashboard backed by a BigQuery view that always reflects the latest streamed data.
- **Apache Beam unified batch/streaming model**: the same pipeline code processes a bounded (batch) or unbounded (streaming) PCollection.
  - When to use: whenever you want one codebase for both historical reprocessing and live processing.
  - How: swap only the source (`ReadFromBigQuery`/`ReadFromText` vs. `ReadFromPubSub`), the runner config (`--streaming` flag), and the windowing strategy; transforms themselves (Map, FlatMap, GroupByKey) stay the same.
- **Sliding-window streaming aggregation**: compute moving statistics (e.g., 60-minute average delay) over unbounded data.
  - When to use: dashboards/metrics needing a rolling time window rather than all-time or per-batch aggregates.
  - How: `beam.WindowInto(beam.window.SlidingWindows(window_size_sec, period_sec))` → `GroupByKey` (keyed, e.g., by airport) → compute per-window stats; attach an explicit event-timestamp attribute at publish time (`timestamp_attribute`) so windows key off event time, not wall-clock/publish time.
- **Causality-respecting multi-event publication**: emit several partial events per entity instead of one complete event at the end.
  - When to use: any entity whose full record is only knowable progressively over time (e.g., a flight's lifecycle).
  - How: for each entity, emit one event per state transition (scheduled, departed, wheelsoff, wheelson, arrived), stripping fields not yet knowable at that point — mirrors the Chapter 2 causality constraint applied to streaming design.

## Key Concepts
- **PCollection**: Beam's immutable, distributable collection abstraction — not an in-memory list; can represent bounded or unbounded data.
- **Side input**: a small PCollection (as a Python dict/list) broadcast to every worker to join against a larger PCollection (e.g., airport time zones joined onto flight records) — must fit in memory; use `CoGroupByKey` for large-to-large joins.
- **Speed-up factor**: ratio of simulated event time to real system time used when replaying historical data as a simulated real-time feed; >1 runs faster than real time, <1 slower.
- **Federated BigQuery query**: querying a CSV file on Cloud Storage (or Google Sheets) directly as an external table definition, without first loading it into BigQuery's native columnar storage — best for small, frequently changing datasets joined against larger native tables.
- **At-least-once delivery / out-of-order messages**: Pub/Sub guarantees messages won't be lost but may be delivered more than once or out of order; downstream consumers (Dataflow) must and do handle both transparently.
- **Horizontal autoscaling (Dataflow)**: Dataflow automatically scales worker count based on pipeline throughput; can be disabled (`--autoscaling_algorithm=NONE`) or capped (`--max_num_workers`).
- **Materialized views vs. views vs. tables** (carried over from Ch. 3): used here to decide where to put the "latest per airport" aggregation for the dashboard.

## Mental Models
- **"It is only as you start to explore a dataset that you discover you need ancillary datasets"**: time-zone and airport-location data were not anticipated during ingestion (Ch. 2) but became necessary once building a location-aware real-time feed — treat schema/dataset scope as something that evolves with the use case.
- **Store UTC + offset, never just local time**: the chapter's central time-handling lesson — always persist both a UTC timestamp and the then-active local time zone offset so you can merge data globally yet still analyze local-time patterns.
- **Batch pipeline code is streaming pipeline code with a different source/runner**: reinforces the "write once, run as batch or stream" Beam philosophy as a practical cost-reduction principle, not just an abstraction.
- **Event time vs. processing time**: aggregation windows must be based on the event's own timestamp (when it actually happened), not when Dataflow happens to process or Pub/Sub happens to deliver it — otherwise simulated/replayed data at non-1x speed produces windows that span the wrong real-world duration.

## Anti-patterns
- **Publishing one event only at flight arrival**: makes the data useless for real-time decisions needed before the flight lands (violates causality) — must publish incremental events as the flight progresses.
- **Running the strictly-sequential, sleep-based simulator in Cloud Dataflow**: Dataflow is not fit for a pipeline whose correctness depends on sequential ordering and sleeping between steps; the simulator must be a standalone Python program instead (accepting the trade-off of no automatic fault-tolerant restart).
- **Relying on Pub/Sub's publish/arrival time for windowing**: without an explicit event-timestamp attribute, sliding windows reflect wall-clock arrival time, producing misleading window spans when replaying data faster/slower than 1x.
- **`SELECT *` on a columnar BigQuery table when only a few columns are needed**: inefficient; the chapter mitigates this for event-replay use cases by storing a single packed `EVENT_DATA` JSON column, trading storage for query speed when the full record is genuinely needed downstream.
- **Treating "FL_DATE" naively as the arrival date**: local departure date can precede UTC arrival date (e.g., HNL→DFW); must detect and add 24 hours when converted arrival time appears to precede departure time — another argument for always storing times in UTC from the start.

## Code Examples
```python
def addtimezone(lat, lon):
    import timezonefinder
    tf = timezonefinder.TimezoneFinder()
    return (float(lat), float(lon), tf.timezone_at(lng=float(lon), lat=float(lat)))
```
- **What it demonstrates**: computing IANA time zone from lat/lon offline (via `timezonefinder`), the first step in UTC time correction; note the Beam-recommended pattern of importing inside the function to avoid pickling issues when the pipeline runs distributed.

```python
flights = (pipeline
   | 'flights:read' >> beam.io.ReadFromBigQuery(query='SELECT * FROM dsongcp.flights', use_standard_sql=True)
   | 'flights:tzcorr' >> beam.FlatMap(tz_correct, beam.pvalue.AsDict(airports))
)
events = flights | beam.FlatMap(get_next_event)
```
- **What it demonstrates**: joining flights against an airports side-input dict to UTC-correct times, then fanning each flight record out into multiple causality-respecting lifecycle events via `FlatMap`.

```python
stats = (all_events
   | 'byairport' >> beam.Map(by_airport)
   | 'window' >> beam.WindowInto(beam.window.SlidingWindows(60 * 60, 5 * 60))
   | 'group' >> beam.GroupByKey()
   | 'stats' >> beam.Map(lambda x: compute_stats(x[0], x[1]))
)
```
- **What it demonstrates**: the canonical Beam streaming-aggregation pattern — key by entity, apply a sliding time window, group, then reduce — to produce rolling 60-minute averages updated every 5 minutes per airport.

```python
argv = [
   '--project={0}'.format(project), '--job_name=ch04avgdelay', '--streaming',
   '--staging_location=gs://{0}/flights/staging/'.format(bucket),
   '--temp_location=gs://{0}/flights/temp/'.format(bucket),
   '--runner=DataflowRunner'
]
```
- **What it demonstrates**: the only changes needed to move a Beam pipeline from local `DirectRunner` batch execution to a fully managed, autoscaled `DataflowRunner` streaming job on GCP.

```sql
CREATE OR REPLACE VIEW dsongcp.airport_delays AS
WITH delays AS (
    SELECT d.*, a.LATITUDE, a.LONGITUDE
    FROM dsongcp.streaming_delays d
    JOIN dsongcp.airports a USING(AIRPORT)
    WHERE a.AIRPORT_IS_LATEST = 1
)
SELECT AIRPORT, CONCAT(LATITUDE, ',', LONGITUDE) AS LOCATION,
    ARRAY_AGG(STRUCT(AVG_ARR_DELAY, AVG_DEP_DELAY, NUM_FLIGHTS, END_TIME)
              ORDER BY END_TIME DESC LIMIT 1) AS a
FROM delays GROUP BY AIRPORT, LONGITUDE, LATITUDE
```
- **What it demonstrates**: a "latest per key" BigQuery view (using `ARRAY_AGG ... ORDER BY ... LIMIT 1`) that Data Studio can poll for an always-current geo dashboard, without needing a separate materialization step.

## Reference Tables
| Pipeline stage | Batch (historical) | Streaming (real-time) |
|---|---|---|
| Source | `ReadFromBigQuery` / `ReadFromText` | `ReadFromPubSub` |
| Runner flag | (none) | `--streaming` |
| Windowing | N/A (whole dataset) | `SlidingWindows(size, period)` keyed by event timestamp |
| Sink | `WriteToBigQuery` (batch load) | `WriteToBigQuery` (streaming insert) |

| GCP streaming component | Role |
|---|---|
| Cloud Pub/Sub | Durable, at-least-once, scalable async message bus (ingest boundary) |
| Cloud Dataflow (Apache Beam) | Unified batch/streaming transform & windowed aggregation engine |
| BigQuery | Queryable sink for raw events and/or aggregated stats, even mid-stream |
| Data Studio | Dashboard layer polling a BigQuery view for the latest data |

## Worked Example
Builds the real-time flight-delay dashboard end to end: (1) obtain airport lat/lon via a BigQuery federated query against `airports.csv` on GCS, then compute each airport's IANA time zone offline with `timezonefinder`; (2) write a Beam pipeline (`df01.py`→`df07.py`, incrementally) that joins flights against airport time zones as a side input, converts all times to UTC (handling DST ambiguity and date-rollover edge cases), and stores both UTC time and local offset; (3) fan each flight into lifecycle events (scheduled, departed, wheelsoff, wheelson, arrived), stripping not-yet-knowable fields per event; (4) run this pipeline first with `DirectRunner` locally on a BigQuery-sampled subset, then with `DataflowRunner` at full scale, writing to `dsongcp.flights_simevents`; (5) build `simulate.py`, a sequential, sleep-based Python program (not a Beam pipeline) that replays these historical events to three Pub/Sub topics (`departed`, `wheelsoff`, `arrived`) at a configurable speed-up factor with an explicit `EventTimeStamp` publish attribute; (6) write a second streaming Beam pipeline (`avg01.py`→`avg03.py`) that reads from Pub/Sub, applies a 60-minute sliding window updated every 5 minutes keyed by airport, computes average arrival/departure delay and flight counts, and streams results into `dsongcp.streaming_delays`; (7) create the `dsongcp.airport_delays` BigQuery view joining the latest per-airport stats with airport coordinates; (8) build a Data Studio Geo Chart dashboard pointed at that view, showing live arrival/departure delay heat maps across US airports.

## Key Takeaways
1. Design event publication around causality — emit incremental lifecycle events with only the fields knowable at that moment, not one event at the end.
2. Always store both a UTC timestamp and the local time zone offset for any datetime field that might need local-time analysis later.
3. Apache Beam's batch and streaming code are nearly identical — change only the source, the `--streaming` flag, and windowing, not the transform logic.
4. Use an explicit event-timestamp attribute on Pub/Sub messages and tell Beam to key windows off it, not off wall-clock arrival time, especially when replaying data at non-1x speed.
5. Prefer sliding windows keyed by entity (e.g., airport) + `GroupByKey` for rolling real-time aggregates.
6. Build dashboard data sources as BigQuery views (not inline Data Studio queries) for reusability, single-point-of-change, and IAM-based access control.
7. Use federated BigQuery queries on Cloud Storage files for small, frequently-changing reference data joined against large native tables; don't overuse this pattern for primary analytics data.

## Connects To
- **Ch 2**: Reuses the causality-constraint principle established there, now applied to streaming event design.
- **Ch 3**: Extends the static contingency-table dashboard into a real-time, location-aware one.
- **Ch 11**: Time-windowed streaming features developed here are revisited and extended for real-time ML feature computation.
- **Ch 5**: The time-corrected `flights` dataset (UTC + offsets) produced here becomes the basis for subsequent exploratory analysis and modeling chapters.
