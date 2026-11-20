# Chapter 11: Time-Windowed Features for Real-Time Machine Learning

## Core Idea
Apache Beam's unified batch/streaming programming model lets the same transform code compute time-windowed aggregate features (e.g., trailing-hour average departure delay per airport) on historical BigQuery data for training and on live Pub/Sub events for serving — eliminating training-serving skew — while Cloud Dataflow's serverless autoscaling, sliding-window deduplication, batched model invocation, and watermark/trigger mechanisms handle the real engineering complexity of late/out-of-order streaming data.

## Frameworks Introduced
- **Shared transform module for training and serving**: factor feature-computation logic into a standalone package called identically from the batch training pipeline and the streaming prediction pipeline.
  - When to use: any ML feature that must be computed identically regardless of whether the data source is historical (BigQuery) or live (Pub/Sub) — directly solves the Ch. 8/9 causality and training-serving-skew problem for time-windowed aggregates.
  - How: put `assign_timestamp`, `is_normal_operation`, `add_stats`, `create_features_and_label`, etc. in a shared Python module (`flightstxf/flights_transforms.py`); call `transform_events_to_features(events)` from both `create_traindata.py` (batch) and `make_predictions.py` (streaming); gate training-only outputs (label, data_split) behind a `for_training` flag.
- **Sliding-window deduplication via window-position check**: emit each entity only once per true "new" window, not once per overlapping pane.
  - When to use: any sliding-window aggregation (e.g., hourly average every 5 minutes) where the entity itself (not just the aggregate) must be re-emitted downstream.
  - How: inject `window=beam.DoFn.WindowParam` into the aggregation function; compute `emit_end_time = window.start + WINDOW_EVERY`; only yield an event if its own timestamp falls before that boundary — ensures it's emitted from the first (earliest) window it appears in, not from every one of the (WINDOW_DURATION/WINDOW_EVERY) overlapping panes it's part of.
- **Batched, reusable model invocation in a streaming pipeline**: avoid per-record network calls to a deployed ML endpoint.
  - When to use: any Beam pipeline calling an external prediction service (Vertex AI endpoint, REST API) per element.
  - How: strip time-window metadata with `beam.WindowInto(beam.window.GlobalWindows())`, then `beam.BatchElements(min_batch_size=1, max_batch_size=64)` before the prediction `ParDo` (Dataflow dynamically tunes the actual batch size, e.g., settling near 30); initialize the endpoint stub once per worker in a `DoFn.setup()` override (not `__init__`, which can be passivated/reactivated on a different machine) since creating a lightweight endpoint stub is cheap enough to not need the `Shared()` handle pattern (reserve `Shared()` for genuinely expensive/large/non-thread-safe objects like an in-memory loaded model).
- **Watermark/trigger-based late-data handling**: control the correctness/performance trade-off for out-of-order streaming records.
  - When to use: any streaming aggregation where records can legitimately arrive late (network delay, retries).
  - How: default behavior triggers aggregation when the watermark (oldest unprocessed record estimate) passes the window's end, then re-fires individually for every late record (maximizes correctness, costs performance); override with `trigger=AfterWatermark(late=AfterCount(10)), allowed_lateness=1800` to batch late records (10 at a time) and discard anything later than a bounded lateness window (30 minutes here).

## Key Concepts
- **Beam vs. Dataflow vs. serverless Spark**: Dataflow is to Cloud Dataproc what BigQuery is to Cloud SQL — fully serverless with no cluster to manage, autoscaling transparently; choose Beam/Dataflow specifically for streaming use cases since Beam ("Batch and Stream") was designed streaming-first, whereas for pure batch, serverless Spark is an equally valid, more-familiar-to-some alternative.
- **Event timestamp vs. insertion/processing timestamp**: must assign the business-meaningful event time (e.g., WHEELS_OFF time) rather than Pub/Sub insertion time, because insertion time doesn't reflect legitimate network delays and would corrupt windowed averages if a delayed message is finally inserted hours later.
- **Watermark**: the pipeline's estimate of the oldest unprocessed record's event time; a learned heuristic (not a strict guarantee) when event time is user-specified rather than derived from insertion time; indicates pipeline lag and determines when windows "close."
- **Trigger**: the mechanism that fires aggregate computation, usually keyed off the watermark; default fires at watermark-crossing then immediately on each late record (correctness-first); configurable to batch/limit late-record processing for a performance/correctness trade-off.
- **Streaming sink selection criteria**: choose based on access pattern (long-term storage vs. near-real-time query), transactional need (append-only/eventually-consistent vs. strongly consistent single-current-state), throughput, and latency — BigQuery streaming inserts handle up to ~1 GBps at few-second latency (sub-millisecond with BI Engine); Cloud Bigtable scales to petabytes with millisecond latency and linear throughput scaling per node; Cloud SQL/Firestore/Spanner serve transactional, single-current-state needs at different scale/consistency trade-offs.
- **Cloud Bigtable row-key design (tall-and-narrow vs. short-and-wide tables)**: short-and-wide tables exploit sparsity (presence/absence of a column) for entity-attribute data (e.g., car parts); tall-and-narrow tables store one event per row (e.g., flight events) — row key design must balance write parallelism (unpredictable/distributed key prefixes avoid hotspotting) against read efficiency (range-scannable, most-recent-first ordering via reverse timestamps).

## Mental Models
- **"Design assuming you'll have a successful product"**: the chapter explicitly justifies batch-computing predictions once per flight (at takeoff) rather than per user-query, assuming many more end users than flights — architect for the access-pattern ratio you expect at scale, not just today's volume.
- **Always verify element counts through each pipeline stage**: Dataflow's monitoring console reporting element counts at every step (5.819M → 5.730M after timestamp assignment → 5.714M after removing canceled/diverted → matches final CSV row count) is the concrete verification discipline that caught both the sliding-window duplication bug and would catch any future silent data loss.
- **A forgotten upstream filtering decision can silently reappear downstream**: distance was deliberately excluded from the wheels-off event in Chapter 4 (unknowable at that time... except it actually *is* the nominal/scheduled distance, not the actual flown distance) — the author later used distance as a feature anyway, forgot the Ch. 4 omission, and had to patch around it by recomputing distance via the Haversine formula from lat/lon rather than fixing the root-cause pipeline (acknowledged as the "better" but deferred fix).
- **Simulate realistic imperfection, don't trust an idealized simulator**: perfectly-ordered simulated events understate real-world late/out-of-order behavior; deliberately injecting jitter (uniform, exponential, or normal-distributed delay) into the simulator is necessary to exercise and validate watermark/trigger logic.

## Anti-patterns
- **Storing a mutable/expensive remote-object reference as a `DoFn.__init__` instance variable**: worker state can be passivated and reactivated on a different machine hours later, so cached connections/stubs set in `__init__` may silently become invalid — use the `setup()` lifecycle method instead for anything that must be (re)initialized once per worker.
- **Using the `Shared()` handle pattern for a cheap, thread-safe object** (like an endpoint stub, which is "just a URL"): adds unnecessary serialization/contention overhead; reserve `Shared()` for genuinely large, expensive-to-create, or non-thread-safe objects (e.g., an in-memory loaded Keras model).
- **Sending prediction requests to an ML endpoint one record at a time in a streaming pipeline**: wasteful in network overhead, cost, and latency — always batch via `BatchElements` (after discarding time-window keying, since batching is itself an aggregation subject to window semantics).
- **Relying on Pub/Sub insertion time as the event timestamp**: corrupts time-windowed aggregates when network delays or retries insert a record much later than when it actually happened.
- **Treating a perfectly-ordered, zero-latency event simulator as representative of production streaming behavior**: hides the late/out-of-order handling bugs that only show up with realistic jitter; the chapter explicitly builds three jitter models (uniform, exponential, truncated-normal via a BigQuery JS UDF) to stress-test watermark/trigger behavior.
- **Choosing a sink before checking it against access pattern, transactions, throughput, and latency needs**: e.g., defaulting to Cloud Bigtable "because it's powerful" would be overkill for a sub-1000-events/sec, non-transactional, few-second-latency use case where BigQuery streaming inserts suffice and are cheaper/simpler.

## Code Examples
```python
def add_stats(element, window=beam.DoFn.WindowParam):
    airport, events = element
    avg_dep_delay = float(np.mean([float(e['DEP_DELAY']) for e in events]))
    emit_end_time = window.start + WINDOW_EVERY
    for event in events:
        if to_datetime(event['WHEELS_OFF']).timestamp() < emit_end_time:
            event_plus_stat = event.copy()
            event_plus_stat['AVG_DEP_DELAY'] = avg_dep_delay
            yield event_plus_stat
```
- **What it demonstrates**: fixing the sliding-window duplication bug by injecting the window parameter and only emitting an event from the earliest pane it belongs to, collapsing a 6GB→57GB blowup back down to the original record count.

```python
preds = (features
  | 'into_global' >> beam.WindowInto(beam.window.GlobalWindows())
  | 'batch_instances' >> beam.BatchElements(min_batch_size=1, max_batch_size=64)
  | 'model_predict' >> beam.ParDo(FlightsModelInvoker())
)
```
```python
class FlightsModelInvoker(beam.DoFn):
    def __init__(self):
        self.endpoint = None
    def setup(self):
        from google.cloud import aiplatform
        self.endpoint = aiplatform.Endpoint.list(filter='display_name="flights-ch10"', order_by='create_time desc')[0]
    def process(self, input_data):
        predictions = self.endpoint.predict(input_data).predictions
        for idx, input_instance in enumerate(input_data):
            result = input_instance.copy()
            result['prob_ontime'] = predictions[idx][0]
            yield result
```
- **What it demonstrates**: the correct per-worker endpoint-stub lifecycle pattern (via `setup()`, not `__init__`) combined with batched, globally-windowed model invocation — Dataflow auto-tunes batch size (observed settling around ~30 flights per call).

```python
beam.WindowInto(
    beam.window.SlidingWindows(WINDOW_DURATION, WINDOW_EVERY),
    trigger=AfterWatermark(late=AfterCount(10)),
    allowed_lateness=1800)  # 30 minutes
```
- **What it demonstrates**: overriding the default per-record-late-firing trigger to batch late records (10 at a time) and bound how long the pipeline waits for stragglers (30 min), trading strict correctness for performance.

```python
key = "{}#{}#{}#{}".format(origin, dest, carrier, reverse_ts(event_time))
# row key: ORIGIN#DEST#CARRIER#ReverseTimeStamp
```
- **What it demonstrates**: Cloud Bigtable row-key design balancing write-parallelism (alphabetically-distributed origin-airport prefix) against read-efficiency (scheduled-departure-based reverse timestamp keeps most-recent flights at the top of a range scan).

## Reference Tables
| Model / dataset | RMSE | Notes |
|---|---|---|
| AutoML, Ch. 9 features only | 0.199 | Baseline |
| AutoML + time-windowed averages | 0.198 | Minuscule improvement; averages matter less once other features/real-time context exist |
| Custom wide-and-deep, Ch. 10 features only | 0.196 | Baseline |
| Custom wide-and-deep + time-windowed averages | 0.195 | Essentially unchanged |

| Streaming sink | Best for |
|---|---|
| Cloud Storage (sharded files) | Long-term storage, delayed/batch access, staging for later BigQuery/Cloud SQL import |
| BigQuery (streaming inserts) | Near-real-time analytics, up to ~1 GBps, few-second latency (sub-ms with BI Engine); no transactions needed |
| Cloud Bigtable | Petabyte-scale, millions of ops/sec, millisecond latency, linear scaling per node; no transactions needed |
| Cloud SQL | Frequently-updated, low-throughput, medium-scale, broad tool/language ecosystem, transactional |
| Cloud Firestore | NoSQL object store, high throughput, eventual consistency with strong consistency within entity groups |
| Cloud Spanner | Globally available, strongly consistent, transactional, SQL-queryable, 99.999% availability, extremely large scale |

## Worked Example
Builds the full real-time flight-delay prediction system: (1) write a Beam pipeline reading `flights_tzcorr` from BigQuery (with a local-JSON-sample mode for fast iteration), filter out canceled/diverted flights, assign each event the WHEELS_OFF timestamp, apply a 60-minute sliding window updated every 5 minutes, group by origin airport, and compute average departure delay and taxi-out time per window; (2) discover and fix a window-overlap duplication bug (each flight appeared in up to 12 overlapping panes) by only emitting from the earliest pane; (3) extract the common feature-transform logic into a shared `flightstxf` module usable by both the training-data-creation pipeline and the real-time prediction pipeline, gating training-only fields (label, data_split) behind a flag; (4) discover and patch a missed Ch. 4 omission where DISTANCE wasn't carried in the wheels-off event, recomputing it via the Haversine formula from airport lat/lon as a stopgap; (5) run the full pipeline on Dataflow (2h50m, autoscaled to 20 workers), verifying element counts at every stage match expectations; (6) retrain the Chapter 9/10 AutoML and custom wide-and-deep models with the two new time-averaged features, finding only marginal RMSE improvement (0.199→0.198 for AutoML, 0.196→0.195 for custom); (7) build the real-time prediction pipeline: subscribe to the `wheelsoff` Pub/Sub topic, reuse the shared transform module, batch requests (`GlobalWindows` + `BatchElements`) to a per-worker-cached Vertex AI endpoint stub (initialized in `setup()`), and stream results into `dsongcp.streaming_preds` in BigQuery via `STREAMING_INSERTS`; (8) inject realistic jitter into the Ch. 4 simulator (uniform, exponential, and truncated-normal via a BigQuery JavaScript UDF using the Marsaglia polar rule) to test watermark/trigger behavior, observing the Dataflow console's learned watermark lag change accordingly (42s for uniform jitter, 22s for exponential); (9) as a hypothetical high-throughput/low-latency alternative, design a Cloud Bigtable schema (`ORIGIN#DEST#CARRIER#ReverseTimeStamp` row key) for the same predictions table and demonstrate HBase-shell range scans for "latest flights between two specific airports."

## Key Takeaways
1. Factor time-windowed feature computation into a shared module callable identically from batch (historical BigQuery) and streaming (live Pub/Sub) pipelines — this is Beam's core advantage for eliminating training-serving skew.
2. When using sliding windows, explicitly check the window's position (`window.start`) and only emit each entity from its earliest matching pane, or you'll get N-fold duplication (N = window_duration / window_every).
3. Batch requests to an external ML prediction endpoint using `BatchElements` after stripping time-window keying with `GlobalWindows`; let Dataflow auto-tune the batch size rather than hardcoding it.
4. Initialize per-worker stateful resources (endpoint stubs, connections) in a `DoFn.setup()` override, never in `__init__`, since worker instances can be passivated and reactivated elsewhere.
5. Use the `Shared()` handle pattern only for genuinely expensive/large/non-thread-safe shared objects (e.g., an in-memory ML model); for cheap, thread-safe objects (endpoint stubs), prefer per-worker instances.
6. Tune watermark/trigger behavior (`AfterWatermark(late=...)`, `allowed_lateness`) to balance correctness (processing every late record) against performance (batching or bounding late-record handling).
7. Choose a streaming sink by explicitly weighing access pattern, transactional needs, throughput, and latency — don't default to the most powerful (Bigtable) or most familiar (BigQuery) option without checking the fit.

## Connects To
- **Ch 4**: Directly reuses and extends the Pub/Sub simulator, the time-correction pipeline, and the Beam fundamentals (windowing, timestamps) introduced there; also surfaces and patches a bug from the original Ch. 4 wheels-off event design.
- **Ch 8**: Resolves the exact causality/training-serving-skew problem (how to provide a moving-average feature at prediction time) that Ch. 8 identified but deferred.
- **Ch 9 & 10**: Directly extends and re-trains the wide-and-deep and AutoML models and the MLOps pipeline built in those chapters with the two new time-windowed features.
- **Streaming Systems** (Akidau et al.): recommended as the deeper foundational reference for the watermark/trigger/windowing concepts used throughout this chapter.
