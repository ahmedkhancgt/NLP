# Chapter 11: Streaming Analytics and Machine Learning

## Core Idea
Real-world data arrives continuously, not as a static S3 dataset; Kinesis (Firehose, Data Analytics, Data Streams) and Amazon MSK let you enrich streaming product-review messages with real-time BERT sentiment predictions (via Lambda + SageMaker Endpoints) and compute continuous descriptive analytics (average rating, anomaly score, approximate counts) using SQL windowed queries.

## Frameworks Introduced
- **Kinesis Data Firehose**: fully managed, near-real-time delivery stream from a source (DirectPut, IoT, CloudWatch) to a destination (S3, Redshift, Elasticsearch, custom HTTP endpoint), with optional Lambda-based record transformation.
  - When to use: zero-admin ingestion tolerant of some latency; pay only for data processed.
  - How: configure a Lambda transform that calls a SageMaker Endpoint per record, then deliver enriched records to S3.
- **Kinesis Data Analytics**: SQL (ANSI 2008 + streaming extensions) or Apache Flink engine for continuous analytics over streaming data, using "in-application streams" and "pumps" (`CREATE STREAM` + `CREATE PUMP ... INSERT INTO ... SELECT STREAM`).
  - When to use: continuous aggregation (averages, counts), anomaly detection, or other descriptive analytics directly on a live stream.
  - How: define windowed SQL queries (tumbling/sliding/stagger) against the Firehose-fed `SOURCE_SQL_STREAM_001`.
- **Kinesis Data Streams**: custom, real-time stream ingestion requiring self-managed throughput (shards).
  - When to use: use cases needing custom per-record processing in real time (vs. Firehose's managed near-real-time delivery).
- **Amazon MSK (Managed Streaming for Apache Kafka)**: managed Kafka cluster, alternative to Kinesis for the same Lambda+SageMaker enrichment pattern.
  - How: `create_event_source_mapping()` maps a Kafka topic to a Lambda function that invokes the SageMaker Endpoint.
- **RANDOM_CUT_FOREST (RCF) in Kinesis Data Analytics**: built-in anomaly-detection SQL function producing an anomaly score (and per-column attribution scores) for numeric stream values.
- **COUNT_DISTINCT_ITEMS_TUMBLING**: built-in HyperLogLog-based approximate distinct-count function over a tumbling window.
- **Enhanced Fan-Out (EFO)** for Kinesis Data Streams: gives each shard-consumer pair its own dedicated full-throughput HTTP/2 push pipe, instead of consumers contending for a shared per-shard read-throughput limit.

## Key Concepts
- **Online (incremental) vs. offline learning**: online learning updates a model from streaming data without a full retrain (e.g., `scikit-learn`'s `partial_fit()` on linear models, Spark streaming ML, or deep learning fine-tuning from a checkpoint); offline learning retrains from a full static dataset.
- **The 3 Vs of streaming data**: volume, velocity, variety — characteristics that make streaming data unsuited to traditional REST/relational-DB processing patterns.
- **Total order (or lack thereof)**: distributed streaming systems generally cannot guarantee messages are consumed in the exact order they were produced; enabling strict total order trades away performance.
- **Windowed queries**: streaming SQL aggregations bound by time or row count, using the auto-added `ROWTIME` column.
  - **Stagger windows**: open per partition key when its first event arrives, close after a fixed window age — best for late/out-of-order data arriving at inconsistent times.
  - **Tumbling windows**: fixed, non-overlapping intervals; each record processed exactly once — best for regular periodic aggregation (`GROUP BY ... STEP(ROWTIME BY INTERVAL ...)`).
  - **Sliding windows**: fixed size/interval windows that continuously slide; a record can belong to multiple overlapping windows (`WINDOW ... RANGE INTERVAL ... PRECEDING` or `ROWS ... PRECEDING`).
- **Proxy sentiment mapping**: predicted star rating 4–5 → positive, 3 → neutral, 1–2 → negative, used as a sentiment proxy metric in streaming dashboards.

## Mental Models
- **Streaming = the same ML lifecycle, applied continuously**: the chapter reuses the fine-tuned BERT SageMaker Endpoint from Ch. 7-9 unchanged — streaming is an ingestion/serving pattern layered on top of the same trained model, not a different modeling approach.
- **Choose window type by arrival pattern, not just by time length**: late/out-of-order grouped data → stagger; regular fixed-interval reporting → tumbling; continuously-updating rolling metrics → sliding.
- **Firehose vs. Data Streams trade-off**: Firehose = zero administration, managed scaling, tolerable latency, pay-per-data; Data Streams = custom per-record real-time processing, self-managed shard throughput — pick based on whether you need custom processing logic per record.
- **Lambda auto-scaling mirrors SageMaker Endpoint auto-scaling (Ch. 9)**: both dynamically add/remove compute (Lambda interpreters vs. endpoint instances) to match load — a consistent AWS elasticity pattern worth recognizing across services.

## Anti-patterns
- **Forcing strict total-order guarantees on a distributed stream without need**: significantly hurts performance and can negate the benefits of building a streaming system in the first place — only enable it when genuinely required.
- **Using small, unaggregated PUT requests to Kinesis Data Streams**: wastes cost since pricing is based on shard count and payload units (25 KB); use the Kinesis Producer Library (KPL) to aggregate/compress multiple records into one PUT.
- **Scaling many consumers on a single Kinesis stream without EFO**: consumers contend for each shard's shared read-throughput limit, forcing wasteful multi-stream fan-out instead of one stream with dedicated per-consumer throughput.
- **Treating RCF/approximate-count results as exact**: like HyperLogLog in Ch. 5, these are probabilistic estimates — fine for dashboards/alerts, not for exact financial or compliance reporting.

## Code Examples
```sql
CREATE OR REPLACE STREAM "AVG_STAR_RATING_SQL_STREAM" (avg_star_rating DOUBLE);
CREATE OR REPLACE PUMP "AVG_STAR_RATING_SQL_STREAM_PUMP" AS
INSERT INTO "AVG_STAR_RATING_SQL_STREAM"
SELECT STREAM AVG(CAST("star_rating" AS DOUBLE)) AS avg_star_rating
FROM "SOURCE_SQL_STREAM_001"
GROUP BY STEP("SOURCE_SQL_STREAM_001".ROWTIME BY INTERVAL '5' SECOND);
```
- **What it demonstrates**: a tumbling 5-second window computing the continuous average star rating for a real-time sentiment dashboard.

```sql
CREATE OR REPLACE STREAM "ANOMALY_SCORE_SQL_STREAM" (anomaly_score DOUBLE);
CREATE OR REPLACE PUMP "ANOMALY_SCORE_STREAM_PUMP" AS
INSERT INTO "ANOMALY_SCORE_SQL_STREAM"
SELECT STREAM anomaly_score
FROM TABLE(RANDOM_CUT_FOREST(
    CURSOR(SELECT STREAM "star_rating" FROM "SOURCE_SQL_STREAM_001")));
```
- **What it demonstrates**: built-in Random Cut Forest anomaly detection flagging unexpected values (e.g., an impossible star rating of 100) for root-cause investigation.

```python
def lambda_handler(event, context):
    outputs = []
    for record in event['records']:
        inputs = [{"features": [review_body]}]
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME, ContentType='application/jsonlines',
            Accept='application/jsonlines', Body=json.dumps(inputs).encode('utf-8'))
        output_record = {'recordId': record['recordId'], 'result': 'Ok', 'data': ...}
        outputs.append(output_record)
    return {'records': outputs}
```
- **What it demonstrates**: a Kinesis Firehose transformation Lambda that enriches each streaming review record with a BERT-based star-rating prediction before delivery to S3.

```python
response = client.create_event_source_mapping(
    EventSourceArn='<MSK_CLUSTER_ARN>', FunctionName='<LAMBDA_FUNCTION_NAME>',
    Enabled=True, Topics=['reviews'])
```
- **What it demonstrates**: wiring an Amazon MSK (Kafka) topic directly to a Lambda function for the same real-time enrichment pattern as Kinesis.

## Reference Tables
| Window type | Behavior | Best for |
|---|---|---|
| Stagger | Opens per partition key on first event, closes after window age | Late/out-of-order grouped data |
| Tumbling | Fixed, non-overlapping intervals, each record processed once | Regular periodic aggregation |
| Sliding | Fixed size, continuously slides, records can belong to multiple windows | Rolling/overlapping metrics |

| Service | Admin overhead | Processing | Best for |
|---|---|---|---|
| Kinesis Data Firehose | None (managed) | Near-real-time | Simple delivery + transform to a destination |
| Kinesis Data Streams | Self-managed shards | Real-time | Custom per-record processing |
| Kinesis Data Analytics | None (managed) | Real-time SQL/Flink | Continuous descriptive analytics |
| Amazon MSK | Self-managed cluster (managed infra) | Real-time | Teams already invested in Kafka |

## Worked Example
A full streaming pipeline for product-review messages: Kinesis Data Firehose receives raw review messages (DirectPut) and invokes a Lambda transform that calls the fine-tuned BERT SageMaker Endpoint (from Ch. 7-9) to predict `star_rating`, enriching each record before Firehose delivers it to S3 (with a backup of originals to a second bucket). In parallel, Kinesis Data Analytics runs three continuous SQL queries over the same stream: a 5-second tumbling window computing `AVG(star_rating)` for a live sentiment dashboard (CloudWatch), a Random Cut Forest anomaly-score stream flagging malformed/out-of-range ratings for the engineering team, and a `COUNT_DISTINCT_ITEMS_TUMBLING` query giving the marketing team an approximate real-time message-volume count to gauge social-campaign effectiveness. The same Lambda+SageMaker enrichment pattern is shown again using Amazon MSK/Kafka via `create_event_source_mapping()`, demonstrating portability between Kinesis and Kafka architectures.

## Key Takeaways
1. Match the windowing strategy (stagger/tumbling/sliding) to how your data actually arrives — late/out-of-order grouped data needs stagger windows, not tumbling.
2. Reuse the same trained SageMaker model for both batch and streaming inference — streaming is an ingestion/serving pattern, not a separate model.
3. Use RCF-based anomaly detection to catch schema/data errors in a stream before they corrupt downstream analytics.
4. Choose Firehose for managed, low-admin delivery; choose Data Streams when you need custom real-time per-record processing control.
5. Aggregate small records (via KPL) before putting them on a Kinesis stream to control cost.
6. Enable Enhanced Fan-Out when scaling many consumers on a single stream, instead of fanning out to multiple streams.
7. Treat RCF anomaly scores and approximate counts as probabilistic signals for dashboards/alerts, not as exact figures.

## Connects To
- **Ch 1/2**: Directly implements the streaming analytics/ML vision (Kinesis, anomaly detection, sentiment dashboards) previewed there.
- **Ch 5**: Reuses HyperLogLog approximate-counting concepts, now applied to streaming data via `COUNT_DISTINCT_ITEMS_TUMBLING`.
- **Ch 7-9**: Reuses the fine-tuned BERT SageMaker Endpoint trained, tuned, and deployed in those chapters as the real-time prediction service.
- **Ch 9**: Lambda auto-scaling and SageMaker Endpoint auto-scaling patterns mirror each other across both chapters.
- **Ch 12**: Security for streaming infrastructure (IAM roles, encryption) is covered in depth next.
