# Chapter 2: Ingesting Data into the Cloud

## Core Idea
Ingesting raw data correctly — respecting causality and avoiding training-serving skew, choosing the right storage/compute scaling strategy, and automating the pipeline with serverless components (Cloud Storage, BigQuery, Cloud Run, Cloud Scheduler) — is the foundation that everything later in the book builds on.

## Frameworks Introduced
- **Hub-and-spoke (ELT) architecture**: land raw data as-is into an enterprise data warehouse (the hub), then transform per use case (the spokes).
  - When to use: default ingestion pattern instead of ETL.
  - How: extract from source (e.g., BTS website) → load unmodified into Cloud Storage/BigQuery → transform later via SQL views; avoids dependency on the source system's own transformation/export reliability.
- **Scaling decision framework (up / out-sharded / out-in-place)**: choose a compute/storage strategy based on job shape.
  - When to use: any time a dataset or job outgrows a single laptop.
  - How: scale up (bigger VM + persistent disk) for jobs that re-read data many times (e.g., ML training); scale out with sharded data (Dataproc + HDFS, MapReduce) for splittable batch jobs without a high-bandwidth data center; scale out with data-in-place (BigQuery, Dataflow, Vertex AI on GCS/Colossus) when the data center's network makes compute and storage fungible — the GCP-native, most agile option.
- **Serverless ingestion pipeline (Cloud Run + Cloud Scheduler + service accounts)**: replace cron jobs with a fully managed, auditable, retryable pipeline.
  - When to use: any periodic/scheduled data ingestion job.
  - How: wrap ingestion logic in a Flask app → containerize with Docker → deploy to Cloud Run with `--no-allow-unauthenticated` and a dedicated least-privilege service account → trigger via Cloud Scheduler hitting an authenticated HTTPS endpoint with configurable retry/backoff.
- **Causality and training-serving skew checks**: a data-quality gate applied before any field is used as a model input.
  - When to use: during feature selection for any predictive model.
  - How: ask "will this field be known at prediction time?" (causality); ensure identical computation logic for a feature in training and serving (no skew) — e.g., use great-circle distance (knowable) instead of actual flown distance (not knowable until landing).

## Key Concepts
- **Causality constraint**: a field usable as a model input only if it will be known at the moment the prediction must be made (e.g., taxi-in duration is unusable because it's unknown at takeoff).
- **Training-serving skew**: inconsistency between how a feature is computed during training versus during live prediction (e.g., unit mismatches, timezone differences, differing library behavior).
- **Single-region vs. multi-region/dual-region bucket**: single-region buckets are cheaper and lower-latency for compute co-located in the same region; multi/dual-region trade cost and latency for availability across a wider geography.
- **CAP theorem (Brewer's theorem)**: a distributed system cannot simultaneously guarantee Consistency, Availability, and Partition resilience — informs the choice between strong-consistency single-region buckets and eventually-consistent, highly available alternatives.
- **Columnar database (BigQuery)**: data stored column-by-column in compressed format, optimized for full-table-scan SELECT/aggregation queries rather than row-level INSERT/UPDATE/DELETE.
- **BigQuery slots**: on-demand, elastically allocated parallel compute units used to execute a SQL query — not statically pre-provisioned.
- **Service account**: an automation-friendly identity with scoped permissions, used instead of a personal account to preserve least privilege and auditability for automated jobs.
- **Cloud IAM policy hierarchy**: permissions set at organization → project → resource level are inherited downward and unioned, never restricted at a lower level.

## Mental Models
- **"Read the documentation first"**: before selecting dataset fields (e.g., choosing `Reporting_Airline` over `IATA_CODE_Reporting_Airline`), consult the source's own field dictionary and analysis tools rather than guessing from the raw data.
- **Download vs. direct cloud access trade-off**: public internet (~3–10 MBps) vs. local/cloud-network disk access (two orders of magnitude faster) dictates whether to stage data locally or process in place.
- **"Serverless = you supply only code"**: the recurring litmus test the book applies to every architectural choice in this chapter (Cloud Run vs. self-managed cron VM).
- **Separation of compute and storage**: made viable by GCP's high-bisection-bandwidth data center network (Jupiter/Andromeda) and the Colossus filesystem, letting any node access any data without presharding — the architectural reason BigQuery/Dataflow/Vertex AI don't require you to manage clusters.

## Anti-patterns
- **Relying on a source server's on-demand field-filtered export**: BTS's per-query ZIP generation is an SPOF; download the full raw dataset instead and transform downstream.
- **Using unknowable-at-prediction-time fields as model inputs**: e.g., actual taxi-in time or actual flight distance — violates the causality constraint and silently corrupts the model's real-world performance.
- **Scheduling ingestion via a traditional cron job on a fixed VM**: fragile to server reboots, hard to port environments, no built-in retries, and no remote monitoring — replaced by Cloud Scheduler + Cloud Run.
- **Running automated ingestion as your personal user account**: pollutes audit logs and over-grants permissions; always use a dedicated, narrowly scoped service account instead.
- **Concatenating monthly files into one large blob in Cloud Storage**: Cloud Storage is a blob store with no append — keep files separate per month to allow incremental updates and partial processing.
- **Letting BigQuery `--autodetect` stand for production schemas**: autodetection samples ~100 rows and can silently mis-type columns (e.g., `NA` strings) — move to an explicit schema once past initial exploration.

## Code Examples
```bash
BTS=https://transtats.bts.gov/PREZIP
BASEURL="${BTS}/On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
curl -k -o temp.zip ${BASEURL}_${YEAR}_${MONTH}.zip
unzip temp.zip
```
- **What it demonstrates**: reverse-engineering a web form's static download URL pattern to script monthly BTS flight-data downloads instead of using the fragile interactive form.

```bash
gsutil mb -l $REGION gs://$BUCKET
gsutil -m cp *.csv gs://$BUCKET/flights/raw/
bq load --autodetect --source_format=CSV \
   dsongcp.flights_auto \
   gs://${BUCKET}/flights/raw/201501.csv
```
- **What it demonstrates**: the standard GCS-as-staging-ground → BigQuery load pattern recommended over direct on-prem-to-BigQuery ingestion for larger files.

```python
def ingest(year, month, bucket):
    tempdir = tempfile.mkdtemp(prefix='ingest_flights')
    try:
        zipfile = download(year, month, tempdir)
        bts_csv = zip_to_csv(zipfile, tempdir)
        gcsloc = f'flights/raw/{year}{month}.csv.gz'
        gcsloc = upload(bts_csv, bucket, gcsloc)
        return bqload(gcsloc, year, month)
    finally:
        shutil.rmtree(tempdir)
```
- **What it demonstrates**: converting ad hoc Bash exploration scripts into a reusable, idempotent Python ingestion function (download → gzip → upload → load) suitable for wrapping in a service.

```bash
gcloud run deploy $NAME --region $REGION --source=$(pwd) \
    --platform=managed --service-account ${SVC_EMAIL} \
    --no-allow-unauthenticated --timeout 12m

gcloud scheduler jobs create http monthlyupdate \
    --schedule="8 of month 10:00" --time-zone "America/New_York" \
    --uri=$SVC_URL --http-method POST \
    --oidc-service-account-email $SVC_EMAIL --oidc-token-audience=$SVC_URL \
    --max-backoff=7d --max-retry-attempts=5 --max-retry-duration=2d --min-backoff=12h
```
- **What it demonstrates**: deploying an authenticated, service-account-scoped Cloud Run service and wiring Cloud Scheduler to invoke it monthly with full retry/backoff configuration — the complete serverless ingestion automation pattern.

## Reference Tables
| Scaling strategy | Best for | Google Cloud implementation | Drawback |
|---|---|---|---|
| Scale up | Jobs re-reading data many times (ML training) | Deep Learning VM + persistent/local SSD | Limited by largest available machine |
| Scale out, sharded | Splittable batch jobs (e.g., search index) | Cloud Dataproc + HDFS, MapReduce | Wasteful unless sustained high utilization |
| Scale out, data-in-place | Most GCP-native analytics/ML | BigQuery, Cloud Dataflow, Vertex AI on GCS/Colossus | Requires GCP-class data center bisection bandwidth |

| Storage | Use when |
|---|---|
| Cloud Storage | Non-tabular data, or custom code reading entire files/datasets |
| BigQuery | Tabular data with interactive SQL query access patterns |

## Worked Example
End-to-end ingestion pipeline for the Airline On-Time Performance dataset: (1) identify BTS as the authoritative source, read field documentation to select causally-valid columns (e.g., `Reporting_Airline`, `FlightDate`, `Origin`, `Dest`); (2) reverse-engineer the BTS download form's URL pattern via Chrome DevTools to script monthly ZIP downloads; (3) download, unzip, and gzip each month's CSV; (4) create a single-region GCS bucket (`${PROJECT}-dsongcp`) and upload raw CSVs with `gsutil -m cp`; (5) create a BigQuery dataset `dsongcp`, load CSVs with `bq load`, moving from `--autodetect` to an explicit schema with `--time_partitioning_field=FlightDate`; (6) convert the Bash exploration scripts into a Python `ingest_flights.py` module (download/zip_to_csv/upload/bqload) with a `next_month()` helper that inspects what's already in GCS; (7) wrap it in a Flask app + Dockerfile, deploy to Cloud Run under a dedicated `svc-monthly-ingest` service account with `--no-allow-unauthenticated`; (8) schedule it with Cloud Scheduler to POST to the Cloud Run URL on the 8th of each month, with multi-day retry/backoff. End state: 2015 flight data lives in both GCS (raw, durable) and BigQuery (queryable), refreshed automatically going forward.

## Key Takeaways
1. Before using any field as a model input, verify it passes the causality test — will it be known at prediction time?
2. Design features so computation is identical between training and serving to avoid training-serving skew.
3. Prefer ELT (hub-and-spoke) over ETL: land raw data first, transform later via SQL views.
4. Choose single-region GCS buckets for internal analytics/ML workloads (cheaper, lower latency, strong consistency) unless cross-region availability is a hard requirement.
5. Stage external data in Cloud Storage before loading into BigQuery, especially for larger files; use `gsutil` for resumable multithreaded uploads.
6. Move off `--autodetect` to an explicit BigQuery schema (and consider date partitioning) once past initial exploration.
7. Automate periodic ingestion with Cloud Run + Cloud Scheduler + a least-privilege service account instead of a cron job on a VM — gets resilience, retries, and auditability for free.

## Connects To
- **Ch 1**: Implements the data acquisition step needed for the flight-delay probabilistic decision framework introduced there.
- **Ch 3**: The BigQuery tables created here (`dsongcp.flights_auto`/`flights_raw`) are the data source for the dashboards built next.
- **Ch 12**: The same ingestion pipeline is reused and extended to ingest the full 2015–2019 multi-year dataset.
- **BigQuery: The Definitive Guide** (same author): referenced as the deeper reference for BigQuery internals covered briefly here.
