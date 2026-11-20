# Chapter 12: The Full Dataset

## Core Idea
Scaling the entire pipeline from one year (2015, ~5M flights) to four years (2015–2018, ~30M flights) for training and holding out a truly independent year (2019) for evaluation reveals that model performance transfers reasonably well across time, but surfaces real-world operational necessities — continuous/retraining, handling schema drift (new airports), and far larger infrastructure — that a single-year prototype never exposes; it closes with a practitioner's governance framework for sensitive data in ML datasets.

## Frameworks Introduced
- **True temporal holdout evaluation**: train on several past years, evaluate on a subsequent, never-touched year to simulate real production deployment.
  - When to use: final validation of a model before or instead of claiming production-readiness, especially for datasets dependent on human/world behavior that drifts over time.
  - How: train on 2015–2018 (using `FARM_FINGERPRINT`-based 95/5 TRAIN/VALIDATE split within those years), hold out 2019 entirely as TEST (`fl_date_str > '2019'` → TEST); explicitly avoid spanning the training data across a known regime change (the chapter stops at 2019, excluding 2020–2021 COVID-19-disrupted data, since no model trained on pre-pandemic data could be expected to generalize through it).
- **Graceful degradation for schema drift**: handle entities unseen during pipeline design (e.g., new airports) without crashing the pipeline.
  - When to use: any production pipeline ingesting multi-year or ever-growing categorical vocabularies.
  - How: wrap category lookups in a function that falls back to a sensible default when the key is missing (e.g., unknown airport → coordinates of the US population center) rather than raising an exception; treat this as a stopgap, not a substitute for keeping reference data (e.g., `airports.csv`) current via continuous training.
- **Global reshuffling before writing out training data**: break up data clumping introduced by upstream grouped/windowed processing.
  - When to use: any dataset produced by a pipeline that groups/windows by key (e.g., by airport and hour) before writing records out, where downstream training relies on batch-level representativeness.
  - How: `beam.WindowInto(beam.window.GlobalWindows())` followed by `beam.util.Reshuffle()` right before writing output — must discard time-window keying first, or the reshuffle only shuffles within each window, not globally.
- **Sliced evaluation via exported BigQuery evaluation tables**: analyze model behavior broken out by categorical slices (airline, airport, error type) rather than only aggregate metrics.
  - When to use: diagnosing where and why a model errs, verifying it has learned sensible relationships, and catching illogical behavior before shipping.
  - How: use AutoML's `export_evaluated_data_items_bigquery_destination_uri` (from Ch. 10) to get a queryable table of predictions + ground truth; compute RMSE, confusion matrices at multiple thresholds (`UNNEST([...]) AS thresh`), feature-vs-prediction relationships, and error-vs-feature relationships, all in SQL, then visualize in a notebook.
- **Sensitive-data governance framework** (from the appendix): a three-goal process for handling sensitive fields in ML datasets — identify, protect, govern.
  - When to use: any ML dataset that may contain personally identifying or otherwise sensitive information.
  - How: (1) identify sensitive data across five common scenarios (columns, natural-language text via known patterns, free-form unstructured content via NLP/Vision/Speech APIs or the DLP API, combinations of fields, embedded contextual information); (2) protect it via removal (views excluding columns), masking (hashing/encryption/tokenization, with the caveat that salted hashes break repeatability needed for ML and all masking remains vulnerable to frequency-analysis attacks), or coarsening (reduce precision — zip5→zip3, lat/lon rounding, age→decade, IP address octet zeroing); (3) establish and document a governance policy (secure documentation location, known sensitive-data sources/locations, access-request process, periodic access review, off-boarding process).

## Key Concepts
- **Continuous training vs. retraining-from-scratch vs. fine-tuning**: real-world ML models can't be trained once and used forever — entities change (new airports, new carriers, economic conditions); retraining from scratch uses all historical data and yields the best overall benefit but is slow; fine-tuning (resuming from a checkpoint, running a few more batches, possibly freezing some layers) is much faster but gives a smaller benefit — the typical practice is to fine-tune frequently (daily/weekly) and fall back to full retraining once enough new data has accumulated (e.g., ~5% of the original dataset size).
- **Dataflow pipeline bottleneck scaling at multi-year volume**: grouping/windowing operations (`group_by_airport`) that worked fine on 1 year of data became the throughput bottleneck at 4 years' scale (processing ~2,500–800 elements/sec, implying tens of hours), requiring far larger, higher-memory machines (`m1-ultramem-40`) — a concrete illustration that pipeline performance characteristics don't scale linearly and must be re-profiled at new data volumes.
- **Why AutoML was chosen over the custom wide-and-deep model for the full-scale run**: not because it was more accurate, but because AutoML's evaluation-data export to BigQuery made sliced/detailed evaluation dramatically easier to implement — an explicit engineering-cost-vs-marginal-accuracy trade-off.
- **Global feature importance stability across data scale**: the top 5 most important features (dep_delay, taxi_out, origin, dest, carrier) were identical between the 2015-only model and the 2015–2018 model, with only features 4 and 5 swapping rank — evidence the model's learned structure is robust to the 6x data-volume increase.
- **Frequency analysis attack**: using known real-world frequency distributions (e.g., of first names, zip codes) to de-anonymize masked/hashed/tokenized/coarsened fields in an aggregate dataset — the central threat model that makes every masking technique (encryption, hashing, tokenization, generalization, PCA-based dimensionality reduction) imperfect and why governance/access-control must always back up technical masking.

## Mental Models
- **A model that seems to "lose" to a simpler baseline on small data might win on more data, and vice versa**: AutoML's sophisticated architectures (neural architecture search, TabNet-style sequential attention) are explicitly data-hungry and may close the gap with, or surpass, a hand-engineered model as data volume grows — "there is no way to know; we'd have to try both."
- **Performance degradation on a temporally-independent test set is expected and informative, not a failure**: RMSE rising slightly (0.198 on 2015 holdout → 0.1998 on 2019) reflects genuine real-world drift (new carriers, schedule changes, economic conditions) rather than a bug — the point of the exercise is to quantify how much drift costs, not to eliminate it.
- **"80% of the time used to be spent on plumbing" — infrastructure cost has inverted**: the book's closing reflection frames the entire project's lesson: serverless, managed GCP services collapsed what used to be the dominant cost of a data science project (ingest, dashboards, stream processing, operationalization) into something foldable into model development itself.
- **Removing sensitive data is not free — it trades security for utility**: the appendix is explicit that blanket removal, random tokenization, or over-aggressive masking can degrade ML training by destroying genuinely predictive signal; the practitioner's job is to find the point on that trade-off curve appropriate to the specific risk, not to default to maximal removal.

## Anti-patterns
- **Assuming a pipeline's runtime and machine-sizing from the prototype scale will linearly extrapolate to production scale**: the group-by-airport bottleneck emerged only at 4-years'-scale and required a completely different machine class (`m1-ultramem-40` vs. default `n1-standard-1`), not just "the same machine running longer."
- **Crashing on unseen categorical values instead of degrading gracefully**: a rigid lookup that throws on an unknown airport code breaks the entire pipeline the moment reference data (airports.csv) goes stale — always have a documented fallback/default.
- **Reshuffling data without first discarding windowing/grouping metadata**: reshuffling "within the window" instead of globally silently fails to fix the clumping problem it was meant to solve.
- **Relying solely on encryption/hashing/tokenization/coarsening as sufficient protection for sensitive ML data**: all of these remain vulnerable to frequency-analysis attacks, especially when multiple masked fields can be jointly analyzed — masking must always be paired with access governance, not treated as a complete solution on its own.
- **Using a salted hash for a field that needs to remain usable for ML training**: destroys the repeatability (same input → same output) that ML models and joins depend on; use a strong unsalted cryptographic hash (e.g., SHA-256) instead when repeatability is required, while accepting the residual frequency-analysis risk.
- **Treating "the dataset is anonymized" as settled once sensitive columns are removed or masked once**: a combination of seemingly innocuous fields (e.g., zip3-of-work + zip3-of-home, or a zip3 trend over time) can still re-identify individuals — governance and ongoing risk review are required, not a one-time pass.

## Code Examples
```python
def get_data_split_2019(fl_date):
    fl_date_str = str(fl_date)
    if fl_date_str > '2019':
        return 'TEST'
    x = np.abs(np.uint64(farmhash.fingerprint64(fl_date_str)).astype('int64') % 100)
    return 'TRAIN' if x < 95 else 'VALIDATE'
```
- **What it demonstrates**: extending the date-hash-based split methodology (Ch. 5/8/10) to carve out an entire future year as a true independent test set, while still splitting the remaining years 95/5 for train/validate.

```python
features = (features
    | 'into_global' >> beam.WindowInto(beam.window.GlobalWindows())
    | 'shuffle' >> beam.util.Reshuffle()
)
```
- **What it demonstrates**: globally reshuffling training data written out by a windowed/grouped Beam pipeline, preventing the model from seeing clumped batches (same airport/same hour) that could trap optimization in a local minimum.

```python
def airport_timezone(airport_id, airport_timezones_dict):
    if airport_id in airport_timezones_dict:
        return airport_timezones_dict[airport_id]
    else:
        return ('37.41', '-92.35', u'America/Chicago')  # US population center fallback
```
- **What it demonstrates**: graceful degradation for a categorical lookup against reference data that has gone stale (new airports not present in the original `airports.csv`), rather than letting the pipeline fail outright.

```sql
WITH counts AS (
    SELECT thresh,
      COUNTIF(CAST(ontime AS FLOAT64) > 0.5 AND predicted_ontime.scores[OFFSET(0)] > thresh) AS num_1_as_1,
      ...
    FROM UNNEST([0.5, 0.7, 0.8]) AS thresh, dsongcp.ch10_automl_evaluated
    GROUP BY thresh
)
SELECT *, ROUND(num_1_as_1/(num_1_as_1+num_1_as_0), 2) AS frac_1_as_1, ... FROM counts ORDER BY thresh ASC
```
- **What it demonstrates**: computing confusion-matrix fractions across multiple decision thresholds in a single BigQuery pass, extending the Ch. 3/5 contingency-table technique to the final full-scale model evaluation.

## Reference Tables
| Evaluation | RMSE | Context |
|---|---|---|
| AutoML on 2015 data only (Ch. 9/10, random split subset) | 0.199 / 0.198 | Smaller-scale baseline |
| AutoML trained 2015–2018, evaluated on independent 2019 | 0.1998 (~0.200) | Slight degradation, expected given real-world drift |

| Masking technique | Trade-off |
|---|---|
| Removal (view excludes column, or auto-redaction via DLP API) | Simplest; loses signal if field was predictive |
| Hashing/encryption (unsalted, strong e.g. SHA-256) | Repeatable for ML joins, but vulnerable to frequency analysis |
| Tokenization (separate mapping DB) | Adds a layer of indirection, same frequency-analysis vulnerability as hashing if token reused per value |
| PCA / dimensionality reduction | Combines fields into one vector; harder to reverse-engineer but trades accuracy for security |
| Coarsening (zip5→zip3, lat/lon rounding, age→decade, IP octet zeroing) | Reduces precision while preserving aggregate utility; still has residual re-identification risk at large scale |

## Worked Example
Scales the entire flight-delay pipeline to its final, most realistic form: (1) change the Beam data-split logic to reserve all of 2019 as TEST and split 2015–2018 95/5 into TRAIN/VALIDATE via farmhash; (2) add global reshuffling after removing time-window keying, to prevent batch clumping by airport/hour; (3) discover the time-correction pipeline crashes on airports missing from the original `airports.csv` (new airports built since 2015) and patch it with a graceful fallback to the US population-center coordinates, explicitly flagging this as a symptom of needing continuous training rather than one-off fixes; (4) re-provision the Dataflow pipeline with much larger `m1-ultramem-40` machines after discovering the `group_by_airport` step bottlenecks at ~2,500 elements/sec (and the downstream stats step at ~800 groups/sec), taking 26 hours total to process 30M flights; (5) retrain using AutoML (chosen specifically for its BigQuery evaluation-export capability) on the full 2015–2018 dataset (~5 hours), finding the same top-5 feature importance ranking as the 2015-only model; (6) evaluate the resulting model against the independent 2019 test set, computing RMSE (0.1998, slightly worse than in-period evaluation), confusion matrices across multiple thresholds, feature-vs-prediction sanity checks (departure delay vs. predicted probability moves smoothly and sensibly), error analysis by taxi-out time, and categorical-slice analysis (the model learned that a 25-minute delay means something different at JFK vs. SEA, and for Alaska Airlines vs. Delta) — concluding the 2015–2018 model would have performed well in 2019, before COVID-19 upended the entire aviation system in 2020. The chapter (and book) closes with an appendix co-written with Brad Svee on identifying, protecting, and governing sensitive data within ML datasets.

## Key Takeaways
1. Always validate a model against a truly independent, temporally-later holdout (not just a held-out slice of the same training period) before trusting it reflects real production performance.
2. Expect infrastructure that worked at prototype scale to hit new bottlenecks at full production scale — re-profile pipeline steps and re-provision machine types rather than assuming linear scaling.
3. Build graceful fallbacks for categorical/reference-data lookups that may go stale (new entities appearing over time); treat this as a signal that continuous training/retraining is a required capability, not a one-time build step.
4. Reshuffle globally (discarding any windowing/grouping key first) before writing out training data produced by a grouped/windowed pipeline, to avoid batch clumping.
5. Choose between a custom model and AutoML based on total engineering cost (e.g., ease of sliced evaluation via BigQuery export), not solely on which currently has the lower error metric — and re-test the comparison at new data scales, since relative performance can flip.
6. Perform sliced/segmented evaluation (by airline, airport, feature value, error type) in addition to aggregate metrics to confirm the model has learned sensible, explainable relationships.
7. Treat sensitive-data protection as a three-part discipline — identify, protect (remove/mask/coarsen), govern (document and review access) — and never rely on masking alone, since nearly every masking technique remains vulnerable to frequency-analysis attacks at scale.

## Connects To
- **Ch 5 & 8 & 10**: Extends the date-hash-based train/test split methodology established in those chapters to a multi-year, truly-independent final evaluation.
- **Ch 9 & 10**: Directly reuses and retrains the AutoML and custom wide-and-deep models, and the evaluation-export capability, at full scale.
- **Ch 2**: The need for continuous training and graceful handling of new/changed reference data (airports) echoes the Ch. 2 emphasis on automated, scheduled ingestion pipelines (Cloud Run + Cloud Scheduler) as a model for how retraining should eventually be automated.
- **Data Governance: The Definitive Guide** (same author, Lakshmanan et al.): a natural follow-on reference for the sensitive-data governance material in this chapter's appendix.
