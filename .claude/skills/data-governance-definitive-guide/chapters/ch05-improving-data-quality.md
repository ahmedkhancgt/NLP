# Chapter 5: Improving Data Quality

## Core Idea
Data quality — accuracy, completeness, and timeliness, always evaluated relative to a specific business use case — directly determines the trustworthiness of decisions and AI/ML models built on the data; quality issues compound as data moves through a pipeline, so they must be addressed as early (upstream) as possible.

## Frameworks Introduced
- **Three-Dimensional Data Quality Model**: Accuracy, Completeness, Timeliness
  - When to use: as the baseline rubric for evaluating any dataset or source.
  - How: Accuracy = correctness (catches entry errors, duplicates); Completeness = no missing required fields; Timeliness = freshness/staleness and correct event-ordering relevance to the use case.

- **Three Key Data Quality Techniques: Prioritization, Annotation, Profiling**
  - **Prioritization**: use lineage to identify which sources feed the most business-critical outputs; spend cleanup effort there first.
  - **Annotation**: attach a standardized quality signal to datasets ("vetted" / "not vetted"); can be crowdsourced via user "star rating" plus curator review of highly-rated data before it's promoted as trusted.
  - **Profiling**: generate a data profile (min/max/cardinality/missing/outliers) to determine legal values and value meaning before using the data.

- **Data Quality Scorecard**
  - When to use: before using any new data source in a pipeline.
  - How: capture origin, accuracy, completeness, timeliness, administrative owner, and requester — used by pipeline builders to decide how/where to use the source.

## Key Concepts
- **Data acquisition chain**: gathering → landing-zone normalization/aggregation → analytics workload; errors can be introduced and amplified at any step before reaching the business user.
- **Dataset source quality ranking**: assign a trust rank to each source so conflicting field values (e.g., differing "zero" vs. "negative-1" sentinel conventions) can be resolved deterministically.
- **Data deduplication / entity resolution**: collapsing multiple records referring to the same real-world entity (name variants, address variants) into a canonical identifier.
- **Natural experiment**: a dataset where "treatment" and "control" groups arise organically (not researcher-assigned) — enables statistically valid comparison without a controlled lab study (used in the Nike Vaporfly case, Ch. 4).
- **Positive feedback loop (ML)**: when an ML model's predictions become inputs to future training, amplifying any original data errors into systematically worse predictions over time.
- **Training/validation/test split**: the three non-overlapping datasets used to build, tune, and evaluate ML models; errors in any one degrade the resulting model.

## Mental Models
- **Quality beats quantity**: even in the AI era of "more data is better," a small amount of correctly-labeled data outperforms a large amount of noisy/fabricated data (Italian water-meter RNN case).
- **Fix quality as far upstream as possible**: cleanup gets progressively harder and more costly to backtrack once data is aggregated or fed into ML training.
- **Business-case-specific quality bar**: "reviewing transactions" means something different to a marketing team (big customers) than a fraud team (provable fraud) — there's no universal "clean" standard independent of use case.

## Anti-patterns
- **Treating data cleanup as a one-time, source-agnostic action**: billing corrections that aren't propagated back to original source data (Italian water-meter case) corrupt all downstream model training.
- **Automatically discarding statistical outliers without investigation**: some "outliers" are real phenomena the use case must account for (e.g., legitimate late-night online orders) — blind elimination introduces bias.
- **Ignoring full business context when deploying automated filters**: a profanity filter trained for a K-12 education context, deployed for a paleontology conference, banned "bone," "jerk," and flagged "Wang" as inappropriate while allowing "Johnson" — showing unintended bias from context mismatch.
- **Relying on "tribal knowledge" for dataset trustworthiness**: breaks down post-acquisition when the people who knew the data have left, leaving valuable data unusable (healthcare company case).
- **Merging datasets without reconciling special-value conventions**: "null" vs. "zero" sentinel mismatches between sources silently corrupt joins if not normalized and documented.

## Code Examples
No executable code; the chapter instead specifies **profiling/cleanup techniques** as a procedure:
1. Compute min/max/cardinality per field; flag missing and out-of-bounds values.
2. Decide whether outlier records are deleted outright or investigated (bell-curve edge spike = investigate; bimodal distribution = may be legitimate).
3. Deduplicate entities (names/places) using canonicalization services (e.g., Google Knowledge Graph API, Google Places API + Geocoding API to resolve different place IDs to the same geolocation).
4. Document special sentinel values during merges (null vs. zero) and normalize them, recording the chosen convention.
- **What it demonstrates**: a concrete order-of-operations for cleaning a dataset before it enters analytics or ML pipelines.

## Reference Tables
No formal table; Figure 5-1 ("simple data acquisition chain": gather → landing zone → analytics workload) functions as the chapter's core diagram for where errors enter and compound.

## Worked Example
**Italian water-meter failure detection (RNN case)**: Goal: detect malfunctioning mechanical water meters from readings alone (avoiding costly site visits), using 15M readings from 1M meters, training an RNN. Initial attempt without quality scrutiny gave poor results. Root cause: (1) ~1% of readings were simply *wrong* due to billing-correction propagation gaps (technician reading → ERP entry → bill; bill got corrected but the underlying meter reading record did not) — the model was trained on random noise for these; (2) 31% of "measurements" were actually *interpolated/made-up* values (skipped readings extrapolated and corrected later), adding further noise. After removing billing errors and interpolated values (discarding a large % of original data), RNN accuracy reached 85% (vs. 79% for a simpler linear regression) — demonstrating that correctly governed, smaller data beat a larger noisy dataset.

**MaxMind IP geolocation default-location bug**: MaxMind defaulted unknown-location IP addresses to a point in northern Kansas. Downstream systems and law-enforcement users were unaware of this default's meaning, leading law enforcement to serve warrants to innocent residents near that default coordinate for activity attributed to "unknown" IPs — illustrating how an undocumented default/sentinel value propagating downstream causes real-world harm.

**California COVID-19 case reporting**: Multiple successive data-quality regressions — switching from per-county to state reporting (introduced inconsistency), switching from "people tested" to "specimens tested" (double-counting risk), and a failed data-lab link causing underreporting — affected 296,000 records and took nearly a month to fully recover from, illustrating cascading, compounding real-world data quality failure.

## Key Takeaways
1. Data quality must always be assessed relative to the specific business use case — there is no context-free definition of "clean."
2. Fix data quality issues as early (upstream) in the pipeline as possible; later-stage fixes are costlier and sometimes impossible after aggregation.
3. In ML specifically, quality beats quantity — correct, small datasets outperform large noisy/fabricated ones.
4. Use lineage to prioritize quality investment toward the sources feeding the most business-critical outputs.
5. Replace tribal knowledge with standardized annotation/scorecards before institutional memory is lost (e.g., via acquisition).
6. Always propagate corrections (e.g., billing fixes) back to the original source record, not just the downstream derivative.
7. Investigate statistical outliers in context before deleting — some are errors, some are real and use-case-relevant signal.

## Connects To
- **Ch 2**: Lineage tracking tooling introduced there is the mechanism used here for source prioritization.
- **Ch 4**: The "Document Data Quality Expectations" callout during the Data Processing phase is expanded fully here.
- **Ch 6**: Data-in-flight governance depends on the quality/validation concepts from this chapter.
- **Ch 8**: Data Quality Monitoring in Ch. 8 operationalizes this chapter's scorecard/baseline concepts into ongoing alerting.
