# Chapter 5: Explore the Dataset

## Core Idea
Before feature engineering, use SageMaker Studio notebooks plus Athena/Redshift SQL to surface correlations, trends, and class imbalances in the dataset, then use Deequ (data quality), SageMaker Data Wrangler/Clarify (bias detection), and Glue DataBrew (visual analysis) to systematically detect and start mitigating data-quality and bias issues before they propagate into model bias.

## Frameworks Introduced
- **PyAthena + pandas (`read_sql`)**: run Athena SQL directly from a SageMaker Studio notebook into a DataFrame.
  - When to use: ad hoc exploratory queries on S3 data from within Jupyter.
  - How: `connect(region_name=..., s3_staging_dir=...)` then `pd.read_sql(sql, conn)`.
- **SQLAlchemy + Redshift**: query the warehouse directly from a notebook.
  - When to use: deeper, relational analysis on warehouse-resident subsets (e.g., 2015 data).
  - How: `pd.read_sql_query(sql, engine)`.
- **AWS Deequ / PyDeequ (via SageMaker Processing)**: Spark-based unit-testing framework for data quality at scale, producing constraints, metrics, and suggestions.
  - When to use: validating 100M+ row datasets for completeness, uniqueness, type conformity, and correlations, and continuously monitoring for "training-serving skew."
  - How: launch a `PySparkProcessor` SageMaker Processing Job with the Deequ JAR; define `AnalysisRunner` analyzers and `VerificationSuite` checks; optionally persist to a `FileSystemMetricsRepository` for trend tracking.
- **SageMaker Data Wrangler + SageMaker Clarify**: ML-focused data analysis/bias-detection tooling with full lineage tracking.
  - When to use: measuring class imbalance and other pre-training bias metrics on a sensitive "facet" column (e.g., `product_category`).
  - How: configure a `BiasConfig` (facet name/values, label values) and `DataConfig`, run `clarify_processor.run_pre_training_bias()`, or use the open-source `smclarify` library (`report.bias_report()`) directly in Python.
- **AWS Glue DataBrew**: no-code visual data analysis tool for correlations, missing cells, duplicates, and class imbalance — not integrated with SageMaker lineage.
  - When to use: quick, codeless exploratory analysis; prefer SageMaker Data Wrangler when ML lineage tracking matters.
- **HyperLogLog (HLL) approximate counting**: probabilistic cardinality estimation supported natively by Athena/Redshift.
  - When to use: fast COUNT DISTINCT on huge datasets where <2% error is acceptable (not for financial/exact reporting).
  - How: `SELECT APPROXIMATE COUNT(DISTINCT col) ...`.

## Key Concepts
- **Training-serving skew**: a model trained on one data distribution encounters new data at prediction time that violates the learned constraints — a sign of data drift.
- **Covariate shift / label shift / concept shift**: three drift types — shift in input distribution, shift in output/label distribution, and change in what a label actually means given context (e.g., regional word usage).
- **Class imbalance (CI)**: bias metric ranging [-1, 1]; near 0 = balanced, near ±1 = imbalanced — triggers the need for balancing before training.
- **Facet**: the sensitive or scrutinized column (e.g., `product_category`, or more typically age/race) used as the grouping variable in bias analysis.
- **Selection bias**: when overrepresented classes get better model performance than underrepresented ones due to imbalanced training data.
- **AQUA (Advanced Query Accelerator)**: hardware-accelerated distributed cache for Redshift that computes directly in cache to reduce storage-to-compute network I/O.
- **QuickSight SPICE**: "Super-fast, Parallel, In-memory Calculation Engine" powering low-latency QuickSight dashboards.

## Mental Models
- **Exploration precedes engineering**: correlation/imbalance/bias findings from this chapter directly drive feature selection and engineering decisions in Ch. 6, and model bias/fairness analysis in Ch. 7/9 — EDA is not a throwaway step.
- **Notebooks vs. clusters division of labor**: notebooks are fine for small-scale exploratory analytics; large-scale data-quality jobs (150M+ rows) belong on ephemeral, serverless Spark clusters via SageMaker Processing Jobs.
- **AWS Data Wrangler (library) vs. SageMaker Data Wrangler (ML service) — same name, different purpose**: the open-source library moves data between AWS storage services; the SageMaker feature is for ML-specific ingestion/analysis/bias/transformation with lineage tracking. Always disambiguate by context.
- **Drift detection as a closed loop**: baseline statistics captured at training time → live metrics compared against baseline in production (Ch. 9) → threshold breach triggers automatic retraining/alerting.

## Anti-patterns
- **Treating approximate counts as exact**: HLL-based APPROXIMATE COUNT is unsuitable for financial reporting or any use case requiring precise numbers.
- **Ignoring class imbalance before training**: leads to models that disproportionately favor overrepresented classes (e.g., Gift Card reviews) at the expense of underrepresented ones (e.g., Digital_Software).
- **Running huge Deequ/Spark jobs inside a notebook**: notebooks lack the memory/compute for datasets of hundreds of millions of rows — use SageMaker Processing Jobs with a dedicated Spark cluster instead.
- **Skipping continuous data-quality monitoring**: data quality degrades over time as schemas evolve and new app features are introduced; one-time validation isn't sufficient.

## Code Examples
```python
from pydeequ.checks import *
from pydeequ.verification import *

verificationResult = VerificationSuite(spark) \
        .onData(dataset) \
        .addCheck(
            Check(spark, CheckLevel.Error, "Review Check") \
                .hasSize(lambda x: x >= 150000000) \
                .hasMin("star_rating", lambda x: x == 1.0) \
                .hasMax("star_rating", lambda x: x == 5.0) \
                .isComplete("review_id") \
                .isUnique("review_id") \
                .isComplete("marketplace") \
                .isContainedIn("marketplace", ["US", "UK", "DE", "JP", "FR"])) \
        .run()
```
- **What it demonstrates**: defining explicit data-quality assertions (size, bounds, completeness, uniqueness, allowed values) that halt the pipeline if violated.

```python
data_bias_config = clarify.BiasConfig(
        label_values_or_threshold=[5, 4],
        facet_name='product_category',
        facet_values_or_threshold=['Gift Card'],
        group_name=product_category)

clarify_processor.run_pre_training_bias(
    data_config=data_config,
    data_bias_config=data_bias_config,
    methods='all',
    wait=True)
```
- **What it demonstrates**: running a full SageMaker Clarify pre-training bias report (CI, DPL, JS, KL, KS, LP, TVD, CDD) as a managed SageMaker Processing Job.

```python
# Mitigate class imbalance by downsampling to the smallest group size
df_grouped_by = df.groupby(["product_category", "star_rating"])[["product_category", "star_rating"]]
df_balanced = df_grouped_by.apply(
        lambda x: x.sample(df_grouped_by.size().min()).reset_index(drop=True)
        )
```
- **What it demonstrates**: a simple downsampling strategy to balance a dataset across facet/label combinations before re-running bias metrics (which drop to near 0 after balancing).

```sql
SELECT APPROXIMATE COUNT(DISTINCT customer_id)
FROM {}.{}
GROUP BY product_category
```
- **What it demonstrates**: HyperLogLog-based approximate COUNT DISTINCT, ~160% faster than exact COUNT with <2% error, on Redshift/Athena.

## Reference Tables
| Bias metric | What it measures |
|---|---|
| CI (Class Imbalance) | Imbalance in facet representation, range [-1,1], 0=balanced |
| DPL (Difference in Positive Proportions in Labels) | Disparity in positive-label rate between facet groups |
| JS (Jensen-Shannon Divergence) | Distributional distance between facet groups |
| KL (Kullback-Leibler Divergence) | Asymmetric distributional distance |
| KS (Kolmogorov-Smirnov) | Max distance between cumulative distributions |
| LP (Lp-norm) | Generalized distance metric between distributions |
| TVD (Total Variation Distance) | Distributional distance metric |
| CDD (Conditional Demographic Disparity) | Disparity in labels conditioned on a grouping variable |

| Tool | Best for |
|---|---|
| Athena + PyAthena | Ad hoc SQL EDA on S3 data from notebooks |
| Redshift + SQLAlchemy | Deeper relational EDA on warehouse subset |
| Deequ/PyDeequ (via SageMaker Processing) | Large-scale data-quality unit tests, metric tracking over time |
| SageMaker Data Wrangler + Clarify | ML-focused bias/imbalance detection with lineage |
| Glue DataBrew | Quick, no-code correlation/imbalance visualization (no lineage) |
| QuickSight | BI dashboards across many data sources |

## Worked Example
The Amazon Customer Reviews Dataset is explored via a sequence of Athena queries answering: highest-rated category (Gift Cards, 4.73 avg), most-reviewed category (Books, ~20M reviews), category launch dates (via MIN(year)), star-rating distribution (62% are 5-star — revealing imbalance), rating trends over time, helpfulness by rating (1-star reviews are most "helpful"), and review-length distribution (80% ≤63 words). The same dataset is then validated with a 10-node Deequ/Spark SageMaker Processing Job (`ml.r5.8xlarge` × 10) checking completeness, uniqueness, allowed values, and correlations (finding `helpful_votes` and `total_votes` strongly correlated, `star_rating` weakly correlated with votes). Finally, SageMaker Clarify's pre-training bias report on the `product_category` facet reveals a 0.45 class imbalance for Gift Card reviews; downsampling each (category, star_rating) group to the smallest group size brings CI down to the balanced value of 0.33 (for 3 categories).

## Key Takeaways
1. Always profile for correlations, trends, and class imbalance before feature engineering — these findings directly shape later decisions.
2. Move large-scale data-quality checks (Deequ) to SageMaker Processing Jobs/Spark clusters, never run them in-notebook.
3. Measure class imbalance (CI) and other bias metrics on sensitive facets before training to avoid selection bias in the model.
4. Use HyperLogLog approximate counts for fast, large-scale cardinality estimation — but never for exact/financial reporting.
5. Distinguish covariate shift, label shift, and concept shift when reasoning about why a model's performance degrades over time.
6. Prefer SageMaker Data Wrangler over Glue DataBrew when ML pipeline lineage tracking is required.

## Connects To
- **Ch 4**: Builds directly on the Athena/Redshift ingestion pipeline and Amazon Customer Reviews Dataset introduced there.
- **Ch 6**: Feature selection/engineering will directly use the imbalance and correlation findings from this chapter.
- **Ch 7/9**: Post-training bias metrics and model-explainability/drift monitoring (SageMaker Clarify, Model Monitor) build on the pre-training bias concepts introduced here.
