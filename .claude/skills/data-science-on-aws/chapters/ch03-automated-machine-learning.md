# Chapter 3: Automated Machine Learning

## Core Idea
SageMaker Autopilot and Amazon Comprehend both offer AutoML — but Autopilot is "transparent" AutoML, generating fully reproducible notebooks/scripts/pipelines alongside the model, in contrast to black-box AutoML tools that hide algorithm choice, transformations, and hyper-parameters.

## Frameworks Introduced
- **Amazon SageMaker Autopilot**: transparent AutoML that analyzes tabular S3 data, auto-detects problem type (regression/binary/multiclass classification), and generates 10 candidate pipelines (data-preprocessor + algorithm combos).
  - When to use: tabular ML problems where you want a fast, explainable, reproducible baseline.
  - How: provide S3 CSV + target column (+ optional problem type) via UI or `create_auto_ml_job()`; Autopilot runs Data Exploration + Candidate Definition notebooks, then feature engineering, training, and hyper-parameter tuning; select/deploy the best candidate by objective metric.
- **SageMaker Experiments**: automatic experiment/trial tracking underlying Autopilot, enabling lineage tracking across S3 locations, algorithms, hyper-parameters, and performance metrics.
  - When to use: any need to organize, compare, and reproduce many training runs (used natively by Autopilot).
- **Amazon Athena + SageMaker Autopilot (Athena ML)**: invoke a SageMaker endpoint as a SQL UDF.
  - When to use: scoring S3 data in bulk from SQL without exporting it.
  - How: `USING FUNCTION predict_x(...) RETURNS ... TYPE SAGEMAKER_INVOKE_ENDPOINT WITH (sagemaker_endpoint = '<name>')`, then reference the function in a `SELECT`.
- **Amazon Redshift ML + SageMaker Autopilot**: train and deploy a model directly from a Redshift query.
  - When to use: data already lives in a Redshift warehouse and needs in-warehouse training/prediction.
  - How: `CREATE MODEL ... FROM (SELECT ...) TARGET <col> FUNCTION <fn> IAM_ROLE ... SETTINGS (S3_BUCKET ...)`, which exports data to S3, triggers Autopilot, and deploys a predict function back into Redshift.
- **Amazon Comprehend Custom (AutoML for NLP)**: fine-tunes Comprehend's built-in NLP model on custom-labeled text.
  - When to use: text classification tasks (e.g., star-rating prediction) needing a custom label set.
  - How: `create_document_classifier()` on labeled CSV → `create_endpoint()` → `classify_document()`.

## Key Concepts
- **Transparent vs. nontransparent AutoML**: transparent AutoML (Autopilot) exposes algorithm choices, data transformations, and hyper-parameters via generated notebooks; nontransparent AutoML hides them, making models hard to explain/reproduce.
- **Data Exploration notebook**: Autopilot-generated notebook documenting data-quality issues (e.g., missing values) found during analysis.
- **Candidate Definition notebook**: Autopilot-generated notebook with the full code/config for each of the 10 model-candidate pipelines.
- **dpp[0-9] (data preprocessor pipelines)**: Autopilot's 10 numbered feature-engineering + algorithm pipeline variants, each producing its own training/tuning job.
- **Inference pipeline (3 containers)**: a deployed Autopilot model is actually 3 chained containers — data transformation (request handler), algorithm (serving), and inverse label transformer (response handler mapping output back to original labels).
- **TF/IDF**: term frequency–inverse document frequency; downweights common words, upweights unique ones, used as the Autopilot-generated text feature representation.
- **Trial components**: a SageMaker Autopilot trial = Processing Job (split data) → Training Job (per dpp) → Batch Transform Job (featurize) → Tuning Job (HPO), preserving full lineage.

## Mental Models
- **AutoML as baseline, not endpoint**: use Autopilot/Comprehend Custom to establish baseline performance quickly, then apply domain expertise to improve — not necessarily the final production model.
- **"Dry run" option**: you can stop Autopilot after the data-analysis phase (just the two notebooks) before committing to full training — cheap way to validate data quality first.
- **ARN as universal resource addressing**: `arn:partition:service:region:account-id:resource-id` is the consistent naming scheme across all AWS services — useful for reasoning about any AWS resource reference.
- **Bring ML to the data, not data to ML**: reinforced via Athena/Redshift ML UDFs — SQL queries call the model in place rather than exporting data to a separate ML environment.

## Anti-patterns
- **Treating AutoML models as inherently production-ready**: Autopilot results should be inspected (accuracy, AUC, F1) and validated against business objectives before deployment, not auto-trusted.
- **Using nontransparent AutoML when auditability matters**: if you need to explain, reproduce, or modify a model, black-box AutoML services are the wrong choice.
- **Skipping the Data Exploration notebook review**: missing values and data-quality issues surfaced there should be addressed, not ignored, before trusting downstream candidate rankings.

## Code Examples
```python
# Launch a SageMaker Autopilot job via the Python SDK
input_data_config = [{
    'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix',
                                     'S3Uri': 's3://<BUCKET>/amazon_reviews.csv'}},
    'TargetAttributeName': 'star_rating'
}]
output_data_config = {'S3OutputPath': 's3://<BUCKET>/autopilot/output/'}

sm.create_auto_ml_job(AutoMLJobName=auto_ml_job_name,
                      InputDataConfig=input_data_config,
                      OutputDataConfig=output_data_config,
                      AutoMLJobConfig=job_config,
                      RoleArn=role)
```
- **What it demonstrates**: fully programmatic AutoML job creation — no UI needed — targeting `star_rating` as the label.

```sql
USING FUNCTION predict_star_rating(review_body VARCHAR)
    RETURNS VARCHAR TYPE
    SAGEMAKER_INVOKE_ENDPOINT WITH (sagemaker_endpoint = 'reviews')
SELECT review_id, review_body,
       predict_star_rating(REPLACE(review_body, ',', ' ')) AS predicted_star_rating
FROM dsoaws.product_reviews
```
- **What it demonstrates**: real-time model inference embedded directly in an Athena SQL query over S3 data.

```sql
CREATE MODEL dsoaws.predict_star_rating
FROM (SELECT review_body, star_rating FROM dsoaws.amazon_reviews_tsv_2015)
TARGET star_rating
FUNCTION predict_star_rating
IAM_ROLE '<ROLE_ARN>'
SETTINGS (S3_BUCKET '<BUCKET_NAME>');
```
- **What it demonstrates**: Redshift ML training a SageMaker Autopilot model directly from a warehouse query, deploying a callable SQL function.

```python
# Amazon Comprehend Custom: train, deploy, and predict
training_job = comprehend.create_document_classifier(
    DocumentClassifierName='Amazon-Customer-Reviews-Classifier-'+id,
    DataAccessRoleArn=iam_role_comprehend_arn,
    InputDataConfig={'S3Uri': 's3://<bucket>/<path>/amazon_reviews.csv'},
    OutputDataConfig={'S3Uri': 's3://<bucket>/<path>/model/outputs'},
    LanguageCode='en')
```
- **What it demonstrates**: training a custom multiclass text classifier via Comprehend's AutoML SDK with just a handful of parameters.

## Reference Tables
| Trial component | Role |
|---|---|
| Processing Job | Splits data into train/validation, separates headers |
| Training Job | Trains model with dpp[0-9].py preprocessor per candidate |
| Batch Transform Job | Converts raw data into algorithm-specific features |
| Tuning Job | HPO across transformed features to find best candidate |

| Inference pipeline container | Function |
|---|---|
| Data transformation | Converts raw input (text) into model format (e.g., TF/IDF vectors) |
| Algorithm | Produces raw prediction (e.g., index 0-4) |
| Inverse label transformer | Maps index back to original label (e.g., star_rating 1-5) |

## Worked Example
End-to-end: the Amazon Customer Reviews Dataset (`review_body`, `star_rating`) is fed to SageMaker Autopilot via UI or SDK, pointing at S3 and target column `star_rating`. Autopilot auto-detects multiclass classification, builds 10 dpp[0-9] pipelines (mixing preprocessors with algorithms like XGBoost, linear learner), tunes each via HPO, and ranks candidates by validation accuracy (best observed: ~0.44). The best candidate's 3-container inference pipeline is deployed as a SageMaker REST endpoint; invoking it with `"I loved it!"` returns `"5"`, and `"It's OK."` returns `"3"`. The same dataset is independently run through Amazon Comprehend Custom's `create_document_classifier()`, yielding a model that is 97% confident `"It's OK."` deserves a 3-star rating — showing two independent AutoML paths converging on comparable results.

## Key Takeaways
1. Prefer transparent AutoML (SageMaker Autopilot) over black-box AutoML whenever reproducibility/auditability matters.
2. Always inspect the Data Exploration notebook for data-quality issues before trusting candidate rankings.
3. A deployed Autopilot model is a 3-container inference pipeline (transform → algorithm → inverse-transform) — know this when debugging prediction issues.
4. Use Athena/Redshift ML UDFs to score data in place via SQL instead of building separate inference pipelines.
5. Comprehend Custom is the fastest path to a custom NLP classifier when text classification is the entire problem.
6. Use AutoML results as a baseline to calibrate how much lift custom modeling (Ch. 7+) can realistically add.

## Connects To
- **Ch 2**: Builds directly on Amazon Comprehend and Redshift ML/Athena ML examples introduced there.
- **Ch 6**: Feature engineering (TF/IDF, data preprocessors) referenced here is covered in depth.
- **Ch 7**: Custom BERT-based text classifier (the "non-AutoML" alternative to this chapter's approach) built next.
- **Ch 8**: Standalone SageMaker Hyper-Parameter Tuning Jobs (used internally by Autopilot here) explained in depth.
