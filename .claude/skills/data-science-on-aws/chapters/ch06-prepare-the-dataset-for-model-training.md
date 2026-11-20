# Chapter 6: Prepare the Dataset for Model Training

## Core Idea
Feature engineering converts raw review text into BERT embeddings (via DistilBERT) and TFRecords at scale with SageMaker Processing Jobs, while balancing classes, splitting data without leakage, and tracking full lineage through SageMaker Feature Store, Lineage, and Experiments.

## Frameworks Introduced
- **SageMaker Data Wrangler "Quick Model"**: trains a disposable baseline model to rank feature importance.
  - When to use: deciding which columns actually matter before committing to feature engineering (here, `review_body` ranked highest).
- **BERT / DistilBERT (Hugging Face Transformers)**: bidirectional transformer language model producing contextual word embeddings; DistilBERT is 60% faster, 40% smaller, 97% as accurate as BERT.
  - When to use: NLP feature engineering where word meaning depends on surrounding context (unlike Word2Vec or TF-IDF).
  - How: `DistilBertTokenizer.from_pretrained('distilbert-base-uncased')` → `tokenizer.encode_plus(text, pad_to_max_length=True, max_length=MAX_SEQ_LENGTH)` produces `input_ids`, `input_mask` (attention_mask); `segment_ids` are generated manually (all 0s for single-sequence tasks).
- **TFRecord format**: binary, protobuf-based TensorFlow-optimized file format for sequential, row-based training data access.
  - When to use: any TensorFlow-based large-scale model training pipeline.
  - How: wrap `input_ids`/`input_mask`/`segment_ids`/`label_ids` in `tf.train.Feature`/`tf.train.Example`, write via `tf.io.TFRecordWriter`.
- **SageMaker Processing Jobs (scikit-learn or Apache Spark)**: parallelize feature engineering (balancing, splitting, BERT transformation) across a cluster.
  - When to use: scaling feature engineering beyond what a single notebook/instance can handle (150M+ reviews).
  - How: `SKLearnProcessor`/`PySparkProcessor` with `instance_count`/`instance_type`, `ShardedByS3Key` input distribution, and `ProcessingInput`/`ProcessingOutput` for train/validation/test BERT output.
- **SageMaker Feature Store**: managed, centralized, discoverable feature repository with offline (batch/training) and online (low-latency/inference) stores.
  - When to use: sharing expensive-to-compute features (like BERT embeddings) across teams/models and preventing train/inference feature skew.
  - How: `FeatureGroup(name=...)`, `load_feature_definitions()`, `.create(s3_uri=..., record_identifier_name=..., event_time_feature_name=..., enable_online_store=True)`, `.ingest(data_frame=...)`; query offline store via `.athena_query()` or online store via `get_record()`.
- **SageMaker Lineage Tracking + Experiments API**: automatic graph of artifacts/actions/associations across the ML lifecycle, plus Experiment/Trial/TrialComponent abstractions for comparing training runs.
  - When to use: auditing/reproducing which dataset, transformations, and hyper-parameters produced a given model.
  - How: `LineageTableVisualizer().show(processing_job_name=...)`; `Experiment.create()`, pass `experiment_config` to a Processing/Training Job, inspect with `ExperimentAnalytics`.

## Key Concepts
- **Feature selection vs. creation vs. transformation**: the three logical categories of feature engineering — picking the best attributes, combining attributes into new ones, and converting representations (e.g., binning, normalization, standardization).
- **Normalization vs. standardization**: normalization scales to [0,1]; standardization transforms to mean 0 / std 1 — standardization is generally preferred since it handles outliers better and allows cross-feature comparison.
- **Undersampling vs. oversampling**: undersampling discards majority-class rows down to minority size (fast but loses signal); oversampling (e.g., SMOTE) synthesizes new minority-class data (needs larger datasets to work well).
- **Data leakage**: information from validation/test leaking into training (e.g., via k-folds reuse or peeking into future time-series data), producing artificially inflated accuracy.
- **max_seq_length**: BERT hyper-parameter capping/padding token count per example; must stay consistent between feature engineering and model training; chosen here as 64 based on the 80th percentile of review word counts.
- **WordPiece tokenization**: subword-level tokenization BERT uses, with a ~30,000-token vocabulary and special tokens like `[CLS]`.
- **Sharding**: splitting a large dataset across multiple SageMaker Processing instances (`ShardedByS3Key`) to parallelize transformation.

## Mental Models
- **Train/validate/test must be independent**: any shared examples across splits (via k-folds on large data, or ungrouped customer data) risks leakage and inflated apparent accuracy — large datasets make clean splits (90/5/5 here) preferable to k-folds.
- **Classification vs. regression decision rule**: if the target has a small number of discrete categorical values (star_rating 1–5), treat it as classification, not regression, even though it's numeric.
- **Stratified splitting**: always pass `stratify=<label column>` to `train_test_split` to preserve class balance established during the balancing step.
- **Feature store as the "data lake for ML features"**: distinct from the data warehouse — stores computationally expensive features (BERT vectors) for reuse, consistency between training and inference, and governance.

## Anti-patterns
- **Training on an imbalanced dataset naively**: a classifier can achieve 62% "accuracy" by always predicting the majority class (5 stars), which is actually a failure mode, not real skill.
- **Using k-folds cross-validation on very large datasets**: reuses rows across splits, risking leakage when simple independent splits are already statistically sufficient.
- **Letting time-series models "peek into the future"**: historical back-testing splits must respect temporal order or risk overstating accuracy.
- **Changing max_seq_length without regenerating embeddings**: the same sequence length must be used consistently across feature engineering and training, or features become invalid.
- **Splitting data without grouping by entity (e.g., customer_id) when that grouping matters**: spreading one customer's data across train/val/test can leak customer-specific patterns.

## Code Examples
```python
from transformers import DistilBertTokenizer
MAX_SEQ_LENGTH = 64
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encode_plus_tokens = tokenizer.encode_plus(
    text_input.text, pad_to_max_length=True, max_length=MAX_SEQ_LENGTH)
input_ids = encode_plus_tokens['input_ids']
input_mask = encode_plus_tokens['attention_mask']
segment_ids = [0] * MAX_SEQ_LENGTH
```
- **What it demonstrates**: converting raw review text into the three numeric arrays (input_ids, input_mask, segment_ids) BERT requires, with automatic padding/truncation to a fixed length.

```python
from sklearn.model_selection import train_test_split
df_train, df_holdout = train_test_split(
        df_balanced, test_size=0.10, stratify=df_balanced['star_rating'])
df_validation, df_test = train_test_split(
        df_holdout, test_size=0.50, stratify=df_holdout['star_rating'])
```
- **What it demonstrates**: stratified 90/5/5 train/validation/test split preserving class balance.

```python
from sagemaker.sklearn.processing import SKLearnProcessor
processor = SKLearnProcessor(framework_version='<SCIKIT_LEARN_VERSION>',
                              role=role, instance_type='ml.c5.4xlarge', instance_count=2)
processor.run(code='preprocess-scikit-text-to-bert.py',
  inputs=[ProcessingInput(input_name='raw-input-data', source=raw_input_data_s3_uri,
                           destination='/opt/ml/processing/input/data/',
                           s3_data_distribution_type='ShardedByS3Key')],
  outputs=[ProcessingOutput(output_name='bert-train', s3_upload_mode='EndOfJob',
                             source='/opt/ml/processing/output/bert/train'), ...],
  arguments=['--train-split-percentage', str(train_split_percentage), ...])
```
- **What it demonstrates**: scaling feature engineering (balance, split, BERT transform) to a 2-instance cluster via SageMaker Processing Jobs, sharding input data across instances.

```python
from sagemaker.feature_store.feature_group import FeatureGroup
reviews_feature_group = FeatureGroup(name="reviews_distilbert_max_seq_length_64",
                                      sagemaker_session=sagemaker_session)
reviews_feature_group.load_feature_definitions(data_frame=df_records)
reviews_feature_group.create(s3_uri="s3://{}/{}".format(bucket, prefix),
    record_identifier_name="review_id", event_time_feature_name="date",
    role_arn=role, enable_online_store=True)
reviews_feature_group.ingest(data_frame=df_records, max_workers=3, wait=True)
```
- **What it demonstrates**: registering and ingesting computed BERT features into SageMaker Feature Store for reuse across teams/models.

## Reference Tables
| Feature engineering type | Purpose | Example |
|---|---|---|
| Selection | Remove redundant/irrelevant attributes | Drop `total_votes` if `helpful_votes` is highly correlated |
| Creation | Combine attributes into new features | Merge `review_headline` + `review_body` |
| Transformation | Change representation | Bin timestamps; normalize/standardize numeric columns |

| Split | Purpose | % in this example |
|---|---|---|
| Train | Fit model parameters | 90% |
| Validation | Tune hyper-parameters | 5% |
| Test | Final unbiased evaluation | 5% |

| BERT input array | Meaning |
|---|---|
| input_ids | Token IDs from pretrained vocabulary |
| input_mask (attention_mask) | 1 = attend, 0 = padding |
| segment_ids | Sentence index (0 for single-sequence tasks, 1 for second sentence in pair tasks) |

## Worked Example
The Amazon Customer Reviews Dataset's `review_body` and `star_rating` columns are selected (via Data Wrangler Quick Model feature importance) as the sole feature/label pair, since the production use case only has raw review text "in the wild." The dataset is balanced by undersampling each star_rating group to the smallest group size (reducing ~100K to ~35K rows), split 90/5/5 with stratification on `star_rating`, then each review is tokenized with DistilBERT's WordPiece tokenizer, padded/truncated to `max_seq_length=64` (chosen from the Ch. 5 finding that 80% of reviews are ≤63 words), and written as TFRecords. This pipeline is then scaled from a notebook sample to the full dataset using SageMaker Processing Jobs — first with scikit-learn (2× `ml.c5.4xlarge`) for balancing/splitting plus TensorFlow/Transformers for BERT embeddings, then alternatively with Apache Spark (2× `ml.r5.xlarge`) using a PySpark UDF wrapping the same Transformers tokenizer, writing output directly to S3 in TFRecord format via a Spark TFRecord connector. The resulting BERT feature vectors are ingested into SageMaker Feature Store under feature group `reviews_distilbert_max_seq_length_64` for reuse in Chapter 7's model training.

## Key Takeaways
1. Always rank feature importance (e.g., Data Wrangler Quick Model) before committing to expensive feature engineering.
2. Balance classes (undersampling/oversampling) before training to avoid a model that just learns to predict the majority class.
3. Use stratified splitting and avoid k-folds on large datasets to prevent data leakage; respect temporal order for time-series data.
4. Keep `max_seq_length` and other feature-engineering hyper-parameters consistent between feature engineering and model training.
5. Scale feature engineering with SageMaker Processing Jobs (scikit-learn or Spark) rather than running large transformations in a notebook.
6. Store expensive, reusable features (like BERT embeddings) in SageMaker Feature Store to ensure train/inference consistency and cross-team reuse.
7. Track lineage (SageMaker Lineage API, Experiments API) from the very first feature-engineering step so every downstream model is fully auditable.

## Connects To
- **Ch 5**: Builds directly on the imbalance/bias findings and review-length distribution discovered there.
- **Ch 7**: The BERT-embedded TFRecords and Feature Store data produced here feed directly into model training.
- **Ch 10**: SageMaker Pipelines (referenced here via Data Wrangler .flow export) automate this feature engineering step end-to-end.
