# Chapter 9: Machine Learning with TensorFlow in Vertex AI

## Core Idea
TensorFlow/Keras, trained via GPU-accelerated notebooks on Vertex AI, solves the portability, embedding, and scalability problems that stymied Spark ML (Ch. 7) and the expressiveness limits of BigQuery ML (Ch. 8) — culminating in a wide-and-deep model (linear part for sparse/categorical features, deep neural network part for continuous/embedded features) that gives the best RMSE yet, deployed as an autoscaling REST endpoint via Vertex AI.

## Frameworks Introduced
- **Wide-and-deep model architecture** (Cheng et al., 2016): combine a linear model over sparse/categorical features with a deep neural network over continuous/embedded features, concatenated before the final sigmoid output.
  - When to use: structured/tabular data with both genuinely continuous signals and high-cardinality categorical/interaction signals.
  - How: build `linear_feature_columns` (sparse, crossed) and `dnn_feature_columns` (real-valued + embeddings) separately; pass each through its own `DenseFeatures` layer, run the DNN branch through stacked `Dense(relu)` layers, concatenate with the wide branch, then a final `Dense(1, sigmoid)` output.
- **Keras Input → Feature Column → DenseFeatures → Model pipeline**: TensorFlow's structured-data feature engineering pattern.
  - When to use: any TensorFlow/Keras model consuming structured/tabular inputs with a mix of numeric and categorical columns.
  - How: declare `tf.keras.layers.Input` per raw input column (matching dtype); define `real` (numeric_column) and `sparse` (categorical_column_with_vocabulary_list / categorical_column_with_hash_bucket) feature-column dicts; apply `tf.keras.layers.DenseFeatures(columns)(inputs)` to produce the feature vector fed into Dense layers.
- **Hashed-bucket categorical encoding**: avoid needing a pre-known vocabulary for a high-cardinality string column.
  - When to use: categorical columns (e.g., airport code) whose full vocabulary is large, unknown, or likely to grow (new airports).
  - How: `categorical_column_with_hash_bucket(colname, hash_bucket_size=N)` where N comfortably exceeds the true cardinality — trades a small hash-collision risk for not needing to enumerate the vocabulary, and tolerates unseen values gracefully (the "hashed feature" design pattern).
- **Vertex AI model deployment pipeline**: upload trained model artifacts → create a reusable endpoint → deploy model(s) to the endpoint with a traffic split.
  - When to use: serving any trained model (TensorFlow, BigQuery ML export, XGBoost, PyTorch, etc.) as a scalable REST prediction service.
  - How: `gcloud ai models upload` with a pre-built per-framework container (`tf2-cpu...`, `xgboost-cpu...`) and `--artifact-uri`; `gcloud ai endpoints create` once (reuse across model versions so the URL stays stable); `gcloud ai endpoints deploy-model` with `--traffic-split` to support canary/A-B rollout between model versions.

## Key Concepts
- **Input vs. feature (TensorFlow)**: the input is the raw column as the client supplies it; the feature is what's actually fed into the model after feature-column transformation (bucketizing, hashing, embedding, crossing) — same distinction introduced in Ch. 8's TRANSFORM clause, now explicit in Keras' Input/FeatureColumn separation.
- **Embedding**: a learned low-dimensional dense representation of a high-cardinality categorical feature (e.g., 300 airports → 10 dimensions), which — unlike one-hot encoding — can capture similarity between categories (e.g., similar airports cluster near each other in embedding space); the direct fix for the Spark ML (Ch. 7) and BigQuery ML feature-cross (Ch. 8) sparsity problems.
- **Bucketizing + feature crossing continuous geography**: discretizing latitude/longitude into grid buckets and crossing them encodes FAA-style "air traffic corridor" structure (e.g., Chicago–New York congestion) without hand-coding geographic rules.
- **ReLU activation**: a piecewise-linear activation (`max(0, x)`) used in hidden layers; trades sigmoid's bounded-output safety for a risk of unbounded large outputs, offset by modern initialization/training techniques.
- **Vertex AI Model vs. Endpoint**: a Model is an uploaded, versioned artifact (any supported framework via pre-built or custom containers); an Endpoint is the stable, reusable URL that splits traffic across one or more deployed Models — create the Endpoint once, redeploy/version Models against it.
- **MirroredStrategy (implicit default)**: Keras/TensorFlow's default distribution strategy for 0+ GPUs on a single machine, requiring no explicit code when training on one machine with however many GPUs are attached.

## Mental Models
- **"The biggest return for your effort is in more/better data and features, not a fancier model"**: reiterated from Ch. 8 — plain logistic regression in TensorFlow (with more features than Spark could handle) barely moved RMSE versus BigQuery ML's logistic regression; switching to a deep neural network alone also didn't help (0.205, no meaningful change) — only adding domain-informed features (bucketized lat/lon, feature crosses, embeddings) via the wide-and-deep architecture produced a real gain (RMSE 0.196).
- **Logistic regression is a one-node neural network**: framing it this way (Figure 9-1) makes the conceptual jump from Spark/BigQuery-ML logistic regression to Keras/TensorFlow deep networks a natural extension rather than a new paradigm.
- **Encode human domain insight as features, not hope the model infers it from raw values**: raw lat/lon alone shouldn't meaningfully predict delay, but bucketized+crossed lat/lon captures real air-traffic-corridor congestion patterns the FAA already organizes around.
- **Shuffle before distributing**: in distributed training, shuffling input data before sharding across workers prevents a persistently slow/faulty worker from always getting (and having discarded) the same batch of data, which would otherwise bias training.

## Anti-patterns
- **Scaling up model complexity (logistic regression → DNN) before investing in better features**: neither the pure linear nor plain DNN model improved on BigQuery ML's score — complexity alone doesn't fix a feature-engineering gap.
- **Loading all of BigQuery into a Pandas dataframe (`tf.convert_to_tensor`) for large datasets**: only safe for data that fits comfortably in memory; for large data, export to GCS CSV (used in this chapter) or stream via `BigQueryReader` (fast but billed per read) instead.
- **Repeatedly recreating a Vertex AI endpoint per model version**: breaks client URLs; create the endpoint once, upload new model versions, and use `--traffic-split` to manage rollout between them under the same stable endpoint.
- **Mixing ad hoc notebook Python with separate shell `gcloud` deployment commands as a long-term workflow**: explicitly flagged as "hard to automate" — motivates moving to a pure-Python, MLOps-friendly pipeline in Chapter 10.
- **Treating one-hot encoding as the default for every categorical column regardless of cardinality**: reiterates Ch. 7/8's lesson — use hashed buckets or embeddings for high-cardinality categoricals instead.

## Code Examples
```python
real = {colname: numeric_column(colname) for colname in
        'dep_delay,taxi_out,distance,dep_hour,is_weekday,dep_airport_lat,dep_airport_lon,arr_airport_lat,arr_airport_lon'.split(',')}
sparse = {
    'carrier': categorical_column_with_vocabulary_list('carrier', vocabulary_list='AS,VX,F9,UA,...'.split(',')),
    'origin': categorical_column_with_hash_bucket('origin', hash_bucket_size=1000),
    'dest': categorical_column_with_hash_bucket('dest', hash_bucket_size=1000),
}
```
- **What it demonstrates**: splitting raw columns into numeric vs. categorical feature-column dicts, using an explicit vocabulary where known (carrier) and a hash bucket where the vocabulary is large/unenumerated (airport codes).

```python
disc.update({'d_{}'.format(key): tf.feature_column.bucketized_column(real[key], latbuckets) for key in ['dep_lat', 'arr_lat']})
sparse['dep_loc'] = tf.feature_column.crossed_column([disc['d_dep_lat'], disc['d_dep_lon']], NBUCKETS*NBUCKETS)
embed = {'embed_{}'.format(colname): tf.feature_column.embedding_column(col, dimension=10) for colname, col in sparse.items()}
real.update(embed)
```
- **What it demonstrates**: the full wide-and-deep feature-engineering pipeline — bucketize continuous lat/lon, feature-cross the buckets into a grid-cell categorical, then embed all sparse columns into 10-dimensional vectors for the deep branch.

```python
def wide_and_deep_classifier(inputs, linear_feature_columns, dnn_feature_columns, dnn_hidden_units):
    deep = tf.keras.layers.DenseFeatures(dnn_feature_columns, name='deep_inputs')(inputs)
    for layerno, numnodes in enumerate(dnn_hidden_units):
        deep = tf.keras.layers.Dense(numnodes, activation='relu', name=f'dnn_{layerno+1}')(deep)
    wide = tf.keras.layers.DenseFeatures(linear_feature_columns, name='wide_inputs')(inputs)
    both = tf.keras.layers.concatenate([deep, wide], name='both')
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(both)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```
- **What it demonstrates**: the full wide-and-deep Keras model function — linear branch and DNN branch concatenated before the final sigmoid output; achieved the chapter's best RMSE (0.196).

```bash
CONTAINER=us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest
gcloud ai models upload --region=$REGION --display-name=$MODEL_NAME \
     --container-image-uri=$CONTAINER --artifact-uri=$EXPORT_PATH
gcloud ai endpoints create --region=${REGION} --display-name=${ENDPOINT_NAME}
gcloud ai endpoints deploy-model $ENDPOINT_ID --region=$REGION --model=$MODEL_ID \
     --machine-type=n1-standard-2 --min-replica-count=1 --max-replica-count=1 --traffic-split=0=100
```
- **What it demonstrates**: the three-step Vertex AI deployment flow (upload model → create reusable endpoint → deploy with traffic split), the same pattern used for TensorFlow, BigQuery ML exports, and XGBoost models alike (just swap the container image).

## Reference Tables
| Model | RMSE | Notes |
|---|---|---|
| Keras logistic regression (all features) | ~same as BQML linear | No improvement despite more features — linear model expressiveness ceiling |
| Keras DNN (64, 8 hidden units) | 0.205 | No meaningful improvement over linear |
| Wide-and-deep (bucketized lat/lon, feature crosses, embeddings) | 0.196 | Best result in the chapter |

| BigQuery ML model type | Export format for Vertex AI |
|---|---|
| Linear, DNN, AutoML, KMeans, Matrix Factorization, PCA | TensorFlow SavedModel |
| Boosted tree (XGBoost) | `.bst` XGBoost 0.82 format — use the XGBoost pre-built container instead |

## Worked Example
Builds and deploys a wide-and-deep flight-delay model: (1) export a flattened `flights_train_data` table from BigQuery (adding carrier, departure/arrival airport lat/lon, hour, is_weekday) to GCS CSV files; (2) build a `tf.data.experimental.make_csv_dataset` pipeline with shuffling/repeat for training and single-pass for eval, splitting features from the `ontime` label; (3) build a plain Keras logistic-regression-equivalent model (single Dense+sigmoid layer over all DenseFeatures) — RMSE doesn't improve over BigQuery ML's linear model; (4) try a small DNN (64→8 hidden ReLU layers) — still no meaningful RMSE improvement; (5) engineer wide-and-deep features: bucketize departure/arrival lat/lon into a grid, cross the bucket pairs into `dep_loc`/`arr_loc` sparse columns, cross those into a `dep_arr` route feature and cross raw origin/destination into `ori_dest`, then embed all sparse columns (10 dimensions) for the deep branch while keeping them raw for the wide branch; (6) build and train the `wide_and_deep_classifier`, achieving RMSE 0.196 — the chapter's best; (7) export the trained model with `tf.saved_model.save`, upload it to Vertex AI with the TensorFlow 2.6 CPU prediction container, create a reusable `flights` endpoint, and deploy the model with 100% traffic split; (8) invoke the deployed endpoint with an HTTP POST of a JSON `instances` payload and receive back per-instance on-time probabilities.

## Key Takeaways
1. Logistic regression is just a one-node neural network — TensorFlow/Keras lets you grow from there (DNN, wide-and-deep) within the same framework rather than switching tools.
2. Don't expect model-complexity increases (linear → DNN) alone to improve performance; invest first in feature engineering that encodes domain insight (e.g., geographic bucketing/crossing for air traffic corridors).
3. Use hashed-bucket categorical columns for high-cardinality or growing vocabularies (e.g., airport codes) instead of requiring an exhaustive, static vocabulary list.
4. Use embeddings (not one-hot encoding or raw feature crosses) to handle high-cardinality categorical interactions without the sparsity explosion that broke Spark ML (Ch. 7) and stalled BigQuery ML's feature cross (Ch. 8).
5. Build wide-and-deep models by cleanly separating linear (sparse/categorical) and deep (continuous/embedded) feature columns, concatenating before the output layer.
6. Create a Vertex AI Endpoint once and reuse it across model versions (via `--traffic-split`) so client URLs never need to change.
7. Export almost any BigQuery ML model type to Vertex AI as a TensorFlow SavedModel (or XGBoost format for boosted trees) to get autoscaling, managed REST serving, and built-in explainability/drift detection.

## Connects To
- **Ch 7**: Directly resolves the categorical-column explosion and one-hot encoding scalability problem that crashed Spark ML.
- **Ch 8**: Resolves BigQuery ML's inability to create embeddings for sparse feature crosses, and reuses the same `trainday`/feature-engineering lineage.
- **Ch 10**: The chapter explicitly flags the hybrid notebook+shell deployment workflow as hard to automate, setting up Chapter 10's move to pure-Python MLOps pipelines.
- **Ch 11**: Time-windowed aggregate features (deferred since Ch. 8 due to causality risk) are addressed with Apache Beam's unified batch/stream code, consistent with TensorFlow's portability goals here.
