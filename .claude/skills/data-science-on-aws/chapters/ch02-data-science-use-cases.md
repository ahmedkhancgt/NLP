# Chapter 2: Data Science Use Cases

## Core Idea
AI/ML has become ubiquitous across industries because AWS packages decades of Amazon's own ML expertise into "1-click" managed AI services — recommendations, forecasting, fraud/anomaly detection, NLP, computer vision, speech, and even quantum computing — that require little or no ML expertise to integrate.

## Frameworks Introduced
- **Amazon Personalize**: managed recommendation engine built on Amazon.com's personalization expertise.
  - When to use: product/content recommendations, targeted marketing, handling the "cold start" problem via a dedicated neural+RL algorithm.
  - How: feed an activity stream (clicks, purchases) + item inventory + optional metadata; call `get_recommendations()` for real-time results.
- **Amazon Forecast**: managed time-series demand forecasting service (DeepAR++, ARIMA, ETS, NPTS, Prophet).
  - When to use: inventory, staffing, cash-flow, or any demand forecasting problem needing metadata beyond raw time series.
  - How: register historical dataset + schema, create a predictor with an algorithm ARN, query forecasts via API.
- **Amazon Fraud Detector**: managed fraud-detection model builder trained on AWS/Amazon fraud expertise.
  - When to use: detecting fake accounts/fraudulent transactions with limited historical data.
  - How: `create_model_version()` on historical event data, then `get_event_prediction()` per transaction.
- **Amazon Comprehend / Comprehend Medical**: managed NLP for sentiment, entities, key phrases, custom classifiers/entity recognizers.
  - When to use: text classification (e.g., sentiment), custom entity extraction (e.g., resume "SKILLS"), or medical text mining.
  - How: `create_document_classifier()` / `create_entity_recognizer()` then `detect_sentiment()` or `start_entities_detection_job()`.
- **Amazon Rekognition**: computer vision AI service (AutoML under the hood) for image/video analysis and content moderation.
  - When to use: detecting objects, people, text, or inappropriate (violent/weapons) content in media.
  - How: `start_content_moderation()` on an S3 video, poll `get_content_moderation()` for labels + confidence + timestamps.
- **Amazon Textract**: OCR + NLP service to extract structured text from scanned documents.
  - When to use: parsing semistructured documents (resumes, forms) into indexable text, often paired with Comprehend custom entities.
- **In-database ML (Redshift ML, Aurora ML, Athena UDFs)**: invoke SageMaker/Comprehend models directly from SQL.
  - When to use: applying ML to data already in a warehouse/relational DB/S3 without exporting data.
  - How: `CREATE ... USING FUNCTION ... SAGEMAKER_INVOKE_ENDPOINT` (Athena) or built-in `aws_comprehend_detect_sentiment()` (Aurora) or Redshift ML UDFs trained via SageMaker Autopilot.

## Key Concepts
- **Cold start problem**: recommender/fraud models have little/no historical signal for new users/items; mitigated with metadata or specialized cold-start algorithms.
- **Popularity trap**: recommenders that only surface popular items miss long-tail relevant content; countered with explore/exploit algorithms like multi-armed bandits.
- **Predictive vs. reactive scaling**: AWS Auto Scaling uses ML to predict future traffic and provision ahead of demand, vs. EC2 Auto Scaling's reactive metric-based scaling.
- **Random Cut Forest (RCF)**: anomaly-detection algorithm used by Kinesis Data Analytics and QuickSight ML Insights; produces an anomaly score plus per-column attribution scores.
- **Edge inference**: running trained models on local devices (via AWS IoT Greengrass) to reduce latency/cost versus round-tripping to the cloud.
- **GAN (generator/discriminator)**: two competing CNNs — generator creates content, discriminator distinguishes real vs. synthetic — used in AWS DeepComposer.
- **Qubit / superposition / quantum supremacy**: quantum bits hold probabilistic state until read; "quantum supremacy" (2019) marked the point where classical computers can no longer simulate quantum computer performance gains.

## Mental Models
- **Two categories of AI value**: improving business operations (forecasting, fraud, optimization) vs. creating new customer experiences (recommendations, rich media).
- **"1-click" AI services ladder**: use managed AI Services first (no ML expertise needed) → SageMaker Autopilot/AutoML for tabular baselines → custom SageMaker models only when necessary (mirrors Ch1's framework).
- **In-database ML principle**: bring the model to the data (SQL UDFs) rather than exporting data to the model, minimizing data movement/duplication.
- **GPU-to-QPU analogy**: QPUs are "off-chip" accelerators like GPUs, needing specialized skills/hardware/software bridges — useful mental model for reasoning about emerging accelerators generally.

## Anti-patterns
- **Rule-based recommendation systems at scale**: become unmanageable as users/items grow; don't provide meaningful personalization.
- **Pure time-series forecasting without metadata**: ignores product/store/price context and performs poorly on irregular trends.
- **Relying on business-rule fraud detection**: too slow to adapt to new fraud techniques compared to ML-based detection.
- **Ignoring sensitive-data monitoring**: loosely configured IAM + sensitive data in S3 is "an easy target" for attackers — continuous monitoring (Macie) is required, not one-time checks.

## Code Examples
```python
# Amazon Personalize: real-time recommendations
get_recommendations_response = personalize_runtime.get_recommendations(
        campaignArn = campaign_arn,
        userId = user_id
)
item_list = get_recommendations_response['itemList']
```
- **What it demonstrates**: calling a trained Personalize campaign for real-time per-user recommendations.

```sql
USING FUNCTION predict_sentiment(review_body VARCHAR(65535))
    RETURNS VARCHAR(10) TYPE
    SAGEMAKER_INVOKE_ENDPOINT WITH (sagemaker_endpoint = '<ENDPOINT_NAME>')
SELECT predict_sentiment(review_body) AS sentiment
     FROM "dsoaws"."amazon_reviews_tsv"
     WHERE predict_sentiment(review_body)="POSITIVE";
```
- **What it demonstrates**: invoking a SageMaker endpoint as a UDF directly from an Athena SQL query — no data export needed.

```python
# Amazon Comprehend sentiment on product reviews
response = comprehend.detect_sentiment(Text="I loved it! I will recommend this to everyone.")
# -> {"Sentiment": "POSITIVE", "SentimentScore": {...}}
```
- **What it demonstrates**: out-of-the-box sentiment classification requiring zero model training.

## Reference Tables
| Use case | AWS Service | Core algorithm/tech |
|---|---|---|
| Product recommendations | Amazon Personalize | Deep learning + cold-start RL |
| Demand forecasting | Amazon Forecast | DeepAR++, ARIMA, ETS, NPTS, Prophet |
| Fraud detection | Amazon Fraud Detector | Pre-trained + custom ML |
| Content moderation | Amazon Rekognition | CNN-based object/activity detection |
| Sensitive-data detection | Amazon Macie | ML-based PII classification |
| Text sentiment/entities | Amazon Comprehend | NLP classifiers |
| Document OCR | Amazon Textract | OCR + NLP |
| Speech-to-text / text-to-speech | Transcribe / Polly | ASR / neural TTS |
| Enterprise search | Amazon Kendra | Cognitive search (NLU) |
| Anomaly detection on streams | Kinesis Data Analytics | Random Cut Forest |
| Predictive infra scaling | AWS Auto Scaling | ML-based traffic prediction |
| Graph predictions | Amazon Neptune ML | Graph Neural Networks |

## Worked Example
A customer-support sentiment pipeline (reused later in the book): product reviews stored in S3/Aurora/Athena are run through Amazon Comprehend's `detect_sentiment()` to tag POSITIVE/NEGATIVE/NEUTRAL sentiment with confidence scores; this can be done directly in Aurora SQL via `aws_comprehend_detect_sentiment()`, or in Athena by defining a SAGEMAKER_INVOKE_ENDPOINT UDF to call a custom-trained SageMaker sentiment model — foreshadowing Chapter 3's deeper dive into training a 5-star-rating classifier on the Amazon Customer Reviews Dataset.

## Key Takeaways
1. Default to managed AI services (Personalize, Forecast, Comprehend, Rekognition) before building custom models for common problems.
2. Address cold start and popularity-trap failure modes explicitly in any recommender design.
3. Use in-database ML (Athena/Redshift/Aurora UDFs) to apply models without exporting data from the warehouse.
4. Treat anomaly/fraud/sensitive-data detection as continuous, automated monitoring, not a one-time audit.
5. Edge inference (IoT Greengrass) reduces latency/cost for real-time decisions when round-tripping to the cloud is too slow.
6. CodeGuru Reviewer/Profiler and DevOps Guru extend ML into software performance and operational monitoring, not just data science.

## Connects To
- **Ch 3**: Deep dive into SageMaker Autopilot and training a custom star-rating sentiment classifier referenced here with Comprehend.
- **Ch 9**: Multi-armed bandits for A/B testing recommenders and reinforcement learning depth (DeepRacer example).
- **Ch 11**: Streaming anomaly detection (Kinesis Data Analytics / RCF) revisited in depth.
- **Ch 12**: Post-quantum cryptography and sensitive-data security practices (Macie) revisited.
