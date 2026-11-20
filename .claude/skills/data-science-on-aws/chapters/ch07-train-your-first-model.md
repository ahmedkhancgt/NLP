# Chapter 7: Train Your First Model

## Core Idea
Fine-tuning a pre-trained BERT/DistilBERT model with TensorFlow/Keras on SageMaker — rather than training from scratch — gives strong accuracy with low cost and fast iteration, and SageMaker's infrastructure (containers, Debugger, Clarify, Experiments/Lineage) provides full visibility, profiling, bias detection, and explainability throughout training.

## Frameworks Introduced
- **SageMaker Training Job containers**: ephemeral Docker containers reading from S3, training, writing model artifacts back to S3, with predefined environment variables (`SM_MODEL_DIR`, `SM_CHANNELS`, `SM_NUM_GPUS`, etc.) mapping container paths to S3 locations.
  - When to use: any SageMaker-managed training job; understand the env vars to write portable training scripts.
- **SageMaker JumpStart**: one-click fine-tuning/deployment of pre-trained models (BERT, etc.) from AWS/TensorFlow Hub/PyTorch Hub.
  - When to use: quickly testing whether a pre-trained model is a good fit before investing in custom training.
- **Three SageMaker model-development options**: built-in algorithms, Script Mode (bring your own script + managed framework container), Bring Your Own Container (BYOC).
  - When to use: built-in for standard tasks with no custom code; Script Mode (used throughout this book) for a good customization/maintenance balance; BYOC for full control or compliance needs.
- **BERT fine-tuning (DistilBERT + Keras classifier head)**: add a trainable classification layer on top of a frozen pre-trained BERT body.
  - How: load `TFDistilBertForSequenceClassification`, freeze BERT layers (`trainable=False`), add Bidirectional LSTM + GlobalMaxPool + Dense classifier layers, compile with `SparseCategoricalCrossentropy`.
- **SageMaker Debugger**: zero-code-change training monitoring/profiling — captures tensors/gradients, detects vanishing/exploding gradients, overfitting, bottlenecks; can trigger `StopTraining()`, email/SMS actions.
  - When to use: debugging training instability, profiling CPU/GPU/I/O bottlenecks, implementing early stopping to save cost.
- **SageMaker Clarify (post-training)**: computes post-training bias metrics (comparing model predictions vs. labels per facet) and SHAP-based feature attribution/explainability.
  - When to use: verifying a trained model (even one trained on unbiased data) doesn't produce biased predictions across facets like `product_category`.
- **PyTorch/MXNet/DJL training alternatives**: SageMaker supports converting a TensorFlow BERT model to PyTorch (`from_pretrained(path, from_tf=True)`) or training natively in PyTorch, MXNet, or Java via AWS Deep Java Library.

## Key Concepts
- **Attention mechanism / self-attention**: weighted average of input vectors where weights come from normalized (softmax) dot products of query and key vectors — lets BERT "pay attention" to relevant tokens regardless of position.
- **Multi-headed attention**: multiple parallel attention computations handling polysemy (same word, different meanings in different contexts).
- **Masked Language Model (Masked LM)**: BERT pre-training task masking 15% of tokens and predicting them, forcing bidirectional context learning.
- **Next Sentence Prediction (NSP)**: parallel BERT pre-training task predicting whether two sentences are a valid consecutive pair.
- **Fine-tuning vs. pre-training**: fine-tuning is supervised, fast, needs few examples (masking/NSP don't happen); pre-training is unsupervised and expensive, done once on huge corpora.
- **Knowledge distillation**: training a smaller "student" model (DistilBERT) to mimic a larger "teacher" model (BERT), reducing size/compute while retaining ~97% accuracy.
- **SHAP (SHapley Additive exPlanations)**: game-theory-based method for attributing a model's prediction to each input feature's contribution.
- **Mixed precision / bfloat16**: 16-bit "half precision" speeds up computation and halves memory, at some risk of vanishing gradients and numerical instability; "mixed" precision keeps 32-bit storage but computes in 16-bit.
- **Vanishing/exploding gradients**: numerical instabilities from repeated multiplication of very small/very large values exceeding float representable range in deep networks.

## Mental Models
- **BERT vs. prior NLP generations**: Word2Vec/GloVe (static embeddings, model discarded after training) → ELMo (bidirectional via two unidirectional LSTMs, not true bidirectional) → GPT-n (unidirectional masked self-attention, Transformer-based) → BERT (true bidirectional self-attention, preserves full model) — know this evolution to justify architecture choices.
- **Freeze base model, train the head**: freezing BERT's pre-trained layers and training only the new classifier layers is faster/cheaper and often "good enough"; unfreezing BERT layers can improve accuracy at higher training cost.
- **Right-sizing compute**: start small, ramp up instance size/count empirically rather than guessing; only add instances if the algorithm is actually parallelizable (adding instances to non-parallelizable algorithms adds overhead, not speed).
- **Debugger-as-safety-net**: configure rules (loss_not_decreasing, overtrain, class_imbalance, vanishing/exploding gradient) with automated actions (StopTraining, Email, SMS) instead of manually babysitting training runs.
- **Bias can appear even on an unbiased dataset**: hyper-parameters and biased pre-trained weights (BERT inherits Wikipedia's societal biases) can introduce model bias independent of data bias — always run post-training Clarify checks.

## Anti-patterns
- **Running GPU-heavy computation inside the notebook**: wastes money on an expensive, long-running notebook instance; move heavy compute to Processing/Training/Batch Transform Jobs instead.
- **Relying on external dependencies (pip installs, model zoos) at training time**: creates availability risk; "burn" dependencies into Docker images/AMIs instead (Netflix's practice, cited directly).
- **Letting Spot Instances run without checkpointing**: Spot Instance interruption without checkpoints means lost training progress; always pair `use_spot_instances=True` with `checkpoint_s3_uri`.
- **Blindly trusting attention visualization as explainability**: there is active industry debate about whether attention equals explainability; gradient-based saliency maps (e.g., AllenNLP Interpret, integrated gradients) are considered a more rigorous alternative.
- **Ignoring SageMaker Debugger's profiling suggestions**: e.g., ignoring a CPU bottleneck warning (add data loaders/prefetching) or low GPU utilization warning (increase batch size) wastes training budget.

## Code Examples
```python
from transformers import DistilBertConfig, TFDistilBertForSequenceClassification
CLASSES=[1, 2, 3, 4, 5]
config = DistilBertConfig.from_pretrained('distilbert-base-uncased',
    num_labels=len(CLASSES),
    id2label={0:1,1:2,2:3,3:4,4:5}, label2id={1:0,2:1,3:2,4:3,5:4})
transformer_model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", config=config)
```
- **What it demonstrates**: loading a pre-trained DistilBERT with a remapped 5-class label space for star_rating classification.

```python
from sagemaker.debugger import Rule, rule_configs, ActionList
actions = rule_configs.ActionList(
   rule_configs.StopTraining(), rule_configs.Email("<EMAIL>"), rule_configs.SMS("<PHONE>"))
rules = [Rule.sagemaker(base_config=rule_configs.loss_not_decreasing(),
    rule_parameters={'collection_names': 'losses,metrics', 'num_steps': '10', 'diff_percent': '50'},
    actions=actions)]
```
- **What it demonstrates**: configuring an automated rule that stops training and alerts a human when loss stops decreasing adequately — cost control via early stopping.

```python
estimator = TensorFlow(entry_point='tf_bert_reviews.py',
                        use_spot_instances=True,
                        max_wait=120,
                        checkpoint_s3_uri='s3://<BUCKET>/<CHECKPOINT_PREFIX>/',
                        ...)
```
- **What it demonstrates**: using Spot Instances for training cost savings with checkpoint-based resumption if Spot capacity is reclaimed.

```python
clarify_processor.run_post_training_bias(
    data_config=data_config, data_bias_config=bias_config,
    model_config=model_config, model_predicted_label_config=predictions_config,
    methods=['DPPL','DI','DCA','DCR','RD','DAR','DRR','AD','CDDPL','TE'],
    wait=True)
```
- **What it demonstrates**: running post-training bias analysis (comparing model predictions against labels per facet) via a SageMaker Clarify Processing Job, spinning up a shadow endpoint automatically.

## Reference Tables
| BERT variant | Layers | Hidden units | Parameters |
|---|---|---|---|
| BERT base | 12 | 768 | 110M |
| BERT large | 24 | 1024 | 340M |

| SageMaker model-dev option | Customization | Maintenance |
|---|---|---|
| Built-in algorithms | Low | Lowest (no code) |
| Script Mode (bring your own script) | High | Low-medium (managed container) |
| Bring Your Own Container | Highest | Highest (full Docker control) |

| Post-training bias metric | Measures |
|---|---|
| AD (Accuracy Difference) | Accuracy gap between facet groups |
| DRR (Difference in Rejection Rates) | Gap in negative-outcome rates |
| RD (Recall Difference) | Gap in recall between facet groups |
| TE (Treatment Equality) | Ratio of false positive/negative rates across groups |

## Worked Example
A product-review classifier is fine-tuned by loading pre-trained DistilBERT, freezing its layers, and adding a Bidirectional LSTM + Dense classifier head (332,905 trainable params out of 66.7M total) trained via Keras `model.fit()` on BERT-embedded TFRecords (from Ch. 6) using `ml.p4d.24xlarge` (8× A100 GPU) instances with hyper-parameters set manually (epochs=500, learning_rate=0.00001, batch_size=128). SageMaker Debugger rules watch for loss plateau/overtraining and can auto-stop the job; SageMaker Clarify then runs post-training bias analysis on a `product_category` facet and SHAP-based feature attribution, generating a bias baseline later reused by SageMaker Model Monitor (Ch. 9) to detect live drift. The resulting model, saved via `model.save('./tensorflow/', save_format='tf')`, correctly predicts "This is great!" as 5-star (92% confidence) and is later convertible to PyTorch via the Transformers library for serving with TorchServe.

## Key Takeaways
1. Fine-tune a pre-trained model (BERT/DistilBERT) rather than training from scratch unless your domain's vocabulary is drastically different.
2. Freeze the base model's layers initially — only unfreeze/retrain them if accuracy with a frozen base isn't sufficient.
3. Use SageMaker Debugger rules with automated actions (StopTraining, Email, SMS) to catch instability early and save training cost.
4. Always run post-training bias analysis (SageMaker Clarify) even if pre-training data bias was already addressed — biased pre-trained weights and hyper-parameters can introduce bias independently.
5. Move all GPU-heavy computation out of the notebook and into Training/Processing Jobs to control cost.
6. Pair Spot Instance training with checkpointing to capture the cost savings without losing progress on interruption.
7. Treat mixed precision, instance type/count, and quantization as tunable hyper-parameters, not fixed defaults.

## Connects To
- **Ch 6**: Directly consumes the BERT-embedded TFRecords produced there.
- **Ch 5**: Builds on the pre-training bias/Clarify concepts introduced there, now applied post-training.
- **Ch 8**: Hyper-parameter tuning (epochs, learning_rate, etc., set manually here) becomes automated via SageMaker Hyper-Parameter Tuning Jobs.
- **Ch 9**: The saved TensorFlow/PyTorch model is deployed to SageMaker Endpoints/Batch Transform; the bias/explainability baselines generated here feed SageMaker Model Monitor.
