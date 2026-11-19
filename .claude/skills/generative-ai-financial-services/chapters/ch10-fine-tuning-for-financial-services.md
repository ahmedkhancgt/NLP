# Chapter 10: Use-case: Fine-tuning for Financial Services

## Core Idea
Unsupervised domain-adaptation fine-tuning of Llama 3 8B on SEC filings (10-K reports) using PEFT/LoRA + 4-bit quantization on Amazon SageMaker measurably improves the model's depth and specificity on financial-domain questions, without needing labeled input/output training pairs.

## Frameworks Introduced
- **End-to-end fine-tuning architecture**: Fine-tuning dataset (SEC filings) → Data Preparation (Python tokenization scripts) → Hugging Face Hub (pretrained Llama 3 download) → Fine-tuning (SageMaker + PEFT/LoRA/quantization) → Model Deployment (SageMaker endpoint) → Evaluation (base vs. fine-tuned comparison).
  - When to use: domain-adaptation fine-tuning when you have a large corpus of unlabeled domain documents (not Q&A pairs) and want the model to internalize domain knowledge/terminology/style, enabling better zero-shot answers with less RAG context needed.
- **Seven fine-tuning use-case categories for financial services**: Financial analysis & reporting, Regulatory compliance, Risk assessment, Investment research, Customer service, Fraud detection, Mergers & Acquisitions analysis — each with concrete example tasks (see Reference Tables).
- **Unsupervised domain-adaptation workflow** (this chapter's specific implementation, contrasted with supervised fine-tuning from Ch3): no labeled input/output pairs — instead, raw documents are tokenized/chunked and the model is trained to predict next tokens, absorbing domain vocabulary/style/facts.

## Key Concepts
- **SEC 10-K filing**: an annual regulatory disclosure document financial institutions use as a rich, structured source of company financial/operational detail — used here as the fine-tuning corpus (Amazon, Apple, Meta 2022-2023 filings from EDGAR).
- **Domain adaptation**: adapting a pre-trained model to perform well on a specific domain by fine-tuning on domain-specific (often unlabeled) data.
- **bitsandbytes / 4-bit quantization**: library used alongside PEFT/LoRA to shrink the model's memory footprint during training (BitsAndBytesConfig with nf4 quant type, bfloat16 compute dtype).
- **AutoTokenizer**: Hugging Face transformers class that automatically loads the correct tokenizer for a given model ID, simplifying tokenization code across models.
- **merge_weights hyperparameter**: controls whether trained LoRA adapter weights are merged back into the base model after training, producing a single deployable model artifact.

## Mental Models
- Think of unsupervised fine-tuning as "teaching the model to read like a financial analyst" (absorbing vocabulary, structure, facts) vs. supervised fine-tuning as "teaching the model to answer like a specific assistant" (input→output mapping) — pick based on whether you have labeled Q&A pairs or just raw domain documents.
- Evaluate fine-tuning success by comparing base vs. fine-tuned model responses side-by-side on the same domain question — look for added specificity (named partnerships, store counts, cited strategies) rather than just "sounds more confident."
- Treat fine-tuning as additive knowledge injection: it lets the model answer well with zero-shot prompts and less RAG context, reducing prompt/context engineering burden for frequently-asked domain questions.

## Anti-patterns
- Note: this chapter is primarily a positive implementation walkthrough; see Ch6 for the formal anti-patterns ("fine-tuning when a pre-trained FM already performs well" or "when there isn't enough quality labeled data") which this chapter's SEC-filing use-case is designed to avoid (plenty of high-quality unlabeled public filing data available, and a genuine domain-knowledge gap to fill).

## Code Examples
```python
# --- Data preparation ---
from langchain_community.document_loaders.web_base import WebBaseLoader, default_header_template
annual_reports = [
    "https://www.sec.gov/Archives/edgar/data/1018724/000101872424000008/amzn-20231231.htm",  # Amazon 2023
    "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",  # Apple 2023
    "https://www.sec.gov/Archives/edgar/data/1326801/000132680124000012/meta-20231231.htm",  # Meta 2023
]
sec_header_template = default_header_template.copy()
sec_header_template["User-Agent"] = "<YOUR COMPANY NAME & EMAIL>"
loader = WebBaseLoader(annual_reports, header_template=sec_header_template)
data = loader.load()

from datasets import Dataset
def strip_spaces(doc):
    return {"text": doc.page_content.replace("  ", "")}
stripped_data = list(map(strip_spaces, data))
train_dataset = Dataset.from_list(stripped_data)

from transformers import AutoTokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

def chunk(sample, chunk_length=2048):
    # Chunking logic here
    ...
lm_train_dataset = train_dataset.map(
    lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(train_dataset.features)
).map(
    partial(chunk, chunk_length=chunk_size),
    batched=True,
)

training_input_path = f's3://{sess.default_bucket()}/processed/sec_data/train'
lm_train_dataset.save_to_disk(training_input_path)

# --- Fine-tuning (train.py excerpt) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False if args.gradient_checkpointing else True,
)
peft_config = LoraConfig(
    r=64, lora_alpha=16, target_modules=modules,
    lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_checkpointing=args.gradient_checkpointing,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    bf16=args.bf16,
)
trainer = transformers.Trainer(
    model=model, train_dataset=dataset, args=training_args,
    data_collator=default_data_collator
)
trainer.train()

# --- SageMaker training job setup ---
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
hyperparameters = {
    'model_id': model_id,
    'dataset_path': '/opt/ml/input/data/training',
    'epochs': 10,
    'per_device_train_batch_size': 2,
    'lr': 2e-4,
    'hf_token': HfFolder.get_token(),
    'merge_weights': True,
}
huggingface_estimator = HuggingFace(
    entry_point='train.py', source_dir=source_dir,
    instance_type='ml.g5.12xlarge', instance_count=1, role=role,
    transformers_version='4.28', pytorch_version='2.0', py_version='py310',
    hyperparameters=hyperparameters,
)
huggingface_estimator.fit(data, wait=True)

# --- Deployment ---
model = HuggingFaceModel(
    image_uri=image_uri,
    model_data=f"s3://{bucket_name}/{job_name}/output/model.tar.gz",
    role=role,
)
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.12xlarge",
    container_startup_health_check_timeout=3600,
)
```
- **What it demonstrates**: a complete unsupervised domain-adaptation fine-tuning pipeline — scraping SEC EDGAR filings via LangChain's WebBaseLoader, tokenizing/chunking with AutoTokenizer, fine-tuning Llama 3 8B with 4-bit quantized LoRA (r=64, alpha=16) on a SageMaker `ml.g5.12xlarge` instance over 10 epochs, then deploying the merged model to a SageMaker real-time endpoint.

## Reference Tables
**7 fine-tuning use-case categories for financial services**

| Category | Example tasks |
|---|---|
| Financial analysis & reporting | Earnings call summaries, automated trend/anomaly report analysis, executive summaries of annual reports |
| Regulatory compliance | Compliance report drafting from transaction data, regulation-impact summaries, compliance-issue detection in internal docs |
| Risk assessment | Loan application credit-risk analysis, market-risk scanning from news/reports, risk assessment reports for new products |
| Investment research | Company analysis reports, sector investment outlooks, analyst-opinion/sentiment summaries |
| Customer service | Complex product explanations tailored to literacy level, account-specific query handling, personalized financial advice |
| Fraud detection | Transaction-log anomaly flagging, financial-statement fraud signal detection, fraud investigation report drafting |
| M&A analysis | Merger proposal summaries, synergy/risk analysis, comparative historical M&A deal analysis |

## Worked Example
**Apple Asia growth strategy (base vs. fine-tuned comparison)**: Asked "What is Apple's strategy for growth in Asia?", the base Llama 3 model gives a generic answer (market expansion into India/Indonesia/SE Asia, product innovation like iPhone SE, Apple Pay). The SEC-filing-fine-tuned model gives a markedly more specific answer: named manufacturing/logistics partnerships (Foxconn in India), exact retail store counts by region (39 in Greater China, 91 in Japan, 24 in India, 48 in SE Asia as of 2022), and named local partnerships (Tencent, Alibaba in China; local banks/payment companies in India) — details traceable to the actual 10-K filings. A second example ("key priorities for Amazon in 2023") shows the same pattern: the fine-tuned model surfaces specific named initiatives ("Fulfilment by Design") drawn from Amazon's actual shareholder letters, while the base model gives generic industry-trend guesses (sustainability, omnichannel). This demonstrates fine-tuning's value: more specific, source-grounded answers with zero-shot prompting and no RAG context needed.

## Key Takeaways
1. Unsupervised domain-adaptation fine-tuning works well when you have abundant unlabeled domain documents (e.g., public SEC filings) but no labeled Q&A training pairs.
2. PEFT + LoRA + 4-bit quantization makes fine-tuning Llama 3 8B tractable on a single `ml.g5.12xlarge` SageMaker instance rather than requiring a massive GPU cluster.
3. Evaluate fine-tuning gains concretely by comparing base vs. fine-tuned responses on the same domain-specific questions — look for added specificity and traceability to source documents.
4. Fine-tuning reduces the need for large RAG context windows on frequently-asked domain questions, since the knowledge is now baked into model weights.
5. The seven fine-tuning use-case categories (financial analysis, compliance, risk, investment research, customer service, fraud detection, M&A) give a structured way to scope which domain corpus to fine-tune on for a given business goal.

## Connects To
- **Ch3**: Implements the PEFT/LoRA/quantization fine-tuning concepts and the supervised-vs-unsupervised distinction defined there.
- **Ch5**: Addresses the "need for specialized knowledge" challenge identified in Ch5 directly.
- **Ch6**: This use-case exemplifies when fine-tuning IS the right approach per Ch6's decision criteria (new domain language, insufficient context-window coverage via RAG alone).
- **Ch11**: Continues into agentic systems that can use tools (DB queries, research, knowledge-base search) built atop LLMs, potentially including fine-tuned ones.
- **Ch14**: Model deployment/evaluation practices here feed into the broader FMOps lifecycle.
