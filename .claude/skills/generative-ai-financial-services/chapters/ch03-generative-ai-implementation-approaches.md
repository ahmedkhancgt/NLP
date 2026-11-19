# Chapter 3: Generative AI Implementation Approaches

## Core Idea
There are five ways to turn a foundation model into a working application, in increasing order of cost/complexity and decreasing order of frequency of use in financial services: prompt engineering (no weight change) → RAG (no weight change, adds your data) → agentic workflows (orchestrates tools/RAG/APIs autonomously) → fine-tuning (changes weights) → training from scratch (full lifecycle, rarely justified).

## Frameworks Introduced
- **Prompt structure (5 components)**: Task, Context, Input, Examples, Output format.
  - When to use: always — this is the base template for any prompt.
- **Prompt engineering techniques**: Zero-shot, Few-shot, Chain of Thought (CoT), ReAct (Reasoning + Acting), Self-consistency, Tree of Thought (ToT).
  - When to use ReAct: tasks needing reduced hallucination and auditability — leaves reasoning + action traces useful for compliance.
  - When to use Self-consistency: explores multiple reasoning paths and picks the most consistent answer (improves on CoT's single greedy path).
  - When to use ToT: complex logical/computational reasoning where CoT/ReAct/self-consistency struggle; builds and evaluates multiple reasoning sub-trees (e.g., multi-expert simulation) and consolidates.
- **RAG (Retrieval Augmented Generation)**: two workflows — Data Ingestion (chunk → embed → store in vector DB) and Response Generation (embed query → ANN similarity search → merge retrieved context with query → send to FM).
  - When to use: unlocks the majority of financial-services generative AI use-cases by grounding answers in organizational data without changing model weights.
  - How (ingestion): identify document sources → chunk (fixed-size e.g. 512/1024 tokens with overlap, or context-aware/recursive chunking by headings) → embed each chunk → store in vector store (Pinecone, Weaviate, OpenSearch, Faiss, Redis, Chroma).
  - How (generation): embed user query with same embedding model → ANN search (product quantization, LSH, HNSW) using cosine similarity/Euclidean distance/dot product → retrieve context → merge with query → prompt the FM.
- **Agentic workflows**: agents autonomously Plan (build action plan using CoT/ReAct/ToT) → Execute (call tools/APIs/RAG/DB as needed) → Memorize (retain task, steps, outcomes, conversation history via buffered memory or persistent cache/DB).
  - When to use: tasks requiring multi-step orchestration across tools/APIs/data sources that static RAG-chatbot business logic cannot handle (e.g., opening a credit account end-to-end).
- **Multi-agentic workflows**: multiple agents collaborate, each potentially acting as a "tool" for another agent, in hierarchical or associative relationships.
- **Fine-tuning taxonomy** — two orthogonal dimensions, combinable:
  - *Parameter update-driven*: Full fine-tuning (all weights updated; slow, expensive, risks catastrophic forgetting) vs. Parameter-Efficient Fine-Tuning / PEFT (freeze base weights, train small added parameter set) via LoRA, QLoRA, DoRA.
  - *Methodology-driven*: Unsupervised fine-tuning (no labels, predicts next token — good for new language/domain style), Supervised/instruction fine-tuning (labeled data for specific tasks like summarization/Q&A), Reinforcement Learning (RLHF with PPO), Policy optimization (DPO).
  - When to use LoRA: reduce trainable parameters ~10,000x and GPU memory ~3x by injecting small rank-decomposition matrices instead of updating full weight matrices.
  - When to use QLoRA: further reduce memory by backpropagating through a frozen 4-bit quantized base model into LoRA adapters (e.g., 65B model: 780GB → <48GB).
  - When to use DoRA: decomposes pretrained weights into low-rank (general knowledge, frozen via LoRA) + task-specific components (fine-tuned) — claims to outperform LoRA on commonsense reasoning/visual instruction tasks.
  - When to use RLHF/PPO: need human-preference-aligned responses via a trained reward model refining the policy.
  - When to use DPO: want RLHF-like alignment without training a separate reward model — model acts as its own reward model, optimizing a cross-entropy loss directly on human-rated response pairs.
- **Training from scratch (pre-training)**: full generative AI lifecycle (Ch1); justified only with strong ROI case (e.g., BloombergGPT).

## Key Concepts
- **Chunking strategy**: fixed-size (token count + overlap) vs. context-aware (recursive, heading-aware) — directly determines RAG output quality.
- **ANN (Approximate Nearest Neighbor)**: algorithm family (product quantization, LSH, HNSW) for fast vector similarity search at scale.
- **Catastrophic forgetting**: a model fine-tuned on a new task forgets how to perform a previously learned task — the key risk of full fine-tuning.
- **Reward model**: a separate model trained on human-rated response pairs to estimate how a human would rate any given response; central to RLHF.
- **Boto3 / Amazon Bedrock / Amazon SageMaker**: AWS's serverless FM-access service (Bedrock) vs. model-hub + custom-hosting service (SageMaker JumpStart) — book's chosen stack, with Llama 3 as the example open model.

## Mental Models
- Think of the four implementation approaches as a cost/control ladder: prompt engineering (cheap, no control over knowledge) → RAG (moderate cost, grounds knowledge) → agentic (adds autonomous orchestration) → fine-tuning (expensive, changes behavior/style) → pre-training (extreme cost, changes everything).
- Use "does this change model weights?" as the first branching question: prompt engineering, RAG, and agentic workflows do not; fine-tuning and pre-training do.
- For reasoning techniques, escalate only as needed: start zero-shot → add few-shot examples → add CoT for reasoning tasks → add ReAct when external tools/facts are needed → use self-consistency/ToT only for the hardest multi-path reasoning problems (cost increases each step).

## Anti-patterns
- **Reaching for full fine-tuning without ROI justification**: expensive, slow, and risks catastrophic forgetting; the book notes it is "not very common in most financial organizations" for this reason.
- **Reaching for training-from-scratch by default**: extremely time/cost-intensive (BloombergGPT: ~$1M+ in compute alone, data prep >60% of effort); only justified with strong business case (rare).
- **Static chatbot logic for multi-step workflows**: trying to hand-code routing/path-selection logic for complex multi-step tasks (e.g., "open a credit account") instead of using agentic workflows wastes developer effort and doesn't scale.

## Code Examples
```python
# Unsupervised fine-tuning with LoRA + 4-bit quantization on Amazon SageMaker
@remote(volume_size=100, job_name_prefix="fine-tune-llama2-unsupervised")
def train_fn(model_name, train_ds, test_ds=None, lora_r=64, lora_alpha=16,
             lora_dropout=0.1, per_device_train_batch_size=8,
             per_device_eval_batch_size=8, learning_rate=2e-4,
             num_train_epochs=1, chunk_size=2048, merge_weights=False, token=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        quantization_config=bnb_config, device_map="auto")
    modules = find_all_linear_names(model)
    config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=modules,
                         lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    trainer = transformers.Trainer(
        model=model, train_dataset=lm_train_dataset,
        eval_dataset=lm_test_dataset if lm_test_dataset is not None else None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            logging_steps=2, num_train_epochs=num_train_epochs,
            learning_rate=learning_rate, bf16=True,
            save_strategy="no", output_dir="outputs"),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()
    if merge_weights:
        output_dir = "/tmp/model"
        trainer.model.save_pretrained(output_dir, safe_serialization=False)
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model = model.merge_and_unload()
        model.save_pretrained("/opt/ml/model", safe_serialization=True, max_shard_size="2GB")
    else:
        model.save_pretrained("/opt/ml/model", safe_serialization=True)
```
- **What it demonstrates**: End-to-end PEFT/LoRA unsupervised fine-tuning job: 4-bit quantize the base model, attach LoRA adapters to all linear layers, train only the adapters, then optionally merge adapter weights back into the base model in fp16 for deployment.

## Reference Tables
**Fine-tuning approach decision matrix**

| Approach | Weights changed? | Data needed | Use when |
|---|---|---|---|
| Prompt engineering | No | None | Simple tasks: classification, sentiment, quick Q&A |
| RAG | No | Unlabeled docs (chunked/embedded) | Need grounded, up-to-date org-specific knowledge |
| Agentic workflows | No (orchestration layer) | Tool/function specs | Multi-step workflows needing autonomous tool orchestration |
| Full fine-tuning | Yes (all) | Thousands of labeled examples | Strong ROI, long re-training cycle acceptable, mature data science team |
| PEFT (LoRA/QLoRA/DoRA) | Yes (small added set) | Labeled or unlabeled depending on methodology | Need near-full-fine-tuning performance at a fraction of cost/compute |
| RLHF (PPO) | Yes | Human-rated response pairs + reward model | Need human-preference alignment, can afford reward-model training |
| DPO | Yes | Human-rated response pairs (no reward model) | Want RLHF-like alignment, simpler pipeline |
| Train from scratch | Yes (from zero) | Massive curated corpus (100s of billions of tokens) | Domain is fundamentally different and ROI is very strong (e.g., BloombergGPT) |

## Worked Example
**BloombergGPT (training from scratch case study)**: Bloomberg built a 50B-parameter decoder-only model (BLOOM architecture, 70 transformer decoder layers, AdamW optimizer) on a ~700B token dataset (363B finance tokens curated from 40 years of Bloomberg data + 345B general tokens), using unigram tokenization. Trained on 64 p4d.24xlarge SageMaker instances (512 NVIDIA A100 40GB GPUs total, NVSwitch 600GB/s intra-node, EFA 400Gb/s inter-node, FSx for Lustre storage) for ~1.3 million GPU hours (~$1M+ compute cost alone; data prep was >60% of total effort). Result: outperformed GPT-NeoX, OPT-66B, and BLOOM-176B on financial benchmarks (Financial_phrasebank, FiQA, ConvFinQA). Lesson: only pursue this path with an equivalently strong business case.

## Key Takeaways
1. Default to the cheapest approach that meets the bar: prompt engineering first, then RAG, then agentic orchestration, before touching fine-tuning or pre-training.
2. RAG's quality hinges on chunking strategy and embedding/vector-store choice — get these right before blaming the FM.
3. ReAct-style reasoning traces materially help financial services compliance/audit requirements by making model decisions explainable.
4. PEFT methods (LoRA/QLoRA/DoRA) have made fine-tuning dramatically cheaper and should be preferred over full fine-tuning in almost all cases.
5. DPO simplifies RLHF by removing the separate reward-model training step.
6. Training a foundation model from scratch is a multi-million-dollar, multi-month commitment that needs rare, explicit ROI justification (see BloombergGPT).

## Connects To
- **Ch1**: Implements the "customization" step of the generative AI lifecycle.
- **Ch2**: Licensing/weight-access attributes from Ch2 determine whether fine-tuning approaches here are even feasible for a chosen model.
- **Ch10**: Deep-dives fine-tuning for a concrete financial-services use-case.
- **Ch11**: Builds a full agentic workflow (ESG analysis) using the agent Plan/Execute/Memorize framework introduced here.
- **Ch12**: Revisits this chapter's approaches as a formal decision framework for choosing among them.
