# Chapter 6: Implementation of LLMs

## Core Idea
LLMs can be adapted to specific tasks via three complementary paths — fine-tuning/training (including PEFT techniques like LoRA/QLoRA), prompt engineering (zero-shot, few-shot, CoT, ToT), and RAG — and this chapter builds complete open-source chat-with-PDF implementations for both Llama 2 and Falcon.

## Frameworks Introduced
- **Three adaptation paths for LLMs**: (1) training/fine-tuning base LLMs/FMs, (2) prompt engineering, (3) RAG.
  - When to use: fine-tuning for deep task/domain specialization with available labeled data and compute; prompt engineering for fast, no-retraining behavior steering; RAG for grounding in external/current knowledge.
- **LLM training pipeline**: pretraining (data collection → cleaning → tokenization → architecture selection) → training (instruction tuning or supervised fine-tuning: choose pretrained model → prepare labeled data → select framework → define task → choose fine-tuning approach (full vs. layer-freezing) → train → evaluate) → reinforcement learning (RLHF or DPO).
- **RLHF (Reinforcement Learning from Human Feedback)**: generate multiple outputs per prompt → human labeler ranks them → train a reward model → use RL to align the LLM to maximize helpfulness/minimize harm.
- **DPO (Direct Preference Optimization)**: directly optimizes the policy to satisfy human preferences via a simple binary cross-entropy classification objective — no separate reward model or RL needed; more stable/lightweight than RLHF/PPO.
  - When to use: as a simpler, more compute-efficient alternative to RLHF for preference alignment.
- **Prompt engineering technique catalog**: explicit prompting, zero-shot, few-shot, chain-of-thought (CoT), self-consistency sampling, tree-of-thought (ToT), sequential prompting, positive/negative prompting, ReAct, recursive prompting, graph-of-thought, emotion prompting.
  - When to use explicit prompting: need precise, formatted, rule-constrained output (e.g., "Never cite sources older than 2020").
  - When to use few-shot: want consistent output format/style demonstrated via n examples.
  - When to use CoT: multi-step reasoning tasks (math, commonsense reasoning) — add "think step by step."
  - When to use self-consistency: CoT tasks where sampling multiple reasoning paths and picking the most frequent answer improves accuracy (at higher compute cost).
  - When to use ToT: complex tasks needing exploration/lookahead/backtracking (e.g., Game of 24) — maintains a tree of intermediate "thoughts" evaluated via BFS/DFS.
- **PEFT (Parameter-Efficient Fine-Tuning) family**: LoRA, QLoRA, quantization.
  - LoRA: freezes pre-trained weights, injects trainable low-rank decomposition matrices (via SVD) into transformer attention layers — reduces trainable parameters up to 10,000x and GPU memory 3x vs. full fine-tuning (vs. GPT-3 175B + Adam).
  - Quantization: converts weights/activations from high precision (32-bit float) to low precision (8-bit/4-bit) to shrink memory footprint; tunable parameters are kept at higher precision during training.
  - QLoRA: combines LoRA with 4-bit quantization of adapter matrices — fine-tunes a 65B model on a single 48GB GPU while preserving 16-bit fine-tuning performance.
  - When to use PEFT: cost/compute constraints, need for better generalization (less overfitting), deployment on resource-constrained devices.
- **LLM evaluation framework**: benchmarking (standard datasets: GLUE, MMLU, SQuAD 2.0, HELM; Hugging Face Open LLM Leaderboard) + human evaluation (expert annotation, crowdsourcing) + task-specific metrics (BLEU for translation, ROUGE for summarization, perplexity for next-word prediction quality) + RAG-specific evaluation (retriever: cosine/Euclidean distance, similarity search with score, LangChain's maximum marginal relevance/MMR; generator: BLEU/ROUGE/perplexity).

## Key Concepts
- **PEFT (Parameter-Efficient Fine-Tuning)**: fine-tuning approaches that update only a small subset of parameters instead of the full model.
- **LoRA (Low-Rank Adaptation)**: injects small trainable low-rank matrices into frozen pretrained model layers.
- **QLoRA**: LoRA + 4-bit quantization of the base model for extreme memory efficiency.
- **Quantization**: reducing numeric precision of model weights (e.g., FP32 → INT4/INT8) to reduce memory/compute.
- **bitsandbytes**: lightweight CUDA wrapper library providing 8-bit optimizers and quantization functions.
- **BLEU score**: 0-1 metric measuring machine translation quality against reference translations (1 = perfect overlap).
- **ROUGE**: metric for evaluating summarization quality.
- **Perplexity**: measures how well a model predicts the next word; lower is better.
- **Chain-of-Thought (CoT) prompting**: guiding the LLM to articulate step-by-step reasoning.
- **Tree-of-Thought (ToT)**: maintains and searches a tree of intermediate reasoning steps with lookahead/backtracking.

## Mental Models
- Think of the three adaptation paths (fine-tune, prompt-engineer, RAG) as a decision tree based on what you have: labeled data + compute → fine-tune; neither, but need behavior control → prompt engineer; need grounding in external/current facts → RAG. They are also combinable.
- Use PEFT (LoRA/QLoRA analogy: "tweaking a cake recipe with a pinch of cinnamon rather than rewriting it") whenever full fine-tuning's cost is prohibitive — small targeted parameter updates instead of retraining everything.
- Treat prompt engineering techniques as a toolbox matched to task shape: explicit/few-shot for format control, CoT/self-consistency/ToT for reasoning depth, sequential/positive-negative for structured multi-step or constrained generation.
- Evaluation is multidimensional: benchmark score alone is insufficient — pair automatic metrics (BLEU/ROUGE/perplexity) with human evaluation and, for RAG, evaluate retriever and generator separately.

## Anti-patterns
- **Full fine-tuning a billion-parameter LLM without considering PEFT**: computationally and financially wasteful when LoRA/QLoRA can achieve comparable results at a fraction of the trainable-parameter count and GPU memory.
- **Relying on RLHF when a simpler alternative exists**: DPO achieves similar preference alignment without a separate reward model or RL loop — consider it first for efficiency.
- **Using naive greedy decoding for complex reasoning tasks**: single-path CoT can settle on a suboptimal answer; self-consistency sampling (multiple paths, pick most frequent) improves accuracy at added compute cost.
- **Evaluating an LLM/RAG system with only one metric**: a good BLEU/ROUGE score doesn't guarantee a good retriever, and vice versa — evaluate retriever and generator components separately.
- **Skipping config-driven setup for retrieval parameters** (chunk size, overlap, vector count): hardcoding these makes it hard to tune retrieval quality (k=3 in the book's `VECTOR_COUNT` example is a tunable, not fixed, value).

## Code Examples
```yaml
# config.yml — Llama 2 RAG app configuration
RETURN_SOURCE_DOCUMENTS: True
VECTOR_COUNT: 3
CHUNK_SIZE: 1000
CHUNK_OVERLAP: 100
DATA_PATH: 'source_data/'
DB_FAISS_PATH: 'vector_storage/db_faiss'
MODEL_TYPE: 'llama'
MODEL_BIN_PATH: 'models/llama-2-7b-chat.ggmlv3.q8_0.bin'
MAX_NEW_TOKENS: 256
TEMPERATURE: 0.2
```
- **What it demonstrates**: externalized, config-driven RAG parameters (chunking, retrieval count, model path) for reproducible experimentation.

```python
# llm.py — load a local quantized Llama 2 model via CTransformers
from langchain.llms import CTransformers
from dotenv import find_dotenv, load_dotenv
import box, yaml

load_dotenv(find_dotenv())
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def build_llm():
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                         model_type=cfg.MODEL_TYPE,
                         config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
                                 'temperature': cfg.TEMPERATURE})
    return llm
```
- **What it demonstrates**: loading a local GGML-quantized Llama 2 model through LangChain's `CTransformers` wrapper using config-driven parameters.

```python
# vectordb_build.py — ingest PDFs into a FAISS vector store
import box, yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def run_db_build():
    loader = DirectoryLoader(cfg.DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                     chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)

if __name__ == "__main__":
    run_db_build()
```
- **What it demonstrates**: directory-level PDF ingestion (vs. single-file in Ch2) using `DirectoryLoader`, config-driven chunking, and sentence-transformer embeddings.

```python
# utils.py — assemble the RetrievalQA chain
import box, yaml
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompt_template import qa_template
from src.llm import build_llm

with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def set_qa_prompt():
    return PromptTemplate(template=qa_template, input_variables=['context', 'question'])

def build_retrieval_qa(llm, prompt, vectordb):
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
        return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
        chain_type_kwargs={'prompt': prompt})

def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm()
    qa_prompt = set_qa_prompt()
    return build_retrieval_qa(llm, qa_prompt, vectordb)
```
- **What it demonstrates**: wiring embeddings, vector store, LLM, and prompt template into a single config-driven `RetrievalQA` chain.

```python
# QLoRA 4-bit quantization + inference with Mistral-7B-Instruct
import torch, transformers
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", quantization_config=quantization_config, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

trans_pipeline = transformers.pipeline(
    "text-generation", model=model_4bit, tokenizer=tokenizer, use_cache=True,
    device_map="auto", max_length=296, do_sample=True, top_k=3, num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id
)
llm = HuggingFacePipeline(pipeline=trans_pipeline)
prompt_template = """Question: {question} Answer: You are a Question & answer bot, You have to think step by step and provide the answer for the question."""
prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
llm_chain("How to start a car?")
```
- **What it demonstrates**: loading Mistral-7B-Instruct with 4-bit NF4 quantization (`BitsAndBytesConfig`) to shrink memory footprint (~15GB → <6GB), then wrapping it in a LangChain `LLMChain` for Q&A.

## Reference Tables
None (this chapter uses figures rather than tables; Figure 6.17/6.18 reference the 26 prompting principles categorized into prompt structure/quality, specificity, user interaction, content style, and complex tasks/coding).

## Worked Example
Two parallel open-source chat-with-PDF CLI apps are built: (1) **Llama 2 7B** — config.yml defines chunking/retrieval params → `vectordb_build.py` ingests PDFs into FAISS using sentence-transformer embeddings → `llm.py` loads the local GGML-quantized Llama 2 model via `CTransformers` → `utils.py` builds a `RetrievalQA` chain (`k=3`) → `main.py` answers "How does the transformer architecture work?" with top-3 sourced answers and timing. (2) **Falcon 7B** (WizardLM-Uncensored-Falcon-7B-GGML) — identical pipeline, just swap `MODEL_BIN_PATH` in config.yml.

## Key Takeaways
1. LLM adaptation has three levers — fine-tuning, prompt engineering, RAG — chosen by available data, compute, and need for external grounding.
2. PEFT (LoRA, QLoRA, quantization) makes fine-tuning billion-parameter models tractable on modest hardware by updating only a small parameter subset or reducing numeric precision.
3. DPO is a simpler, more stable alternative to RLHF for preference alignment, avoiding a separate reward model and RL loop.
4. Prompt engineering technique choice should match task shape: formatting/consistency (few-shot), multi-step reasoning (CoT/self-consistency/ToT), constrained output (explicit/positive-negative prompting).
5. Evaluate LLMs and RAG systems with task-appropriate metrics (BLEU for translation, ROUGE for summarization, perplexity for language modeling) plus human evaluation — and evaluate retriever and generator separately in RAG.
6. Config-driven architecture (YAML files for chunk size, vector count, model paths) makes RAG pipelines easy to swap between models (Llama 2 ↔ Falcon) without code changes.

## Connects To
- **Ch2**: Directly extends the Falcon chat-with-PDF implementation from Chapter 2 into a more structured, config-driven, multi-model (Llama 2 + Falcon) RAG pipeline.
- **Ch5**: Reuses LangChain's `RetrievalQA` chain pattern introduced in Chapter 5, now with config-file-driven parameters.
- **Ch7**: PEFT/quantization techniques (LoRA, QLoRA, bitsandbytes) here set up the Hugging Face-centric implementation chapter that follows.
- **Ch9**: Model evaluation and efficient deployment (quantization) concerns here connect directly to the deployment chapter's production considerations.
