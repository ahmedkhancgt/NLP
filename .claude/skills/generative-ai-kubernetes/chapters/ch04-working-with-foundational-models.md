# Chapter 4: Working with Foundational Models

## Core Idea
Choosing and operating a foundational model efficiently requires evaluating it against your specific use case (not just leaderboard scores), tuning sampling parameters (temperature, top-k/top-p, context length), applying prompt-engineering techniques (few-shot, chain-of-thought), and — when more compression is needed — applying quantization/pruning/distillation/PEFT to fit resource constraints, including GPU-sharing strategies.

## Frameworks Introduced
- **Model evaluation question framework**: before picking a model, answer — What problem? How will users interact (batch vs real-time)? What context is needed? How domain-specific/bespoke? How much knowledge must it contain? What license/budget constraints apply?
  - When to use: at the start of any new GenAI use case, before benchmarking.
  - How: answer each question, then filter Hugging Face's model search by task/license/library (e.g. safetensors) and cross-reference the Open LLM Leaderboard.
- **LLM benchmark suite**: ARC (reasoning), HellaSwag (commonsense sentence completion), MMLU (57-subject multitask knowledge), TruthfulQA (truthfulness vs. popular-but-false statements), Winogrande (coreference reasoning), GSM8K (grade-school math); plus MBPP/HumanEval (code, using Pass@k) for coding tasks.
  - When to use: choose benchmarks matching your task type (reasoning vs. knowledge vs. code vs. truthfulness) rather than a single aggregate score.
  - How: filter Hugging Face's Eval-metrics tab or consult the Open LLM Leaderboard / HumanEval leaderboard for the relevant benchmark.
- **Sampling control framework**: Top-k (fixed-size candidate pool), Top-p/nucleus (smallest set summing to probability p), Temperature (<1 sharpens/more deterministic, >1 flattens/more creative), Repetition Penalty, Context Length (memory window), Max Tokens (output cap).
  - When to use: tune per use case — low temperature + low top-k for deterministic/factual tasks; higher temperature for creative tasks.
  - How: set library-specific params (e.g. llama-cpp-python/LangChain defaults: temperature 0.8, top_p 0.95, top_k 40) and adjust empirically.
- **Prompt engineering techniques (intro)**: zero-shot (no examples) → few-shot (provide labeled examples in-prompt via `FewShotPromptTemplate`) → chain-of-thought ("Let's think step by step" to force stepwise reasoning).
  - When to use: zero-shot for simple completion; few-shot when you need a specific output format/behavior (e.g. sentiment classification); CoT when multi-step reasoning is required and the model is getting confused.
  - How: construct instruction + examples + user-query templates with LangChain's `PromptTemplate`/`FewShotPromptTemplate`.
- **Model-shrinking toolkit**: Quantization (reduce weight precision — GGUF/GGML for CPU, GPTQ/AWQ for GPU), Pruning (zero out/remove low-impact weights or layers — "structured pruning" targets specific layers), Knowledge Distillation (train a small "student" model on a larger "teacher" model's outputs), PEFT (fine-tune only a small added parameter subset, freezing the base model).
  - When to use: quantization first (broadly applicable, low effort); pruning/distillation/PEFT when quantization alone doesn't meet resource constraints and you have ML expertise to invest.
  - How: quantization — download/use a pre-quantized GGUF (CPU) or GPTQ/AWQ (GPU) model variant from Hugging Face (e.g. TheBloke repos).

## Key Concepts
- **Instruct-tuned model**: a model with an added supervised fine-tuning phase for following instructions/Q&A (naming convention: "-Instruct" suffix); raw/base models complete text but don't reliably follow directions.
- **Tokens / vocabulary**: models choose from a fixed vocabulary of tokens at each generation step via a softmax probability distribution.
- **Context window**: total tokens (input + prior output) the model can "remember"; larger windows cost more memory/compute.
- **GGUF**: GPT-Generated Unified Format — quantized model file format for CPU inference (successor to GGML).
- **GPTQ**: Post-Training Quantization for GPT models — quantizes to 4-bit, dynamically dequantizes to 16-bit during GPU inference.
- **AWQ (Activation-aware Weight Quantization)**: quantizes based on which weights are most activated during inference, preserving important weights at higher precision.
- **LoRA (Low-Rank Adaptation)**: PEFT technique adding small low-rank adapter weight layers on top of a frozen base model.
- **IA3**: PEFT technique injecting learned vectors to amplify/suppress activations between layers, more parameter-efficient than LoRA.

## Mental Models
- Think of **quantization as "fewer bits per weight, same relationships"** — an 8-bit int vs 32-bit float stores the same learned relationships in a fourth of the space, often with small accuracy loss.
- Use **temperature as a creativity/consistency dial, not a quality dial** — lower is more deterministic/repeatable, higher is more varied but less coherent; neither is universally "better."
- Think of **PEFT as "fine-tune the add-on, not the engine"** — you freeze most pretrained parameters and update only a small adapter layer, dramatically cutting compute cost vs. full fine-tuning.
- GPU sharing today is **immature and trades off maintenance burden vs. isolation vs. flexibility** — pick time-slicing for simplicity, MIG for isolation, MPS (not production-ready) for finer efficiency.

## Anti-patterns
- **Judging models by a single leaderboard number**: different benchmarks test different skills (reasoning vs. truthfulness vs. code) — picking a model on one metric alone can mismatch your actual use case.
- **Deploying a non-instruct raw model for chat/Q&A without fine-tuning or strong prompting**: the chapter's Orca-2-7B (non-instruct) vs Mistral-7B-Instruct comparison shows raw models often fail to follow instructions or stay concise.
- **Assuming overfitting-free benchmark scores**: training data can leak benchmark test questions, inflating scores without real capability gains.
- **Running GPU workloads without any sharing strategy**: naively splitting workloads one-per-node is inefficient given GPU scarcity/cost — evaluate time-slicing/MIG/MPS options.
- **Ignoring disk space on model-heavy dev boxes**: large model downloads fill disk fast; use `docker system prune` and `huggingface-cli delete-cache` proactively.

## Code Examples
```dockerfile
FROM python:3.10-slim
WORKDIR /llmcompare
RUN apt-get update -y && apt-get install -y git-lfs && git lfs install
RUN apt-get install -y build-essential && apt-get install -y gcc-11 g++-11
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
    mistral-7b-instruct-v0.2.Q5_K_M.gguf --local-dir Mistral-7B-Instruct-v0.2-GGUF \
    --local-dir-use-symlinks False
COPY . .
EXPOSE 7800
CMD ["gradio", "app.py"]
```
- **What it demonstrates**: pre-downloading quantized GGUF models at build time (via `huggingface-cli` + `HF_HUB_ENABLE_HF_TRANSFER=1`) so containers start fast in Kubernetes — critical for scale-up/recovery speed.

```python
#Prepare prompt
template = """{instructions}
Question: {query}
Answer: """
prompt_template = PromptTemplate(
    input_variables=["instructions", "query"],
    template=template
)
prompt = prompt_template.format(instructions=instruction, query=prompt)
```
- **What it demonstrates**: LangChain `PromptTemplate` separating system instructions from user query for `llama-cpp-python`'s `create_chat_completion`.

```python
few_shot_prompt_template = FewShotPromptTemplate(
    examples=review_examples,
    example_prompt=example_prompts,
    prefix=instructions_template,
    suffix=user_prompt_template,
    input_variables=["instructions", "query"],
    example_separator="\n\n"
)
```
- **What it demonstrates**: few-shot prompting — embedding labeled examples (sentiment classification) directly in the prompt template to shape model output format.

## Reference Tables
| GPU Sharing Approach | Support | Maintenance | Memory Protection | Flexibility |
|---|---|---|---|---|
| Time-slicing | NVIDIA K8s Device Plugin | Low | No isolation/limits | High |
| Multi-Instance GPUs (MIG) | NVIDIA K8s Device Plugin | High (ConfigMap-managed) | Isolation and limits | Low (needs MIG-capable GPUs) |
| Multi-Process Service (MPS) | Not yet in Device Plugin; planned via DRA driver | Medium (unofficial, DRA not production-ready) | No isolation, but memory limits | High |
(Table 4.1: Options for GPU sharing in Kubernetes)

## Worked Example
**CPU-based Mistral-7B-Instruct vs Orca-2-7B prompt comparison:** Build the `chapter4/llmcompare` Dockerfile (pre-downloads both GGUF quantized models). Run identical prompts through both models via Gradio UI:
1. Text completion ("What is the color of") with vague instructions → Mistral (instruct-tuned) asks for clarification; Orca (non-instruct) attempts a direct completion.
2. Same prompt with explicit instruction ("complete sentences that I start") → both models perform much better, demonstrating that instruction clarity matters more than raw model capability for this task.
3. Q&A ("What is the biggest city in the world?") → Mistral gives polished natural answers; adding "Please be concise" measurably shortens Mistral's response but Orca ignores the instruction (no instruct-tuning).
4. Chain-of-thought math problem (cups-on-shelves) → both models get the wrong answer with zero-shot prompting; adding "Let's think step by step" to the instruction prompt produces the correct step-by-step answer from Mistral.
This demonstrates concretely that instruction-tuning + prompt phrasing can matter more than model size for a given task.

## Key Takeaways
1. Select benchmarks that match your actual task (reasoning, truthfulness, code, etc.) rather than relying on one aggregate leaderboard score.
2. Instruction-tuned ("-Instruct") model variants follow directions far better than raw/base models — check the model card for fine-tuning status before choosing.
3. Quantization (GGUF for CPU, GPTQ/AWQ for GPU) is the first, lowest-effort lever for reducing model resource needs; PEFT/pruning/distillation require more ML expertise but shrink further.
4. Temperature, top-k/top-p, context length, and max tokens are the core sampling knobs — tune per use case rather than trusting library defaults blindly.
5. Prompt engineering (few-shot, chain-of-thought, explicit conciseness instructions) can yield large quality/cost improvements without touching model weights at all.
6. GPU sharing in Kubernetes remains immature in 2024-2025 — weigh time-slicing (simple, no isolation) vs. MIG (isolated, high maintenance) vs. MPS (not production-ready) deliberately.

## Connects To
- **Ch1**: revisits tokens, transformers, and quantization concepts first introduced there, now with hands-on application.
- **Ch5-6**: the prompt-comparison patterns here become the basis for the MLflow-based systematic model/parameter comparison pipeline.
- **Ch8**: few-shot and chain-of-thought prompting are expanded into more advanced techniques (least-to-most, plan-and-solve, self-consistency, role prompting) and combined with RAG.
