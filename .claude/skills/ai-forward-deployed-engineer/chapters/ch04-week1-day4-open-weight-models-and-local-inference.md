# Week 1, Day 4: Open-Weight Models, Local Inference and Token Economics

## Lessons Covered
- Lesson 27: Inference Basics - GPUs, VRAM, Unified Memory and Memory Bandwidth
- Lesson 28: Proprietary, Open-Source and Open-Weight LLMs - Privacy, Cost and Security
- Lesson 29: Dense vs Mixture of Experts (MoE) Models and Active Parameters
- Lesson 30: Quantization, GGUF, Safetensors and How to Read a Models Name
- Lesson 31: Ollama, LM Studio, vLLM and Choosing Hardware to Run LLMs Locally
- Lesson 32: Install Ollama and Chat with Qwen 3.5 and Gemma 4 on Your Own Computer
- Lesson 33: OpenRouter Provider Routing, Token Pricing and the Cost per Claim
- Lesson 34: Testing DeepSeek, Kimi K3, GLM, MiMo, Nemotron and Mistral on Claims
- Lesson 35: Ollama vs OpenRouter vs Cerebras - Comparing Accuracy, Cost and Speed

## Core Idea
Open-weight and local models give you real deployment options (privacy, fine-tuning, avoiding vendor cost) but are never automatically cheaper or safer than proprietary frontier models — the right choice always comes from running your own accuracy/cost/speed experiment on the business task, exactly as with proprietary models on Day 2, and a tighter, leaner prompt consistently helps smaller models perform closer to their larger cousins.

## Frameworks Introduced
- **Proprietary vs. Open-Source vs. Open-Weight taxonomy**: proprietary = weights never released (OpenAI, Anthropic, Gemini); true open-source = weights + training data + methodology released (rare, e.g. K2 from a UAE university); open-weight = only the weights released (the common case, e.g. Qwen, Kimi K3, GLM, MiMo, DeepSeek); open-weight-with-restrictions = weights released but with commercial-use limitations above a revenue threshold (e.g. some Meta/Llama licenses).
  - When to use: to correct the loose, inaccurate industry habit of calling everything "open source" — precise terminology matters when discussing licensing/legal implications with a client.
- **Dense vs. Mixture-of-Experts (MoE) Model Architecture**: dense = all parameters activate on every token; MoE = only a subset of "expert" sub-networks activate per token, chosen by a per-layer learned router, with one or more experts "shared"/always-active.
  - When to use: when memory is your binding constraint and you must choose between similarly-capable dense vs. MoE variants of the same model family (e.g. Qwen 3.6 27B dense vs. 35B-A3B MoE) — MoE gives speed at the cost of total memory footprint (must still fit the whole model) and sometimes slightly lower benchmark intelligence.
  - How: read the model name suffix `<total>B-A<active>B` (e.g. `35B-A3B` = 35B total parameters, ~3B active per token); MoE is strictly an internal inference-efficiency mechanism, unrelated to agentic routing between separate LLM calls.
- **Quantization**: reducing the numeric precision (bit-width) used to store each parameter, trading a little intelligence for large reductions in memory and compute.
  - When to use: whenever model size or speed is the binding constraint and you're choosing between a model's full-precision and quantized release.
  - How: memory (GB) ≈ (billions of parameters × bits per weight) ÷ 8; going 16-bit → 8-bit typically costs under ~1% intelligence; going to 4-bit is noticeably felt, especially in coding-agent use; QAT (quantization-aware training) trains a model specifically to tolerate later quantization well.
- **OpenRouter Provider Pinning**: because OpenRouter routes the same model name to many different backend providers (which may differ in quantization, feature support, and price), you can constrain and order which providers are used.
  - When to use: whenever reliability (consistent precision, guaranteed structured-output support, stable pricing) matters more than maximum redundancy.
  - How: pass `extra_body={"provider": {"order": [...], "allow_fallbacks": False, "require_parameters": True, "ignore": [...]}}` to pin a provider list, require feature support, or exclude a misbehaving provider.
- **Unit Economics / Cost-per-Business-Task**: think in terms of "cost per claim processed," not "cost per token" or "cost per model."
  - When to use: whenever presenting a model-selection recommendation to a non-technical business stakeholder.
  - How: compute total cost across a representative batch of the real task, divide by task count, and multiply by expected monthly/annual volume to get a business-relevant dollar figure (e.g. $0.00024/claim × 10,000 claims/month ≈ $2.40/month).

## Key Concepts
- **Inference**: running a trained model on new input to generate output (vs. training, which sets the parameters); for LLMs this means repeated autoregressive calls, one token at a time.
- **GPU / VRAM / Unified Memory**: GPUs are parallel-matrix-math chips (originally for graphics) now central to LLM inference; VRAM is a PC/Linux GPU's dedicated memory; Unified Memory (modern Macs) shares one memory pool between CPU and GPU.
- **Memory bandwidth**: how fast data can move into the GPU — a major, under-discussed factor in inference speed alongside raw memory capacity.
- **Distillation**: training a smaller "student" model on outputs generated by a larger "teacher" model to approximate its behavior at a fraction of the size — unlike quantization, this is a "sledgehammer" tradeoff that meaningfully reduces intelligence, not a near-free efficiency gain.
- **Safetensors / GGUF / MLX**: three model file formats — safetensors (Hugging Face's standard publishing format), GGUF (llama.cpp-originated, used by Ollama/LM Studio, packs weights + tokenizer + chat template into one file), MLX (Apple Silicon format used by Ollama/LM Studio on Mac).
- **Model-name decoding**: suffixes like `550B-A55B` (MoE total/active parameters), `IT` (instruction-tuned, i.e. good at chat), `E4B` (Gemma's "effective" parameter trick, a distinct speed optimization), `QAT` (quantization-aware trained), `Q4_K_M`/`BF16` (quantization scheme and bit-width tags), `:free` (OpenRouter's no-cost, rate-limited tier).
- **Ollama / LM Studio**: consumer-friendly local LLM runners exposing OpenAI-compatible APIs on localhost (Ollama defaults to port 11434); both use C++ inference libraries (originally llama.cpp for Ollama) plus Apple's `mlx-lm` on Macs. Watch out for "Ollama Cloud," a separate paid cloud-inference offering that is not local.
- **vLLM**: the production-grade inference server for NVIDIA/AMD GPU deployments (not Mac-native), exposed via an OpenAI-compatible API — the typical target for renting GPUs (e.g. via RunPod) to self-host at scale.
- **llmfit**: a CLI tool (`uv tool install llmfit`) that analyzes your machine's hardware and reports which models will fit and at what expected tokens/second.
- **Groq (with a Q) vs. Grok (with a K)**: Groq is a high-performance open-model inference provider (a potential OpenRouter backend); Grok is xAI's proprietary model — easily confused, deliberately called out as a trap.
- **Information-security framing for third-party LLM calls**: treat sending data to a proprietary-model vendor the same as any other third-party technology dependency (e.g. Workday, Office 365) — apply standard due diligence/SOC 2 review rather than treating it as uniquely risky because it "feels" conversational.
- **Prompt tightening strategy**: Day 4's lab deliberately shrinks the Day 2 prompt (~2,900 → ~1,200 tokens) via a shorter policy-wording document plus claim-type-specific section selection, improving small-model accuracy by raising prompt signal-to-noise, not merely saving tokens.

## Mental Models
- "Having your cake and eating it" (quantization) vs. "a sledgehammer" (distillation): use this contrast to quickly explain to a colleague why a quantized 16B model usually beats a distilled-down smaller model of similar size at the same task.
- Treat the official explanation for why quantization works so well ("it's empirical") as an honest admission of an open research question — useful shorthand, not false confidence.
- Apply the same Pareto-front mental model from Day 2 to open-weight models: plot your own accuracy-vs-cost and accuracy-vs-speed scatterplots on your actual task rather than trusting that "open source = cheap" or "bigger = smarter."
- When sizing local hardware, think in two separate axes — "will it fit" (total memory) and "will it be fast" (memory bandwidth, active-parameter count for MoE) — a model that fits can still be commercially unusable if too slow.
- For OpenRouter specifically, think of the model name as a contract and the provider as an implementation detail you must pin down if you need consistent behavior — the same model name can silently vary in quantization, feature support, and price across providers.

## Anti-patterns
- **Assuming open-weight/open-source models are automatically cheaper**: someone always pays the compute bill; a free OpenRouter model costs you your training data, a locally-run model costs electricity and hardware depreciation, and even paid open-weight models can cost more than a cheap proprietary model like GPT-6 Luna for the same task.
- **Assuming more total parameters means more intelligence**: Qwen 3.6 35B-A3B (MoE) scored lower on the Artificial Analysis Intelligence Index than the smaller 27B dense Qwen 3.6 — total parameter count without architecture context is a misleading signal.
- **Treating OpenRouter's single model name as a fixed, stable contract**: different backend providers behind the same model name can silently change precision, feature support (e.g. structured outputs), max output tokens, and price — pin providers explicitly if your use case needs stability.
- **Obsessing over shaving a word or two from a prompt for token savings**: micro-optimizing prompt wording for cost is rarely worth the effort; a wholesale prompt rewrite for signal-to-noise (as done in this lab) is worth it, but nickel-and-diming tokens usually isn't.
- **Forgetting to `/compact` before walking away from a coding-agent session**: leaving a session mid-task and compacting only after returning forfeits prompt-caching savings on everything that accumulated in between — compact right before stepping away, not right after returning.
- **Conflating MoE's internal "router" with agentic multi-LLM routing**: the per-layer expert router inside an MoE model is pure inference-time math optimization, unrelated to agent frameworks that route between separate LLM calls (or to OpenRouter's provider routing, which is yet a third, different kind of "routing").

## Code Examples
```python
def get_sample_data(tiny=False, held_out=False):
    """Return a list of claims, and the handlers' decisions on them keyed by claim id."""
    claims = load("evals/held_out_claims.json" if held_out else "claims.json")
    decisions = load("evals/held_out_decisions.json" if held_out else "evals/decisions.json")
    if tiny:
        chosen = {decision["claim_id"] for decision in decisions if decision["tiny"]}
        claims = [claim for claim in claims if claim["claim_id"] in chosen]
        decisions = [decision for decision in decisions if decision["tiny"]]
    labels = {decision["claim_id"]: decision for decision in decisions}
    return claims, labels

claims, labels = get_sample_data(tiny=False)
claims = [claim for claim in claims if labels[claim["claim_id"]]["difficulty"] != "manual"]
```
- **What it demonstrates**: deliberately excluding the 5 "manual"-difficulty claims (unanswerable without a claims manual not yet provided) so the benchmark isn't polluted by inherently unsolvable cases.

```python
def split_sections(wording):
    """Split the wording into its opening lines and a dict of its sections, keyed by heading."""
    parts = wording.split("\n## ")
    sections = {part.split("\n")[0].strip(): "## " + part for part in parts[1:]}
    return parts[0], sections

BENEFITS = {
    "trip_cancellation": "Cancellation and Interruption",
    "trip_interruption": "Cancellation and Interruption",
    "trip_delay": "Trip Delay",
    # ...
}
```
- **What it demonstrates**: progressive-disclosure-style prompt construction — splitting the policy document into sections and selecting only the section relevant to a given claim type, rather than sending the whole policy every time.

```python
response = openrouter.chat.completions.parse(
    model=model,
    messages=make_messages(claim),
    response_format=ClaimDecision,
    reasoning_effort=effort,
    extra_body={"provider": {"require_parameters": True, "ignore": [ignore] if ignore else []}},
)
```
- **What it demonstrates**: `require_parameters: True` forces OpenRouter to only route to providers supporting the requested features (e.g. structured outputs); `ignore` excludes a specific misbehaving provider (used here to exclude a provider not honoring the `low` reasoning-effort setting for GLM 5.3).

```python
response = ollama.chat(
    model=model,
    messages=make_messages(claim),
    format=ClaimDecision.model_json_schema(),
    think=False,
)
answer = ClaimDecision.model_validate_json(response.message.content)
```
- **What it demonstrates**: calling a local Ollama model with a JSON-schema-constrained response, then validating the raw JSON text back into the same Pydantic `ClaimDecision` object used for the cloud-model experiments, keeping the evaluation code identical across local and cloud paths.

## Reference Tables
| Model | Where run | Decisions right (of 20) | Payout also right | Cost (all 20) | Time (all 20) |
|---|---|---|---|---|---|
| MiMo (Xiaomi) | OpenRouter | 16 | 14 | 2 cents | 565s (~9m 24s) |
| GLM 5.3 Flash (Z.ai) | OpenRouter | 16 | 13 | 0.4 cents | 179s |
| Kimi K3 (Moonshot AI) | OpenRouter | 16 | 14 | 11 cents (priciest) | 99s |
| DeepSeek V4.1 Flash | OpenRouter | 17 (best accuracy) | 16 | 1.7 cents | ~5s/claim |
| Nemotron (NVIDIA) | OpenRouter | 15 | 11 | 2 cents | 37s (very fast) |
| Mistral | OpenRouter | 15 | more than Nemotron | 5 cents | 23s (fastest OpenRouter run) |
| Qwen 3.5 4B | Ollama (local) | 14 | 8 | $0 API cost | ~3.5 min |
| Gemma 4 E2B-IT-QAT | Ollama (local) | 15 | 8 | $0 API cost | 73s |
| Gemma 4 E4B | Ollama (local) | 14 | 11 | $0 API cost | 240s |
| Qwen 3.8 27B via Cerebras, no reasoning | Cerebras (direct) | 13 | 11 | — | 9s (fastest overall) |
| Qwen 3.8 27B via Cerebras, low reasoning | Cerebras (direct) | 17 (tied best) | 14 | ~6 cents | 19s |
| Qwen 3.8 27B via Cerebras, high reasoning | Cerebras (direct) | fewer than low | fewer than low | more | longer |

| Quantization level | Typical intelligence impact |
|---|---|
| 16-bit → 8-bit | Very small (as low as ~1% degradation in some benchmarks) |
| 16-bit → 4-bit | Noticeably felt in practice (e.g. coding-agent quality), though benchmark scores often don't show it as much |

| Local hardware option | Approx. price | Capability |
|---|---|---|
| Mac Studio M5 Ultra | ~$10,000 | 256GB unified RAM, 48 tokens/sec even on a big model |
| NVIDIA DGX Spark | ~$5,000 | 128GB RAM, suited to MoE models |
| PC tower + NVIDIA GPU | ~$10,000 | More flexible memory configuration |
| Mac mini / Beelink mini PC | ~$2,000–$2,500 | Comfortably runs Qwen 27B as a competent (non-frontier) coding assistant |
| RunPod rented GPU (e.g. RTX PRO 6000) | ~$2/hour | Try frontier-scale local inference briefly and cheaply |

## Worked Example
The Day 4 lab (`labs/week1day4.ipynb`, with free and CPU-only variants) runs 20 of UnderCover's claims (excluding the 5 "manual" ones, which remain unanswerable without data not yet provided) through nine open-weight models plus one optional direct-provider test, using a deliberately shortened and section-targeted prompt (down from ~2,900 to ~1,200 tokens versus Day 2). Step 1: establish the baseline — 12 of 20 claims are "approve," so any model that blindly approves everything scores a meaningless 60%. Step 2: run six OpenRouter models (MiMo, GLM 5.3 Flash, Kimi K3, DeepSeek V4.1 Flash, Nemotron, Mistral) with structured outputs and provider pinning/ignoring as needed; DeepSeek V4.1 Flash wins on accuracy (17/20, 16/20 with correct payout) while also being reasonably fast and cheap, whereas Kimi K3 ties on accuracy but costs roughly 6x more. Step 3: run three models locally via Ollama (Qwen 3.5 4B, Gemma 4 E2B-IT-QAT, Gemma 4 E4B) — all free in API cost but visibly taxing the instructor's GPU, landing in the 14-15/20 accuracy range, impressive for models 25-100x smaller than the frontier-scale OpenRouter models. Step 4 (optional): a direct Cerebras connection running Qwen 3.8 27B shows that reasoning effort isn't monotonically helpful — "none" effort finished in 9 seconds (13/20 correct), "low" effort hit the overall tied-best accuracy of 17/20 in 19 seconds, but "high" effort actually did slightly worse while taking longer and costing more, illustrating that more reasoning is not reliably better and must be tested per task. Final step: 2D accuracy-vs-cost and accuracy-vs-speed scatterplots (plus an optional 3D plot) reveal DeepSeek V4.1 Flash dominating this particular task's Pareto front across both cost and speed, with GLM 5.3 Flash and Gemma 4 E4B as the only sensible picks if the budget is below roughly 0.1 cent per claim.

## Key Takeaways
1. "Open source" is usually a misnomer for "open-weight" — know the distinction (weights only vs. weights + data + methodology) when discussing licensing with stakeholders.
2. MoE architecture trades a little intelligence and extra total memory footprint for substantially faster, cheaper inference per token by activating only a subset of "expert" parameters per layer.
3. Quantization is a near-free efficiency win (especially 16-bit → 8-bit); distillation is a real intelligence cut in exchange for a much smaller model — don't confuse the two tradeoffs.
4. Decode model names methodically: total/active parameters, instruction-tuning, quantization scheme, and QAT status are all encoded in the suffix conventions, though labs are inconsistent (no universal standard).
5. OpenRouter's convenience comes with provider variability — pin providers, require feature parameters, and exclude misbehaving providers when reliability matters.
6. Think and report in cost-per-business-task (e.g. cost per claim), not cost-per-token, when talking to business stakeholders about model economics.
7. A tighter, more targeted prompt (less noise, more signal) measurably improves smaller/local models' accuracy — this is a higher-leverage optimization than micro-trimming words for token savings.
8. Reasoning effort is not reliably monotonic with accuracy even on strong models (Cerebras/Qwen's high-reasoning run underperformed its low-reasoning run in this test) — always validate empirically per task.
9. Running models locally (Ollama, vLLM) remains essential for strict data-residency/on-premises requirements, independent of whether it's actually cheaper than a cloud API call.

## Connects To
- **Week 1, Day 2**: directly reuses and extends the structured-outputs/claims-decision evaluation harness built there, applying the same accuracy/cost/speed methodology to open-weight and local models instead of proprietary frontier models.
- **Week 1, Day 3 and Day 5**: Day 4 deliberately sets the UnderCover pilot product aside to focus purely on model experimentation in notebooks; Day 5's harness engineering returns to building the product, with an explicit stretch challenge here to connect a chosen local/open-weight model into the Day 3 pilot.
- **vLLM / RunPod**: foreshadows the optional stretch exercise (and later production deployment themes) of self-hosting inference at scale on rented GPU infrastructure rather than consumer hardware.
- **AWS Bedrock (Week 3)**: the proprietary-model "trusted partner" routing pattern described here (calling a lab's model via AWS Bedrock, billed through AWS) foreshadows Week 3's deeper AWS AgentCore deployment coverage.
