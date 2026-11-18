# Patterns

## Transfer Learning + Fine-Tuning
**When to use**: You need domain/task-specific performance but lack the data or compute to train a model from scratch.
**How**: Start from a pre-trained foundation model matched to your modality/task; freeze most layers and tune only the top layers (or use PEFT/LoRA to update a small parameter subset); optionally continue pre-training on domain data first, then fine-tune on labeled task data.
**Trade-offs**: Dramatically cheaper than pre-training but risks catastrophic forgetting with full fine-tuning; PEFT avoids this at the cost of slightly lower ceiling accuracy versus full fine-tuning.

## Retrieval-Augmented Generation (RAG)
**When to use**: You need an LLM to answer from current, proprietary, or domain-specific knowledge without retraining, and want to reduce hallucination.
**How**: Chunk and embed your documents into a vector store; at query time, retrieve the top-k relevant chunks via similarity search; stuff/synthesize them into the LLM's prompt alongside the user query; generate the final answer (optionally with citations).
**Trade-offs**: Avoids fine-tuning cost and staleness but adds retrieval-pipeline complexity (chunking strategy, embedding model choice, vector store latency) and is only as good as retrieval precision/recall.

## Small-to-Big (Auto-Merging) Retrieval
**When to use**: An answer spans multiple sections of a document and single fixed-size chunks lose needed context.
**How**: Build a hierarchy of parent chunks (large) and child chunks (small, multiple sizes) with parent-child references; retrieve with small chunks for precision, then expand to parent chunks for synthesis context.
**Trade-offs**: Better answer quality on multi-section queries at the cost of more complex indexing and storage overhead.

## Chain Strategy Selection for Long Documents
**When to use**: Content exceeds the LLM's context window.
**How**: Use stuffing when one/few chunks suffice; map-reduce for parallelizable independent documents; map-rank when you need the single best-confidence source; refine when sequential, incrementally-built context improves accuracy.
**Trade-offs**: Stuffing is fastest but loses information beyond the window; map-reduce parallelizes but loses cross-document synthesis; refine is most accurate but slowest (sequential).

## Prompt Engineering Escalation Ladder
**When to use**: Any LLM task — scale technique to task complexity.
**How**: Start with zero-shot; add few-shot examples if needed; add explicit instructions for behavior/style control; use Chain-of-Thought for reasoning tasks; escalate to Tree-of-Thought only for tasks needing strategic exploration/look-ahead.
**Trade-offs**: More sophisticated techniques (ToT) cost more tokens/latency and complexity — reserve for genuinely hard reasoning tasks, not routine queries.

## Prompt Engineering Process (Design → Evaluate → Refine → Templatize)
**When to use**: Any production prompt that will be reused or scaled across users.
**How**: Set audience/tone/structure objectives; design the prompt (openness/length combination matched to task); evaluate on accuracy/relevance/coherence/factuality/bias; refine iteratively in a sandbox; version-control the final template with metadata (model, params, cost, latency).
**Trade-offs**: Adds process overhead versus ad hoc prompting but is essential for consistent, auditable production behavior.

## Domain-Adaptive Pretraining + Task Fine-Tuning (Two-Stage Adaptation)
**When to use**: Adapting a general LLM to a narrow domain and a specific task with limited labeled data.
**How**: Stage 1 — continue pre-training (causal LM objective) on raw domain text (manuals, documents) to build contextual relevance. Stage 2 — fine-tune the domain-adapted model on a small labeled set of (input, desired-output) pairs for the specific task format.
**Trade-offs**: Produces much better task-specific output than either stage alone, but requires enough domain text for Stage 1 to be worthwhile; skip it for tasks with abundant labeled data where direct fine-tuning suffices.

## Agent Tool-Use Pattern (ReAct)
**When to use**: The LLM needs to interact with external systems (databases, APIs) rather than just generate text.
**How**: Define a `Tool` wrapping a function with a description; register it with an agent (e.g., OpenAIFunctionsAgent); the LLM decides whether/how to call the tool based on the user query and tool description, the application executes it, and the result is incorporated into the final response.
**Trade-offs**: Extends LLM capability to real actions but introduces new risks (excessive agency, unauthorized actions) requiring scoped permissions and human-in-the-loop for sensitive operations.

## Classifier-Free Guidance (Diffusion Conditioning)
**When to use**: Conditioning diffusion-model generation (e.g., text-to-image) without training a separate classifier.
**How**: During training, randomly drop the conditioning label to teach the model both conditional and unconditional generation; at sampling time, blend the two predictions using weight ω (0=unconditional, 1=standard conditional, >1=strongly conditional).
**Trade-offs**: Cheaper and simpler than classifier guidance or CLIP guidance; higher ω increases prompt fidelity at some cost to output diversity.

## Latent Diffusion (Compress-Then-Diffuse)
**When to use**: Image generation at high resolution where pixel-space diffusion would be too slow/expensive.
**How**: Encode the image into a compressed latent space with a VAE encoder; run the diffusion (forward/reverse) process entirely in latent space; decode the final latent back to pixel space.
**Trade-offs**: Major speed/cost reduction (Stable Diffusion's core technique) at the cost of a small quality ceiling imposed by the VAE's compression fidelity.

## CycleGAN Unpaired Domain Translation
**When to use**: Image-to-image translation tasks where paired training examples (source image + matching target image) don't exist.
**How**: Train two GANs with cyclic-consistency loss (`F(G(x))≈x` and `G(F(y))≈y`) so translating to the target domain and back reconstructs the original image.
**Trade-offs**: Enables translation without paired data but generally needs more training and can introduce artifacts versus paired (pix2pix/cGAN) approaches when paired data is actually available.

## PEFT-First Alignment (LoRA Before Full Fine-Tuning)
**When to use**: Adapting an LLM to a new task/domain.
**How**: Default to LoRA (or another PEFT technique) updating a small low-rank parameter subset; reserve full fine-tuning for cases with abundant supervised data and a demonstrated need for complete weight updates.
**Trade-offs**: LoRA is far cheaper in compute/storage and avoids catastrophic forgetting, but may have a slightly lower accuracy ceiling than full fine-tuning on very large labeled datasets.

## Preference Optimization Without a Reward Model (DPO/ORPO over RLHF)
**When to use**: Aligning a model to human preferences when the cost/complexity of RLHF (separate reward model + PPO) isn't justified.
**How**: Collect (prompt, chosen response, rejected response) preference triples; train directly on a preference loss (DPO) or combine language-modeling loss with an odds-ratio term (ORPO) — no separate reward model needed.
**Trade-offs**: Much cheaper and simpler than RLHF with comparable alignment quality in many cases; RLHF may still be preferred when a dedicated, reusable reward model benefits multiple downstream objectives.

## Continuous (In-Flight) Batching for LLM Serving
**When to use**: Serving LLM inference requests with varying generation lengths at scale.
**How**: Execute multiple requests concurrently, injecting newly completed requests into the batch immediately rather than waiting for the longest sequence in a static batch to finish.
**Trade-offs**: Significantly improves GPU utilization and throughput over static/dynamic batching; requires serving infrastructure (vLLM, TGI, Ray Serve) that supports it.

## Model Merging for Multi-Task Capability (No Retraining)
**When to use**: Combining multiple task-specialized fine-tuned models into one capable model without additional training compute.
**How**: Choose a merge technique by need — linear merge (simple weighted average), SLERP (pairwise, preserves curvature), task arithmetic (vector addition/subtraction of fine-tuning deltas), TIES/DARE (handle parameter interference via trimming/pruning + sign election).
**Trade-offs**: Zero additional training cost but quality depends heavily on model compatibility (same architecture/ancestor) and the chosen merge technique's handling of parameter interference.

## Layered LLM Guardrails (Defense in Depth)
**When to use**: Any production LLM application exposed to untrusted input, external data, or acting with real-world permissions.
**How**: Combine input/output content filtering, malicious-prompt classification, structural defenses (delimiters separating instructions from data, meta-prompt templates), scoped agent permissions with human-in-the-loop approval, continuous monitoring/abuse-pattern detection, and pre-deployment red-teaming.
**Trade-offs**: No single control is sufficient; layered defense adds engineering overhead but is necessary given documented jailbreak/injection techniques (direct role-play, indirect poisoned data, DAN-style prompts).

## Carbon-Aware Model Selection
**When to use**: Choosing between a general-purpose foundation model and a smaller task-specific model for a GenAI feature.
**How**: Measure emissions (CodeCarbon, ML CO2 Impact, LLMCarbon) for candidate approaches across the full lifecycle (training + inferencing); prefer the smallest model/architecture that meets task requirements; select cleaner-energy cloud regions and higher-PUE data centers.
**Trade-offs**: Reduces environmental and often financial cost, but requires measurement tooling and may trade off against marginal accuracy gains from larger general-purpose models.
