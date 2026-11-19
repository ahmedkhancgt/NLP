# Chapter 11: Cost Optimization Strategies

## Core Idea
Cost optimization for enterprise LLMs splits cleanly into two peripheries — "inside" (optimizing the model itself during training: parameters, data quality, architecture, hardware, skills, algorithms) and "outside" (optimizing consumption: model selection, prompt/fine-tuning, token minimization, GPU/compute choices, auto-scaling, temperature, caching) — and treating cost as a strategic lever, not just an expense to cut, drives sustainable AI value.

## Frameworks Introduced
- **Inside Periphery / Outside Periphery Cost Model**: Inside periphery = cost levers specific to training one's own model (parameters/model size, data quality, architecture, modern GPUs/hardware efficiency, skills/training, algorithms/training techniques, inference testing mode). Outside periphery = cost levers for consuming/operating a model (model selection and accuracy, prompt tuning and fine-tuning, minimized input/output tokens, GPU compute and discounted compute, auto-scalability and resource utilization, temperature setting and application caching, skills).
  - When to use: As a complete map of where cost can be optimized, whether an enterprise trains its own model or consumes a third-party one.
  - How: If training your own model, focus on the inside-periphery levers; if consuming a model as-a-service, focus on outside-periphery levers; most enterprises need both.
- **Pruning and Quantization**: Techniques to reduce model size/precision without substantially compromising performance, cutting compute load and memory requirements, enabling cheaper hardware.
  - When to use: During training/architecture-optimization phase (inside periphery) to control acquisition and operational cost.
  - How: Apply pruning to remove less-important connections and quantization to reduce numeric precision; validate performance impact is acceptable before deploying.
- **Token and Temperature Discipline**: Minimize input/output tokens per transaction; set temperature close to zero for more deterministic, accurate, and token-efficient outputs (vs. higher temperature for creative diversity at higher token/compute cost).
  - When to use: Any production LLM consumption scenario, especially customer-facing or high-volume interactions.
  - How: Design prompts/interactions to use the minimum necessary tokens; tune temperature per use case — lower for accuracy-critical, deterministic tasks; higher only when creative variety justifies the cost.

## Key Concepts
- **Outside periphery**: Cost optimization opportunities external to the LLM itself — model selection, tuning, operational settings, scaling.
- **Inside periphery**: Cost optimization opportunities intrinsic to training the LLM itself — parameters, data, architecture, hardware.
- **Pruning**: Removing less-important model connections to reduce size/compute.
- **Quantization**: Reducing numeric precision of model operations to cut compute/memory cost.
- **Prompt tuning vs. fine-tuning (cost lens)**: Prompt tuning adjusts input to elicit better output cheaply; fine-tuning adjusts model weights at higher resource cost but adapts the model more deeply.
- **Discounted/spot compute**: Cloud provider discounted instances (spot/reserved) that lower GPU compute cost at the expense of usage/availability flexibility.
- **Application caching**: Reusing previously generated responses for duplicate queries to avoid redundant model calls.

## Mental Models
- Picture cost optimization as two concentric circles: the "inside" circle (training-time levers) and the "outside" circle (consumption-time levers) — both need tending, but they use different tools.
- Use "garbage in, garbage out, expensively" — poor data quality doesn't just hurt accuracy, it drives costly retraining and continuous tuning cycles.
- Treat temperature as a cost dial, not just a creativity dial: lower temperature isn't only "more accurate," it's also "fewer wasted tokens."

## Anti-patterns
- **Optimizing only inside-periphery levers while ignoring consumption-time waste (e.g., unbounded tokens, no caching)**: Misses major cost savings available without touching the model itself.
- **Always choosing the largest, most capable model regardless of task**: A smaller model might fully satisfy lower-accuracy-tolerant tasks at a fraction of the cost.
- **Running online inference when batch processing would suffice**: Pays for continuous operational load unnecessarily when off-peak batch scheduling could use cheaper energy/compute rates.
- **Fixed-size compute provisioning instead of auto-scaling**: Leaves resources over-provisioned during low demand and under-provisioned during spikes.
- **Treating cost optimization as a one-time exercise rather than continuous**: Model updates, usage pattern shifts, and new discount options require ongoing reassessment.

## Worked Example
The chapter's illustrative framing: if the enterprise need is coding assistance, selecting a pre-trained model with a coding-specific dataset is more cost-effective than a general text-generative model (right-sized model selection, outside periphery). Once selected, prompt tuning and fine-tuning further reduce the tokens needed to reach desired accuracy (outside periphery). Compute sizing should match the LLM's actual requirement rather than being oversized, with longer-duration commitments unlocking double-digit discounts, and auto-scalability configured based on utilization/idle time. During inference, setting temperature close to zero yields more accurate results with fewer generated tokens, directly lowering cost. Application caching then reuses prior responses to minimize repeat API calls. Underlying all of this, skilled personnel making informed choices about tuning type, hardware, and configuration determine whether these levers are actually pulled correctly — making skills investment itself a cost-optimization lever.

## Key Takeaways
1. Map cost optimization opportunities into inside periphery (model training) and outside periphery (model consumption) — most enterprises need to act on both.
2. Use pruning and quantization to shrink model size/compute without materially hurting performance.
3. Match model size/capability to actual task accuracy requirements — don't default to the largest available model.
4. Minimize tokens per interaction and tune temperature deliberately; both directly affect per-transaction cost.
5. Use auto-scalability and discounted/spot compute instances to avoid paying for idle or over-provisioned infrastructure.
6. Application caching eliminates redundant model calls for duplicate queries — a simple, high-leverage cost lever.
7. Invest in skilled personnel; the right cost-optimization technique applied incorrectly (or not at all) negates the potential savings.

## Connects To
- **Ch 2**: Directly expands the ROI/TCO analysis first introduced in the production-scaling framework into concrete optimization levers.
- **Ch 3**: CapEx/OpEx and cost-per-token selection criteria established there are the decision inputs this chapter optimizes against.
- **Ch 6 / Ch 7**: Prompt engineering and fine-tuning techniques covered earlier are revisited here explicitly through a cost lens.
- **Ch 13**: The pay-as-you-go vs. real-time-endpoint cost trade-off in Model-as-a-Service builds directly on the outside-periphery framework here.
