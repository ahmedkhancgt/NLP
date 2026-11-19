# Chapter 4: Large Language Models as Cognitive Cores

## Core Idea
LLMs come in three tiers — **base** (fast, shallow, unaligned pattern-matching) → **post-trained** (safe, instruction-following, the agentic workhorse) → **reasoning** (slow, deep, multi-pass, state-of-the-art for complex logic) — and choosing among them is a four-way trade-off between reasoning quality, speed, safety, and cost, not a single "best" pick.

## Frameworks Introduced

### The Three-Tier LLM Taxonomy
1. **Base models**: pure next-token pattern matching, single-pass, completion-focused, unaligned/unfiltered (GPT-3 Ada/Babbage/Curie/Davinci, base Llama, base Claude). "Fast thinking, shallow reasoning."
2. **Post-trained models**: base + SFT/instruction tuning + RLHF + safety training → direct helpful answers, conversational, guardrailed (ChatGPT/GPT-4, Claude Sonnet, Gemini Pro). Still fast/shallow, but aligned. **Current workhorse of agentic AI.**
3. **Reasoning models**: post-trained + chain-of-thought training + deeper RLHF/RLVR + Constitutional AI/reflection + multi-pass processing (OpenAI o1/o3-mini, DeepSeek R1). "Slow thinking, deep reasoning" — best at math/code/logic/multi-step planning, at the cost of latency and compute.
- When to use which: **reasoning > post-trained > base** for complex reasoning and safety-critical tasks; **base > post-trained > reasoning** for speed and cost-effectiveness. In a multi-agent system, give the **orchestrator** a reasoning model (it must understand intent and build a plan) and let specialist sub-agents use cheaper post-trained models.
- Why it works: these tiers are not separate architectures but cumulative training layers on the same base — explaining why reasoning models inherit post-trained safety *and* add deliberation.
- Emerging direction: rather than statically picking a tier per agent, the industry is moving toward **dynamic fast/slow switching within one system** (directly analogous to System 1/System 2 human cognition) — the designer may stop choosing a tier explicitly at all.

### Four-Dimension Model Selection
Pick an LLM along four axes simultaneously: **reasoning quality, speed, safety, cost**. No model wins on all four — reasoning models top quality/safety but lose badly on speed/cost; base models invert that.

## Key Concepts
- **Temperature**: controls output randomness; 0 = deterministic (good for code-generation agents), high = varied/creative (good for brainstorming/ideation agents).
- **System prompt vs. user prompt**: the system prompt sets persona/tone/context and can dramatically reshape the answer to the *same* user prompt (the chapter's running example: a "Brazilian fan" persona answers "Pelé," an "Argentinian fan" persona answers "Messi," to an identical question) — a lightweight lever for agentic multi-perspective analysis.
- **Structured output** (via Pydantic + `with_structured_output`): forces the LLM to emit a typed object (e.g., name/age/occupation) instead of free text, eliminating most post-processing — a major reason LLMs work well as cognitive cores in a larger software pipeline.
- **`invoke` vs. `generate`**: `invoke` returns one response; `generate` with `n=N` returns N sampled responses from one prompt — useful when one agent needs to offer several candidate options to another agent (meaningful only at temperature > 0).
- **Evolutionary lineage**: n-gram/Maximum Entropy statistical LMs → Word2Vec/RNN/LSTM/GRU neural LMs (still fundamentally Markovian, limited long-range context) → Transformer + attention (global, parallel, solved long-range dependency) → modern multimodal LLMs.

## Mental Models
- Treat the three-tier taxonomy as a spectrum of "how much deliberation does this sub-task deserve," not a one-time global architecture choice — assign per-agent-role, not per-application.
- Use the system prompt as a cheap "perspective-injection" knob before reaching for more expensive techniques (multiple agents, fine-tuning) when you need balanced/multi-angle answers.
- When debugging "the agent keeps repeating the same mistake," check first whether you've assigned a reasoning model where the task actually needs persistent learning (LLMs don't update weights from experience — no amount of model-tier upgrade fixes that; you need RLHF/RLVR or an external memory/learning loop).

## Anti-patterns
- **Assuming a bigger/smarter model fixes a memory problem**: LLMs are stateless per-inference regardless of tier; forgetting across sessions requires external memory systems (Ch 7), not a better cognitive core.
- **Using a reasoning model everywhere "to be safe"**: wastes latency and cost on sub-agents that don't need deep multi-step logic — reserve reasoning models for orchestration/planning and genuinely hard reasoning steps.
- **Trusting LLM confidence as a calibration signal**: current LLMs have poor metacognition — "breathless AGI claims" aside, they can be confidently wrong and fail to recognize their own knowledge gaps, so don't gate escalation-to-human purely on the model's stated confidence.
- **Expecting consistent multi-modal reasoning**: spatial/cross-modal reasoning is still inconsistent across frontier LLMs — don't assume vision+text performance matches text-only performance, especially for robotics/spatial tasks.

## Code Examples

```python
from langchain_openai import AzureChatOpenAI
import textwrap
# Step 1: Create an LLM client with parameters suitable for your task
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-08-01-preview",
    azure_endpoint=ENDPOINT_URL,
    api_key=OPENAI_API_KEY,
    temperature=0.0,  # 0 for deterministic answers, higher for more creative outputs
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# Step 2: Set up the messages
system_prompt = SystemMessage(content="You are a well informed sports analyst of the soccer world.")
user_prompt = HumanMessage(content="Give me a short answer: Who is the best soccer player of all time?")
# Step 3: Send the messages to the LLM and get a response
response = llm.invoke([system_prompt, user_prompt])
# Step 4: Display the response
print(f"Response:\n{textwrap.fill(response.content, width=80)}")
```
- **What it demonstrates**: the four-step LLM invocation pattern (client → prompts → invoke → use response) that recurs throughout agentic code.

```python
# Structured output via Pydantic
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke([system_prompt, user_prompt])
# result is a typed Person object, not raw text — no parsing needed
```
- **What it demonstrates**: eliminating post-processing by forcing typed output — critical when an LLM's response feeds directly into downstream code.

```python
# Multiple sampled responses for agent-to-agent option generation
multiple_responses = llm.generate([[system_prompt, user_prompt]], n=3)
# At temperature=0.0, all 3 responses are identical (deterministic)
# At temperature=0.7, responses vary meaningfully — useful as candidate options
```
- **What it demonstrates**: `generate(..., n=N)` as the mechanism for producing a diverse candidate pool, e.g. for one agent to hand several options to another.

## Reference Tables

**LLM selection thumb rules**

| Priority | Best → Worst tier ordering |
|---|---|
| Complex reasoning | Reasoning > Post-trained > Base |
| Speed | Base > Post-trained > Reasoning |
| Safety | Reasoning ≈ Post-trained > Base |
| Cost-effectiveness | Base > Post-trained > Reasoning |

**LLM limitation categories and mitigations**

| Limitation | Root cause | Mitigation (and its limit) |
|---|---|---|
| Memory/state | Stateless inference, finite context window | External vector DB / RAG memory (Ch 7) — disconnected from core reasoning |
| Grounding/world model | Episodic training, data cutoffs | Multi-agent consistency checking; live tool/web search injection (Ch 6) |
| Planning (>10 steps) | Poor long-horizon planning, weak contingency reasoning | Explicit planning tool producing a plan the LLM executes stepwise |
| Metacognition | Poor confidence calibration | Human-in-the-loop escalation — don't rely on self-reported confidence |
| Real-time decisions | High latency, no streaming/interrupt handling | Must be separately engineered; not solved by the LLM itself |
| Safety in open-ended contexts | Training distribution ≠ deployment distribution | Guardrails + adversarial testing; generalization still limited |
| Learning/adaptation | No weight updates from experience | Fine-tuning + RLHF/RLVR loop — trades agility vs. engineering complexity |
| Multi-modal integration | Inconsistent cross-modal performance | None complete — still an open research gap, esp. spatial/robotics |
| Economic/scaling | Model size → compute/energy cost | Hybrid architectures, smaller specialist modules |
| Evaluation/debugging | Stochastic, low introspection, complex interaction effects | Better agentic evaluation frameworks (Ch 10) |

## Worked Example

**Reasoning model vs. post-trained model on a logic puzzle** (the chapter's own comparison, June 2025, Alice/Bob/Clara pet-and-house constraint puzzle):
- **GPT-4o (post-trained)** found only **2 of the 3** valid configurations satisfying all constraints.
- **o3-mini (reasoning)** correctly found **all 3** valid configurations.
- Lesson: for constraint-satisfaction / exhaustive-case logic tasks, reasoning models' multi-pass, self-correcting search finds solutions post-trained models miss in a single pass — concretely validating the "reasoning > post-trained for complex reasoning" thumb rule, not just asserting it.

## Key Takeaways
1. Pick LLM tier per agent-role (orchestrator vs. specialist), not once for the whole system — orchestrators benefit most from reasoning models.
2. The four selection axes (quality/speed/safety/cost) always trade off — there's no universally "best" model, only best-for-this-role.
3. Use system prompts as a cheap lever for perspective/persona control before reaching for multi-agent or fine-tuning solutions.
4. Structured output (Pydantic + `with_structured_output`) should be the default when an LLM response feeds downstream code — it removes a whole class of parsing bugs.
5. None of today's LLM limitations (memory, grounding, planning, metacognition, real-time, safety, learning, multi-modal, cost, debugging) are solved by picking a "smarter" model — each needs its own architectural mitigation (later chapters: memory systems, tools, multi-agent, human-in-the-loop).
6. A worked side-by-side test (not just a stated heuristic) is the right way to validate model-tier choices for your own task — the puzzle example shows exactly how to run that comparison.

## Connects To
- **Ch 2 (Theoretical Frameworks)**: the connectionist-architecture discussion there is the theory behind what's implemented here.
- **Ch 5 (Agent Patterns)**: next chapter builds single/multi-agent architectures on top of the cognitive cores chosen here.
- **Ch 6 (Tools and Interfaces)**: tool/web-search grounding is the named mitigation for the "grounding and world model accuracy" limitation.
- **Ch 7 (Memory Systems)**: external memory is the named mitigation for the "memory and state management" limitation.
- **Ch 10 (Testing and QA)**: "better evaluation frameworks" is named here as a mitigation and becomes its own chapter.
