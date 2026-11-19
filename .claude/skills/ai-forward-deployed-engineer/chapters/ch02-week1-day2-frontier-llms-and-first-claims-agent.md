# Week 1, Day 2: Frontier LLMs, Model Selection and the First Claims Agent

## Lessons Covered
- Lesson 9: Recap of Day 1 and Todays Plan for Frontier LLMs and Agents
- Lesson 10: From GPT to ChatGPT - 6 Tricks, Tokens, Context Windows and API Costs
- Lesson 11: Comparing Frontier LLMs - Artificial Analysis, the Pareto Front and METR
- Lesson 12: What Is an AI Agent Tool Calling, the Harness and Agent Frameworks
- Lesson 13: Introducing UnderCover - The Course Project and the Claims Agent Pilot
- Lesson 14: Lab Prep - Chat Completions, Structured Outputs and LangChain create_agent
- Lesson 15: Exploring UnderCovers Claims Data and the Handlers Decisions
- Lesson 16: Testing GPT-6 Luna, Gemini 3.5 Flash Lite, GPT-6 Sol and Jev on Claims
- Lesson 17: Build a Claims Agent with LangChain create_agent and Tools

## Core Idea
An LLM is "predictive text on steroids" made to feel human-like by six specific, learnable tricks (autoregression, chat, reasoning, illusion of memory, context, tool calling) — and the question "which model/approach is best?" is never answered by a benchmark alone, but by running your own accuracy/cost/speed experiment against your actual business task, because an agent is not automatically better than a single structured-output LLM call.

## Frameworks Introduced
- **The Three-Layer AI Product Stack**: Model (GPT, Claude, Gemini — statistical pattern matchers) → Inference platform/software (OpenAI Platform, AWS Bedrock, OpenRouter — runs the model repeatedly) → Product (ChatGPT, Claude Code, Duolingo Premium — the end-user experience).
  - When to use: to diagnose where a problem actually lives when something behaves unexpectedly — is it the model, the inference layer, or the product wrapper?
- **Six Tricks from GPT to ChatGPT**: autoregression (generate one token at a time in a loop), chat (RLHF-trained turn-taking format), reasoning (RLVR-trained "think step by step" before answering), illusion of memory (full history resent each call — the model is stateless), context (inject retrieved/background data, e.g. via RAG), tool calling (let the model request external actions).
  - When to use: as a checklist to explain to a stakeholder why an LLM "feels" intelligent/human, or to debug why a model seems to "forget" something (check what's actually in the resent history).
- **Artificial Analysis Intelligence Index / Pareto Front**: a blended benchmark score plotted against cost; the Pareto front is the set of models offering the best intelligence at each price point.
  - When to use: as a first-pass shortlist tool for model selection — never as the final decision criterion. Always follow with your own task-specific test.
  - How: plot cost (log x-axis) vs. intelligence (y-axis); favor models on or near the dotted Pareto-front line for your budget; expect labs to compete for Pareto-front placement (e.g. OpenAI's Luna, MiMo 2.6, Claude Opus 5.5 all occupy different points on it).
- **METR Time-Horizon Benchmark**: measures AI capability not by quiz score but by "how many human-hours of software-engineering work can the model+harness complete at a 50% success rate," tracked over time.
  - When to use: to gauge harness + model capability growth specifically for verifiable, measurable tasks (chiefly coding) — not for subjective tasks like "is this strategy deck good."
  - How: because the task is verifiable, a 50%-reliable model can be made highly reliable by running multiple parallel attempts (ensemble) and picking a success.
- **Structured Outputs (Pydantic + `.parse()`)**: force an LLM's reply into a JSON schema defined as a Pydantic `BaseModel`, with field order chosen deliberately (rationale before decision) so the "thinking" happens before the commitment.
  - When to use: whenever you need a reliably parseable decision/classification from an LLM, especially for downstream business logic.
  - How: define `class X(BaseModel): rationale: str; decision: Literal[...]; payout: float`, call `client.chat.completions.parse(model=..., messages=..., response_format=X)`, read `response.choices[0].message.parsed`.
- **LangChain `create_agent` + `@tool`**: a framework-level abstraction over the hand-rolled agent loop from Day 1.
  - When to use: when you want tool-calling/agent-loop boilerplate handled for you, and you're fine depending on a specific framework rather than raw API calls.
  - How: (1) wrap the model in `ChatOpenAI(model=..., base_url=..., api_key=...)`, (2) decorate plain Python functions with `@tool`, (3) call `create_agent(model=llm, tools=[...], system_prompt=...)`, (4) run with `agent.invoke({"messages": [...]})`. Internally this builds a LangGraph dependency graph.
- **Fixed Workflow vs. Agent Decision Rule**: if the sequence of tool calls is always the same (e.g., always fetch policy, then always record decision), wrapping it in an agent adds autonomy you don't need — and costs more time and money for no accuracy gain.
  - When to use: before reaching for an agent framework, ask whether the task truly needs the LLM to *choose* which tool to use and when, or whether a fixed sequence (simple structured-output call, or deterministic code) would do.

## Key Concepts
- **Token**: the unit of text sent to/from an LLM, typically a few letters; ~1,000 tokens ≈ 750 English words.
- **Context window**: the hard cap on input + output tokens a model can handle in one call; exceeding it fails the call.
- **Compacting**: a product-level feature (e.g. in Claude Code) that rewrites conversation history more concisely when the context window fills up.
- **Prompt caching**: providers charge less (often ~4x less) for input tokens that repeat an earlier call's prefix; typically expires after about an hour of inactivity. GPT-6 Luna caches automatically; Gemini required explicit caching in the instructor's test.
- **RLHF (Reinforcement Learning from Human Feedback)**: post-training technique behind the "chat" trick that enabled ChatGPT's question-answering behavior (vs. raw GPT-3's text-continuation behavior).
- **RLVR (Reinforcement Learning with Verifiable Rewards)**: post-training technique behind the "reasoning" trick, starting with o1 (late 2024); rewards the model for chains of thought that lead to verifiably correct answers.
- **Harness**: the software scaffolding (loop, tool wiring, retries) that drives multiple LLM calls and turns a token predictor into an agent — distinguished from the model itself.
- **Agent (final course definition)**: "an LLM with tools in a loop to achieve a goal" — equivalently, "an LLM plus a harness."
- **Confusion matrix / false positive vs. false negative**: the asymmetry in how costly different mistake types are to a business; e.g., wrongly declining a valid claim is typically more reputationally costly than wrongly approving an invalid one — the COVID-testing false-positive/false-negative tolerance tradeoff is used as the analogy.
- **UnderCover**: the course's fictional travel-insurance company and three-week capstone project; today's pilot scope is an AI agent that works a claims queue, auto-deciding payouts under $1,000 and escalating larger ones to a human.
- **Difficulty tiers in the claims dataset**: "routine" (policy wording alone suffices), "hard" (still solvable from wording, but trickier), "manual" (requires an internal claims manual not yet given to the model — expected to fail until later weeks).
- **Jev (TypeSafe AI)**: a non-autoregressive "decision" model that returns a probability/confidence per answer option instead of generating free text — framed as unable to "confidently hallucinate" because it can't fabricate a convincing wrong narrative, only a wrong but calibrated choice.
- **Chat Completions API**: the original OpenAI-defined request/response shape (`messages` list of `{role, content}` dicts) now used across virtually all LLM providers, including via OpenRouter.

## Mental Models
- Treat benchmark charts (Artificial Analysis, METR) as a shortlist generator, never a purchase order — the only benchmark that matters for a production decision is your own task, your own data, your own cost/accuracy/latency tradeoff.
- When evaluating "is this model good enough," separate the question into three independent axes — accuracy, cost, speed — and negotiate the tradeoff with business stakeholders using their actual budget and latency tolerance, not your own intuition.
- Use "premature optimization is the root of all evil" (Knuth) as your model-selection order of operations: solve for accuracy first, then optimize cost/speed — a cheaper model that fails the task is not actually cheaper.
- See reasoning effort (low/medium/high) as a dial you must test empirically per task: in the Day 2 experiment, medium and high reasoning on GPT-6 Luna cost significantly more and took much longer but barely moved accuracy — don't assume "more reasoning = more worth it."
- The "fixed workflow vs. autonomy" test: before building an agent, ask "would a human need judgment here, or just execute the same two steps every time?" If it's the latter, an agent is over-engineering.

## Anti-patterns
- **Assuming open-source or "free" models cost nothing**: even free OpenRouter models cost you your data (used for training), and locally-run open models cost electricity and amortized hardware — "someone is paying for it" if trillions of calculations are happening.
- **Obsessing over token/input costs before accuracy**: tuning prompts for efficiency before you've nailed down accuracy is backwards — a correct answer from a cheaper model beats an efficient prompt to a model that's wrong.
- **Treating benchmark leaderboard position as a procurement decision**: Artificial Analysis and similar sites are described explicitly as needing "a heavy pinch of salt" — useful for narrowing candidates, not for final selection.
- **Forcing agent autonomy onto a deterministic process**: the Day 2 lab explicitly shows the LangChain agent costing more, running slower, and performing no better (sometimes slightly worse) than a single structured-output call, precisely because the "choice" of tool order was never actually in question.
- **Putting decision before rationale in a structured-output schema**: if `decision` is defined before `rationale` in the Pydantic model, the model's displayed "reasoning" becomes a post-hoc justification of an already-made decision rather than genuine deliberation.

## Code Examples
```python
class ClaimDecision(BaseModel):
    """A claims handler's decision on one claim."""
    rationale: str = Field(description="2-3 sentences to explain the decision along with the relevant section of the policy")
    decision: Literal["approve", "decline", "refer"]
    payout: float = Field(description="The amount to pay in US dollars, or 0 for a decline or a referral")

def decide(claim, model, effort):
    """Ask one model to decide one claim."""
    start = time.time()
    response = openrouter.chat.completions.parse(
        model=model,
        messages=make_messages(claim),
        response_format=ClaimDecision,
        reasoning_effort=effort,
        extra_body={"provider": {"require_parameters": True}},
    )
    answer = response.choices[0].message.parsed
    return {
        "model": model, "effort": effort, "technique": "LLM",
        "claim_id": claim["claim_id"], "decision": answer.decision,
        "payout": answer.payout, "rationale": answer.rationale,
        "cost": response.usage.cost, "seconds": round(time.time() - start, 1),
    }
```
- **What it demonstrates**: structured outputs with reasoning-effort control and rationale-before-decision field ordering; every call's cost/time is captured for later accuracy/cost/speed comparison.

```python
@tool
def get_policy(policy_id: str) -> str:
    """Look up a customer's policy record: the plan, the add-ons, the purchase date, the trip and the travelers."""
    policy = policies.get(policy_id)
    return json.dumps(policy, indent=2) if policy else f"There is no policy {policy_id}"

@tool
def record_decision(claim_id: str, decision: Literal["approve", "decline", "refer"], payout: float, rationale: str) -> str:
    """Record the decision on a claim in the claims system."""
    recorded[claim_id] = {"decision": decision, "payout": payout, "rationale": rationale}
    return f"Recorded {decision} on {claim_id}"

agent = create_agent(model=luna, tools=[get_policy, record_decision], system_prompt=AGENT_SYSTEM_PROMPT)
result = agent.invoke({"messages": [{"role": "user", "content": make_claim_prompt(claims[0])}]})
```
- **What it demonstrates**: LangChain's `@tool` decorator eliminates hand-written JSON tool schemas; `create_agent` + `invoke` replaces the Day 1 hand-rolled loop.

```python
def decide_with_jev(claim):
    """Ask Jev for the decision on one claim, with its confidence."""
    state = {"policy_wording": policy_wording, "policy": policies[claim["policy_id"]], "claim": claim}
    response = openrouter_decisions.alpha.decisions.create(model=JEV, state=state, questions={"decision": JEV_QUESTION})
    answer = response.answers["decision"]
    return {"claim_id": claim["claim_id"], "decision": answer.choice, "confidence": answer.confidence}
```
- **What it demonstrates**: a non-autoregressive decision model called through OpenRouter's separate Decisions endpoint (needs OpenRouter's own client, not the OpenAI client), returning a choice plus calibrated confidence instead of free text.

## Reference Tables
| Model / Technique | Decisions right (of 25) | Cost | Time |
|---|---|---|---|
| GPT-6 Luna, low reasoning | 18 | $0.004 | 38s |
| GPT-6 Luna, medium reasoning | 18 | $0.63 | 90s |
| GPT-6 Luna, high reasoning | 19 | higher still | 98s |
| Gemini 3.5 Flash Lite, low | 14 | $0.03 (~10x Luna) | ~30s (fastest LLM call) |
| GPT-6 Sol, high reasoning | 20 | $0.12 | ~2 min |
| Jev (decision only) | 17 of 25 decisions | $0.003, very low | 5s (fastest overall) |
| LangChain agent (Luna, low, via tools) | 17 | more than direct call | 1m55s (slowest) |

| Jev confidence | Accuracy |
|---|---|
| ≥ 90% confident | 5 of 6 correct |
| < 90% confident | 12 of 19 correct |

## Worked Example
The Day 2 lab (`labs/week1day2.ipynb`, with a free-model variant `week1day2free.ipynb`) runs UnderCover's 25-claim queue through a progression of techniques to build intuition about model selection tradeoffs. Step 1: load and visualize the claims data (type, amount, difficulty, handler's ground-truth decision) via pandas/plotly, including two "Easter egg" claims modeled on real-world stories (a child left on a plane; a bachelor-party dental mishap in Vegas) deliberately seeded to test policy-reading rigor. Step 2: define a `ClaimDecision` Pydantic schema and call GPT-6 Luna at low/medium/high reasoning via `chat.completions.parse`, observing that low reasoning got 18/25 right for under half a cent in 38 seconds, while medium and high reasoning cost 100x+ more and took much longer for the same or marginally better accuracy. Step 3 (optional): Gemini 3.5 Flash Lite scored worse (14/25) yet cost ~10x more than Luna-low (likely due to lacking Luna's automatic prompt caching); GPT-6 Sol at high reasoning scored best of the LLM calls (20/25) but at noticeably higher cost/time; Jev, a non-generative decision model, matched speed records (5 seconds) and decent accuracy (17/25 on decision alone), with its confidence score proving a reliable proxy for correctness. Step 4: the same 25 claims were run through a LangChain `create_agent` equipped with `get_policy` and `record_decision` tools, using the same Luna-low model — producing essentially the same accuracy (17 vs. 18) but at higher cost and nearly 2 minutes of runtime, because the task's tool-call sequence was always fixed (fetch policy, then record), giving the agent's autonomy no actual decision to make. The lab's assignment challenge: find the cheapest model/config that still gets at least 15/25 right using the agent approach, and contribute results back via PR.

## Key Takeaways
1. Explain any "smart-seeming" LLM behavior through the six tricks (autoregression, chat, reasoning, illusion of memory, context, tool calling) — it demystifies apparent intelligence into specific, named engineering techniques.
2. Benchmarks like Artificial Analysis and METR are shortlisting tools, not procurement decisions — always validate with your own task-specific experiment.
3. The Pareto front identifies the best intelligence-per-dollar tradeoff available at any given budget — use it to narrow candidates before testing.
4. Reasoning effort is not free lunch: test empirically whether higher reasoning actually improves accuracy for your task before paying its cost/latency premium.
5. Structured outputs with rationale-before-decision field ordering gets you both a parseable answer and a genuine (not post-hoc) chain of thought.
6. An agent is not automatically superior to a single LLM call — only use agent autonomy where the tool-call sequence genuinely depends on the model's judgment.
7. Mistake costs are asymmetric (false positive vs. false negative) and that asymmetry, not raw accuracy alone, should drive model/threshold selection — discuss it explicitly with business stakeholders.
8. Fast, cheap, non-generative "decision" models (like Jev) paired with a confidence threshold can be a strong hybrid strategy: auto-resolve confident cases, escalate uncertain ones to a stronger (more expensive) model.
9. Prompt caching behavior varies by provider (automatic with Luna, opt-in with Gemini in this test) and can be a major cost driver independent of base-model price.

## Connects To
- **Week 1, Day 1**: formalizes and benchmarks the hand-rolled agent-loop concept introduced there; the UnderCover claims task replaces the maze as the running example for the rest of the course.
- **Week 1, Day 3**: the claims-decision experimentation done here in notebooks becomes the specification a coding agent (Claude Code) is asked to build into a real backend/dashboard/agent product.
- **LangGraph**: `create_agent` is revealed to build a LangGraph dependency graph under the hood, previewing Week 2's deeper LangGraph coverage.
- **OpenRouter's Decisions endpoint / Jev**: a non-chat-completions API surface, foreshadowing later discussions of specialized inference platforms beyond standard chat completions.
