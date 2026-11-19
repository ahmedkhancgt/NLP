# Week 2, Day 1: Agent Architecture Traps, Evals, and Observability with Langfuse

## Lessons Covered
- Lesson 45: Welcome to Week 2 and a Recap of the Seven Parts of an Agent Harness
- Lesson 46: Three Traps to Avoid When Designing an AI Agent Architecture
- Lesson 47: When to Add Sub-Agents and When Business Logic Belongs in Code
- Lesson 48: Agent Evals, LLM as a Judge, the Confusion Matrix, Precision and Recall
- Lesson 49: Scoring the Claims Agent - The Cost of Mistakes and Cohens Kappa
- Lesson 50: Agent Observability with Langfuse - Traces, Experiments and Account Setup
- Lesson 51: Tracing a LangChain Agent in Langfuse on UnderCovers Harder Claims
- Lesson 52: Scoring the Baseline Agent with Langfuse Datasets and run_experiment
- Lesson 53: Business Logic in a Payout Tool - Comparing Experiments in Langfuse

## Core Idea
Agent architecture decisions (sub-agents, workflows, roles) should never be drawn on a whiteboard before you have a measurable business problem and a baseline; every architectural change — adding a sub-agent, adding a tool — is only justified when an experiment, scored with evals and viewed in an observability platform like Langfuse, shows it measurably improves outcomes.

## Frameworks Introduced
- **The Three Traps** (agent architecture anti-patterns):
  - When to use: as a checklist before starting any agentic AI project.
  - How: (1) **The Agent for X** — reject any "I need an agent for X" request until the underlying business problem is named and measurable; solve it without agents/LLMs first if possible. (2) **Human-like agents (anthropomorphizing)** — don't draw org charts of agent "roles and responsibilities" (Claims Manager, Fraud Check, Payout agents) before running experiments; start with one agent, measure, then add agents only if they improve the measured outcome. (3) **Plausible vs. accurate** — LLMs generate believable output, not necessarily output that moves the real business metric (e.g., "fewer meetings" insight sounds right but doesn't address attrition); always validate with evaluations tied to the actual objective.
- **Priority ordering for AI architecture**: Data science concerns (data quality, evals, observability) > System design concerns (integrations, context layer, API/MCP gateways, governance — covered Week 3) > Agent architecture concerns (roles, design patterns, communication like A2A). Start simple and only elevate agent-architecture concerns once the first two are grounded.
- **Five legitimate reasons to add a sub-agent**:
  - When to use: only when one measurably improves an outcome.
  - How: (1) Context isolation — focused prompt/tools for one task, keeps the main agent's context clean (e.g., summarizing a big document, returning only the summary). (2) Independent check / LLM-as-a-judge — a second agent evaluates the first's output without seeing its reasoning, ideally using a different model/provider for diversity. (3) Cost efficiency — cheap model for routine steps, strong model for hard reasoning. (4) Speed — fan out parallel work (e.g., multiple research threads). (5) Security — ring-fenced, tightly scoped permissions per task.
- **Business logic belongs in code, not prompts**: When to use: whenever a process is a fixed sequence (do A, then B, then C) or involves deterministic math (deductibles, limits, caps). How: implement as Python functions, expose to the agent as a tool (e.g., `calculate_payout`), and reserve the LLM for genuine natural-language judgment calls (is this covered? which benefit applies?).
- **Three Kinds of Evals**: Direct measurement (curated test dataset or real-world outcomes), LLM as a judge (second model scores against a rubric, can also verify citations/facts autonomously), Human review (expert graders). Best practice: use a *portfolio* of metrics — some close to the model, some closer to noisy business outcomes (e.g., revenue, alpha) — rather than one single metric.
- **The Confusion Matrix / Precision / Recall**: When to use: any yes/no (or multi-class) agent decision. How: Precision = TP/(TP+FP) — of predicted positives, how many were truly positive; Recall = TP/(TP+FN) — of actual positives, how many were caught. Business tolerance for false positives vs. false negatives must be set explicitly (UnderCover tolerates false positives — wrongly approving — more than false negatives — wrongly declining — because wrongful declines damage reputation/loyalty).
- **Cost-of-mistakes scoring (3x3 confusion matrix)**: When to use: multi-class business decisions (approve/decline/refer) where errors have different dollar costs. How: agree a cost matrix with the business (e.g., wrongful decline = $1,500 reputational cost; wrongful payment = amount paid; needless referral = $50 handler time; correct = $0) and sum it across all cases as the primary business metric.
- **Cohen's Kappa**: When to use: whenever raw accuracy could be inflated by a dominant class (e.g., approving everything scores 56% "accuracy" in UnderCover's data). How: rescale observed agreement against chance agreement so that a trivial "always approve" strategy scores 0, and perfect agreement scores 1.
- **Langfuse observability model**: When to use: any production or experimental agent you want to measure and debug. How: four building blocks — Traces (every model/tool call, tokens, cost, latency, inputs/outputs per run), Datasets (curated items with expected outputs), Experiments (a named run of a task over a dataset), Scores (recorded per-item and per-run evaluations). Use `langfuse.run_experiment(name=, data=, task=, evaluators=, run_evaluators=, max_concurrency=)`.

## Key Concepts
- **Agent harness (7 parts recap)**: Context, Workflow, State, Sub-agents, Permissions, Observability, Self-improvement.
- **False positive**: predicted positive (e.g., approved) but actually negative.
- **False negative**: predicted negative (e.g., declined) but actually positive.
- **Decision accuracy**: share of cases where the agent's categorical decision matches the human/ground truth, collapsing multi-class into a diagonal match.
- **Straight-through rate**: share of claims the agent resolves itself without referring to a human handler.
- **LLM as a judge**: a second model/agent that scores the first agent's output against a rubric; can be a different model family for independence, and can be asked to validate output against a curated "golden" reference even when wording differs.
- **Model metrics vs. business/outcome metrics**: model metrics (accuracy, cost of mistakes) are tightly coupled to agent changes; outcome metrics (revenue, attrition) are closer to the real goal but noisier and influenced by factors outside the agent's control.
- **Explainability**: surfacing an agent's reasoning/trace directly inside the business product (not just in an observability tool) for internal power users — functions like an audit trail tied to business outcomes.
- **run_experiment parameters**: `data` (dataset items with expected outputs), `task` (function called once per item that runs the agent), `evaluators` (score one item), `run_evaluators` (score the whole run).
- **Dataset item**: a claim/input bundled with its `expected_output` (ground truth) in Langfuse, referenced by a stable `id`.
- **Google-style docstring + `@tool(parse_docstring=True)`**: in LangChain, tool argument descriptions are only attached per-parameter if you explicitly pass `parse_docstring=True`; otherwise the whole docstring is used as one generic description.

## Mental Models
- Think of agent architecture decisions as hypotheses to test, not designs to draw: "start with experiments, not boxes and lines."
- Use "would a human org chart justify this split?" as a red flag test — if your sub-agent boundaries mirror human job titles rather than measured outcomes, you're in Trap 2.
- Treat an LLM's fluent explanation as evidence of plausibility, not accuracy — always ask "is this also *true* and *useful* for the metric I actually care about?"
- When deciding false-positive vs. false-negative tolerance, ask the business which error is more reputationally costly, not which is cheaper in dollars terms only.
- A metric that a trivial baseline (e.g., approve everything, refer everything) can win at is a bad primary metric — pressure-test every metric against the "do-nothing" or "do-the-simplest-thing" strategy (this is exactly what Cohen's kappa and straight-through-rate catch).

## Anti-patterns
- **"I need an Agent for X"**: a solution statement disguised as a requirement; always trace back to the measurable business problem first.
- **Drawing agent org charts (Claims Manager agent, Fraud Check agent, Payout agent) before any experiment**: encodes human role structures into software for no measured reason — architecture-by-anthropomorphism.
- **Treating plausible LLM output as validated**: e.g., accepting "fewer meetings would improve culture" without checking it's actually correlated with the real driver (attrition), which the input data couldn't even surface.
- **Putting fixed business logic or deterministic math (deductibles, caps, multi-step payout rules) inside a prompt**: slow, error-prone, and unauditable compared to a Python function exposed as a tool.
- **Single-number accuracy as the only metric**: hides that a trivial strategy (approve everything) already gets 56%; always pair with kappa, cost-of-mistakes, precision/recall.
- **Forgetting `parse_docstring=True`** on a LangChain `@tool`: silently degrades per-argument descriptions into one blob, hurting the model's tool-use accuracy.
- **Not agreeing the cost-of-mistakes matrix with the business before running evals**: without agreed dollar weights for each error type, your "improvement" may optimize the wrong thing.

## Code Examples
```python
DECISIONS = ["approve", "decline", "refer"]
WRONGFUL_DECLINE = 1500
NEEDLESS_REFERRAL = 50

def mistake_cost(answer, expected):
    """What the agent's answer costs UnderCover, against the handler's decision."""
    if answer["decision"] == expected["decision"]:
        return abs(answer["payout"] - expected["payout"]) if answer["decision"] == "approve" else 0
    if answer["decision"] == "decline":
        return WRONGFUL_DECLINE
    if answer["decision"] == "approve":
        return answer["payout"]
    return NEEDLESS_REFERRAL
```
- **What it demonstrates**: encoding a business-agreed cost matrix (wrongful decline costs $1,500, wrongful payment costs the amount paid, needless referral costs $50) directly as the scoring function used in every eval.

```python
def kappa(table):
    """Cohen's kappa for the agent's decisions against the handlers'."""
    observed = (table["agent"] == table["handler"]).mean()
    agent_share = table["agent"].value_counts(normalize=True)
    handler_share = table["handler"].value_counts(normalize=True)
    expected = sum(agent_share.get(d, 0) * handler_share.get(d, 0) for d in DECISIONS)
    return (observed - expected) / (1 - expected)
```
- **What it demonstrates**: rescaling raw agreement so a trivial "approve everything" strategy scores 0, not the misleadingly high 56% raw accuracy.

```python
@tool(parse_docstring=True)
def calculate_payout(policy_id: str, benefit: Benefit, lines: list[Line], paid_by_others: float = 0.0) -> Payout:
    """Work out the payout on an approved claim with UnderCover's payout rules.

    Args:
        policy_id: The customer's policy id
        benefit: The benefit the claim is paid under
        lines: The covered items and expenses, one for each line of the claim
        paid_by_others: Anything a carrier, hotel or other party has paid for the same loss

    Returns:
        The calculation, the payout and whether it needs sign-off
    """
```
- **What it demonstrates**: moving deterministic payout math out of the prompt and into a Python tool (`work_out_payout`), so the agent only judges coverage/line-items, not arithmetic.

```python
baseline = langfuse.run_experiment(
    name="Baseline",
    description="The agent works out each payout itself",
    data=items,
    task=baseline_task,
    evaluators=[score_claim],
    run_evaluators=[score_run],
    max_concurrency=1,
)
print(baseline.format())
```
- **What it demonstrates**: the canonical Langfuse experiment call — a dataset, a per-item task, an item evaluator, and a run-level evaluator, repeatable across model/tool variants for apples-to-apples comparison.

## Reference Tables
| Metric | Baseline (agent does payout math) | With payout tool |
|---|---|---|
| Decision accuracy | 84% | 96% |
| Accuracy (decision + payout) | 52% | 89% |
| Precision | 1.0 | 1.0 |
| Recall | lower | higher |
| Kappa | lower | higher |
| Cost of mistakes | $5,300 | $640 |
| Straight-through rate | 65% | 73% |
| LLM cost per claim | ~0.6¢ | ~1¢ |

| UnderCover's 3x3 cost matrix | Agent: Approve | Agent: Decline | Agent: Refer |
|---|---|---|---|
| Handler: Approve | $0 | $1,500 (wrongful decline) | $50 (needless referral) |
| Handler: Decline | Amount paid (wrongful payment) | $0 | $50 (needless referral) |
| Handler: Refer | Amount paid (wrongful payment) | $1,500 (wrongful decline) | $0 |

## Worked Example
The Week 2 Day 1 lab (`labs/week2day1.ipynb`) introduces 25 "harder" UnderCover claims — longer item lists, deductions already paid by other insurers, deductibles, and caps — deliberately designed to trip up the Week 1 agent. Four parts: (1) load the harder dataset and inspect a representative claim (a Rome burglary with multi-line losses and a partial refund from another insurer already applied); (2) run the baseline LangChain agent (`create_agent` with GPT-6 Luna via OpenRouter, reasoning effort "none") wrapped with a `CallbackHandler` from Langfuse, and inspect the resulting trace tree in Langfuse's tracing UI (LangGraph run → ChatOpenAI call → tool call to `record_decision` → final ChatOpenAI call); (3) score the baseline with `mistake_cost`, kappa, and a 3x3 confusion matrix — result: decision accuracy 84%, raw accuracy 52%, cost of mistakes $5,300, because the agent kept getting the dollar math wrong on deductibles and multi-party reimbursements; (4) add a `calculate_payout` tool backed by a pure-Python `work_out_payout` function encoding the actual business rules, rerun the identical experiment as "payout tool," and compare in Langfuse's experiment-comparison view. Result: decision accuracy jumped to 96%, raw accuracy to 89%, cost of mistakes collapsed from $5,300 to $640, straight-through rate rose from 65% to 73%, and the only metric that got (slightly) worse was LLM cost (0.6¢ → ~1¢ per claim) — a clear, data-backed case for moving business-logic math out of the prompt and into a tool.

## Key Takeaways
1. Never accept "I need an agent for X" as a requirement — trace it back to a measurable business problem, and try solving it without agents first.
2. Start with one agent; add sub-agents only when they measurably improve a tracked outcome, never to mirror a human org chart.
3. Agree a dollar cost matrix for each kind of mistake with the business before running evals — it is the real objective function, not raw accuracy.
4. Always pair accuracy with Cohen's kappa so a dominant-class baseline (approve/refer everything) can't masquerade as a good model.
5. Business logic that is deterministic math or a fixed sequence belongs in Python code exposed as a tool, not in the agent's prompt.
6. LLM-as-a-judge is strongest when it can't see the first agent's reasoning (judges the outcome only) and optionally uses a different model family for independence.
7. Langfuse's four building blocks — traces, datasets, experiments, scores — let you run and compare architecture changes as controlled experiments, not guesses.
8. Remember `@tool(parse_docstring=True)` in LangChain or your argument descriptions silently collapse into one blob.
9. Consider surfacing agent traces/explainability directly in the business product for power users — sometimes more effective than a separate observability dashboard.

## Connects To
- **Week 2, Day 2**: the same evals/observability discipline (Langfuse experiments) is reused to compare the UnderCover agent with and without long-term memory via agentic RAG over MCP.
- **Week 1**: builds directly on the LangChain `create_agent` and tool-calling patterns from Week 1's UnderCover baseline agent, now adding rigorous measurement.
- **LangSmith / Langfuse**: Langfuse is presented as one of several interchangeable observability platforms (LangSmith is the named alternative); the concepts (traces, datasets, experiments, scores) generalize across tools.
