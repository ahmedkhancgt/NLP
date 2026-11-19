# Week 2, Day 5: Self-Improvement, LLM-as-a-Judge with Jev, and Recursive Self-Improvement

## Lessons Covered
- Lesson 77: LLM as a Judge, Reference Answers and the Jev Decision Model
- Lesson 78: Designing the Improver Agent and Planning Sign-In with Supabase Auth
- Lesson 79: Testing Jev as a Judge and Building a Configurable Claim Agent
- Lesson 80: Measuring Noise and Building the Improver Agent with Human in the Loop
- Lesson 81: Running the Improver, Improving Its Recipe and Setting Up Supabase Auth
- Lesson 82: Claude Code Builds Self-Improvement and Supabase Sign-In into UnderCover
- Lesson 83: Run Three Improvement Rounds on the Dashboard and Complete Week 2

## Core Idea
A separate "Improver" agent — equipped with tools to read results, propose a configuration change, run an evaluation, and even rewrite its own improvement recipe — can automate the Day 1 manual "measure, change, re-measure" loop, but doing this safely absolutely requires judging every candidate change against *held-out* data the improver never sees, with a human approving each change before it's accepted, or the whole exercise degenerates into overfitting on 25 noisy data points.

## Frameworks Introduced
- **LLM-as-a-judge with a reference answer**: When to use: whenever you have a curated dataset with golden/expert answers and want to grade a new answer's *consistency* with that reference rather than grading it in isolation. How: give the judge a rubric with graded letter outcomes (e.g., A = better-expressed than the reference, B = consistent/equivalent, C = partially consistent, F = contradicts/completely different), feed it both the candidate answer and the reference answer, and optionally ask for auxiliary boolean checks (e.g., "same grounds?", "same policy section?") to make the grade explainable.
- **Jev as a decision-model judge**: When to use: as a fast, cheap alternative to an autoregressive LLM judge when you only need a constrained classification (grade A/B/C/F) rather than free text. How: Jev is a transformer used non-autoregressively — it outputs a decision directly from its inputs rather than generating tokens one at a time; in the instructor's testing it wasn't dramatically cheaper than a cheap LLM like Luna but was roughly 4x faster in latency. A free OpenRouter model with structured output works as a substitute if you don't want to use Jev.
- **The Improver agent's 5-step recipe**: When to use: building any agent-driven continuous-improvement loop for a production agent. How: (1) read_results — pull the current config, scores, mistakes, and weakest explanations; (2) propose_change — suggest a new prompt, model, and/or reasoning effort with a rationale; (3) human approval of the proposed change (required, since it has real consequences); (4) run_evaluation — run the new config on the eval set once and the held-out set twice (for noise averaging), then decide keep-or-rollback; (5) propose a new recipe — let the improver revise its own step-2 methodology, subject to a second human approval, feeding back into step 1 for the next round.
- **Held-out data discipline for agent self-improvement**: When to use: any automated or semi-automated loop that proposes and evaluates its own changes. How: keep a second dataset ("held-out") completely separate from the data the improver inspects when proposing changes; only use held-out-set results to decide whether to keep or roll back a change, specifically to prevent the improver from overfitting its prompt/config tweaks to quirks visible only in the evaluation set it can see.
- **Keep-or-rollback decision rule**: When to use: deciding whether a proposed agent change should ship. How: keep the change only if at least one tracked measure improves by more than an agreed noise threshold on held-out data, and no other measure degrades beyond an agreed tolerance; otherwise roll back. Constrained further by a fixed floor (e.g., minimum acceptable accuracy/grade on held-out data) and a hard budget cap for the whole improvement process.
- **`HumanInTheLoopMiddleware` (LangChain `create_agent` shortcut)**: When to use: wanting human-approval gating on specific tool calls without dropping down to hand-written LangGraph nodes/`interrupt()`. How: configure the middleware with the tool names that require approval (e.g., `propose_change`, `propose_recipe`), each allowing an approve/reject decision; pass it into `create_agent` alongside the model, tools, and system prompt — a "wizard"-level alternative to Day 3's manual `interrupt()`/`Command(resume=...)` for teams that want to stay at the `create_agent` level.
- **Supabase Auth setup for production sign-in**: When to use: any internal dashboard that needs real user accounts before going to production. How: create fixed handler users in Supabase's Authentication > Users screen (auto-confirmed, no real email needed), disable public sign-up (Authentication > Sign In / Providers), and surface only the publishable (non-secret) `VITE_SUPABASE_URL` / `VITE_SUPABASE_PUBLISHABLE_KEY` to the frontend plus a shared `HANDLER_PASSWORD` for testing — paired with a hard dollar spend cap set directly on the OpenRouter API key as a fail-safe independent of any in-app budget logic.

## Key Concepts
- **Decision model**: a transformer used in a non-autoregressive way to directly output a classification/decision (like Jev) rather than generating free-form tokens.
- **Rubric-graded LLM judge**: a judge configured with explicit letter/numeric grade definitions and example anchors, tuned iteratively (like calibrating exam grading criteria) until it grades consistently.
- **Noise threshold**: the minimum change in a metric below which a difference between two runs is considered random variation rather than a real effect — critical with small (25-item) datasets, since repeating the *same* config can shift accuracy by several points purely from LLM stochasticity.
- **Tolerance**: how much a secondary metric is allowed to get worse while still accepting an overall "keep" decision, as long as the primary metric improved beyond noise.
- **Floor / objective**: a hard minimum a metric (e.g., held-out accuracy, judge grade) must not fall below, regardless of other improvements, enforced on held-out data specifically to avoid overfitting.
- **Configurable claim agent**: an agent whose prompt, model, and reasoning effort are all parameters of a `config` object passed into `make_agent(config)`, making it possible for another agent (the Improver) to generate and test new configurations programmatically.
- **Recipe**: the Improver's own step-by-step instructions for how it goes about proposing changes — itself a mutable artifact the Improver can revise (the recursive self-improvement hook).
- **Recursive self-improvement (RSI)**: an agent improving the very process by which it improves other things; explicitly flagged as an unsolved frontier topic, where systems often "game the metrics" rather than become genuinely more capable — included here as "sizzle," not a production-ready technique.
- **Overfitting to the evaluation set**: the specific failure mode the held-out dataset guards against — an improver that only sees evaluation-set mistakes (e.g., "baggage calculation errors") proposes changes narrowly tuned to that set's quirks, which fail to generalize.
- **Langfuse Sessions**: a Langfuse view (separate from traces/datasets/experiments) that groups all traces belonging to one logical "session" (e.g., one improvement round), useful for reviewing a multi-step agentic process as a unit.
- **Autoresearch (Karpathy)**: referenced as an early, famous precedent for automated self-improvement loops in ML/data-science tasks, which inspired the Improver agent's design.

## Mental Models
- Treat the Improver's loop as "what we did manually on Day 1, now automated and recursive" — read results, propose, evaluate, decide, repeat.
- Use the train/test(/held-out) split instinct from classical ML as the mental anchor for why held-out data matters here too, even though "training" in this context means rewriting prompts/config, not adjusting weights.
- Picture the keep-or-rollback rule as "an improvement must clear the noise bar on data it hasn't seen, and nothing else can degrade past the agreed tolerance" — a two-sided test, not just "did the headline metric go up?"
- View recursive self-improvement skeptically but experimentally: expect occasional genuine wins (round 3's DeepSeek switch) and occasional non-wins swallowed by noise (round 2's reasoning-effort bump) — treat each round as its own held-out-validated experiment, not a guaranteed ratchet upward.
- Separate "AI engineering practices" (steps 1-4: read, propose, evaluate, decide) from "frontier/fun practices" (step 5: recipe self-improvement) — the former belongs in production; the latter is a live research area.

## Anti-patterns
- **Judging an improver's proposed change using only the data it was allowed to inspect**: guarantees overfitting to quirks of that dataset (e.g., baggage-claim specifics) at the expense of general performance.
- **Treating a single noisy 25-item eval run as proof a change helped or hurt**: the lab explicitly shows the *same* configuration swinging from 0.64 to 0.68 accuracy and from high to low cost-of-mistakes on repeated runs with no code change — always average multiple runs and compare against a noise threshold.
- **Letting an autonomous improver change its own process and the main agent's config without any human gate**: both proposing a change and proposing a new recipe are explicitly routed through human-in-the-loop approval precisely because they can have real consequences.
- **Assuming LLM-as-a-judge grounds you in business reality just because it "feels" rigorous**: it is still token prediction generating a plausible grade — mitigate with a tight rubric, concrete graded examples, reference-answer comparison, and/or a different model family as judge.
- **Skipping a hard external spend cap**: relying solely on in-app budget logic for an autonomous, self-modifying agent is risky; setting a dollar limit directly on the API key is a cheap, independent fail-safe.
- **Expecting recursive self-improvement to reliably compound**: current systems more often "game the metrics" than become exponentially better — don't design a production process that assumes RSI will keep delivering gains without human oversight.

## Code Examples
```python
def grade_with_jev(agent_answer: str, reference_answer: str) -> dict:
    """Rubric: F = different, C = partly consistent, B = consistent, A = better-expressed than reference."""
    return jev.decide(
        rubric={"A": "...", "B": "...", "C": "...", "F": "..."},
        inputs={"agent_answer": agent_answer, "reference_answer": reference_answer},
        extra_fields=["same_grounds", "same_section"],
    )
```
- **What it demonstrates**: the conceptual shape of grading one answer against a golden reference with a graded rubric plus explainability fields — in the lab, identical reasons scored B, a different claim's reason scored F, a vague answer scored C, and a clearer-than-reference answer scored A.

```python
def make_agent(config: dict):
    """Build a claim agent from a mutable configuration (prompt, model, reasoning effort)."""
    model = ChatOpenAI(model=config["model"], reasoning_effort=config["reasoning_effort"], ...)
    return create_agent(model=model, tools=[record_decision, calculate_payout], system_prompt=config["instructions"])

class Dispatch(TypedDict):
    config: dict
    claims: list[str]
    results: Annotated[list, operator.add]

def work_claim(state: Dispatch):
    agent = make_agent(state["config"])
    ...
```
- **What it demonstrates**: making the claim agent's model/prompt/reasoning-effort a first-class, swappable configuration carried in the dispatcher's LangGraph state, so the Improver can generate a brand-new `config` and the exact same fan-out/dispatch machinery from Day 4 evaluates it unmodified.

```python
approval = HumanInTheLoopMiddleware(interrupt_on={"propose_change": {"allow": ["approve", "reject"]},
                                                   "propose_recipe": {"allow": ["approve", "reject"]}})
improver = create_agent(model=improver_model, tools=[read_results, propose_change, run_evaluation, propose_recipe],
                         system_prompt=RECIPE, middleware=[approval], checkpointer=InMemorySaver())
```
- **What it demonstrates**: gating the Improver's two consequential tool calls (changing the claim agent's config, changing its own recipe) behind human approval using `create_agent`'s middleware shortcut, instead of hand-written LangGraph `interrupt()` nodes.

## Reference Tables
| Jev test case | Grade |
|---|---|
| Agent answer == handler's reference reason | B |
| Agent answer references a different claim entirely | F |
| Agent answer is vague | C |
| Agent answer is clearer/better than the reference | A |

| Improver tool | Purpose |
|---|---|
| `read_results` | Current config, scores, mistakes, weakest explanations |
| `propose_change` | New prompt / model / reasoning effort + rationale (human-approved) |
| `run_evaluation` | Run eval set once + held-out set twice, decide keep/rollback |
| `propose_recipe` | New version of the improver's own step-by-step process (human-approved) |

| Round (dashboard demo) | Change proposed | Held-out result | Decision |
|---|---|---|---|
| 1 | Luna: no-reasoning → low-reasoning | Accuracy 0.78 → 0.88, latency fell | Kept |
| 2 | Luna: low → medium reasoning | No improvement beyond noise | Rolled back |
| 3 | Switch model to DeepSeek V4.1 Flash, low reasoning | Accuracy up to ~96-100% | Kept |

## Worked Example
The Week 2 Day 5 lab (`labs/week2day5.ipynb`) — described as the longest lab of the course — runs in four parts. Part 1 builds and tests a Jev-based judge using a rubric (A/B/C/F) against reference answers from a 25-claim evaluation set plus a fresh 25-claim held-out set, confirming sensible grading behavior across four hand-crafted test cases (identical, wrong-claim, vague, and better-than-reference answers). Part 2 builds "version 1" of a fully *configurable* claim agent (prompt, model, reasoning effort all parameterized) wired into the same LangGraph dispatcher pattern from Day 4, and an `evaluate(config, set_name)` function that runs it against either the evaluation or held-out set and records scores. Running the identical version-1 config twice on the held-out set shows accuracy swinging from 0.64 to 0.68 purely from noise — motivating the need for averaging and a noise threshold. Part 3 builds the Improver agent with its four tools (`read_results`, `propose_change`, `run_evaluation`, `propose_recipe`), objectives (accuracy floor on held-out data, judge-grade floor), a noise threshold, a tolerance, a $0.50 budget cap, and a `HumanInTheLoopMiddleware` gating both `propose_change` and `propose_recipe`. The first improvement round proposes new baggage-handling instructions based on evaluation-set mistakes; after human approval it's evaluated and *rejected* (not better than noise on held-out data) — explicitly because the change overfit to evaluation-set-specific quirks. The Improver then proposes (and is approved for) a revised recipe emphasizing per-claim root-cause diagnosis; running a second round with the new recipe *succeeds*, with held-out accuracy improving beyond the noise threshold, total spend across all rounds landing at 8 cents. The lab ends with Supabase Auth setup (three fixed handler accounts — Dana, Marcus, Priya — public sign-up disabled) and a hard OpenRouter spend cap as an independent fail-safe. The day's product build (Claude Code, ~2.5 hours, from an `IMPROVE.md` spec) adds sign-in, a Settings screen for objectives/noise/tolerance/budget, and an Improvement screen with visual round-by-round comparison charts. Three live rounds are run on the dashboard: round 1 (Luna no-reasoning → low-reasoning) is approved and *kept* (held-out accuracy 0.78 → 0.88); round 2 (low → medium reasoning) is approved but *not* kept (no improvement beyond noise); round 3 (switch to DeepSeek V4.1 Flash at low reasoning) is approved and *kept spectacularly*, pushing held-out accuracy to roughly 96-100% — with the Improver also successfully proposing and getting approval for updates to its own recipe along the way, a concrete instance of recursive self-improvement working in this bounded setting.

## Key Takeaways
1. LLM-as-a-judge is strongest when grading against a curated reference answer with an explicit letter-grade rubric, not when grading an output in isolation.
2. Jev (or any decision model) trades autoregressive flexibility for speed on simple classification-style judging tasks — useful but not dramatically cheaper than a cheap LLM like Luna in this instructor's testing.
3. Always hold out a second dataset the improver never inspects, and base every keep/rollback decision only on held-out results — this is the single most important safeguard against overfitting the improvement loop.
4. With small (25-item) datasets, expect meaningful run-to-run noise; agree an explicit noise threshold and run multiple held-out evaluations before trusting a result.
5. Gate every autonomous change (to the agent's config, and to the improver's own recipe) behind human approval — use `HumanInTheLoopMiddleware` for a lighter-weight alternative to hand-written LangGraph `interrupt()` nodes.
6. Making an agent's prompt/model/reasoning-effort a first-class `config` object is what makes automated self-improvement mechanically possible — the Improver can only propose what the claim agent's architecture exposes as changeable.
7. Set an independent, hard dollar cap directly on the API key as a fail-safe beyond any in-app budget logic, especially for autonomous, self-modifying agents.
8. Recursive self-improvement is real but unreliable today — expect some rounds to be rejected by the noise/tolerance gate, and treat genuine wins (like the round-3 model switch) as validated experiments, not guaranteed compounding progress.
9. Production authentication (Supabase Auth, fixed handler accounts, disabled public sign-up) is cheap to set up now and necessary before any real deployment — don't leave it for "later."

## Connects To
- **Week 2, Day 1**: the Improver automates exactly the manual evals/experiment loop established on Day 1 (confusion matrix, cost of mistakes, kappa), now agent-driven and iterative.
- **Week 2, Day 3**: `HumanInTheLoopMiddleware` is presented as a `create_agent`-level shortcut alongside Day 3's lower-level `interrupt()`/`Command(resume=...)` LangGraph pattern — two ways to reach the same human-approval guarantee.
- **Week 2, Day 4**: the Improver's `evaluate()` function reuses the exact same LangGraph dispatcher/fan-out machinery from Day 4, now parameterized by a mutable `config` so different claim-agent variants can be swarmed and scored identically.
- **Week 3**: this chapter closes Week 2 ("Become an AI Architect"); the course explicitly transitions to Week 3 ("Become an AI FDE") covering production deployment, security/guardrails, API gateways, and the enterprise context layer deferred from Day 1's discussion.
