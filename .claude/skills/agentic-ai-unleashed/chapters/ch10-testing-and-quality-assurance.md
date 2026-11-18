# Chapter 10: Testing and Quality Assurance

## Core Idea
Agentic systems are probabilistic, not deterministic, so output-only testing is structurally insufficient — quality assurance requires **trace/span-level instrumentation as a measurement instrument** (not a mockable side effect), a layered combination of **probabilistic evaluation (LLM-as-judge) + deterministic constraints (schema/safety/routing asserts)**, and **continuous evaluation against gold-standard datasets** that never stops after deployment because models, prompts, tools, and input distributions all keep drifting.

## Frameworks Introduced

### Tracing as Testing Infrastructure (the foundational layer)
Borrowed from microservices distributed tracing: every LLM call, tool invocation, and memory read/write is recorded as a **span** within a hierarchical **trace**, capturing inputs, outputs, latency, and parent/sibling linkage. Traces enable three concrete testing capabilities:
1. **Deterministic replay** — reconstruct exact inputs to any pipeline component and re-run it in isolation without invoking the full agent.
2. **Regression testing at the span level** — compare traces before/after a change to detect behavioral drift before it surfaces as an output-level failure.
3. **Trace-grounded evaluation** — score whether the agent used the right tool/context/intermediate decision, not just whether the final answer happened to be correct (catching "right answer, wrong reasoning").
- When to use: assume this infrastructure exists *before* designing any other testing strategy in this chapter — every technique below depends on spans being captured, even in test environments. Tools: LangSmith, Arize Phoenix, W&B Weave, Traceloop, OpenTelemetry-based OpenLLMetry.

### Span-Level Unit Testing (a departure from conventional unit testing)
Conventional unit testing isolates a component from its infrastructure (mocking it away). For agents, the tracing infrastructure must stay *on* even in tests, because the span's contents are part of what's being validated — not noise to eliminate.
- Why it works: an agent that completes a 5-tool workflow cleanly and one that completes it after 3 silent retries of a failed call can produce **identical final outputs** but represent qualitatively different (and differently risky) system behavior. Output-only tests can't distinguish these; span-level assertions (was the right tool selected? were arguments well-formed? was retrieved context non-empty and relevant?) can.
- How: configure an in-memory OpenTelemetry span exporter in the test environment; assert against span attributes (tool-choice rationale, input/output payloads, latency, error/retry state), not just the function's return value.

### Deterministic Constraints + Least-Privilege Testing (hard floor beneath probabilistic evaluation)
LLM-as-judge evaluation is powerful but itself non-deterministic — it can disagree with itself across runs and shares the same prompt sensitivities as the system under test. A robust architecture layers:
- **Deterministic asserts**: schema validation (structured output conforms to Pydantic/JSON schema), safety boundary checks (no prohibited tool invoked, no token-budget overrun, no restricted-category output), routing assertions (correct downstream specialist/endpoint for a known query class). Cheap, fast, perfectly reproducible.
- **Least-privilege testing**: run tests against the actual least-privilege service-account/IAM configuration (not an elevated dev account) so permission errors surface in test, not production; explicitly test that the agent **cannot** act outside its scope, not only that it can act within it.
- Why it works: combining policy-level least privilege with explicit boundary tests bounds the blast radius of a misbehaving agent both architecturally and verifiably.

### LLM-as-Judge (with an independence requirement)
Use a separate, powerful model to automatically score another agent's output quality/correctness — but the judge model must be **distinct** from the agent's cognitive core (preferably different training org/data) to avoid hidden correlated errors. E.g., a fine-tuned Llama handling medical coding, judged by GPT-4o.

### The Four-Flavor Integration Testing Model
1. **Cross-component interaction verification** (innermost loop, dev/local) — data flows correctly and semantically between components.
2. **End-to-end testing** (INT environment, curated/synthetic data) — full workflow validation, run periodically.
3. **Scenario-based testing** (pre-release, often test-in-production behind feature flags) — real-world multi-step sessions; e.g., a refund agent genuinely needs a real purchase + real ticket to validate end-to-end, since production data can't be synthesized adequately.
4. **Regression testing** (ongoing) — a large, human-annotated representative test case set catches behavioral drift from changes elsewhere in an interconnected system; includes systematic edge-case generation.

### Continuous Evaluation Pipeline (the never-stops layer)
Three components, triggered automatically on every change to model/prompt/tool/retrieval config:
1. **Gold-standard dataset** — SME-curated input scenarios with annotated correct/safe/well-cited expected behavior (clinicians for clinical agents, attorneys for legal research agents). The single most human-intensive and most important component — a dataset built without real domain expertise "rewards the wrong behaviors" and silently optimizes the agent away from what matters.
2. **Automated scorer** — exact-match/recall for factual tasks, LLM-as-judge for generative tasks, but **operating on the full execution trace**, not just final output (so a correct answer reached via a hallucinated retrieval step is still flagged as a system failure waiting to recur).
3. **CI/CD integration** — evaluation runs on every relevant PR, surfaces as a required check alongside unit/integration results (long-running evals can be async, linked to the triggering commit, trend-tracked on a dashboard).

### Adversarial / Safety Testing Categories
- **Prompt injection**: tool outputs containing embedded instructions — verify the agent follows its original system instructions, not the injected ones.
- **Jailbreaking**: adversarial requests attempting prohibited-category responses — verify correct refusal, including under multi-turn pressure, role-play framing, and indirect phrasing.
- **Data/memory poisoning**: corrupted retrieval corpus or memory content (outdated info, contradictory facts, injected adversarial content) — verify the agent doesn't propagate corrupted context as fact.
- **Tool failure injection**: timeouts, schema errors, empty results, malformed responses — verify safe, communicative fallback, never silent swallowing or hallucinated recovery.

## Key Concepts
- **Trace archive**: exported (JSON/Parquet) trace history correlated with version-control history — provides the counterfactual record for "did this model/prompt change improve, degrade, or not affect quality at each decision point."
- **AB testing with statistical significance**: compare challenger vs. incumbent on latency and accuracy using t-tests (p < 0.05) before deployment decisions — essential because output variability means naive comparison can mistake noise for signal.
- **Staged deployment rings**: DEV → TEST → INT → PROD, and within PROD, concentric widening user rings (internal users first, general users last) — because agentic behavioral changes may not be immediately apparent and need extended monitoring.
- **Agentic version control scope**: must track prompts, configuration, and (implicitly) model/RAG versions as "code," not just source files — code review must cover prompt and config changes, a commonly missed gap.

## Mental Models
- Treat a passing output-only test as inconclusive, not reassuring, for any multi-step or tool-using agent — always ask "would a span-level check have caught a different, concerning path to the same answer?"
- When choosing an LLM judge, actively check for shared training lineage with the system under test — a judge that shares blind spots with the agent it's grading is not actually independent verification.
- Think of continuous evaluation as the production-time extension of unit testing, not a separate "monitoring" concern — the same gold-standard-dataset-plus-scorer logic that validates a PR should keep running after merge, because input distributions drift.

## Anti-patterns
- **Mocking away the tracing infrastructure in tests** "to isolate the component" — this is the one place agentic testing should deliberately diverge from conventional unit-test isolation practice; the trace *is* what's under test.
- **Trusting LLM-as-judge evaluation alone**: without a deterministic floor (schema/safety/routing asserts), the evaluation layer inherits all the non-determinism of the system it's grading.
- **Testing with an elevated developer account instead of least-privilege credentials**: hides permission-boundary bugs until they reach production.
- **Building a gold-standard dataset without real domain experts**: a layperson-curated dataset for a clinical or legal agent will reward superficially plausible but substantively wrong behavior.
- **Scoring only the final output in continuous evaluation**: misses the "right answer via hallucinated retrieval" failure class — always score the trace.
- **Testing only the happy path**: a program that never exercises prompt injection, jailbreaking, memory poisoning, or tool failure injection is "structurally incomplete," not merely less thorough.

## Worked Example

**AB test scorecard for a support-classification challenger vs. incumbent (the chapter's own run, 50 samples/variant):**
```
⚡ LATENCY: Challenger 151.6ms ± 30.6ms vs. Incumbent 201.0ms ± 39.0ms → +24.6% improvement, p=0.0000 (significant)
🎯 ACCURACY: Challenger 90.0% vs. Incumbent 94.0% → -4.0pp regression, p=0.4660 (NOT significant)
🚀 RECOMMENDATION: DEPLOY CHALLENGER — statistically significant latency win, accuracy difference not statistically distinguishable from noise
```
Lesson: the accuracy drop looks concerning in isolation, but the statistical test reveals it's not distinguishable from random variation at this sample size — without the significance test, a team might wrongly reject a genuinely better (faster, equally accurate) system, or wrongly ship a worse one. Always pair metric deltas with significance testing before a deployment decision.

## Key Takeaways
1. Instrument for tracing from day one — it's the measurement instrument every other testing technique in this chapter depends on.
2. Never test an agent on output alone when tool use, retrieval, or multi-step reasoning is involved — span-level assertions catch failure classes output-only tests structurally cannot.
3. Layer deterministic asserts (schema/safety/routing) beneath probabilistic LLM-as-judge evaluation — the deterministic layer is what keeps your test suite itself from being as unreliable as the system under test.
4. Test against least-privilege credentials, and explicitly test the permission *boundary*, not just permitted actions.
5. A gold-standard dataset is only as good as the domain expertise that built it — invest here first, since everything downstream depends on it.
6. Treat continuous evaluation as mandatory and ongoing, not a pre-launch gate — models, prompts, tools, and input distributions all drift after deployment.
7. Include adversarial/safety test categories (injection, jailbreaking, poisoning, tool failure) in the standard test suite, not as a separate occasional exercise.
8. Pair any AB comparison with statistical significance testing before acting on an observed metric difference.

## Connects To
- **Ch 6 (Tools and Interfaces)**: tool execution spans and error-path testing here directly extend that chapter's tool-calling architecture.
- **Ch 7 (Memory Systems)**: memory consistency/retrieval-accuracy testing here validates the RAG mechanisms built there.
- **Ch 8 (Development Lifecycle)**: the evaluation metrics (resolution rate, escalation quality) specified there are what this chapter's frameworks are built to measure.
- **Ch 9 (Performance Optimization)**: AB testing's latency metrics connect directly to that chapter's optimization techniques.
- **Ch 11 (Monitoring and Observability)**: the next chapter builds the production-time counterpart to this chapter's test-time tracing infrastructure.
- **Ch 13 (Security and Safety)**: this chapter's brief adversarial/safety testing section is a preview of that chapter's full treatment.
