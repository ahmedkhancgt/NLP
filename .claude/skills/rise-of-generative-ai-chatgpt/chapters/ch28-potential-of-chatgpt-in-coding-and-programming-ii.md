# Chapter 28: Potential of ChatGPT in Coding and Programming (II)

## Core Idea
This chapter revisits Ch 16's coding topic with a more skeptical, evidence-driven lens: it reports a concrete benchmark result (ChatGPT solved 19/40 Python problems, on par with CoCoNut and Codex, with a 77.5% success rate across follow-up conversations) and argues explicitly that ChatGPT augments but cannot replace developers, due to its lack of logical reasoning and its reliance on training data that stops in 2022.

## Frameworks Introduced
- **"Can It Replace Developers?" Evaluation Framework**: assess (1) benchmark success rate on real coding problems, (2) ability to reason logically vs. "smoothly combine text fragments," (3) training-data recency vs. current library/tool versions, (4) tendency to fabricate plausible-but-wrong library names/procedures.
  - When to use: evaluating organizational risk/readiness before relying on ChatGPT-generated code in production.
  - How: benchmark against known problem sets, stress-test with edge cases and production-level hardening requirements (not just "straightforward instances"), and always route output through human review before deployment.
- **Practical ChatGPT-for-coding workflow (step-by-step)**: register/log in → verify phone number → select personal usage → choose features → open in playground → describe desired code in natural language.
  - When to use: onboarding a new developer to using ChatGPT as a coding aid.

## Key Concepts
- **Benchmark result**: 19/40 Python problems solved, comparable to CoCoNut and Codex; conventional Automated Program Repair (APR) techniques solved only 7; success rate reached 77.5% through follow-up conversation refinement.
- **Training-data staleness for code**: data only through 2022, meaning ChatGPT can lag behind current libraries/tool versions and may suggest outdated or incompatible code.
- **Fabricated plausibility**: ChatGPT "may fill in blanks with words or code that sound rational," including inventing non-existent libraries/keys/data structures.
- **Supported-language boundary**: formally trained primarily on Python, Java, C++, and JavaScript — may fail outside these.
- **Stack Overflow ban context**: cited as evidence that ChatGPT-generated answers were "of poor quality but convincing sounding" enough to warrant a platform-level ban.

## Mental Models
- Use the **"convincing ≠ correct" heuristic for code**, the coding-specific instance of Ch 10's "confidently wrong" pattern — code that compiles/reads plausibly can still be subtly broken, especially around edge cases and production hardening.
- Treat **human-developer review as non-negotiable before production deployment** — this chapter states it as directly as Ch 20 states it for medical decisions: AI-generated code must always be checked by a qualified human before shipping.

## Key Takeaways
1. ChatGPT augments developer productivity (quicker reviewing, learning, drafting) but does not replace logical reasoning or production-hardening judgment.
2. The 2022 training cutoff is a concrete, practical limitation for any fast-moving library/tool ecosystem.
3. Fabricated-but-plausible code elements (invented libraries, altered sequences) are a specific, named failure mode distinct from simple syntax errors.
4. Formal language-support boundaries (Python/Java/C++/JavaScript) should be checked before relying on ChatGPT for a less common language.

## Connects To
- **Ch 16**: the companion chapter covering the same topic with a more constructive, capability-building lens (generation/debugging/completion/professionalism) — read both together for the full balanced picture.
- **Ch 10**: shares the root-cause framing for confidently-wrong output, here specifically for code rather than general text.
- **Ch 9**: the FIM (Fill-In-the-Middle) training technique described there is the technical mechanism underlying the code-completion capability evaluated here.
