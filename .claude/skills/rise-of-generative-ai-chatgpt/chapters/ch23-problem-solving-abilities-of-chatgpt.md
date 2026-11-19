# Chapter 23: Problem solving abilities of ChatGPT

## Core Idea
Through four hands-on experiments (theorem application, mathematical modeling, neural-network design, formula-writing), the authors demonstrate that ChatGPT doesn't just produce a final answer — it reliably shows the calculation steps, reasoning, and explanation behind the answer, which is the differentiator that "impressed" the authors across every example.

## Frameworks Introduced
- **Step-by-step verification protocol for testing ChatGPT's problem-solving**: pose a specific, non-trivial question → capture the full response → check for (a) correct final answer, (b) shown calculations/equations, (c) explanation of the approach/reasoning, (d) step-by-step structure.
  - When to use: evaluating whether ChatGPT is trustworthy for a new category of quantitative/technical problem before relying on it.
  - How: the authors explicitly test polynomial remainder theorem, a percentage-discount word problem, a neural-network architecture question, and a sphere-volume formula problem — each scored against the same four criteria.

## Key Concepts
- **Neural language model (NLM)**: the chapter's term for ChatGPT's category, used to frame it as fundamentally a problem-solving tool, not a lookup tool (contrasted explicitly with "Google can handle" trivial unit-conversion questions).
- **Sequential problem-solving methodology**: ChatGPT's tendency to replicate a structured, step-by-step solving process rather than jumping straight to an answer.
- **Logical reasoning display**: the chapter treats showing *why* an answer is correct as equally important as the answer's correctness.

## Mental Models
- Use the **"not simple computation, but reasoning demonstration" test** to separate genuinely useful AI problem-solving evaluation from trivial unit-conversion-style benchmarks that any calculator/search engine already handles.
- Treat **"impressed because it explained, not just because it solved"** as the chapter's recurring evaluative refrain — a transferable standard for judging any generative AI's output quality on technical tasks.

## Worked Example
Four documented experiments: (1) Polynomial remainder theorem — "what is the degree of the polynomial 4x+2?" solved with full calculation and explanation. (2) Mathematical model — 20% discount word problem solved with step-by-step logical reasoning. (3) Neural network — "write a neural network that can classify handwritten digits" answered with architecture guidance and training-data advice. (4) Formula application — sphere volume (radius 3) solved with formula, calculation, and explanation.

## Key Takeaways
1. ChatGPT consistently pairs answers with calculations and explanations across very different problem types (theorem, modeling, neural-network design, formula application).
2. Neural network architecture planning plus large/varied labeled datasets are named as the two essential ingredients for successful image-classification tasks when ChatGPT is asked to design one.
3. The "step-by-step" quality is presented as the primary differentiator of value, more than raw correctness alone.
4. This chapter is the empirical companion to Ch 17's abstract quantitative-capability claims — read together for theory + evidence.

## Connects To
- **Ch 17**: supplies the abstract capability-tiering framework that this chapter's four worked experiments substantiate concretely.
- **Ch 24**: continues directly from this chapter's experiments into practical "how should a beginner use ChatGPT for problem-solving" guidance.
