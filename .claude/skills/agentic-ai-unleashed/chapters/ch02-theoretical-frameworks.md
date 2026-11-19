# Chapter 2: Theoretical Frameworks

## Core Idea
Agentic reasoning rests on three interlocking pillars: **deduction vs. induction** (applying vs. building a world view), **three competing reasoning architectures** (symbolic, analogy-based, connectionist — connectionist wins for real-world agentic systems), and **decision theory** (Bayes' law for belief updating → maximum expected utility (MEU) for rational action choice).

## Frameworks Introduced

### Deduction vs. Induction
**Deduction** = applying an existing world view to make predictions/decisions (ML calls this *inference*). **Induction** = building the world view itself from observations (ML calls this *training*).
- When to use: diagnosing whether an agent failure is a reasoning problem (bad deduction from a correct model) or a learning problem (bad induction — the world view itself is wrong/stale).
- How: induction requires **ground truth** — either self-supervision (direct experience, e.g. cutting open the mango) or expert supervision (a teacher's label). Deduction and induction interleave continuously: an agent updates its model as new deductions are checked against ground truth, because the real world drifts and stale models degrade.

### Three Reasoning Architectures (the "three tribes")
1. **Symbolic/knowledge engineering**: knowledge as true/false strings of symbols (theorems); deduction = logical proof (e.g., "All humans are mortal; Socrates is human; therefore Socrates is mortal"). Requires absolute truth — brittle under real-world uncertainty; largely failed as 1980s-era expert systems because experts disagreed and rule sets were never exhaustive.
2. **Analogy-based (kernel regression / nearest-neighbor / SVM)**: inference = find similar neighbors, aggregate their unobserved property via a weighted kernel function. Non-parametric; no separate training phase in pure kernel regression — accuracy improves monotonically as more observations accumulate. SVMs formalize this by keeping only the *support vectors* (the hardest boundary cases) instead of all historical data.
3. **Connectionist (DNNs, transformers, LLMs)**: modeled loosely on neurons/synapses; learning = backpropagation + stochastic gradient descent (forward pass → compute loss → backpropagate error → adjust weights). Winning architecture for agentic cognitive cores because it's scalable, supports probabilistic/approximate inference, has tractable optimization-based learning, and rides the GPU hardware cost curve.
  - When to use which: connectionist for the agent's core reasoning; symbolic for explainability/safety-rule enforcement (chain-of-thought, Constitutional AI); analogy/RAG for grounding against a knowledge base. **All three coexist in practice** — RAG is literally an analogy system (semantic-search nearest-neighbor lookup) feeding a connectionist LLM.

### Bayes' Law (belief updating under uncertainty)
`P(A|B) ∝ P(B|A) · P(A)` — posterior belief in A given evidence B is the likelihood of B under A, times the prior on A (normalization constant doesn't affect the decision).
- How: the **prior** P(A) comes from past experience/experts (assign a max-entropy/uniform prior when none is reliably available); the **likelihood** P(B|A) is exactly the "world view" that machine learning induces from training data — this is why an LLM's next-token prediction is literally applying a learned likelihood function.
- **MAP rule** (maximum a-posteriori): choose the inference `x` that maximizes P(X=x|Y) — provably optimal for minimizing *error rate*, but notably NOT the same as optimal for minimizing *cost* (see MEU below).

### Maximum Expected Utility (MEU) — the rational-decision principle
Decision theory's core rule: choose decision `d` to maximize `E[U(d)] = Σ P(s|d) · U(s)` over possible resulting states `s`.
- Why it works: MAP (inference accuracy) is a **special case** of MEU when utilities are symmetric (equal penalty for both error types, zero reward for correct ones). Real-world decisions usually have asymmetric costs, so MEU — not MAP — is the general rational-choice rule.
- When to use: any time an agent must act on an uncertain inference and the costs of different error types differ (e.g., a false negative on a medical diagnosis costs more than a false positive).
- Failure mode it corrects: **status-quo bias** — "doing nothing" is itself a decision with its own expected utility and opportunity cost; agentic systems must evaluate inaction's utility, not treat it as free.

### Explore-Exploit Trade-off
Learning a complete world view requires observing all regions of state space, but acting optimally on current knowledge locks the agent into a narrow range of states — a local-minimum risk analogous to deterministic optimization. Resolution: sporadically take bounded, limited-frequency suboptimal actions purely to gather new observations.
- When to use: any RLHF/self-supervised learning loop where the agent's own policy determines what data it sees next.

## Key Concepts
- **World view**: the agent's internal model (rules/axioms/parameters) — simpler than reality by necessity, inherently probabilistic/incomplete.
- **Ground truth**: the correctness signal needed for induction — self-supervision (direct outcome) or expert supervision (teacher label).
- **Markov Decision Process (MDP)**: the formalism for decisions whose effects persist into future states, requiring long-term (not just immediate) utility optimization — solved via Bellman's dynamic programming.
- **Nash equilibrium**: the rational-agent solution concept from game theory, assuming all players maximize expected utility (even if non-cooperative); real agentic systems must also handle irrational counterparties.
- **RLHF (Reinforcement Learning from Human Feedback)**: expert-supervised induction applied to agent policy learning.
- **Alignment**: ensuring agent exploration/policy stays within human ethical bounds even when an unconstrained exploration would improve the learned world view (e.g., never withholding treatment from patients just to observe the effect).
- **Constitutional AI**: a symbolic-system technique — an explicit written "constitution" of principles the model uses to self-critique and constrain outputs, reducing reliance on human feedback.

## Reference Tables

**Table 2.1: Theory → Practical Implementation Mapping**

| Theoretical concept | Informs practical implementation of |
|---|---|
| Deduction | Inferencing with AI models in cognitive cores |
| Induction | Training/adaptation of models, prompts, hyperparameters |
| Analogistic reasoning | Semantic search, vector databases, RAG |
| Symbolic reasoning | Explaining agent decisions, enforcing safety/ethics rules |
| Probabilistic reasoning, decision theory, MEU | Rationally choosing actions from model inferences |
| Alignment, explore-exploit trade-off | Choosing RL rewards and RLHF to prevent bias, ensure ethical learning |

## Worked Example

**Mango ripeness decision, worked end-to-end (the chapter's running example):**
- Hidden variable X ∈ {ripe, unripe}; observation Y = color (e.g., green).
- Bayes' law: `P(X=ripe|Y=green) ∝ P(Y=green|X=ripe)·P(X=ripe)`, likewise for unripe. The likelihood `P(Y|X)` is the learned "world view" (what color ripe/unripe mangoes tend to be); the prior `P(X)` comes from past experience.
- Four possible decision outcomes with assigned utilities: cut-open-and-ripe U=2, cut-open-and-unripe U=−2, skip-and-ripe U=−1, skip-and-unripe U=0.
- Expected utility of each decision:
  `E[U(cut open)] = P(ripe|Y)·2 + P(unripe|Y)·(−2)`
  `E[U(skip)] = P(ripe|Y)·(−1) + P(unripe|Y)·0`
- Rational rule: cut open only if `E[U(cut open)] ≥ E[U(skip)]`.
- Sanity check: if utilities are made symmetric (U(s1)=U(s3)=−1, U(s0)=U(s2)=0), the MEU rule algebraically collapses to the plain MAP rule `P(ripe|Y) ≥ P(unripe|Y)` — proving MAP is MEU's special case, not a separate principle.

## Key Takeaways
1. Separate "is my world view wrong" (induction/training problem) from "did I apply it wrong" (deduction/inference problem) when debugging an agent.
2. Connectionist architectures (DNNs/transformers) are the practical default for agentic cognitive cores, but symbolic (explainability, safety rules) and analogy-based (RAG) reasoning remain essential supporting sub-modules, not legacy relics.
3. Bayes' law is how an agent should combine prior belief with new evidence — and the likelihood function is exactly what training teaches a model to approximate.
4. Never use MAP (accuracy-only) when decision costs are asymmetric — use MEU, and remember inaction has a utility too.
5. Build deliberate, bounded exploration into any agent that learns from its own actions, or it will lock into a self-reinforcing, suboptimal policy.
6. Alignment constraints apply *during* exploration, not just at output time — some learning experiments are categorically off-limits regardless of information gain.

## Connects To
- **Ch 1 (Introduction)**: operationalizes the "decide" and "adapt" verbs from the six-verb agency ontology into Bayes'/MEU math.
- **Ch 4 (LLMs as Cognitive Cores)**: elaborates the connectionist/transformer architecture only sketched here.
- **Ch 7 (Memory Systems)** and **RAG discussions throughout**: the analogy-reasoning tribe reappears as the mechanism behind retrieval augmented generation.
- **Ch 13/14 (Security/Ethics)**: alignment and Constitutional AI, introduced here, are developed into full safety frameworks later.
