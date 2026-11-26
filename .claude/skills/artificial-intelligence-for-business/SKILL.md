---
name: artificial-intelligence-for-business
description: "Knowledge base from \"Artificial Intelligence for Business\" by Hadelin de Ponteves and Kirill Eremenko. Use when solving real-world business problems with reinforcement learning and AI — Q-Learning for process optimization, Deep Q-Learning for cost minimization, Thompson Sampling for revenue maximization, neural network fundamentals, or state-of-the-art bonus models (Deep Convolutional Q-Learning, A3C, Augmented Random Search)."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Artificial Intelligence for Business
**Authors**: Hadelin de Ponteves, Kirill Eremenko | **Pages**: ~102 | **Chapters**: 6 | **Generated**: 2026-11-26

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch02`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**The General AI Framework / Blueprint (5 steps)** (Ch2): Build the Environment (as a class) -> Build the Brain (neural network) -> Implement the Deep RL algorithm (DQN) -> Train the AI -> Test the AI. This is the reusable template the book applies to every Deep RL business problem — code sections in CAPITAL LETTERS are the general, transferable steps; lowercase sections are specific to the case study at hand.

**Match the model to the problem shape, not the other way around** (Ch1-3, Ch6): small/discrete state+action space -> tabular Q-Learning; continuous/large state space -> Deep Q-Learning; no evolving state at all, just "try an option, observe an outcome" -> Thompson Sampling (a completely different branch, Online Learning/Multi-Armed Bandits, needing no states, actions, or neural networks). Recognizing which shape a business problem has is the first and most important design decision.

**Markov Decision Process (MDP) as the foundation** (Ch1): a tuple (S, A, T, R) — states, actions, a transition rule, and a reward function — with the key assumption that the future depends only on the current state and action (no memory). Q-Learning finds the optimal policy pi* via the Bellman equation: `Q(s,a) <- Q(s,a) + alpha * TD(s,a)`, where the Temporal Difference TD is `R(s,a) + gamma*max(Q(s',.)) - Q(s,a)`.

**Deep Q-Learning = Q-Learning + a neural network + Experience Replay** (Ch2): replace the Q-table with a neural network predicting Q-values per action; train on random batches sampled from a capped memory of past transitions (never just the latest transition, which is too correlated with the one before it). Add epsilon-greedy exploration (play random actions ~30% of the time during training) and early stopping (halt once reward stops improving).

**Thompson Sampling for online option selection** (Ch3): maintain a Beta distribution per option (parameterized by accumulated successes/failures); each round, sample one value per option and play whichever sampled highest; update that option's counts based on the observed outcome. Needs zero hyperparameters (no learning rate, no discount factor) and automatically balances exploration vs. exploitation through the Beta distribution's own shape. Always benchmark against a trivial Random Selection baseline and report Relative Return + a Regret Curve, not just final score.

**Parallel baseline tracking proves AI value** (Ch2): simulate both the AI-controlled system and an "unintelligent" baseline system on identical inputs at every timestep; define the reward as the measured difference between them — this is what makes "energy saved" or "revenue gained" an auditable, provable number rather than an assumption.

**Neural network fundamentals underlie every model** (Ch4): a neuron = weighted inputs -> activation function -> output. Choose the activation function by what you need: Threshold for binary output, Sigmoid for probability-like output (output layer / last hidden layer), Rectifier for hidden layers with normalized ([0,1]) inputs, Hyperbolic Tangent for hidden layers with standardized ([-1,1]) inputs. Training = forward-propagate -> compute loss (MSE) -> back-propagate -> Gradient Descent updates weights toward lower loss, repeated until early stopping or max iterations.

**Gradient Descent variants and optimizers trade off convergence risk vs. cost** (Ch4): Batch GD (whole dataset per update, fast but risks local minima on the non-convex cost functions typical of real problems) vs. Stochastic GD (one example per update, avoids local minima, less deterministic) vs. Mini-Batch (practical middle ground). Optimizer lineage Momentum -> Nesterov -> AdaGrad -> AdaDelta -> Adam, each fixing the previous one's weakness; default to Adam unless you have a specific reason not to.

**The three "bonus" state-of-the-art models are explicitly NOT the business default** (Ch5): Deep Convolutional Q-Learning (image states via a pooling-free CNN front end), A3C (multiple asynchronous agents sharing actor/critic networks, robust against any one agent getting stuck), and Augmented Random Search (gradient-free linear-policy optimization via finite differences, for continuous-control/robotics problems) are all flagged as closer to research state-of-the-art than to typical business AI — reach for the Ch1-3 case-study approaches first.

**Hyperparameters and architecture choices are empirical, not derived** (Ch1, Ch2, Ch6): the discount factor, network layer sizes, and similar choices come from experimentation and precedent (published architectures), validated empirically — not from a closed-form calculation. Expect to "try, observe, adjust."

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| [ch01](chapters/ch01-optimizing-processes.md) | Optimizing Processes (Q-Learning) | MDP, tabular Q-Learning, Bellman equation, Argmax vs. Softmax, waypoint routing via function composition |
| [ch02](chapters/ch02-minimizing-costs.md) | Minimizing Costs (Deep Q-Learning) | Deep Q-Learning, Experience Replay, the 5-step AI Framework, epsilon-greedy exploration, early stopping, dropout |
| [ch03](chapters/ch03-maximizing-revenues.md) | Maximizing Revenues (Thompson Sampling) | Multi-Armed Bandit / Online Learning, Thompson Sampling, Relative Return, Regret Curve |
| [ch04](chapters/ch04-artificial-neural-networks.md) | Artificial Neural Networks (Annex 1) | Neuron, activation functions, forward/back-propagation, Gradient Descent, optimizers (Adam) |
| [ch05](chapters/ch05-three-extra-ai-models.md) | Three Extra AI Models (Annex 2) | Deep Convolutional Q-Learning, A3C, Augmented Random Search (ARS) |
| [ch06](chapters/ch06-questions-and-answers.md) | Questions and Answers (Annex 3) | Plan vs. Policy, hyperparameter-tuning clarifications, further bandit applications |

## Topic Index

- **A3C (Asynchronous Advantage Actor-Critic)** → ch05
- **Activation functions (Threshold/Sigmoid/Rectifier/Tanh)** → ch04
- **Adam optimizer** → ch02, ch04
- **Argmax vs. Softmax** → ch01, ch02, ch06
- **ARS (Augmented Random Search)** → ch05
- **Bellman equation** → ch01
- **Convex / non-convex cost functions** → ch04
- **Deep Convolutional Q-Learning** → ch05
- **Deep Q-Learning (DQN)** → ch02
- **Discount factor (gamma) tuning** → ch01, ch06
- **Dropout** → ch02
- **Early stopping** → ch02
- **Experience Replay** → ch02
- **Finite differences (gradient approximation)** → ch05
- **Gradient Descent (Batch/Stochastic/Mini-Batch)** → ch04
- **Markov Decision Process (MDP)** → ch01
- **Mean Squared Error (MSE)** → ch02, ch04
- **Multi-Armed Bandit / Online Learning** → ch03, ch06
- **Neural network architecture choices** → ch02, ch04, ch06
- **Plan vs. Policy** → ch01, ch06
- **Q-Learning (tabular)** → ch01
- **Regret curve** → ch03
- **Relative return** → ch03
- **Reward matrix design** → ch01, ch02
- **Temporal Difference (TD)** → ch01, ch02
- **Thompson Sampling** → ch03, ch06
- **The 5-step AI Framework / Blueprint** → ch02

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — reusable techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules, decision tree, and quick reference

---

## Scope & Limits
This skill covers the book's 6 content sections (3 case-study chapters plus 3 annexes), synthesized as structure, not verbatim text. The source document's "Companion download index" (a list of downloadable code archive filenames) is excluded as a non-substantive file listing, not instructional content.
