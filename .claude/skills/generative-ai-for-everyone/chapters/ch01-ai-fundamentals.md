# Chapter 1: AI Fundamentals

## Core Idea
AI is organized by scope (Narrow → General → Super-intelligence) and by how models learn from data (supervised, unsupervised, semi-supervised, reinforcement learning); choosing the right learning paradigm and data-representation/design pattern is the foundation every later GenAI technique builds on.

## Frameworks Introduced
- **Three tiers of AI (ANI/AGI/ASI)**: Artificial Narrow Intelligence (task-specific, "Weak AI"), Artificial General Intelligence (human-like general reasoning), Artificial Super-intelligence (exceeds human cognition in nearly all domains; Bostrom's speed/quality/collective super-intelligence).
  - When to use: scoping what a given AI system can realistically do.
  - How: classify a system by whether it generalizes beyond its training task (ANI if not).
- **Four ML learning paradigms**: supervised, unsupervised, semi-supervised (SSL), reinforcement learning (RL).
  - When to use: choosing a training approach based on label availability and task shape.
  - How: labeled data → supervised; unlabeled data, find structure → unsupervised; mostly unlabeled + small labeled set → SSL; sequential decisions with reward feedback → RL.
- **SSL method taxonomy**: generative, pseudo-labeling, consistency regularization, graph-based, hybrid.
  - When to use: large unlabeled corpora with only a little labeled data (the common real-world case).
  - How: pseudo-labeling (self-training, co-training) bootstraps labels from a supervised model; consistency regularization (Π-model, temporal ensembling, mean teacher) forces stable predictions under perturbation; hybrid (MixMatch, FixMatch) combines both plus MixUp augmentation.
- **RL components and algorithm families**: Agent/Action/State/Environment/Reward/Policy (formalized as an MDP); model-based vs. model-free; model-free splits into value-based (Q-learning, DQN, SARSA) and policy-based (REINFORCE, actor-critic, A3C, A2C).
  - When to use: sequential decision problems with trial-and-error feedback (robotics, self-driving, RLHF for LLMs).
  - How: agent observes state, takes action, receives reward; value-based methods learn Q(s,a) via the Bellman equation; policy-based methods directly parameterize and gradient-ascend the policy.
- **Design patterns for AI projects**: (1) real-world data representation (encoding, embedding), (2) training strategies (transfer learning, multi-task learning, distributed training), (3) performance improvement (ensembling, mixture of experts).

## Key Concepts
- **Narrow/General/Super-intelligence**: scope ladder of AI capability from task-specific to beyond-human.
- **Supervised vs. unsupervised learning**: labeled input→output pairs vs. unlabeled pattern discovery.
- **Semi-supervised learning (SSL)**: transductive learning that fuses a small labeled set with a large unlabeled set, resting on continuity, cluster, and manifold assumptions.
- **Markov Decision Process (MDP)**: the formal environment model (state, action, reward, transition) underlying almost all RL.
- **Q-learning / Bellman equation**: off-policy value-based RL update, `Q(s,a) ← Q(s,a) + α(r + γ·max Q(s',a') − Q(s,a))`.
- **Embedding**: a fixed-size, lower-dimensional float vector representation of complex data (text/image/audio) that preserves semantic proximity; stored in vector stores.
- **Transfer learning / fine-tuning**: reusing a pre-trained model's weights (freezing most layers, tuning a few) instead of training from scratch.
- **Bias-variance trade-off**: high bias → underfitting; high variance → overfitting; ensembling (bagging, boosting, stacking) manages this trade-off.
- **Double descent**: in highly over-parameterized models (e.g., GPT-scale), test error can decrease again past the classical interpolation threshold, explaining why huge models don't simply overfit.
- **Mixture of Experts (MoE)**: routes each input to a subset of "expert" sub-networks for sublinear compute cost relative to parameter count (used in GPT-4-class models).

## Mental Models
- Think of the **ANI/AGI/ASI ladder** as a scope dial, not a quality dial — most deployed AI today (including LLMs) is still ANI in the sense of being trained for specific objective functions, even when broad in apparent capability.
- Use the **label-availability lens** to pick a learning paradigm first, before picking an algorithm: "how much labeled data do I actually have?" determines whether you're in supervised, SSL, or unsupervised territory.
- Think of **ensembling as trading variance for bias (or vice versa)** rather than a free accuracy boost — bagging fights variance, boosting fights bias, stacking learns the combination itself.
- Use the **exploration-exploitation frame** for any RL design: every policy choice is implicitly trading off "use what has worked" against "learn something new."

## Anti-patterns
- **One-hot encoding high-cardinality categorical data**: creates sparse, incomplete-vocabulary vectors and breaks on unseen ("cold start") categories — use feature hashing instead for high-cardinality columns (e.g., Order ID).
- **Treating label encoding as neutral**: assigning arbitrary integers to unordered categories silently injects a false ordinal relationship into the model.
- **Ignoring the bias-variance trade-off when scaling model complexity**: blindly increasing complexity without ensembling or regularization risks overfitting (though double descent shows very large models can recover generalization past the interpolation threshold).
- **Sharing all parameters across tasks in multi-task learning**: in large networks, excessive hard parameter sharing introduces task conflicts that hurt per-task predictions.

## Code Examples
No executable code fences in this chapter (two fenced blocks are prose headers, "The advantages/disadvantages of supervised ML are as follows"); the chapter is conceptual/mathematical rather than code-driven.

## Reference Tables
**Supervised learning: advantages vs. disadvantages**

| Advantages | Disadvantages |
|---|---|
| Knowledge amplification from past experience | Big data complexity at scale |
| Guided learning via labels speeds convergence | Costly data preparation |
| Simpler to interpret than unsupervised | Missing/incomplete values |
| Well-suited to practical problems | Bias in training data |
| Validation/control via held-out labels | Resource-intensive training; label dependency |

**RL algorithm families**

| Category | Sub-type | Examples |
|---|---|---|
| Model-based | — | MCTS, PILCO, MBMF, MBVE, World Models |
| Model-free | Value-based | Q-learning (off-policy), DQN, SARSA (on-policy) |
| Model-free | Policy-based | REINFORCE, Actor-Critic, A3C, A2C |

**Embedding modalities and example models**

| Modality | Classical features | Learned/pre-trained models |
|---|---|---|
| Image | Color/texture/spatial (CBIR) | CNN, BEiT, DINO, MAE, VisualBERT |
| Audio | Zero-crossing rate, pitch, log-mel, MFCC | Wav2Vec, YAMNet, w2v-BERT, Whisper, Conformer, HuBERT |
| Text | TF-IDF, bag of words | Word2Vec, GloVe, BERT, ELMo, GPT, CLIP |

## Worked Example
**Car-purchase prediction (supervised learning walkthrough)**: A model is trained on labeled historical data — customer income, miles driven, location, prior cab-booking records, and previously purchased car type (luxury/sedan/SUV/hatchback/electric). Step 1: train on input→output pairs so the model learns the relationship. Step 2: freeze the learned relationship into a model. Step 3: test on new customer data where the true next purchase is already known, and compare predicted vs. actual to compute accuracy. The resulting model lets car sellers forecast a customer's next purchase and target marketing accordingly — illustrating the three-step supervised-learning cycle (train → build model → test/validate) used throughout the book.

## Key Takeaways
1. Classify any AI system first by scope (ANI/AGI/ASI), then by learning paradigm (supervised/unsupervised/SSL/RL) — this double classification drives every later design decision.
2. SSL is the practical default for most real-world corpora (mostly unlabeled, a little labeled), and underlies how LLMs bootstrap from raw text plus techniques like co-training for prompt learning.
3. RLHF (reinforcement learning from human feedback) — introduced here as an RL application — is the mechanism GenAI chapters later rely on to align LLM outputs with human preference.
4. Embeddings are the universal interface between raw modalities (text/image/audio) and neural models, and recur in every subsequent chapter (vector stores, RAG, LLM tokenization).
5. Transfer learning (reusing pre-trained foundation models) is positioned as the default — not the exception — given the infeasibility of training GenAI-scale models from scratch for every use case.
6. Mixture of Experts and double descent explain why GPT-scale models don't follow naive bias-variance intuition — critical context before later chapters introduce GPT-3/4 architectures.

## Connects To
- **Ch 2**: takes the supervised-learning math introduced here (loss functions, gradient descent) and formalizes it into full deep-learning mechanics (neural network training, backpropagation), then introduces GenAI itself.
- **Ch 5**: the text-embedding techniques (Word2Vec, GloVe, BERT) sketched here become the full transformer tokenization/embedding pipeline.
- **Ch 8**: transfer learning/fine-tuning and distributed training strategies here are operationalized into the LLMOps pre-training, fine-tuning, and LoRA pipeline.
- **Ch 6**: RLHF, first introduced here as an RL application, becomes central to how ChatGPT is aligned and evaluated.
