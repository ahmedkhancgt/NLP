# Chapter 9: Deep Learning

## Core Idea
Deep learning extends basic neural networks (Chapter 8) with specialized architectures — RNNs/LSTMs for sequences, CNNs for images, and transformers (attention-based) for text and multimodal data — each exploiting structure (temporal order, spatial locality, long-range context) that a plain feedforward network ignores, and reinforcement learning extends this further to settings where an agent learns from interaction rather than labeled data.

## Frameworks Introduced
- **RNN (Recurrent Neural Network)**: adds a feedback loop that carries a hidden state from one sequence step to the next, letting the network use prior context.
  - When to use: ordered/sequential data (time series, text) where position/order matters.
  - How: unroll the network across timesteps, train via backpropagation through time (BPTT); suffers from vanishing gradients (short-term memory only).
- **LSTM (Long Short-Term Memory)**: an RNN variant with a separate cell state plus forget/input/output gates that regulate what information is kept, added, or exposed.
  - When to use: sequences with long-range dependencies that vanilla RNNs forget.
  - How: at each timestep, forget gate erases irrelevant cell-state info, input gate writes new info, output gate exposes the updated state as the new hidden state/output. GRUs are a simpler, faster-training variant combining forget+input into one update gate.
- **CNN (Convolutional Neural Network)**: applies small, learned filters (weight sharing) across local regions of an image, stacking convolution + pooling/stride layers to build hierarchical features (edges → shapes → objects).
  - When to use: image data (and other grid-like/spatially-correlated data).
  - How: input → repeated [convolution + activation + pooling/stride] blocks → flatten or global average pooling (GAP) → fully connected layer → output; transfer learning (fine-tuning a model pretrained on ImageNet etc.) avoids training from scratch.
- **Transformers (Attention / Self-Attention)**: replace recurrence with an attention mechanism that lets every token directly weigh the relevance of every other token, fully parallelizable on GPUs.
  - When to use: large-scale language/sequence/multimodal modeling (the basis of LLMs like GPT, Claude, Gemini).
  - How: embed tokens into vectors; project into Query/Key/Value matrices via learned weights; compute attention = softmax(QK^T/√dk)·V; stack many layers of multi-head self-attention + feedforward + layer norm/residual connections; decoder-only architectures use causal masking so each token only attends to preceding tokens.
- **Reinforcement Learning (Multi-Arm Bandits, MDPs, Deep RL)**: an agent learns a policy by taking actions in an environment and receiving reward feedback, balancing exploration vs. exploitation.
  - When to use: sequential decision problems where the model's own actions affect what data/states it encounters next (ad optimization, game playing, RLHF for LLMs).
  - How: multi-arm bandits (epsilon-greedy, epsilon-first, Upper Confidence Bound, contextual bandits) for simpler "choose the best option repeatedly" problems; Markov Decision Processes (states, actions, transition probabilities, rewards, discount factor γ) with the Bellman equation V*(s) = max_action E[r + γV*(s′)] for longer-horizon sequential problems; deep RL uses neural networks to approximate the value function/policy in large/continuous state spaces.

## Key Concepts
- **Token**: a discrete unit of a sequence (word, subword, character, punctuation) — text must be tokenized before deep learning models can process it; rule of thumb: ~1 token per 4 characters, or words × 1.3.
- **Double descent**: in very large (over-parameterized) networks, validation error can rise (classic overfitting) then fall again as capacity keeps increasing — contradicts simple bias-variance intuition and remains an active research topic.
- **Vanishing/exploding gradient (revisited)**: RNN's core limitation — repeated multiplication through many timesteps shrinks (or blows up) gradients exponentially, causing "forgetting" of distant context; mitigated by LSTM/GRU gating, ReLU, batch normalization, and skip/residual connections.
- **Embeddings**: dense vector representations of discrete items (words, etc.) where semantically similar items map to nearby points; learned (not hand-specified) and typically hundreds-to-thousands of dimensions; modern transformer embeddings are context-dependent (same word gets different vectors in different contexts), unlike static Word2vec-style embeddings.
- **Query / Key / Value**: the three learned projections of each token used in attention — Query asks "what am I looking for," Key describes "what I can offer," Value carries "the actual information to pass forward" if attended to.
- **Self-attention**: attention where queries, keys, and values all derive from the same input sequence — the core mechanism transformers use to build context-aware token representations.
- **Multi-head attention**: several attention mechanisms run in parallel, each potentially learning a different kind of relationship (syntax, semantics, position).
- **Masking (causal masking)**: in decoder-only transformers, prevents a token from attending to future tokens, enforcing left-to-right generation.
- **Transfer learning**: fine-tuning a model pretrained on a large dataset (e.g., ImageNet) on a smaller target dataset, reusing learned low/mid-level features.
- **Reward / Agent / Policy / Discount factor (γ)**: RL vocabulary — an agent takes actions in an environment, receives rewards, and learns a policy (state→action mapping) that maximizes expected (discounted) cumulative reward.

## Mental Models
- Think of an RNN as "reading left to right with a notebook" — it updates a running summary (hidden state) as it processes each new token, but the notebook's older entries fade exponentially (vanishing gradient) unless something like an LSTM's cell state actively protects them.
- Think of a CNN's convolution filters as "the same small stencil slid across the whole image" — weight sharing is why a 3×3 filter needs only ~9 weights regardless of image size, versus a standard dense layer needing one weight per pixel.
- Think of self-attention as "every word asking every other word: how relevant are you to me right now?" and getting a weighted blend of their information back — this is why transformers capture long-range dependencies that RNNs struggle with, and why they parallelize (every token's attention can be computed simultaneously, unlike an RNN's sequential dependency).
- Treat reinforcement learning as fundamentally different from supervised learning: the agent's own policy determines what data/states it sees next, so "training distribution" and "learned policy" co-evolve — there's no fixed, external, labeled dataset.

## Anti-patterns
- **Using a vanilla RNN for long sequences expecting it to remember distant context**: short-term memory limits (vanishing gradient) make this unreliable — use LSTM/GRU or, for most modern large-scale tasks, a transformer instead.
- **Training a CNN/transformer from scratch on a small dataset when a suitable pretrained model exists**: wastes enormous compute and typically underperforms transfer learning (fine-tuning).
- **Assuming the number of layers alone makes a network "deep" is the whole story**: both the number of layers (depth) and the regularization/architecture choices (dropout, batch norm, skip connections, GAP) interact to determine whether a deep network generalizes rather than overfits.
- **Ignoring tokenization costs and domain mismatch**: a general-purpose tokenizer may perform poorly on specialized text (e.g., aircraft maintenance logs) — retraining/fine-tuning a tokenizer on domain text may be necessary.
- **Treating RL like supervised learning with a static dataset**: because the agent's actions shape the future data distribution, standard i.i.d. assumptions don't hold; exploration-exploitation trade-offs must be explicitly managed (epsilon-greedy, UCB, etc.).

## Code Examples
```python
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
).to(device)
# After 5 epochs: Test Accuracy: 98.06%
```
- **What it demonstrates**: a simple 3-convolution-layer CNN with GAP trained on MNIST handwritten digits, reaching 98% test accuracy in 5 epochs — but still misclassifying an ambiguous "4" as "6" with high confidence, illustrating the model's remaining brittleness.

```
Attention(Q, K, V) = softmax(Q·K^T / √dk) × V
```
- **What it demonstrates**: the scaled dot-product self-attention formula — query-key similarity (dot product, scaled) is softmaxed into weights that blend the value vectors into a context-aware output.

```
V*(s) = max_action E[r + γ × V*(s′)]
```
- **What it demonstrates**: the Bellman equation for the optimal value function in a Markov Decision Process — the recursive core of reinforcement learning's long-horizon planning.

## Reference Tables
| Architecture | Handles | Key mechanism | Limitation addressed by successor |
|---|---|---|---|
| RNN | Sequences | Hidden state passed forward in time | Short-term memory only (vanishing gradient) → LSTM/GRU |
| LSTM/GRU | Long sequences | Gated cell state | Sequential (not parallelizable) → Transformer |
| CNN | Images (grid data) | Weight-shared convolution filters | — (still dominant for vision, often combined with transformers) |
| Transformer | Sequences, long-range context | Self-attention (parallel) | Quadratic compute cost in sequence length (active research area) |

| RL framework | Scope | Example strategy |
|---|---|---|
| Multi-arm bandit | Single-step repeated choice | Epsilon-greedy, Epsilon-first, Upper Confidence Bound, contextual bandits |
| Markov Decision Process | Multi-step sequential decisions | Bellman equation / value iteration |
| Deep RL | High-dimensional/continuous states | Neural network approximates value function or policy |

## Worked Example
MNIST handwritten digit classification (28×28 grayscale images, 10 classes) via PyTorch: a 3-convolutional-layer CNN (32→64→128 filters, ReLU, max-pooling, global average pooling, final linear layer to 10 classes, ~90,000+ total parameters) trained with Adam optimizer and cross-entropy loss over 5 epochs improves from 94.79% to 98.06% test accuracy. Despite this high aggregate accuracy, a specific test image of a handwritten "4" is misclassified as "6" with 72.2% confidence (vs. 26.8% for the correct "4"), illustrating that even well-performing deep models retain systematic blind spots on visually ambiguous inputs — and that model accuracy summaries can mask instance-level brittleness. A simple self-attention walkthrough on the sentence "The cat sat on the mat" shows how the query from "sat" produces high attention weight (0.45) on "cat" (the subject) and moderate weight (0.30) on "mat" (the object/location), versus near-zero weight (0.02) on function words like "the" — demonstrating how attention naturally recovers grammatically/semantically relevant context without hand-coded linguistic rules.

## Key Takeaways
1. Match architecture to data structure: RNN/LSTM for sequences needing memory, CNN for spatially-correlated grid data (images), transformers for large-scale sequence/language tasks needing long-range context and parallel training.
2. LSTMs solve vanilla RNNs' short-term-memory problem via gated cell state; transformers solve both the memory problem and the parallelization problem via self-attention, which is why they now dominate large-scale language modeling.
3. Transfer learning (fine-tuning pretrained CNNs/transformers) is almost always preferable to training from scratch unless you have very large task-specific data and compute.
4. Self-attention's Query/Key/Value mechanism is the mathematical heart of modern LLMs — understanding the scaled dot-product softmax formula demystifies how "attention" works.
5. Reinforcement learning differs fundamentally from supervised learning because the agent's policy determines its own future training data — exploration/exploitation trade-offs (bandits, MDPs) are a distinct and necessary toolset.
6. High aggregate accuracy does not guarantee robustness on individual ambiguous cases — always sanity-check specific predictions, not just summary metrics.

## Connects To
- **Ch 8 (Neural Networks)**: this chapter's architectures are direct extensions of the feedforward network, backpropagation, and regularization concepts (dropout, L1/L2, ReLU) introduced there.
- **Ch 3 (Statistical Experiments and Significance Testing)**: multi-arm bandits here are the same algorithm introduced in Ch 3's "Multi-Arm Bandit Algorithm" section, now framed within the broader reinforcement learning paradigm.
- **Ch 10 (Generative AI and Large Language Models)**: the transformer/attention/embedding machinery developed here is the direct technical foundation for the generative models (LLMs) covered next.
