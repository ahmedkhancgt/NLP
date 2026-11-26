# Chapter 5: Three Extra AI Models (Annex 2)

## Core Idea
Three bonus, closer-to-state-of-the-art models — explicitly flagged as better suited to games/robotics than typical business problems — extend the book's Deep RL foundation in three different directions: image-based states (Deep Convolutional Q-Learning), multi-agent parallelism (A3C), and gradient-free policy optimization (Augmented Random Search).

## Frameworks Introduced
- **Deep Convolutional Q-Learning**: identical to Deep Q-Learning (Ch2), except the input is now raw screen images (3D: 2D pixel array + color channel) rather than an encoded state vector; a Convolutional Neural Network (CNN) with *no pooling layers* detects objects/state in each image, is flattened into a 1D vector, and fed into the same fully-connected Deep Q-Learning network as before.
  - When to use: whenever preserving spatial structure matters for predicting the next state (e.g. video-game-style inputs) — a plain encoded vector throws away that spatial information.
  - Why no pooling: pooling layers would lose the location of objects inside the image, which the AI needs to track.
- **A3C (Asynchronous Advantage Actor-Critic)**: multiple agents (A1...An), each with its own copy of the environment, share two networks — an *actor* (decides actions) and a *critic* (evaluates how good a state is) — and asynchronously update the shared networks after collecting n steps of experience.
  - When to use: when a single agent risks getting stuck in a suboptimal-but-attractive policy; other agents' asynchronous updates to the shared network can pull it back out, since no single agent is the sole source of policy updates.
  - The three As: **Asynchronous** (multiple agents on their own environment copies, updating at different times), **Advantage** (`Q(s,a) - V(s)`, the gap between the actor's and critic's predictions), **Actor-Critic** (two networks, two losses — policy loss for the actor, value loss for the critic — both reduced via backpropagation + SGD).
- **Augmented Random Search (ARS)**: a gradient-free Policy Gradient method using a *linear* policy (a single-layer Perceptron, not a deep network); optimizes by randomly perturbing the weight matrix in paired positive/negative directions and approximating the gradient via finite differences, rather than computing it analytically.
  - When to use: continuous-control problems (e.g. a simulated robot learning to walk) where the output is a *group* of continuous actions (muscle impulsions) rather than one discrete action, and where you lack (or don't want to derive) an explicit differentiable reward function of the weights.
  - How: initialize weight matrix Theta to zero; sample a small perturbation matrix Delta_k (entries drawn from a Gaussian N(0, sigma)); evaluate the policy with `Theta + Delta_k` and `Theta - Delta_k` over a full episode to get rewards r+ and r-; keep the top-16 (by reward) directions; update Theta by averaging `(r+ - r-) * Delta_k` across those best directions, optionally scaled by the reward's standard deviation and a learning rate alpha.

## Key Concepts
- **Convolutional layer**: detects spatial features (objects) in an image without the location-destroying effect of pooling.
- **Actor vs. Critic**: the actor estimates Q(s,a) (what action to take); the critic estimates V(s) (how good the current state is) — their difference is the "Advantage."
- **Linear policy / Perceptron**: a single-layer network — here, a weight matrix Theta of shape (number_of_outputs x number_of_inputs) mapping environment-state vectors directly to a vector of continuous actions, with no hidden layers.
- **Finite differences**: approximating a gradient you can't compute analytically (no explicit reward-as-a-function-of-weights formula exists) by observing how the reward changes under small positive and negative perturbations of the weights.
- **Noise (sigma)**: the standard deviation of the Gaussian distribution perturbations are sampled from — small by design, since the finite-difference approximation only holds for small perturbations.
- **ARS improvements (V1 vs. V2)**: normalizing input states, scaling the weight update by the reward's standard deviation, and adding a tunable learning rate alpha — all shown in the original research paper to measurably improve performance over the baseline algorithm.

## Mental Models
- **Each model generalizes the book's prior machinery along one specific axis**: Deep Convolutional Q-Learning generalizes *input shape* (vectors -> images), A3C generalizes *agent count/coordination* (one learner -> many asynchronous learners sharing a network), and ARS generalizes *optimization method* (analytic gradient descent -> gradient-free search via paired perturbations) — recognizing which axis a new problem stresses tells you which of these three to reach for.
- **Paired positive/negative perturbations are a substitute for an unavailable analytic gradient**: when you can't write `reward = f(weights)` in closed form, you can still *estimate* its derivative by nudging weights slightly in both directions and measuring the reward difference — the same finite-differences idea used throughout numerical computing, applied here to policy optimization.
- **Asynchronous multi-agent training provides implicit robustness against getting stuck**: no single agent's experience alone determines the shared policy's trajectory, so one agent's bad luck (getting stuck in a locally-attractive but suboptimal policy) is continuously corrected by the others' updates.

## Anti-patterns
- **Using pooling layers in a Deep Convolutional Q-Learning CNN**: standard CNN practice for classification tasks often uses pooling, but doing so here would destroy the object-location information the agent needs to decide its next move.
- **Reaching for these three models on an ordinary business problem**: the chapter explicitly frames them as closer to "State of the Art" and "not necessarily adapted to solve business problems," more suited to games or robotics — the earlier case-study chapters' simpler Q-Learning/DQN/Thompson-Sampling approaches remain the right default for business AI.
- **Expecting ARS to work with large perturbations**: the finite-differences approximation specifically relies on the perturbations delta being small (close to zero); large perturbations break the linear approximation the gradient estimate depends on.
- **Treating ARS's linear policy as a limitation rather than a deliberate choice**: ARS deliberately uses a simple Perceptron instead of a deep network (unlike Evolution Strategies, which does use parallelized deep networks) — the simplicity is part of what makes the gradient-free search tractable.

## Code Examples
This chapter is conceptual/mathematical rather than code-driven — no new from-scratch implementation is provided (the underlying DQN code reused by Deep Convolutional Q-Learning was already shown in Ch2; A3C and ARS are described via their algorithms and point to their original research papers rather than a worked Keras/TensorFlow implementation).

## Reference Tables
| Model | Branch of AI | Input type | Key mechanism |
|---|---|---|---|
| Deep Convolutional Q-Learning | Deep Reinforcement Learning | Images (3D) | CNN (no pooling) + existing DQN network |
| A3C | Deep Reinforcement Learning | State vectors | Multiple async agents, shared actor+critic networks |
| Augmented Random Search | Policy Gradient / Evolution Strategies | State vectors | Linear policy, finite-differences gradient approximation |

| ARS improvement | Effect |
|---|---|
| Normalizing input states (V2 vs. V1) | Clearly improves performance per the original paper |
| Scaling by reward's standard deviation | Stabilizes the update magnitude across episodes |
| Learning rate (alpha) | Standard tunable control over update step size |

## Worked Example
In ARS, the weight matrix Theta (all zeros initially) representing a Half-Cheetah's walking policy is perturbed in 16 different random directions, each applied both positively (`Theta + Delta_k`) and negatively (`Theta - Delta_k`). Each of the 32 resulting policies is run for one full episode on the PyBullet physics simulation, producing a reward. The 16 directions whose (r+, r-) pairs yielded the highest rewards are kept; the weight update then averages `(r+ - r-) * Delta_k` across exactly those 16 best directions (optionally divided by the reward's standard deviation and scaled by a learning rate), and this single averaged step is applied to the entire weight matrix. Repeating this process for roughly 1000 steps trains the simulated Half-Cheetah to walk/run across the field, despite the algorithm never computing an explicit analytic gradient of reward with respect to weights.

## Key Takeaways
1. Deep Convolutional Q-Learning is Deep Q-Learning plus a pooling-free CNN front end — use it whenever the state is naturally an image and spatial structure matters.
2. A3C's asynchronous, multi-agent, actor-critic design provides built-in robustness against any single agent getting stuck in a suboptimal policy.
3. ARS replaces analytic gradient descent with a finite-differences approximation from paired positive/negative weight perturbations — essential when no differentiable reward-vs-weights function exists.
4. All three models are explicitly positioned as closer to research state-of-the-art than to typical business AI — treat the earlier case-study chapters as the default toolkit for business problems, and these three as the "bonus, if you need them" tier.
5. Simplicity is sometimes the point: ARS's linear (single-layer) policy is a deliberate design choice that keeps the gradient-free search tractable, not a limitation to be fixed.

## Connects To
- **Ch2**: Deep Convolutional Q-Learning is explicitly "the same as Deep Q-Learning" from Ch2, just with an added CNN front end for image inputs.
- **Ch4 (Annex 1)**: A3C's actor/critic networks, policy/value losses, and backpropagation-plus-SGD training all build directly on this chapter's neural-network and Gradient Descent fundamentals.
- **External research papers**: A3C traces to DeepMind's "Asynchronous Methods for Deep Reinforcement Learning" (arxiv.org/pdf/1602.01783.pdf); ARS traces to Mania, Guy & Recht's 2018 paper (arxiv.org/pdf/1803.07055.pdf), which the chapter explicitly points readers to for the full V1/V2 algorithm details.
- **Ch6 (Annex 3)**: none of this chapter's three models are directly revisited in the Q&A section, which instead stays focused on Ch1-3's core case studies.
