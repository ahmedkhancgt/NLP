# Cheatsheet

## Decision rules

- **Problem has no evolving state, just "try an option and observe a binary outcome"** -> Thompson Sampling (Ch3), not Q-Learning/DQN.
- **State/action space is small and discrete/enumerable** -> tabular Q-Learning (Ch1). **Continuous or very large** -> Deep Q-Learning (Ch2).
- **State is naturally an image** -> Deep Convolutional Q-Learning: CNN front end, no pooling layers (Ch5).
- **Output must be a group of continuous actions (e.g. robot muscle impulsions), not one discrete action** -> ARS / Policy Gradient methods (Ch5), not standard DQN.
- **No explicit differentiable reward-vs-weights function exists** -> approximate the gradient via finite differences (paired +/- perturbations), as in ARS (Ch5), rather than forcing analytic Gradient Descent.
- **A single agent keeps getting stuck in a suboptimal policy** -> consider a multi-agent asynchronous approach (A3C) rather than tuning one agent harder (Ch5).
- **Choosing an activation function**: binary 0/1 output -> Threshold. Probability-like output / last hidden layer -> Sigmoid. Hidden layers with normalized ([0,1]) inputs -> Rectifier. Hidden layers with standardized ([-1,1]) inputs -> Hyperbolic Tangent (Ch4).
- **Choosing Argmax vs. Softmax**: simple/business problem, little exploration needed -> Argmax. Complex problem (games, robotics) needing real exploration -> Softmax (Ch1, Ch2, Ch6).
- **Cost function is non-convex (the common case for real problems)** -> use Stochastic or Mini-Batch Gradient Descent, not plain Batch Gradient Descent, to avoid getting stuck in a local minimum (Ch4).
- **Choosing an optimizer** -> default to Adam unless you have a specific, well-understood reason not to (Ch4).

## Decision tree: which AI model for which problem

- Is the problem "pick the best of N options via live sequential feedback, no evolving state"?
  - Yes -> **Thompson Sampling** (Ch3)
  - No, there's an evolving state/environment ->
    - Is the state/action space small and fully enumerable?
      - Yes -> **Tabular Q-Learning** (Ch1)
      - No (continuous or very large) ->
        - Is the state an image?
          - Yes -> **Deep Convolutional Q-Learning** (Ch5)
          - No ->
            - Does the output need to be a group of continuous actions (e.g. robot motor control)?
              - Yes -> **ARS / Policy Gradient** (Ch5)
              - No -> **Deep Q-Learning (DQN)** (Ch2), optionally **A3C** if a single agent keeps getting stuck (Ch5)

## Trade-off matrices

| Gradient Descent variant | Local-minimum risk | Memory needs | Determinism |
|---|---|---|---|
| Batch | Higher (non-convex cost) | Highest | Deterministic given init |
| Stochastic (SGD) | Lower | Lowest | Non-deterministic run-to-run |
| Mini-Batch | Lower | Moderate | Mixed |

| Optimizer | Fixes | Known weakness |
|---|---|---|
| Momentum | SGD oscillation | Can overshoot minimum |
| Nesterov | Momentum overshoot | Still uncommon in libraries |
| AdaGrad | Per-parameter learning rate | Learning rate decays to ~0 |
| AdaDelta | AdaGrad's decay | (stepping stone only) |
| Adam | All of the above | None flagged; the default choice |

| Model | Needs hyperparameters? | Scales to large/continuous states? | Business-problem fit |
|---|---|---|---|
| Tabular Q-Learning | gamma, alpha | No | Good (small, discrete problems) |
| Deep Q-Learning | gamma, alpha, epsilon, memory size, batch size | Yes | Good (the book's main business workhorse) |
| Thompson Sampling | None | N/A (no state) | Excellent (online option selection) |
| Deep Conv. Q-Learning | Same as DQN + CNN config | Yes (images) | Rare for business; mostly games |
| A3C | Network + agent-count config | Yes | Rare for business; mostly games/robotics |
| ARS | Noise sigma, learning rate, #directions | Yes (continuous control) | Rare for business; mostly robotics |

## Thresholds & defaults (as used in the book's case studies)

- Q-Learning: gamma (discount) = 0.75, alpha (learning rate) = 0.9, 1000 training iterations (Ch1).
- DQN server-cooling case study: 64+32 neuron hidden layers, Adam optimizer, MSE loss, epsilon = 0.3, discount = 0.9, max memory = 3000, batch size = 512, 100 training epochs of simulated 5-month periods, early-stopping patience = 10, Dropout rate = 0.1 (Ch2).
- Thompson Sampling: 10,000 simulated customers, 9 strategies, Beta(successes+1, failures+1) per arm (Ch3).
- ARS: 16 best directions kept per update step, perturbations sampled from Gaussian N(0, sigma), ~1000 training steps (Ch5).

## Tells & smells

- A "reward" that's really just energy/cost *saved vs. a baseline*, not an absolute number -> you're in a parallel-baseline-tracking setup (Ch2) — make sure you're tracking both systems' state, not just the AI's.
- Seeing accuracy/reward reported without a Random Selection (or equivalent trivial) baseline alongside it -> the reported number is unverifiable; always compute Relative Return or an equivalent lift metric (Ch3).
- A "final score" reported with no regret curve -> can't tell if the algorithm is actually converging or got a lucky run — ask for the regret curve (Ch3).
- Hyperparameters (discount factor, layer sizes) presented as if derived from theory -> they're virtually always chosen empirically in this book's own account; expect "try, observe, adjust," not a formula (Ch1, Ch2, Ch6).
- A business AI pitch reaching for A3C, ARS, or Deep Convolutional Q-Learning -> pause and ask whether the simpler Ch1-3 case-study-style approach (tabular Q-Learning, DQN, or Thompson Sampling) would already solve it — these three are explicitly "closer to the state of the art" and "not necessarily adapted to business problems" (Ch5).
