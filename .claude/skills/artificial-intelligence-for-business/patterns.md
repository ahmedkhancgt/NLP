# Patterns

## The 5-Step AI Framework / Blueprint
**When to use**: starting any Deep Reinforcement Learning business problem from scratch.
**How**: build the Environment as a class (states, actions, rewards, update/reset methods) -> build the Brain (neural network) -> implement the Deep RL algorithm (DQN/Experience Replay) -> train the AI (epsilon-greedy exploration + early stopping) -> test the AI (deployment-mode environment that clips rather than ends on failure).
**Trade-offs**: highly reusable across business problems, but is explicitly tuned for Deep RL — not every business problem is a Deep RL problem (see Thompson Sampling below).

## Tabular Q-Learning for Small, Discrete State Spaces
**When to use**: a finite, enumerable set of states and actions small enough for a full table (here, 12x12).
**How**: encode states/actions as sequential integers; build a reward matrix by hand; initialize a Q-matrix to zero; repeatedly sample random (state, action) pairs, compute the Temporal Difference, and apply the Bellman equation.
**Trade-offs**: simple, fully interpretable, and needs no neural network — but doesn't scale once the state/action space becomes continuous or very large.

## Deep Q-Learning for Continuous/Large State Spaces
**When to use**: states are continuous-valued (e.g. temperature, user count) or too numerous for a lookup table.
**How**: replace the Q-table with a neural network (`Input -> Dense -> Dense -> Q-values`), paired with Experience Replay (store transitions, train on random batches) to avoid training only on highly-correlated consecutive transitions.
**Trade-offs**: scales to realistic business environments, but needs careful hyperparameter choices (memory size, batch size, exploration rate) and more compute than tabular Q-Learning.

## Automatic Goal Parameterization (vs. Manual Reward-Matrix Editing)
**When to use**: whenever the "goal" of a route/decision changes at runtime.
**How**: copy the base reward matrix inside the decision function itself and set the goal cell's reward there, rather than hand-editing the matrix for every new goal.
**Trade-offs**: trivial to implement, removes a whole class of manual-update bugs; the only cost is re-running the Q-Learning process per new goal.

## Waypoint Routing via Function Composition
**When to use**: forcing a path through an intermediary point without biasing the reward matrix.
**How**: call the same single-leg routing function twice (start -> intermediary, then intermediary -> end) and concatenate the results.
**Trade-offs**: far simpler and more robust than trying to automatically compute reward biases for intermediate points; costs only two extra lines of code.

## Parallel Baseline Tracking (AI vs. No-AI)
**When to use**: whenever you need to prove an AI's value against a status-quo system.
**How**: simulate both the AI-controlled system and the "unintelligent"/baseline system in parallel on identical inputs at every timestep, then define the reward/benefit as the measured difference between them.
**Trade-offs**: makes ROI directly measurable and auditable, at the cost of doubling the state-tracking logic in the environment.

## Epsilon-Greedy Exploration
**When to use**: training any DQN-style model, to avoid getting stuck exploiting an undertrained network's early, suboptimal predictions.
**How**: draw a random number each step; if below epsilon (e.g. 0.3), play a random action; otherwise play `argmax(Q(s))`.
**Trade-offs**: cheap and effective, but epsilon itself must be chosen (the book uses 30% without extensive tuning) and typically should decay over training in more advanced setups.

## Early Stopping + Dropout for Overfitting Control
**When to use**: any neural-network training loop showing signs of overfitting or running past the point of real improvement.
**How**: track best-so-far total reward/performance; stop after a "patience" number of epochs with no improvement. Separately, insert `Dropout(0.1)` after hidden Dense layers to randomly deactivate 10% of neurons per pass.
**Trade-offs**: both are nearly free to add in Keras and materially improve generalization; dropout rate and patience still require some judgment/experimentation.

## Thompson Sampling for Online, Stateless Option Selection
**When to use**: choosing the best of several discrete options (ads, strategies, treatments) via live, sequential feedback, with no evolving environment state.
**How**: maintain (successes, failures) counts per option; each round, sample `Beta(successes+1, failures+1)` per option and play the argmax; update the played option's count based on the observed outcome.
**Trade-offs**: needs zero hyperparameters (no learning rate, no discount factor) and converges fast, but only applies to bandit-shaped problems (no sequential state) — don't force it onto an MDP-shaped problem.

## Relative Return + Regret Curve Evaluation
**When to use**: proving a bandit/online-learning algorithm's real value, not just reporting a raw score.
**How**: run the algorithm and a trivial Random Selection baseline on the identical simulated data; report `(algorithm_reward - baseline_reward) / baseline_reward * 100` as Relative Return, and separately plot cumulative `(best_known_reward - algorithm_reward)` per round as the Regret Curve.
**Trade-offs**: requires building and maintaining a baseline and a simulated ground truth, but is the only way to distinguish "got lucky on this run" from "genuinely learned the best option."

## Linear Policy + Finite-Differences Gradient Approximation (ARS)
**When to use**: continuous-control problems with no explicit differentiable reward-vs-weights function available.
**How**: initialize a linear weight matrix to zero; sample small Gaussian perturbations, evaluate paired positive/negative perturbed policies over full episodes, keep the top-N directions by reward, and average `(r+ - r-) * perturbation` across them as the approximate gradient step.
**Trade-offs**: avoids needing an analytic gradient at all, and the linear policy keeps the search space small — but is restricted to a single-layer policy (no deep representational power) and requires many full-episode rollouts per update.

## Image-State Handling via Pooling-Free CNNs
**When to use**: when the environment's state is naturally an image (game screen, camera feed) and spatial/location information about objects matters.
**How**: pass the image through convolutional layers only (no pooling layers), flatten to a 1D vector, and feed that into the existing fully-connected DQN network.
**Trade-offs**: preserves the spatial information pooling would destroy, at the cost of a larger, more compute-intensive network than a hand-encoded state vector.
