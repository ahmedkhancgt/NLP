# Glossary

**A3C (Asynchronous Advantage Actor-Critic)** — multi-agent Deep RL method; several agents share actor/critic networks and update them asynchronously (Ch5).

**Action** — the next move an agent can take from a given state; encoded as an integer index in Ch1-2, as a continuous vector in ARS (Ch5).

**Activation function (phi)** — the function inside a neuron converting the weighted input sum into an output signal; Threshold, Sigmoid, Rectifier, Hyperbolic Tangent are the four covered (Ch4).

**Actor / Critic** — in A3C, the actor predicts Q(s,a) and decides actions; the critic predicts V(s), the value of a state (Ch5).

**Adam Optimizer** — Adaptive Moment Estimation; the book's default optimizer, combining momentum with a per-parameter adaptive learning rate (Ch4).

**Advantage** — `Q(s,a) - V(s)`, the gap between the actor's and critic's predictions in A3C (Ch5).

**AI Framework / Blueprint (5-step)** — Build Environment -> Build Brain -> Implement DQN -> Train -> Test; the reusable template for any Deep RL business problem (Ch2).

**Argmax method** — select the action with the single highest Q-value; preferred for simple/business problems (Ch1, Ch2, Ch6).

**ARS (Augmented Random Search)** — gradient-free Policy Gradient method using a linear policy and finite-differences gradient approximation (Ch5).

**Backpropagation** — propagating the loss error from output back through the network to update weights (Ch4).

**Batch Gradient Descent** — feeding the entire dataset at once before each weight update (Ch4).

**Bellman equation** — `Q(s,a) <- Q(s,a) + alpha * TD(s,a)`; the core Q-Learning update rule (Ch1).

**Beta distribution** — the per-strategy distribution Thompson Sampling samples from, shaped by accumulated successes/failures (Ch3).

**Convex / non-convex cost function** — convex has one global minimum (always findable by Gradient Descent); non-convex may trap Gradient Descent in a local minimum (Ch4).

**Deep Convolutional Q-Learning** — Deep Q-Learning with a pooling-free CNN front end for image-based states (Ch5).

**Deep Q-Learning (DQN)** — Q-Learning where a neural network, not a lookup table, predicts Q-values (Ch2).

**Direction boundary** — `(number_actions - 1) / 2`; splits a discrete action index into a heating/cooling (or equivalent) direction (Ch2).

**Discount factor (gamma)** — weights future reward vs. immediate reward in the Bellman equation; found via experimentation (Ch1, Ch6).

**Dropout** — regularization technique randomly deactivating a fraction of neurons during training to prevent overfitting (Ch2).

**Early stopping** — halting training once reward stops improving for a set "patience" window (Ch2).

**Environment (class)** — the object-oriented container for a case study's states, actions, rewards, and update/reset logic (Ch2).

**Epsilon-greedy exploration** — playing a random action a fixed fraction of the time (e.g. 30%) during training instead of always exploiting the current best-known action (Ch2).

**Experience Replay** — storing a capped buffer of past transitions and training on random batches from it, rather than only the latest transition (Ch2).

**Finite differences (method of)** — approximating a gradient by observing reward differences under small positive/negative weight perturbations (Ch5).

**Gradient Descent** — iterative first-order optimization algorithm following a cost function's negative slope to its minimum (Ch4).

**Learning rate (alpha)** — controls the step size of a weight/Q-value update (Ch1, Ch4, Ch5).

**Markov Decision Process (MDP)** — tuple (S, A, T, R) of states, actions, transition rule, and reward function; assumes no memory of history (Ch1).

**Mean Squared Error (MSE)** — the loss function comparing predicted and target values, used throughout the book's neural networks (Ch2, Ch4).

**Mini-Batch Gradient Descent** — feeding small batches of data per update, combining Batch and Stochastic Gradient Descent's benefits (Ch4).

**Momentum (optimizer)** — carries forward a fraction of the prior update's direction to speed convergence, at the risk of overshoot (Ch4).

**Multi-Armed Bandit / Online Learning** — the branch of AI with no states/actions to define, just "rounds" trying options and observing rewards (Ch3).

**Plan vs. Policy** — the plan is defining the environment; the policy is the function mapping states to actions (Ch1, Ch6).

**Policy function (pi)** — maps a state to the action to play; the optimal policy pi* maximizes accumulated reward (Ch1).

**Q-value (Q(s,a))** — the learned value of taking action a in state s; initialized to 0, updated via the Bellman equation (Ch1).

**Random Selection** — the trivial benchmark algorithm (uniformly random choice) Thompson Sampling is measured against (Ch3).

**Regret curve** — plot of (best-known cumulative reward) minus (algorithm's cumulative reward) over rounds; shows convergence (Ch3).

**Relative Return** — percentage improvement of one algorithm's total reward over another's, on the same simulation (Ch3).

**Reward function (R)** — maps (state, action) to a numeric reward; built as a matrix in Ch1-2, as a 0/1 outcome in Ch3 (Ch1-3).

**Softmax method** — samples an action from a probability distribution over Q-values, rather than always taking the max; used for complex/exploration-heavy problems (Ch1, Ch2, Ch6).

**State** — the input describing the current situation (e.g. a location, a server's temperature/users/data rate) (Ch1-3).

**Stochastic Gradient Descent (SGD)** — updates weights after every single training example, avoiding local minima at the cost of being non-deterministic (Ch4).

**Temporal Difference (TD)** — `R(s,a) + gamma*max(Q(s',.)) - Q(s,a)`; the "surprise" signal driving Q-value updates (Ch1-2).

**Thompson Sampling** — online-learning algorithm sampling per-option Beta distributions and playing whichever samples highest (Ch3).
