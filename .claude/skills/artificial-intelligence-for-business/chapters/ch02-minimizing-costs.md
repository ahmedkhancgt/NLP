# Chapter 2: Minimizing Costs (Deep Q-Learning)

## Core Idea
Modeled directly on DeepMind's real 40%-energy-saving data-center cooling system, this case study builds a Deep Q-Learning (DQN) agent that regulates a server's temperature to save energy vs. an "unintelligent" baseline cooling system — introducing the general 5-step AI Framework/Blueprint (Environment, Brain, DQN algorithm, Training, Testing) reused for every later Deep RL case study in the book.

## Frameworks Introduced
- **Deep Q-Learning (DQN)**: replace Q-Learning's lookup table with a neural network that takes the state vector as input and outputs a Q-value per action; the chosen action is `argmax` (or Softmax) over those outputs.
  - When to use: when the state space is continuous or too large for a table (here: temperature, user count, data rate — all continuous-valued).
  - How: prediction = `Q(s_t, a_t)`; target = `r_t + gamma * max(Q(s_t+1, a))`; loss = squared Temporal Difference (effectively Mean Squared Error), backpropagated to update network weights.
- **Experience Replay**: store the last `m` transitions `(s_t, a_t, r_t, s_t+1)` in a memory buffer; train on *random batches* sampled from that memory, not just the single most recent transition.
  - When to use: always, for DQN — consecutive transitions are highly correlated, so training only on the latest one teaches the network very little.
  - How: `remember()` appends to a capped-size list (oldest dropped once full); `get_batch()` samples a random subset and builds matched `(inputs, targets)` arrays from it.
- **The General AI Framework / Blueprint (5 steps)**: Step 1 Build the Environment (as a class) -> Step 2 Build the Brain (the neural network) -> Step 3 Implement the Deep RL algorithm (DQN) -> Step 4 Train the AI -> Step 5 Test the AI.
  - When to use: as the template for *any* Deep Reinforcement Learning business problem, not just this one — the book reuses this exact 5-file structure for every subsequent case study.
  - How: code sections that are part of the reusable "general framework" are written in CAPITAL LETTERS; code specific to this one case study is in regular case — a convention for separating "always do this" from "this problem's specifics."
- **Exploration vs. Exploitation (epsilon-greedy)**: during training, play a *random* action some fraction of the time (here epsilon = 0.3, i.e. 30%) instead of always trusting the network's current prediction.
  - When to use: always during DQN training — pure exploitation from an undertrained network gets stuck in early, suboptimal behavior.
  - How: draw a random number; if below epsilon, pick a random action; otherwise take `argmax(model.predict(state))`.
- **Early stopping**: stop training once performance (total reward) hasn't improved for a set number of epochs ("patience"), rather than always running the full epoch count.
  - When to use: any neural-network training loop, to prevent wasted compute and overfitting.
  - How: track `best_total_reward`; if `patience` epochs pass with no new best, halt training early.
- **Dropout (regularization)**: during each forward/backward pass, randomly deactivate a fraction of neurons (e.g. 10%) so the network can't over-rely on any single neuron.
  - When to use: whenever a network shows signs of overfitting the training data.
  - How: insert a `Dropout(0.1)` layer after each `Dense` hidden layer in Keras.

## Key Concepts
- **State vector**: here, 3 continuous values — server temperature, number of active users, data transmission rate — scaled to [0,1] before being fed to the network.
- **Action**: one of 5 discrete temperature changes, from -3C to +3C in 1.5C steps; `direction_boundary = (number_actions - 1) / 2` splits actions into "cooling" vs. "heating" directions.
- **Reward**: energy saved by the AI vs. the baseline "unintelligent" cooling system at each timestep — `energy_noai - energy_ai`, scaled by 1e-3.
- **The Brain**: a fully connected neural network (3 inputs -> 64 neurons -> 32 neurons -> 5 Q-value outputs), trained with Mean Squared Error loss and the Adam optimizer.
- **Assumption 1 (linear server temperature model)**: server temperature = a linear function of atmospheric temperature, number of users, and data rate — a simplification the book explicitly flags as a modeling assumption, not an AI-derived result.
- **Assumption 2 (linear energy-cost model)**: the energy any system spends to change the server's temperature is proportional to the absolute temperature change it causes.
- **Game over condition (training only)**: if the AI lets the server's temperature fall outside [-20C, 80C] during training, the epoch ends early as a failure signal; during testing, the environment instead clips the temperature and keeps running (production can't just "end the game").
- **Epochs of 5-months**: one training epoch simulates 5 months of minute-by-minute server operation; the model is trained over 100 such epochs, then tested over one full year.

## Mental Models
- **The Blueprint separates "always do this" from "this problem's specifics"**: capitalized code-section titles for framework steps vs. lowercase for case-study-specific logic is a deliberate signal for which parts of the code transfer directly to a new business problem.
- **Two systems are evaluated in parallel, not sequentially**: the environment tracks `temperature_ai` and `temperature_noai` simultaneously at every timestep, so the AI's performance is always measured against what the dumb baseline would have done in the exact same conditions — this is what makes "energy saved" a meaningful, comparable number.
- **Modeling assumptions are explicitly separated from the AI itself**: the book is careful to say that the linear temperature/energy models (Assumptions 1 and 2) are simplifications made "to focus the maximum on the AI solution," not something intrinsic to Deep Q-Learning — swap in a different (e.g. quadratic) relationship and the same DQN approach still applies.
- **Training and testing environments behave differently on purpose**: a training episode can "fail" and end early (game over) to penalize bad policies quickly; a deployed/testing environment instead just clips temperature at its bounds and keeps running, since a real server can't simply stop existing.

## Anti-patterns
- **Training on only the single most recent transition**: `s_t` and `s_t+1` are highly correlated, so naive online training barely teaches the network anything — always use Experience Replay with random batch sampling instead.
- **Running the full epoch count regardless of convergence**: wastes compute and risks overfitting once performance has plateaued — use early stopping with a patience threshold.
- **Treating the linear temperature/energy assumptions as AI-derived facts**: they are deliberate simplifying assumptions chosen by the engineer (here, a linear regression with coefficients just assumed/plugged in), explicitly flagged as swappable for a more realistic (e.g. nonlinear) relationship.
- **Letting training and testing share the same "game over" logic**: using a hard episode-ending failure condition in a live/testing environment would mean the AI "crashes" instead of recovering — clip values and keep running in testing/production instead.

## Code Examples
```python
class Brain(object):
    def __init__(self, learning_rate=0.001, number_actions=5):
        self.learning_rate = learning_rate
        states = Input(shape=(3,))
        x = Dense(units=64, activation='sigmoid')(states)
        y = Dense(units=32, activation='sigmoid')(x)
        q_values = Dense(units=number_actions, activation='softmax')(y)
        self.model = Model(inputs=states, outputs=q_values)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
```
- **What it demonstrates**: the full "Brain" (DQN's neural network) built in a handful of lines via Keras's functional API — two hidden layers, Q-values as output, MSE loss, Adam optimizer.

```python
class DQN(object):
    def __init__(self, max_memory=100, discount=0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        inputs = np.zeros((min(len_memory, batch_size), num_inputs))
        targets = np.zeros((min(len_memory, batch_size), num_outputs))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            Q_sa = np.max(model.predict(next_state)[0])
            targets[i, action] = reward if game_over else reward + self.discount * Q_sa
        return inputs, targets
```
- **What it demonstrates**: Experience Replay's capped memory buffer and random-batch sampling, producing matched input/target arrays for one training step.

```python
epsilon = .3
direction_boundary = (number_actions - 1) / 2
if np.random.rand() <= epsilon:
    action = np.random.randint(0, number_actions)   # exploration
else:
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])                  # exploitation
direction = -1 if (action - direction_boundary < 0) else 1
energy_ai = abs(action - direction_boundary) * temperature_step
```
- **What it demonstrates**: epsilon-greedy action selection and decoding a discrete action index into a (direction, energy) pair the environment can apply.

## Reference Tables
| Action index | Effect |
|---|---|
| 0 | Cool down by 3C |
| 1 | Cool down by 1.5C |
| 2 | No temperature change |
| 3 | Heat up by 1.5C |
| 4 | Heat up by 3C |

| Hyperparameter | Value used |
|---|---|
| Hidden layers | 64 neurons, then 32 neurons |
| Loss function | Mean Squared Error |
| Optimizer | Adam (learning rate 0.001) |
| Discount factor (DQN) | 0.9 |
| Exploration rate (epsilon) | 0.3 |
| Max memory (Experience Replay) | 3000 transitions |
| Batch size | 512 |
| Training epochs | 100, each a simulated 5-month period |
| Early-stopping patience | 10 epochs |
| Dropout rate | 0.1 (10%) per hidden layer |

## Worked Example
At t=4:00pm the server is at 28C (both with and without AI). The AI predicts action 1 (cool by 1.5C), spending `energy_ai = 1.5`. The baseline "no-AI" cooling system would have brought the temperature from 28C only down to the nearest optimal-range bound (24C), spending `energy_noai = 4`. Reward = `energy_noai - energy_ai = 2.5`. One minute later, intrinsic conditions (new users, new data) raise the temperature further by 5C on both tracks; the AI's temperature becomes 29.5C, the no-AI temperature becomes 27C — tracked completely separately. Over the next minute the AI cools by 3C to 28.5C (reward 2) while the no-AI system (now at 29C, still above the 24C bound) spends more energy correcting it. Total reward over these 2 minutes: 4.5. The same process repeats every minute across 100 training epochs of simulated 5-month periods, then over a full year of testing — the final model is reported to meet DeepMind's real-world benchmark of roughly 40% energy savings vs. the unintelligent baseline.

## Key Takeaways
1. Deep Q-Learning swaps Q-Learning's table for a neural network, letting the same Bellman-equation logic handle continuous/large state spaces.
2. Experience Replay (store many past transitions, train on random batches) is what makes DQN training stable — training on only the latest transition fails due to correlation.
3. The 5-step AI Framework (Environment, Brain, DQN, Train, Test) is the book's reusable blueprint for every Deep RL business problem from here on.
4. Epsilon-greedy exploration and early stopping are both small, cheap additions that materially improve training — explore 20-30% of the time during training, and stop once reward stops improving.
5. Keep "no-AI baseline" state tracked in parallel with "AI" state throughout, so the agent's value is always measurable against a concrete alternative.
6. Clearly separate modeling assumptions (how temperature/energy relate) from the learning algorithm itself — the former can be swapped for more realistic functions without touching the DQN logic.

## Connects To
- **Ch1**: generalizes tabular Q-Learning into Deep Q-Learning once the state space becomes continuous.
- **Ch4 (Annex 1)**: the Dense layers, activation functions, and backpropagation used in "the Brain" are explained in full in the neural-network annex.
- **Ch5 (Annex 2)**: Deep Convolutional Q-Learning and A3C both extend this same DQN foundation to image-based states and parallel/asynchronous training.
- **Ch6 (Annex 3)**: the Q&A section directly revisits this chapter's Argmax-vs-Softmax choice and the 64/32-neuron architecture, including why Early Stopping and Dropout were specifically chosen to prevent overfitting.
