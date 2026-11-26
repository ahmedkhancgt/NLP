# Chapter 4: Artificial Neural Networks (Annex 1)

## Core Idea
A from-first-principles tour of the neural network machinery underlying the "Brain" built in Chapter 2's Deep Q-Learning model: the neuron, activation functions, forward/back-propagation, Gradient Descent and its variants, and the optimizers (ending at Adam) that actually drive weight updates.

## Frameworks Introduced
- **The artificial neuron**: inputs (scaled independent variables) x weights -> summed -> passed through an activation function -> output signal; weights are exactly what gets updated during learning.
  - When to use: the universal building block of every neural network in the book — understanding it is a prerequisite for understanding "the Brain."
- **Activation function selection blueprint**: Threshold (binary 0/1 outcome), Sigmoid (probability output / output layer), Rectifier (hidden layers, normalized [0,1] inputs), Hyperbolic Tangent (hidden layers, standardized [-1,1] inputs).
  - When to use: Sigmoid specifically for the final hidden-to-output step when you want a probability-like output; Rectifier or Tanh for hidden layers depending on whether inputs are normalized or standardized.
  - How: match the activation function to (a) what kind of output you need and (b) how your inputs are scaled — this is presented as a simple, memorizable decision rule, not a tuning exercise.
- **Forward-propagation / Back-propagation cycle**: forward-propagate inputs to get a prediction -> compute loss (Mean Squared Error) against the actual target -> back-propagate that loss right-to-left through the network -> apply Gradient Descent at every neuron to update weights in the loss-reducing direction -> repeat until early stopping or max iterations.
  - When to use: this is literally how every neural network in the book (including "the Brain") learns — the general training loop.
- **Gradient Descent**: an iterative, first-order optimization algorithm that follows the negative slope (first derivative) of a cost function to find its minimum — the "ball rolling down a bowl" intuition.
  - When to use: the universal technique connecting every ML/DL algorithm with learnable parameters, from linear regression up to deep networks; guarantees finding the *global* minimum only if the cost function is convex (a non-convex cost function risks getting stuck in a local minimum).
- **Batch vs. Stochastic vs. Mini-Batch Gradient Descent**: Batch feeds the whole dataset at once (fast, collective updates, but prone to local minima on non-convex costs); Stochastic feeds one row at a time (slower-looking per step, but avoids local minima and needs less memory, at the cost of being non-deterministic run-to-run); Mini-Batch feeds small batches, combining both benefits.
  - When to use: Batch when the cost function is convex and the dataset fits comfortably in memory; Stochastic when the cost is non-convex (the common case for real business problems) or memory/compute is limited; Mini-Batch as the practical default balancing speed and robustness.
- **Optimizer progression (Momentum -> Nesterov -> AdaGrad -> AdaDelta -> Adam)**: each optimizer is introduced specifically to fix the previous one's weakness, building up to Adam as the practical default.
  - When to use: Adam, in essentially all cases — it's described as "one of the most powerful optimizers" and is pre-implemented in every major Deep Learning library (Keras, TensorFlow, PyTorch).

## Key Concepts
- **Weight**: the learnable parameter on each synapse (input connection); larger weight = stronger signal; the thing Gradient Descent actually updates.
- **Activation function (phi)**: the function inside a neuron that converts the weighted sum of inputs into the neuron's output signal.
- **Loss / Cost function (C)**: the discrepancy between prediction and target; Mean Squared Error is the example used throughout the book (the same loss used for "the Brain" in Ch2).
- **Convex vs. non-convex cost function**: convex has exactly one global minimum (bowl-shaped, guaranteed to be found by plain Gradient Descent); non-convex has multiple local minima — "most problems, including business problems" have non-convex costs, which is precisely why Stochastic Gradient Descent matters.
- **Local minimum vs. global minimum**: a local minimum is a point the ball can get stuck at that isn't actually the lowest possible loss; the risk plain Gradient Descent runs on non-convex surfaces.
- **Momentum**: carries forward a fraction of the previous update's direction into the current step, speeding convergence and reducing oscillation, at the cost of risking overshoot past the minimum.
- **Adam (Adaptive Moment Estimation)**: stores momentum changes while computing a per-parameter learning rate — the end result of the Momentum -> Nesterov -> AdaGrad -> AdaDelta lineage, and the book's default optimizer choice.

## Mental Models
- **The neuron mirrors biology on purpose**: the book frames the entire field as "mimic how the human brain works ... in the hope that ... we are going to create something ... as awesome as that" — dendrites/axons map directly onto input/output signals, motivating *why* the architecture looks the way it does.
- **Gradient Descent is "the thing that connects all ML/DL algorithms"**: from simple linear regression to the deepest neural network, the same ball-in-a-bowl, follow-the-negative-slope idea is what makes an algorithm "learn" rather than be hand-tuned by an engineer.
- **Each optimizer exists to patch the previous one's specific flaw**: Momentum fixes SGD's oscillation but overshoots; Nesterov fixes the overshoot; AdaGrad adapts the learning rate but decays it to near-zero; AdaDelta fixes that decay — tracing this lineage makes "why Adam" an answer, not an assumption.
- **Stochastic "looks slower but is faster"**: processing one row at a time seems like it should be slower than the whole dataset at once, but in practice it avoids ever loading the full dataset into memory and avoids waiting for a full pass before any update — a classic case where the intuitive read of an algorithm's speed is wrong.

## Anti-patterns
- **Using plain (Batch) Gradient Descent on a non-convex cost function expecting the global minimum**: it is only guaranteed to find the global minimum when the cost function is convex — on the non-convex surfaces typical of real business problems, it can get stuck in a local minimum.
- **Using the Sigmoid activation inside hidden layers of a deep network**: the book's blueprint reserves Sigmoid specifically for the last hidden-to-output step (where a probability-like output is wanted); Rectifier/Tanh are the hidden-layer choices.
- **Picking Rectifier vs. Hyperbolic Tangent arbitrarily**: the choice should follow directly from how inputs are scaled — Rectifier when inputs are normalized ([0,1]), Hyperbolic Tangent when standardized ([-1,+1]) — not personal preference.
- **Reaching for Momentum or AdaGrad without acknowledging their known drawbacks**: Momentum risks overshoot and is less commonly pre-implemented; AdaGrad's ever-decreasing learning rate risks training effectively halting — know these trade-offs rather than treating any optimizer as a free upgrade over SGD.

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
        self.model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
```
- **What it demonstrates**: the exact same "Brain" code from Chapter 2, revisited here to show where the Adam optimizer (the end result of this chapter's optimizer lineage) and the MSE loss (this chapter's cost-function example) actually plug in.

## Reference Tables
| Activation function | Output shape | Typical placement |
|---|---|---|
| Threshold | Discontinuous, 0 or 1 | Binary (0/1) outcome prediction |
| Sigmoid | Continuous, (0,1) | Last hidden layer / output layer (probability-like) |
| Rectifier (ReLU) | Continuous, [0, infinity) | Hidden layers, normalized [0,1] inputs |
| Hyperbolic Tangent | Continuous, (-1,1) | Hidden layers, standardized [-1,1] inputs |

| Gradient Descent variant | Data fed per update | Risk of local minimum | Memory needs |
|---|---|---|---|
| Batch | Entire dataset | Higher (on non-convex cost) | Highest |
| Stochastic (SGD) | One row | Lower | Lowest |
| Mini-Batch | Small batch | Lower | Moderate |

| Optimizer | Key idea | Known drawback |
|---|---|---|
| Momentum | Carries forward prior update direction | Can overshoot the minimum |
| Nesterov | Reorders momentum calculation | Still less common in libraries |
| AdaGrad | Adapts learning rate per parameter | Learning rate decays toward 0 |
| AdaDelta | Fixes AdaGrad's decay issue | (stepping stone to Adam) |
| Adam | Stores momentum + per-parameter adaptive rate | None flagged — the book's default choice |

## Worked Example
A real-estate price prediction example (inputs: area, bedrooms, distance to city, age; target: price) walks through the full learn loop: forward-propagate the four inputs through weighted hidden-layer connections, get a predicted price, compute the Mean Squared Error against the real sale price, back-propagate that error right-to-left, and apply Gradient Descent at each neuron to nudge weights toward lower loss. The "ball in a bowl" visualization is then used literally: starting at an arbitrary position (initial weights), the first derivative (slope) of the cost function tells the ball which direction is downhill; a negative slope moves it right, a positive slope moves it left, and repeating this converges it to the bottom of the bowl (the minimum loss) in the convex case — while a non-convex 3D cost surface risks it settling into a local dip instead of the true global minimum.

## Key Takeaways
1. Every neural network in the book, including the Deep Q-Learning "Brain," is built from the same primitive: weighted inputs through an activation function.
2. Choose activation functions by a simple rule — Sigmoid for probability-like outputs, Rectifier/Tanh for hidden layers depending on input scaling, Threshold only for pure binary outcomes.
3. Gradient Descent is the single unifying technique behind essentially all trainable ML/DL algorithms — "learning" means iteratively finding the weights that minimize a cost function.
4. Most real (non-toy) cost functions are non-convex, which is exactly why Stochastic/Mini-Batch Gradient Descent (not plain Batch) matters in practice.
5. Default to the Adam optimizer unless you have a specific, well-understood reason to choose otherwise — it's the end of a deliberate lineage of fixes to earlier optimizers' weaknesses.

## Connects To
- **Ch2**: "the Brain" built there is a direct, concrete application of every concept in this chapter — the Dense layers, Sigmoid activations, MSE loss, and Adam optimizer all appear verbatim in Ch2's implementation.
- **Ch5 (Annex 2)**: Deep Convolutional Q-Learning and A3C both build further neural-network architectures on top of these same fundamentals.
- **Keras / TensorFlow / PyTorch**: the external Deep Learning libraries that implement all of this chapter's activation functions and optimizers out of the box, referenced directly as where Adam "is always pre-implemented."
