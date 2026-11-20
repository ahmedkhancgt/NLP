# Chapter 18: Neural Networks

## Core Idea
Building from a single perceptron (which can represent AND/OR/NOT but not XOR) to feed-forward networks with hidden layers using a smooth sigmoid activation, the chapter implements backpropagation from scratch to train networks via gradient descent — solving XOR by hand-design, then re-learning it via training, then tackling "Fizz Buzz" purely through a trained network.

## Frameworks Introduced
- **Perceptron**: `perceptron_output(weights, bias, x) = step_function(dot(weights, x) + bias)` — a single artificial neuron that fires (1) or doesn't (0) based on a weighted sum crossing a threshold; geometrically, separates input space along a hyperplane.
  - When to use: as a building block / conceptual starting point; limited to linearly-separable functions (can do AND, OR, NOT; cannot do XOR).
- **Feed-forward neural network**: layers of neurons (`List[List[Vector]]` = layers of neurons of weight vectors), each neuron computing `sigmoid(dot(weights, inputs_with_bias))`, passing outputs forward layer by layer via `feed_forward`.
  - When to use: problems requiring nonlinear decision boundaries (like XOR) that a single perceptron cannot represent — stacking a hidden layer lets the network compute intermediate "features" (e.g., AND, OR) that the output layer recombines.
- **Backpropagation**: train network weights via gradient descent by (1) forward pass to get all neuron outputs, (2) compute squared-error loss at the output layer, (3) compute output-layer gradients, (4) propagate error backward to compute hidden-layer gradients via the chain rule, (5) gradient-descent step on every neuron's weights.
  - When to use: whenever hand-designing weights (as for the bespoke XOR network) isn't feasible — i.e., essentially always for real problems with many neurons.
- **Encoding non-numeric problems as vectors**: convert a classification problem (Fizz Buzz's 4 output classes) into a one-hot-style vector, and convert an input (an integer) into a representation (binary digits) that gives the hidden layer meaningful features to compute — raw integer magnitude is a poor input encoding.

## Key Concepts
- **Sigmoid (= logistic) activation**: smooth, differentiable approximation of the step function; chosen specifically because training via calculus-based gradient descent requires smooth (continuously differentiable) functions, which the discontinuous `step_function` lacks.
- **Hidden layer**: a layer between input and output that computes intermediate "features" of the input (e.g., AND and OR of the two inputs, in the hand-built XOR network) which the output layer then recombines.
- **Weights vector with bias folded in**: each neuron's weight vector has one more element than its number of inputs, representing the bias term, paired with a constant-1 "bias input" appended to every input vector.
- **Loss (sum of squared errors)**: the training objective backpropagation minimizes, comparing network output to a target vector.
- **argmax**: the index of the largest value in a vector — used to convert a network's multi-dimensional output into a single discrete class prediction.
- **One-hot-style output encoding**: representing a categorical target (e.g., "fizz", "buzz", "fizzbuzz", or neither) as a vector of 0s with a single 1 at the active class's position.

## Mental Models
- **XOR as the canonical "linear models aren't enough" example**: no single perceptron (linear separator) can represent XOR, because its positive and negative examples aren't linearly separable — motivating the entire hidden-layer/multi-layer architecture.
- **Hidden layers compute features, output layers combine them**: the hand-designed XOR network's hidden layer literally computes "AND" and "OR" of the inputs; the output layer computes "second but not first" of those two features — a concrete illustration of what "feature learning" means in a neural net.
- **Input encoding matters as much as architecture**: feeding Fizz Buzz's integer input directly (as a single scalar) would let the hidden layer compute only trivial scalar transformations; encoding as 10 binary digits gives the hidden layer discrete, combinable "features" (bit values) to work with — this design choice is as important as the network's structure.
- **Neural networks as (mostly) black boxes**: the book explicitly flags that "inspecting their details doesn't give you much understanding of how they're solving a problem," and that for most ordinary data science problems a neural network is probably not the right first choice compared to simpler, more interpretable models.

## Anti-patterns
- **Expecting a single perceptron to solve non-linearly-separable problems**: XOR is provably impossible for any single perceptron, however the weights/bias are chosen — requires a hidden layer instead.
- **Using the discontinuous step function as an activation when you need to train via gradient descent**: its non-differentiability breaks calculus-based training — use a smooth approximation (sigmoid) instead.
- **Feeding raw/poorly-chosen input encodings to a network**: e.g., a single scalar integer input for Fizz Buzz, where "intensity" (2 being twice 1) is irrelevant to the actual problem (divisibility) — binary-digit encoding works far better here.
- **Training/testing on the same range of inputs**: the book deliberately trains Fizz Buzz on numbers 101-1023 and evaluates on 1-100 specifically to avoid "cheating" by training on the exact numbers being evaluated.
- **Reaching for a neural network as a first choice**: the chapter explicitly says that for most problems a data scientist encounters, they're "probably not the right choice" given their black-box nature and training difficulty — reserve them for problems (image/handwriting recognition) that actually need them.

## Code Examples
```python
def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    calculation = dot(weights, x) + bias
    return step_function(calculation)

and_weights, and_bias = [2., 2], -3.
# fires (1) only when both inputs are 1
```
- **What it demonstrates**: a perceptron implementing logical AND via hand-chosen weights/bias.

```python
def feed_forward(neural_network, input_vector):
    outputs = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output
    return outputs

xor_network = [[[20., 20, -30], [20., 20, -10]],  # hidden: and, or
               [[-60., 60, -30]]]                  # output: 2nd-but-not-1st
```
- **What it demonstrates**: a hand-designed two-layer network solving XOR by computing AND and OR as hidden features, combined by the output layer.

```python
def sqerror_gradients(network, input_vector, target_vector):
    hidden_outputs, outputs = feed_forward(network, input_vector)
    output_deltas = [output * (1 - output) * (output - target)
                      for output, target in zip(outputs, target_vector)]
    # ... (output_grads, hidden_deltas, hidden_grads derived via chain rule)
    return [hidden_grads, output_grads]
```
- **What it demonstrates**: backpropagation's core gradient computation, chaining the output-layer error backward into the hidden layer.

## Reference Tables
| Gate | Perceptron-solvable? | Weights/bias example |
|---|---|---|
| AND | yes | `[2,2], bias=-3` |
| OR | yes | `[2,2], bias=-1` |
| NOT | yes | `[-2], bias=1` |
| XOR | no (needs hidden layer) | 2-layer network required |

| Fizz Buzz input/output encoding | Representation |
|---|---|
| Input (integer n) | 10-bit binary encoding (`binary_encode`) |
| Output (fizz/buzz/fizzbuzz/none) | 4-dim one-hot vector (`fizz_buzz_encode`) |
| Final prediction | `argmax` of the 4-dim output |

## Worked Example
A hand-designed `xor_network` (hidden layer: AND-neuron `[20,20,-30]`, OR-neuron `[20,20,-10]`; output layer: `[-60,60,-30]` computing "2nd but not 1st") correctly reproduces XOR for all four input combinations. The same XOR function is then *learned* from scratch via 20,000 epochs of gradient descent (learning rate 1.0) starting from random weights, converging to functionally equivalent weights (`[[7,7,-3],[5,5,-8]]` hidden, `[11,-12,-5]` output). For Fizz Buzz, training on binary-encoded integers 101-1023 (avoiding the 1-100 evaluation range) with a 10-25-4 network (10 inputs, 25 hidden units, 4 outputs) over 500 epochs achieves 96/100 correct predictions on numbers 1-100 — "well above the VP of Engineering's hiring threshold."

## Key Takeaways
1. A single perceptron can only represent linearly separable functions (AND, OR, NOT) — XOR requires at least one hidden layer.
2. Smooth activation functions (sigmoid) are required, not optional, for gradient-descent-based training — the step function's non-differentiability rules it out.
3. Backpropagation is just the chain rule applied layer by layer: compute output-layer gradients first, then propagate error backward into hidden layers.
4. How you encode inputs and outputs as vectors (e.g., binary digits, one-hot vectors) is often as important to a neural network's success as its architecture.
5. Neural networks are powerful but opaque and hard to train — reserve them for problems that genuinely need their representational power, not as a default tool.

## Connects To
- **Ch16 (Logistic Regression)**: the sigmoid/logistic function reused verbatim as the neuron activation function.
- **Ch8 (Gradient Descent)**: `gradient_step` is the literal training mechanism, applied per-neuron per-layer.
- **Ch4 (Linear Algebra)**: `dot` underlies every neuron's weighted-sum computation.
- **Ch19 (Deep Learning)**: directly continues and generalizes this chapter's neural network foundation.
- **TensorFlow / the author's "Fizz Buzz in Tensorflow" blog post**: external references for production-grade neural network frameworks.
