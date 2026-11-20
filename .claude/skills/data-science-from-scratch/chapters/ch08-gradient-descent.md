# Chapter 8: Gradient Descent

## Core Idea
Gradient descent is the from-scratch optimization workhorse used throughout the book to fit model parameters: compute the gradient of a loss function, take small steps against it, and repeat until parameters converge — with batch, minibatch, and stochastic variants trading off update frequency against gradient noise.

## Frameworks Introduced
- **Gradient descent (batch)**: pick a random starting vector, repeatedly compute the gradient of the objective over the *entire* dataset, step against it (`-step_size * gradient`), until convergence.
  - When to use: minimizing a differentiable loss function over small-to-moderate datasets where a full-dataset gradient is affordable.
- **Stochastic gradient descent (SGD)**: compute the gradient and update parameters using *one* training example at a time.
  - When to use: large datasets / expensive gradient computations; converges in far fewer epochs on simple problems, at the cost of noisier per-step direction.
- **Minibatch gradient descent**: compute the gradient (and step) on a sampled sub-batch of the dataset.
  - When to use: the practical middle ground between batch and stochastic — common batch sizes trade compute cost against gradient-direction stability.
- **Difference-quotient gradient estimation**: `(f(x+h) - f(x)) / h` approximates a derivative without calculus; `partial_difference_quotient` extends this per-dimension to estimate a full gradient numerically.
  - When to use: only as a fallback/sanity-check — explicitly called out as computationally expensive (2n evaluations for an n-dimensional gradient) compared to deriving the gradient analytically, which the book does everywhere else.

## Key Concepts
- **Gradient**: the vector of partial derivatives; points in the direction of steepest increase of the function.
- **Loss function**: measures how poorly a model with given parameters fits the data (smaller is better); gradient descent minimizes it.
- **theta**: the book's convention for the parameter vector being learned (e.g., `[slope, intercept]`).
- **Epoch**: one full pass through the dataset during training.
- **Learning rate / step size**: scalar multiplier controlling how far each gradient step moves; too small is slow, too big risks divergence or overshooting.
- **Local vs. global minimum**: gradient descent on a function with multiple local minima may converge to the wrong one — mitigated by rerunning from different random starting points.

## Mental Models
- **Analytic gradients over numerical estimation**: once you know calculus, derive `grad` directly (e.g., `sum_of_squares_gradient(v) = [2*v_i for v_i in v]`) rather than paying the 2n-evaluation cost of `estimate_gradient`; the book uses analytic gradients throughout.
- **Step-size as an art, not a science**: three options — fixed step size, shrinking step size over time, or (rarely, since it's costly) exactly minimizing the objective along the gradient direction at each step; experimentation is expected.
- **Terminology is inconsistent across the field**: "batch gradient descent" = full dataset per step; some people call the minibatch version "stochastic," with one-point-at-a-time being a special case of minibatch of size 1 — know the ambiguity exists.
- **Update-frequency vs. direction-accuracy trade-off**: smaller batches let you take more steps per unit of compute, but each step's gradient is a noisier estimate of the true dataset-wide gradient.

## Anti-patterns
- **Using numerical gradient estimation in production loops**: `estimate_gradient` costs `2n` function evaluations per gradient; always derive and hard-code the analytic gradient when feasible.
- **Picking a step size without experimentation**: too small wastes compute converging slowly; too big can make the loss diverge or become undefined.
- **Running batch gradient descent naively on huge datasets**: requires evaluating the gradient over the full dataset before any update — minibatch/stochastic variants exist specifically to avoid this bottleneck.
- **Assuming a single run finds the global minimum**: on multimodal objectives, gradient descent can get stuck in a local minimum; rerun from multiple random starts if this is a concern.

## Code Examples
```python
def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

v = [random.uniform(-10, 10) for i in range(3)]
for epoch in range(1000):
    grad = sum_of_squares_gradient(v)
    v = gradient_step(v, grad, -0.01)
```
- **What it demonstrates**: the canonical from-scratch gradient-descent loop minimizing `sum_of_squares`, converging to `[0,0,0]`.

```python
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = (predicted - y)
    grad = [2 * error * x, 2 * error]
    return grad
```
- **What it demonstrates**: the gradient of squared error for a simple linear model `y = slope*x + intercept`, used to fit slope/intercept via gradient descent (foreshadows Ch14).

```python
def minibatches(dataset, batch_size, shuffle=True):
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle: random.shuffle(batch_starts)
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]
```
- **What it demonstrates**: a generic, type-parameterized (`TypeVar('T')`) minibatch generator reused for minibatch gradient descent.

## Reference Tables
| Variant | Gradient computed over | Updates per epoch | Direction stability |
|---|---|---|---|
| Batch GD | entire dataset | 1 | most stable, slowest per epoch |
| Minibatch GD | a sampled sub-batch | len(dataset)/batch_size | middle ground |
| Stochastic GD (SGD) | 1 example | len(dataset) | noisiest, often fastest convergence in practice |

## Worked Example
Fitting `y = 20x + 5` from 100 synthetic points (`x` in [-50, 49]) via gradient descent on mean squared error: starting from random `theta = [slope, intercept]`, 5,000 epochs of batch gradient descent at `learning_rate=0.001` recovers `slope ≈ 20`, `intercept ≈ 5` (checked with `assert 19.9 < slope < 20.1`). The same problem solved with minibatches of size 20 over 1,000 epochs, or with stochastic gradient descent (one point at a time) over just 100 epochs, both converge to the same parameters — SGD needing far fewer epochs on this simple problem.

## Key Takeaways
1. Gradient descent is the book's universal optimization tool: define a differentiable loss, compute its gradient, step against it repeatedly.
2. Prefer analytic gradients over numerical difference-quotient estimation whenever you can derive them — numerical estimation costs `2n` evaluations per gradient.
3. Choose batch vs. minibatch vs. stochastic gradient descent based on dataset size and the update-frequency/gradient-noise trade-off; SGD often converges faster in practice despite noisier steps.
4. Step size (learning rate) must be tuned experimentally — there's no universal "right" value.
5. Multimodal objectives risk convergence to a local (not global) minimum; consider multiple random restarts.

## Connects To
- **Ch4**: reuses `Vector`, `add`, `scalar_multiply`, `distance`, `vector_mean` directly.
- **Ch14 (Simple Linear Regression)** and **Ch15 (Multiple Regression)**: `linear_gradient` here is exactly the gradient used to fit regression coefficients.
- **Ch16 (Logistic Regression)**, **Ch18-19 (Neural Networks/Deep Learning)**: all use gradient descent (often minibatch/stochastic) as the core training loop, with different loss functions and gradients.
- **Sebastian Ruder's gradient descent survey**: external resource the book points to for a deeper comparison of gradient descent variants (momentum, Adam, etc., covered informally later in Ch19).
