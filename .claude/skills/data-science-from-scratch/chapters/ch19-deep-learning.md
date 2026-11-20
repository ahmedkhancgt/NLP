# Chapter 19: Deep Learning

## Core Idea
The chapter generalizes Chapter 18's hand-rolled neural network into a reusable, composable deep learning framework from scratch: Tensors (nested lists), a `Layer` abstraction (forward/backward/params/grads), `Sequential` composition, pluggable `Loss` and `Optimizer` abstractions, additional activations (Tanh, Relu), softmax + cross-entropy for classification, dropout for regularization, and a full MNIST digit-classification example with model save/load.

## Frameworks Introduced
- **Tensor as nested list, with recursive helpers**: `Tensor = list`; `shape`, `is_1d`, `tensor_sum`, `tensor_apply`, `tensor_combine` recursively handle any-dimensional nested lists uniformly.
  - When to use: representing any n-dimensional array in a from-scratch setting without numpy.
- **Layer abstraction**: `forward(input)`, `backward(gradient)`, `params()`, `grads()` — the uniform interface every network component implements, enabling composability.
  - When to use: building any reusable neural network component (activation, linear transform, dropout).
- **Sequential composition**: `Sequential(layers)` chains layers forward in order and backpropagates gradients in reverse, itself behaving as a `Layer`.
  - When to use: building arbitrarily deep/complex networks by stacking simpler layers.
- **Loss and Optimizer abstractions**: `Loss.loss`/`Loss.gradient` decouple the objective from training; `Optimizer.step(layer)` decouples the update rule (plain gradient descent, or `Momentum`) from the loss and architecture.
  - When to use: experimenting with different objectives (SSE vs. SoftmaxCrossEntropy) or update rules (GradientDescent vs. Momentum) without rewriting the training loop.
- **Weight initialization schemes**: `random_uniform`, `random_normal`, and `xavier` (variance `2/(num_inputs+num_outputs)`-style, via `len(dims)/sum(dims)`) — initialization can be the difference between a network training at all and not.
- **Softmax + cross-entropy for classification**: `softmax(tensor)` turns raw outputs into a probability distribution (numerically stabilized by subtracting the max before `exp`); `SoftmaxCrossEntropy` loss computes negative log-likelihood and has a remarkably simple gradient (`predicted_probability - actual`).
  - When to use: multi-class classification output layers — strictly preferred over sigmoid-output + SSE for classification because it trains faster and interprets outputs as true probabilities.
- **Dropout**: randomly zero each neuron's output with probability `p` during training (forcing the network not to over-rely on any single neuron), scale outputs by `(1-p)` at evaluation time.
  - When to use: regularizing deep networks against overfitting, as an alternative/complement to weight-penalty regularization (Ch15).

## Key Concepts
- **Tensor**: an n-dimensional array, represented here as a (possibly nested) Python list; `shape()` recursively measures dimensions.
- **Xavier initialization**: weight initialization scaled to the layer's fan-in/fan-out, chosen because it "often works nicely" for training stability — poor initialization can make some networks in the chapter fail to train at all.
- **Slice assignment (`param[:] = ...`)**: necessary for in-place parameter updates, since reassigning a Python variable (`param = ...`) wouldn't change the original list referenced by the layer's internal state.
- **Momentum**: optimizer variant that maintains a running (exponentially-weighted) average of past gradients and steps in that average's direction, smoothing out noisy individual gradient updates.
- **Tanh / ReLU activations**: tanh ranges (-1,1), outputs 0 at input 0 (fixing sigmoid's "always-positive-output-at-zero" quirk); ReLU (`max(x,0)`) avoids sigmoid's vanishing-gradient saturation problem in large networks.
- **Cross-entropy / negative log-likelihood**: the loss that corresponds to maximum likelihood estimation when network outputs are probabilities (softmax outputs) — mirrors Ch14's MLE justification for least squares.
- **One-hot encoding**: representing a categorical label as a vector with a single 1 at the correct class's index, used for MNIST's 10 digit classes.

## Mental Models
- **Abstraction enables experimentation**: separating Layer/Loss/Optimizer lets you swap SSE for SoftmaxCrossEntropy, or GradientDescent for Momentum, without touching the training loop — a direct payoff of good software design applied to ML code.
- **Softmax has "more ways to win" than sigmoid+SSE**: to predict class 0 under sigmoid, you need the first output large-positive AND all others large-negative; under softmax, you only need the first output *larger than the others* — a much easier target, which is why SoftmaxCrossEntropy trains faster.
- **Dropout prevents over-reliance on individual neurons**: by randomly "deleting" neurons during training, no single neuron can become a single point of failure the rest of the network depends on — a form of built-in redundancy/ensembling within one network.
- **From-scratch tensors don't scale**: the chapter explicitly acknowledges its lists-as-tensors approach is "very slow" compared to real tensor libraries (which use true matrix multiplication), and that convolutional layers (needed to beat MNIST's best models) are "out of scope."

## Anti-patterns
- **Reassigning a parameter variable instead of slice-assigning**: `param = tensor_combine(...)` only rebinds the local name and leaves the layer's actual stored tensor unchanged — must use `param[:] = ...` for in-place mutation to propagate.
- **Using sigmoid-output + SSE loss for multi-class classification**: works, but trains much slower than softmax + cross-entropy because it demands large-positive-AND-large-negative outputs simultaneously rather than just relative ordering.
- **Ignoring weight initialization**: the book notes some networks "couldn't get to train at all with different initializations" — treat initialization scheme as a real hyperparameter, not an afterthought.
- **Calling `Dropout.backward` outside training mode**: explicitly raises `RuntimeError`, since dropout's backward pass only makes sense when a mask was actually applied during the forward pass.
- **Computing `exp()` on raw large values in softmax**: `exp(1000)` overflows; always subtract the max value first for numerical stability.

## Code Examples
```python
class Layer:
    def forward(self, input): raise NotImplementedError
    def backward(self, gradient): raise NotImplementedError
    def params(self): return ()
    def grads(self): return ()

class Sequential(Layer):
    def __init__(self, layers): self.layers = layers
    def forward(self, input):
        for layer in self.layers: input = layer.forward(input)
        return input
    def backward(self, gradient):
        for layer in reversed(self.layers): gradient = layer.backward(gradient)
        return gradient
```
- **What it demonstrates**: the uniform Layer interface and how Sequential composes arbitrary layers for both forward and backward passes.

```python
def softmax(tensor: Tensor) -> Tensor:
    if is_1d(tensor):
        largest = max(tensor)
        exps = [math.exp(x - largest) for x in tensor]
        sum_of_exps = sum(exps)
        return [exp_i / sum_of_exps for exp_i in exps]
    else:
        return [softmax(tensor_i) for tensor_i in tensor]

class SoftmaxCrossEntropy(Loss):
    def gradient(self, predicted, actual):
        probabilities = softmax(predicted)
        return tensor_combine(lambda p, actual: p - actual, probabilities, actual)
```
- **What it demonstrates**: numerically-stable softmax and the remarkably simple `predicted_prob - actual` gradient of softmax cross-entropy.

```python
class Dropout(Layer):
    def __init__(self, p): self.p, self.train = p, True
    def forward(self, input):
        if self.train:
            self.mask = tensor_apply(lambda _: 0 if random.random() < self.p else 1, input)
            return tensor_combine(operator.mul, input, self.mask)
        else:
            return tensor_apply(lambda x: x * (1 - self.p), input)
```
- **What it demonstrates**: training-time random masking vs. evaluation-time uniform scaling for dropout regularization.

## Reference Tables
| Activation | Range | Notes |
|---|---|---|
| Sigmoid | (0,1) | outputs 0.5 at input 0; saturates for large\|x\| |
| Tanh | (-1,1) | outputs 0 at input 0; still saturates |
| ReLU | [0,inf) | `max(x,0)`; avoids saturation for positive inputs |

| Model (MNIST) | Architecture | Test accuracy |
|---|---|---|
| Multiclass logistic regression | `Linear(784,10)` + softmax | ~89% |
| Deep net w/ dropout | `784->30->10->10`, Tanh, Dropout(0.1) | >92% |

| Loss | Pairs with | Notes |
|---|---|---|
| SSE | Sigmoid output | works but slower to train for classification |
| SoftmaxCrossEntropy | raw (no final activation) output | faster training, true probability interpretation |

## Worked Example
XOR is retrained using the new framework (`Sequential([Linear(2,2), Sigmoid(), Linear(2,1)])`, `GradientDescent(0.1)`, `SSE()`, 3000 epochs) and converges to a different but functionally equivalent solution: `hidden1` approximates NOR, `hidden2` approximates AND, and the output combines them into NOR-of-those, which is exactly XOR. Fizz Buzz is retrained twice: once with Sigmoid output + SSE loss + Momentum optimizer (1000 epochs, 90% test accuracy), and again with no final activation + SoftmaxCrossEntropy loss (only 100 epochs needed, comparable or better accuracy) — demonstrating softmax's faster convergence. The MNIST example flattens 28x28 images to 784-dim vectors, recenters/rescales pixels, one-hot-encodes the 10 digit labels, trains first a baseline multiclass logistic regression (89% accuracy) then a deeper `784->30->10->10` network with Tanh activations and Dropout(0.1) on both hidden layers (>92% accuracy) — illustrating dropout's `train` flag toggling between training (random masking) and evaluation (uniform scaling) modes.

## Key Takeaways
1. A clean Layer/Loss/Optimizer abstraction lets you mix and match activations, losses, and update rules without rewriting training loops — good software design pays off directly in ML experimentation speed.
2. Weight initialization (uniform, normal, Xavier) is a real hyperparameter that can determine whether a network trains at all.
3. Prefer softmax + cross-entropy over sigmoid + SSE for multi-class classification — it trains faster because it only needs relative output ordering, not absolute large-positive/large-negative values.
4. Dropout regularizes by forcing redundancy: randomly disabling neurons during training prevents any one neuron from becoming a single point of failure.
5. From-scratch list-based tensors are illustrative but slow; real deep learning needs true tensor libraries (and architectures like convolutional layers) that are out of this book's scope.
6. Save/load model weights (e.g., via JSON) to avoid re-training expensive models from scratch every time.

## Connects To
- **Ch18 (Neural Networks)**: this chapter is a direct generalization of Ch18's sigmoid-only, two-layer, hand-differentiated network into a reusable, extensible framework.
- **Ch14's Maximum Likelihood Estimation**: cross-entropy's "minimizing loss = maximizing likelihood" justification directly parallels Ch14's least-squares/MLE equivalence.
- **Ch15's Regularization**: dropout is presented as another regularization technique alongside ridge/lasso weight penalties.
- **Keras / PyTorch / "Deep Learning" (Goodfellow, Bengio, Courville)**: external frameworks/textbook the chapter's own abstractions are patterned after (Keras) or that the author personally uses (PyTorch) for real work.
