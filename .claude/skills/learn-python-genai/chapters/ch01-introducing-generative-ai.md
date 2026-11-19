# Chapter 1: Introducing Generative AI

## Core Idea
Generative models learn the underlying probability distribution p(X) that produced a dataset, so they can sample brand-new, realistic data points, whereas discriminative models only learn the decision boundary between classes.

## Frameworks Introduced
- **Discriminative vs. Generative modeling**: Discriminative models learn P(y|x) (the boundary between classes); generative models learn P(x) or P(x,y) and can sample new x.
  - When to use discriminative: classification, regression, ranking, retrieval, pattern recognition.
  - When to use generative: image/video/text generation, data augmentation, outlier detection, density estimation, digital twins, class-imbalance correction.
- **Sample space of a generative model**: X = G(Z; θ) — the generator G maps a noise/latent space Z through parameters θ into the data space X. Formally S = {x | x = G(z), z ~ p(z)}.
  - How: define a prior p(z) over latent noise, learn G so that G(z) matches the true data distribution.
- **Parametric modeling**: Assume the PDF p(X|θ) belongs to a known family (e.g., Gaussian) with finite parameters θ, estimated via MLE or Bayesian methods.
  - When to use: whenever you need a tractable, fittable density instead of searching over all possible density functions.
- **Maximum Likelihood Estimation (MLE)**: Find θ that maximizes L(θ|X) = Πᵢ p(xᵢ|θ), equivalently maximizes the log-likelihood Σᵢ log p(xᵢ|θ).
  - How: optimize log-likelihood numerically (gradient descent, Newton's method) since log avoids underflow and simplifies derivatives.
- **KL Divergence**: D_KL(p‖q) = ∫ p(x) log(p(x)/q(x)) dx — measures information lost approximating true p(x) with model q(x); always ≥ 0, zero only when distributions match exactly.
  - How: used inside VAE and GAN objectives to pull a learned distribution toward a target distribution.

## Key Concepts
- **Generative AI**: AI that produces new content (images, text, audio) with characteristics similar to training data.
- **Discriminative model**: Learns P(y|x) directly; examples: logistic regression, SVM, decision trees, CNNs, RNNs.
- **Generative model**: Learns P(x) or joint P(x,y); examples: Naive Bayes, GMM, HMM, Autoencoders, VAEs, GANs, Boltzmann machines.
- **Probability Density Function (PDF)**: f(x) describing relative likelihood of a continuous variable; integral over the domain = 1; PDF is the derivative of the CDF.
- **Log-likelihood**: log L(θ|X) = log p(X|θ); preferred over raw likelihood for numerical stability and easier optimization.
- **Gaussian Mixture Model (GMM)**: Probabilistic model assuming data is generated from K Gaussian components, each with mixing coefficient πₖ, mean, and covariance; fit via Expectation-Maximization (EM).
- **GAN value function**: minG maxD V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1−D(G(z)))].
- **VAE loss**: reconstruction loss + a regularization (KL) term pulling the learned posterior toward a prior (typically Gaussian).
- **Expectation-Maximization (EM)**: Iterative algorithm alternating an E-step (compute expected cluster-membership probabilities) and M-step (update Gaussian parameters) until convergence.

## Mental Models
- Think of a generative model as reverse-engineering the "unknown recipe" (probability distribution) that produced the training examples, then cooking new dishes from that recipe.
- Use discriminative modeling when you only need to draw a line between categories; use generative modeling when you need to understand or recreate the "shape" of the data itself.
- The sample space equation X = G(Z; θ) is the single unifying lens for GANs, VAEs, and autoregressive models: all of them are different choices of G and Z.
- Log-likelihood vs. likelihood: use likelihood when values aren't extreme; switch to log-likelihood whenever values are very small/large or derivatives are needed — this is why nearly every deep generative loss is expressed in log form.

## Anti-patterns
- **Treating CNNs/RNNs as inherently generative**: They are discriminative feature extractors by default; they only become generative components when embedded inside a GAN/VAE-style generator.
- **Relying on generative models when data imbalance could be fixed more cheaply with a discriminative re-weighting or resampling approach**: generative augmentation is powerful but computationally heavier and not always the best tool.
- **Ignoring the "naive" independence assumption in Naive Bayes** on real-world correlated features, causing degraded accuracy and the zero-probability problem for unseen feature values.

## Code Examples
```python
import numpy as np
def kl_divergence(p, q):
    """Computes KL divergence between two probability distributions p and q."""
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
# Example usage:
p = [0.2, 0.3, 0.5]
q = [0.25, 0.25, 0.5]
kl_div = kl_divergence(p, q)
```
- **What it demonstrates**: Direct computation of KL divergence between two discrete probability distributions, the core building block of VAE regularization and GAN distribution-matching losses.

```python
import tensorflow as tf
import tensorflow_probability as tfp
gmm = tfp.distributions.MixtureSameFamily(
    mixture_distribution=tfp.distributions.Categorical(
        probs=tf.ones([num_components], dtype=tf.float32) / num_components),
    components_distribution=tfp.distributions.MultivariateNormalDiag(
        loc=tf.Variable(tf.zeros([num_components, num_features], dtype=tf.float32)),
        scale_diag=tfp.util.TransformedVariable(tf.ones([num_components, num_features], dtype=tf.float32),
                                                bijector=tfb.Softplus())
    )
)
@tf.function
def nll_loss(x):
    return -tf.reduce_mean(gmm.log_prob(x))
```
- **What it demonstrates**: Building a trainable Gaussian Mixture Model with TensorFlow Probability, optimized via negative log-likelihood — the parametric-modeling + MLE framework applied directly to MNIST.

## Reference Tables
| Approach | Learns | Examples | Typical Use |
|---|---|---|---|
| Discriminative | P(y\|x), decision boundary | Logistic regression, SVM, decision trees, CNN, RNN | Classification, regression, ranking |
| Generative | P(x) or P(x,y) | Naive Bayes, GMM, HMM, Autoencoder, VAE, GAN, Boltzmann machine | Generation, augmentation, density estimation |

## Worked Example
Building a GMM from the MNIST (digits) dataset with `sklearn`: load `load_digits()`, fit `GaussianMixture(n_components=20, covariance_type='full')` to the flattened 8x8 images, then call `gmm.sample(100)` to draw 100 new synthetic digit-like samples, reshape to 8x8, and visualize a 10x10 grid. The generated digits resemble the originals but show blur/variation — illustrating both the promise (new, plausible samples) and the limits (GMM cannot capture fine structure the way deep generative models can) of a classical parametric generative model.

## Key Takeaways
1. Discriminative and generative models answer fundamentally different questions (P(y|x) vs P(x)) — pick based on whether you need to classify or to create/augment data.
2. Every deep generative model (GAN, VAE) can be understood through the same primitives covered here: sample space, PDF, MLE, and KL divergence.
3. Log-likelihood, not raw likelihood, is the practical tool for training generative models at scale.
4. GANs minimize KL-divergence-like quantities adversarially; VAEs minimize KL divergence explicitly between posterior and prior — same math, different mechanics.
5. Class imbalance in classification tasks is a legitimate, practical use case for generative models (train a GAN to synthesize minority-class samples).

## Connects To
- **Ch 2**: The GAN value function introduced here (minG maxD V(D,G)) is dissected in full architectural detail across Vanilla GAN, DCGAN, WGAN, and other variants.
- **Ch 4**: The reconstruction-error / MLE framing here directly motivates the autoencoder loss function (minimizing Δ(x, f(g(x)))).
- **Ch 6-7**: KL divergence introduced here becomes the explicit regularization term in the VAE loss function.
