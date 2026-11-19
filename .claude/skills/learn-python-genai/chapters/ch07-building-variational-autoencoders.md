# Chapter 7: Building Variational Autoencoders for Generative AI

## Core Idea
A VAE's generative quality depends on three compounding engineering choices — encoder/decoder architecture (CNN vs. dense), choice of prior distribution (Gaussian vs. Dirichlet vs. other), and careful mitigation of a long, well-characterized list of training failure modes (vanishing gradients, mode collapse, posterior collapse, latent overcompression) — and the CNN+Gaussian combination is consistently the strongest baseline shown in the chapter's side-by-side experiments.

## Frameworks Introduced
- **Dirichlet-prior VAE (Non-CNN)**: Dense-layer encoder/decoder with standard reparameterization trick, but samples drawn from `np.random.dirichlet` for generation instead of a standard normal — suited to categorical/compositional data (e.g., topic modeling).
  - When to use: data is naturally proportions/compositions over categories rather than continuous pixel intensities.
- **Dirichlet-prior VAE (CNN)**: Same Dirichlet-sampling idea but with Conv2D/Conv2DTranspose encoder/decoder — captures spatial image structure while still modeling categorical uncertainty.
  - When to use: image data where you additionally want calibrated categorical-uncertainty quantification (CNN classification head + Dirichlet over class probabilities), per the chapter's discussion of Dirichlet-distribution uncertainty modeling in CNNs.
- **Non-Dirichlet (standard Gaussian) VAE on Fashion-MNIST**: Conv2D encoder → z_mean/z_log_var → Sampling layer (custom `tf.keras.layers.Layer` subclass) → Conv2DTranspose decoder; custom `loss_func` closure bundling reconstruction (scaled MSE) + KL loss.
  - When to use: general-purpose baseline VAE for image data — this is the version that actually produces visibly correct "Trouser"/"Shirt" generations in the chapter's output figures.
- **ELBO ↔ KL divergence derivation (expanded)**: ELBO = E[log p(x|z)] − KL(q(z|x)‖p(z)); maximizing ELBO ⟺ minimizing the KL divergence between the approximate posterior and the prior — the chapter walks through the full algebraic derivation from the evidence/marginal likelihood.
  - When to use: as the reference derivation whenever a custom VAE loss needs to be justified or modified.
- **Missing-data handling strategies catalogue**: mean/mode imputation, input masking, Conditional VAE, multiple imputation, Bayesian VAE, VAE-GAN hybrid, data augmentation, Transformer-based VAEs (VQ-VAE-2, TransVAE), Joint VAE.
  - When to use: pick based on whether missingness is random (imputation), structural (masking/CVAE), or needs principled uncertainty (Bayesian VAE).

## Key Concepts
- **CNN vs. Non-CNN VAE trade-off**: CNN approach captures spatial/grid structure (better for images); Non-CNN (dense) approach is more versatile for tabular/sequential/non-spatial data — directly determines reconstruction quality (the chapter notes the CNN version "clearly shows better output when trained till 100 epochs").
- **Reconstruction-loss-factor scaling**: multiplying the MSE reconstruction loss by a large constant (e.g., 1000) to rebalance its magnitude against the KL loss term — a practical trick for loss-term weighting shown in the Fashion-MNIST example.
- **Posterior collapse mitigation via KL-warm-up**: gradually increasing the KL term's weight during training so the model first learns to reconstruct, then is regularized — prevents the model from collapsing the posterior to the prior too early.
- **Latent space overcompression**: when too few latent dimensions force the VAE to discard important features; fixed by increasing latent dimensionality or using a more flexible/mixture prior.
- **Importance weighting**: reweighting the KL term to correct for the mismatch between the true posterior and the variational approximation.

## Mental Models
- Treat VAE training issues as a checklist to run through systematically: gradient behavior (vanishing/exploding) → sample diversity (mode collapse) → reconstruction sharpness (poor quality) → latent capacity (overcompression) → posterior behavior (collapse) → data diversity → interpretability (uninformative latent space) → compute budget. Each symptom maps to a specific, named fix in the chapter.
- The CNN-vs-Dense VAE choice should mirror the CNN-vs-Dense Autoencoder choice from Chapter 4 — same underlying logic (spatial locality matters or it doesn't), just now applied to a probabilistic encoder/decoder.
- β-VAE's hyperparameter β is the single knob for the fundamental reconstruction-fidelity ↔ disentanglement/regularization trade-off that runs through this entire chapter's troubleshooting list.

## Anti-patterns
- **Using the plain BCE reconstruction loss unscaled against the analytic KL loss without checking their relative magnitudes**: can silently starve one loss term of influence — the chapter's working example explicitly multiplies the MSE reconstruction loss by 1000 to correct this imbalance.
- **Assuming a Dirichlet prior is a drop-in replacement for a Gaussian prior**: it changes the geometry of the latent space (simplex vs. unbounded Euclidean space) and is only appropriate for compositional/categorical data, not generic image pixels.
- **Treating "more epochs" as a universal fix**: the book's own CNN ViT-encoder VAE experiments (foreshadowing Ch. 9) show overfitting after too many epochs without early stopping — the same risk applies to plain VAEs trained too long without a validation-loss-based stopping criterion.

## Code Examples
```python
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim), mean=0.0, stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss
    def vae_kl_loss(encoder_mu, encoder_log_variance):
        return -0.5 * tf.keras.backend.sum(
            1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance),
            axis=1)
    def vae_loss(y_true, y_predict):
        return vae_reconstruction_loss(y_true, y_predict) + vae_kl_loss(encoder_mu, encoder_log_variance)
    return vae_loss
vae_model.compile(optimizer='adam', loss=loss_func(z_mean, z_log_var))
```
- **What it demonstrates**: A production-style VAE loss built as a Keras `Layer` subclass (for sampling) plus a closure-based loss factory (for the weighted reconstruction+KL loss) — the pattern needed when `add_loss` isn't used and the loss must be passed to `.compile()`.

## Reference Tables
| VAE Training Issue | Root Cause | Primary Fix |
|---|---|---|
| Vanishing/exploding gradients | KL term's gradient instability | Gradient clipping, Wasserstein/JS divergence alternatives |
| Mode collapse | Limited diversity capture | Entropy regularization, Annealed Importance Sampling, MDL |
| Poor reconstruction quality | Insufficient capacity | Deeper encoder/decoder, hierarchical VAE, conv layers |
| Latent overcompression | Latent dim too small | Increase latent dimensionality, flexible/mixture prior |
| Posterior collapse | KL term dominates too early | KL warm-up / annealing schedule |
| Lack of data diversity | Biased training set | Data augmentation, more diverse sampling |
| Uninformative latent space | No disentanglement pressure | β-VAE, loss annealing |
| Computational complexity | Large models/datasets | SGVB, amortized inference, distributed training |

## Worked Example
Training three VAE variants on the same visual task and comparing outcomes: (1) Dense/Non-CNN VAE on flattened MNIST with Dirichlet-sampled generation — baseline, weaker reconstructions; (2) CNN VAE on reshaped MNIST with the same Dirichlet sampling — the chapter states this "clearly shows better output when trained till 100 epochs"; (3) CNN VAE on Fashion-MNIST with standard Gaussian sampling, a custom weighted loss (reconstruction_loss_factor=1000), and EarlyStopping-free 20-epoch training, producing recognizably correct "Trouser" and "Shirt" generated samples. The progression illustrates concretely how architecture choice (CNN vs dense) and prior choice (Dirichlet vs Gaussian) each independently affect output quality.

## Key Takeaways
1. CNN-based encoder/decoder consistently outperforms dense/non-CNN VAEs on image data in this chapter's own comparisons.
2. The Dirichlet distribution is a specialized prior for categorical/compositional data, not a general-purpose Gaussian replacement.
3. ELBO maximization is mathematically equivalent to KL-divergence minimization between the approximate posterior and the prior — understanding this equivalence clarifies why so many VAE fixes revolve around KL-term weighting.
4. A systematic troubleshooting checklist (gradients → mode collapse → reconstruction quality → latent capacity → posterior collapse → data diversity → interpretability → compute) is more productive than ad hoc debugging.
5. Missing-data handling in VAEs has at least eight distinct named strategies — matching the strategy to whether missingness is random, structural, or needs principled uncertainty quantification is key.

## Connects To
- **Ch 6**: Directly extends the Vanilla VAE's challenges list (posterior collapse, blurry reconstructions, hyperparameter sensitivity) into concrete architectural and training fixes.
- **Ch 9**: The CNN VAE architecture built here on MNIST/Fashion-MNIST is scaled up to the STL-10 dataset and then fused with Vision Transformer encoders.
- **Ch 10**: The KL-warm-up, cyclic learning rate, and EarlyStopping techniques previewed here are formalized and extended into a full architectural-refactoring toolkit.
