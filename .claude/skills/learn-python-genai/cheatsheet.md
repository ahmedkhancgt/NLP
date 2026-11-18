# Cheatsheet

## Architecture Selection

| Need | Choose | Why |
|---|---|---|
| Classify/predict on existing data | Discriminative model (CNN, SVM, logistic regression) | Learns P(y\|x) directly |
| Generate new, similar data | Generative model (AE/VAE/GAN) | Learns P(x), can sample |
| Easy training, OK with less-realistic output | Autoencoder (AE) | Single reconstruction loss, end-to-end |
| True novel-sample generation + smooth latent interpolation | VAE | Probabilistic latent space, ELBO objective |
| Maximum output realism, willing to fight training instability | GAN | Adversarial loss, more expressive but harder to train |
| Unpaired image-to-image translation | CycleGAN | Cycle-consistency loss, no paired data needed |
| Paired image-to-image translation | Pix2Pix | PatchGAN + L1 pixel loss, needs paired data |
| Need conditioning/class control | cGAN or CVAE | Conditioning vector concatenated into model |
| High-resolution image generation | Progressive GAN / StyleGAN | Staged resolution growth, style modulation |
| Training instability / mode collapse in a vanilla GAN | WGAN / WGAN-GP | Wasserstein distance replaces BCE loss |
| Image classification/generation with global context | Vision Transformer (ViT) | Patch embedding + self-attention |
| Large/high-res images, need efficiency | Swin Transformer | Windowed attention, linear not quadratic cost |
| Small dataset, need a transformer encoder | Pretrained ViT/Swin backbone | Avoids overfitting a from-scratch transformer |
| Spatial/image data for an autoencoder | CNN autoencoder | Conv2D/Conv2DTranspose preserve locality |
| Tabular/non-spatial data for an autoencoder | Dense (ANN) autoencoder | No spatial structure to exploit |

## GAN Training-Failure Smell Test

| Symptom | Likely Diagnosis | First Fix to Try |
|---|---|---|
| D loss crashes near 0, G loss plateaus/oscillates | Mode collapse | Vary loss function, regularize G, diversify training data |
| Both losses plateau early, little progress | Vanishing gradients | LeakyReLU over sigmoid/tanh, gradient clipping, batch norm |
| Both losses fluctuate erratically, never settle | Oscillation | Balance G/D capacity, tune LR per network, spectral norm |
| Loss never converges at all | Training instability | Skip connections, normalization layers, LR scheduling, progressive growing |
| Generated samples look fine but lack diversity | Hidden mode collapse (loss can mislead) | Check FID/Inception Score, not just loss curves |

## VAE Training-Failure Smell Test

| Symptom | Likely Diagnosis | First Fix to Try |
|---|---|---|
| Latent space ignored, decoder relies only on prior | Posterior collapse | KL-term annealing/warm-up |
| Reconstructions are blurry | Pixel-wise BCE loss limitation | Try MSE/perceptual loss, deeper decoder, conv layers |
| Gradients vanish/explode through KL term | ELBO gradient instability | Gradient clipping, alternative divergence (Wasserstein/JS) |
| Latent space loses important features | Latent overcompression | Increase latent dim, use flexible/mixture prior |
| Train loss keeps dropping, val loss rises | Overfitting | EarlyStopping, regularization, more/better data, pretrained encoder |
| Model fails on small dataset with ViT encoder | ViT needs large pretraining | Switch to a pretrained ViT/Swin backbone |

## Loss Function Defaults

| Data Type | Loss Function |
|---|---|
| Continuous (images, sensor data) | MSE |
| Binary / grayscale pixel (probability-like) | Binary Cross-Entropy |
| Categorical / multi-class | Categorical Cross-Entropy |
| Continuous with outliers | Huber loss (tune δ) |
| VAE latent-distribution regularization | KL divergence (always paired with a reconstruction loss) |

## Attention-Type Decision Rules

| Situation | Attention Type |
|---|---|
| Default / general-purpose | Scaled dot-product |
| Need richer scoring than dot product | Additive attention |
| Need to capture multiple relationship types at once | Multi-head attention |
| Decoder needs to reference encoder outputs (seq2seq) | Cross-attention |
| Need one discrete region selected (captioning, localization) | Hard attention (accept non-differentiability, higher training cost) |
| General vision task, want differentiability | Soft attention (most common default) |
| All image regions equally relevant | Global attention |
| Large image, need efficiency, long-range deps | Local / windowed (Swin-style) attention |

## Key Thresholds & Defaults Committed to in the Text

- WGAN: train critic to convergence before each generator update; clip critic weights to a fixed range for Lipschitz continuity.
- VAE reconstruction-loss scaling example: `reconstruction_loss_factor = 1000` to rebalance MSE against the analytic KL loss term.
- CLR (Cyclic Learning Rate) example: `base_lr=1e-4` to `max_lr=3e-4`, `triangular2` mode, `step_size = 4 * (len(x_train)//batch_size)`.
- EarlyStopping default pattern: `monitor='val_loss', patience=5, restore_best_weights=True`.
- Anomaly-detection reconstruction-error threshold example (MNIST/Fashion-MNIST): `threshold = 0.08`.
- Vanilla VAE latent dimensionality examples: 2 (visualization), 32-128 (practical image tasks), 256 (Swin-backbone VAE).
- GAN evaluation: never trust loss values alone — always pair with FID, Inception Score, or LPIPS, plus visual inspection.

## Quick Reference: Core Equations

- GAN value function: `min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]`
- Non-Saturating GAN generator loss: `max_G E[log(D(G(z)))]`
- WGAN critic objective: approximate Wasserstein (Earth Mover's) distance between real/fake distributions
- CycleGAN cycle-consistency loss: `L_cycle(G,F) = ||F(G(x))-x||_1 + ||G(F(y))-y||_1`
- VAE reparameterization: `z = mu + sigma ⊙ epsilon`, `epsilon ~ N(0,I)`
- VAE ELBO: `L = E[log p(x|z)] - KL(q(z|x) || p(z))`
- Analytic Gaussian KL term: `KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))`
- Scaled dot-product attention: `alpha = softmax(QK^T / sqrt(d_k))`
- Multi-head attention: `MultiHead(q,k,v) = Concat(head_1,...,head_h) . W_o`
