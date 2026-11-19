# Chapter 2: Designing Generative Adversarial Networks

## Core Idea
GANs pit a Generator against a Discriminator in an adversarial minimax game; the specific GAN architecture (DCGAN, WGAN, cGAN, CycleGAN, Progressive GAN, StyleGAN, Pix2Pix) is chosen by what constraint on the output you need (resolution, stability, conditioning, domain translation, style control).

## Frameworks Introduced
- **Vanilla GAN**: min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1−D(G(z)))]. Basic GAN with 2-3 hidden dense layers.
  - When to use: as a baseline / pedagogical starting point only; rarely production-grade.
- **Deep Convolutional GAN (DCGAN)**: Generator uses transposed-conv (deconv) layers + BatchNorm + ReLU, final layer tanh; Discriminator uses conv layers + BatchNorm + LeakyReLU, final sigmoid. No fully-connected layers; strided convolutions instead of pooling.
  - When to use: image generation tasks needing stable, scalable training.
- **Wasserstein GAN (WGAN)**: Replaces binary cross-entropy with the Wasserstein (Earth Mover's) distance via a Critic network (not a classifier); enforces Lipschitz continuity via weight clipping.
  - When to use: when mode collapse or vanishing gradients plague a standard GAN; train critic to convergence before each generator update.
- **Conditional GAN (cGAN)**: Concatenates a conditional vector (labels/text/image) with the noise vector fed to G, and with the image fed to D.
  - When to use: need control over the output class/attribute, e.g. generate "digit 7" specifically.
- **CycleGAN**: Two generators (G: X→Y, F: Y→X) + two discriminators (DX, DY), trained with a cycle-consistency loss L_cycle(G,F) = ||F(G(x))−x||₁ + ||G(F(y))−y||₁.
  - When to use: unpaired image-to-image translation (e.g., horses↔zebras) where paired training examples don't exist.
- **Progressive GAN (PGAN)**: Trains G/D starting at low resolution, progressively adds layers to increase resolution in stages.
  - When to use: need high-resolution image generation while keeping training stable.
- **StyleGAN**: Adds a mapping network (z → intermediate latent w) and style-modulation blocks that inject style at each generator layer via learned scale/bias (AdaIN-like); uses hinge loss for D; adds progressive growing + truncation trick.
  - When to use: need fine-grained, disentangled style control (e.g., human faces with varying expression/style).
- **Pix2Pix**: Conditional GAN with U-Net-like encoder-decoder + skip connections for the generator; PatchGAN discriminator (patch-based, not whole-image); loss = adversarial loss + pixel-wise (L1) loss.
  - When to use: paired image-to-image translation (colorization, sketch→photo, denoising).
- **Hybrid GANs (VAE-GAN, AAE-GAN)**: Combine GAN's generative power with AE/VAE's encoding; VAE-GAN replaces the VAE's encoder output distribution with a GAN generator feeding a discriminator; AAE-GAN replaces the AE encoder with a GAN discriminator.
  - When to use: need more stable training and more control over the latent space than a pure GAN offers.

## Key Concepts
- **Generator (G)**: Network mapping noise z to fake data; trained to minimize log(1−D(G(z))), or practically to maximize log(D(G(z))) (Non-Saturating GAN).
- **Discriminator (D)**: Binary classifier outputting P(real); trained to maximize correctly classifying real vs. fake.
- **Saturating gradient problem**: Early in training, D confidently rejects G's output (D(G(z))≈0), so log(1−D(G(z))) saturates at 0 with no usable gradient for G — the motivation for the Non-Saturating GAN (NS-GAN) reformulation.
- **Critic (WGAN)**: Replaces the Discriminator; outputs an unbounded real-valued score (not a probability) approximating Wasserstein distance.
- **Lipschitz continuity / weight clipping**: WGAN constraint ensuring the critic's gradient stays bounded, enforced by clamping critic weights to a fixed range.
- **Cycle-consistency loss**: Forces a round-trip translation (X→Y→X) to return close to the original — the mechanism enabling unpaired translation in CycleGAN.
- **Mapping network (StyleGAN)**: Fully-connected network mapping z to an intermediate latent w, decoupling noise sampling from the style-injection process.
- **PatchGAN (Pix2Pix)**: Discriminator that classifies local image patches rather than the whole image, focusing on local texture realism.

## Mental Models
- Treat choice of GAN architecture as a decision tree: need resolution and stability → DCGAN/Progressive GAN; need training stability specifically → WGAN; need conditioning/control → cGAN or StyleGAN; need domain translation without paired data → CycleGAN; need domain translation with paired data → Pix2Pix.
- Think of the Discriminator/Critic and Generator as being in a pursuit-evasion game: any architectural change that makes one side "too strong" (an overpowered D saturates gradients; an overpowered G floods D with convincing fakes before D can learn) breaks the balance needed for learning.
- The GAN↔VAE architectural correspondence: Generator ≈ VAE decoder, Discriminator ≈ VAE encoder (both reduce dimensionality of their input), but GANs lack an explicit probabilistic latent space — this is the conceptual bridge used throughout the book's GAN-then-AE-then-VAE progression.

## Anti-patterns
- **Using Vanilla GAN / binary cross-entropy loss on complex, high-resolution data**: prone to mode collapse, vanishing gradients, and training instability — one of several major GAN architecture design challenges.
- **Ignoring the Discriminator-saturation failure mode**: training with the original minimax G-loss (minimizing log(1−D(G(z)))) rather than the Non-Saturating variant (maximizing log(D(G(z)))) stalls the generator early in training.
- **Treating GAN hyperparameter choices (learning rate, batch size, normalization, regularization) as an afterthought**: the chapter explicitly lists architecture/loss-function/normalization/regularization/training-strategy/data-augmentation/evaluation-metric choices as all co-dependent design decisions, not independent knobs.

## Code Examples
No full runnable code blocks appear in this chapter — it is architecture-and-equation focused (code implementations arrive in Chapter 3). Key equations preserved above (GAN value function, Discriminator/Generator loss derivation, WGAN distance, CycleGAN cycle-consistency loss) are the load-bearing content.

## Reference Tables
| GAN Variant | Key Mechanism | Primary Use Case |
|---|---|---|
| Vanilla GAN | Dense layers, binary cross-entropy | Baseline/teaching |
| DCGAN | Conv/deconv + BatchNorm, no FC layers | General image generation |
| WGAN | Wasserstein distance, critic, weight clipping | Training stability, mode-collapse mitigation |
| cGAN | Conditional vector concatenation | Controlled/class-conditioned generation |
| CycleGAN | Dual generators/discriminators + cycle loss | Unpaired image-to-image translation |
| Progressive GAN | Staged resolution growth | High-resolution image generation |
| StyleGAN | Mapping network + style modulation blocks | Fine-grained style control (faces) |
| Pix2Pix | U-Net generator + PatchGAN discriminator | Paired image-to-image translation |

## Worked Example
Deriving the Discriminator loss from the GAN value function: starting from V(D,G) = E[log D(x)] + E[log(1−D(G(z)))], the chapter shows that in a minibatch of size N, p(x)=1/N, and maximizing the expectation of log D(x) is equivalent to minimizing the binary cross-entropy with label y=1 for real images and y=0 for fake images — producing the familiar BCE Discriminator loss used directly in Chapter 3's TensorFlow implementation (`cross_entropy(tf.ones_like(real_output), real_output)` + `cross_entropy(tf.zeros_like(fake_output), fake_output)`).

## Key Takeaways
1. There is no single "best" GAN — architecture choice follows directly from the generation constraint (resolution, stability, conditioning, pairing, style control).
2. The Non-Saturating GAN reformulation (max log D(G(z)) instead of min log(1−D(G(z)))) is the practical default used in almost all real implementations.
3. WGAN's critic + Wasserstein distance is the standard fix when a vanilla GAN suffers mode collapse or instability.
4. Cycle-consistency loss is the key trick enabling GANs to learn from unpaired data (CycleGAN) — contrast with Pix2Pix, which needs paired data but gets cleaner, more targeted translations.
5. StyleGAN's mapping network + style blocks generalize the "conditioning vector" idea of cGANs into continuous, disentangled style control.

## Connects To
- **Ch 1**: Directly operationalizes the GAN value function and KL-divergence-style distribution matching introduced there.
- **Ch 3**: Takes these architectures into actual training code and diagnoses the failure modes (mode collapse, vanishing gradients, oscillation) hinted at here as "major challenges."
- **Ch 7**: Revisits GAN applications, success stories (StyleGAN, BigGAN, MuseNet), and ethical concerns at a higher level after the training mechanics are mastered.
