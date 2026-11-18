# Chapter 3: Fundamentals of Generative Models

## Core Idea
Generative models share a common toolkit (GANs, VAEs, autoencoders, CycleGAN, BERT, DeepDream) built on probability theory and linear algebra, and they are fundamentally distinguished from discriminative models by their goal: learning a data distribution well enough to sample new, realistic instances from it.

## Frameworks Introduced
- **GAN (adversarial generative modeling)**: generator vs. discriminator in continual competition; implemented by NVIDIA (StyleGAN/StyleGAN2), OpenAI (DALL-E), Google Brain, Facebook AI, IBM.
  - When to use: tasks needing high-fidelity, realistic sample generation (images, synthetic data for data augmentation, anomaly detection).
  - How: 6-step pipeline — (1) define the problem, (2) choose architecture (vanilla/DCGAN/StyleGAN/WGAN), (3) prepare data, (4) train generator+discriminator adversarially, (5) optimize/fine-tune, (6) deploy (serialize, containerize, expose via API).
- **VAE (probabilistic encoder/decoder)**: encoder maps input to a latent distribution; decoder reconstructs/generates by sampling it; trained to jointly minimize reconstruction loss and KL divergence.
  - When to use: when a smooth, interpolatable latent space matters (molecule design, music generation, anomaly detection).
  - How: 8-step pipeline — define problem → choose VAE variant (convolutional for images, recurrent for sequences) → prepare data → train with dual (reconstruction + KL) objective → evaluate (FID/task metric) → deploy → iterate → leverage community frameworks (PyTorch, TensorFlow Probability).
- **Autoencoder family**: generic encoder-decoder for compression/representation learning; variants include denoising, convolutional, and variational (VAE) autoencoders.
  - When to use: dimensionality reduction, image compression (JPEG-style), anomaly/fraud detection, unsupervised representation learning.
  - How: encode → compress to latent → decode/reconstruct; minimize reconstruction loss (often MSE).
- **CycleGAN (unpaired image-to-image translation)**: two generators (G: X→Y, F: Y→X) and two discriminators (DX, DY), trained with a cycle-consistency loss so that F(G(x)) ≈ x.
  - When to use: style transfer or domain adaptation when paired training examples are unavailable.
  - How: cycle consistency loss `Lcycle(G,F) = ||F(G(x)) – x||1 + ||G(F(y)) – y||1` enforces a reversible mapping.
- **BERT (bidirectional contextual embeddings)**: transformer-based model using masked-language-model pre-training then task-specific fine-tuning.
  - When to use: NLP tasks needing deep contextual understanding (not pure generation) — question answering, search ranking, sentiment analysis.
- **DeepDream**: not a true generative model, but a CNN-feedback visualization technique that amplifies recognized patterns, used for artistic/creative exploration.

## Key Concepts
- **Generative vs. discriminative models**: generative models learn the underlying data distribution to create new instances (unsupervised, e.g., MLE/variational inference); discriminative models learn a mapping from data to labels (supervised, e.g., logistic regression, SVM).
- **Latent space**: lower-dimensional, often continuous representation of data learned by an encoder, from which new samples can be generated/interpolated.
- **Reconstruction loss**: measures fidelity between input and the model's reconstructed output (used in autoencoders/VAEs).
- **KL divergence**: regularizes the VAE's latent space to be smooth/continuous, enabling meaningful sampling.
- **Adversarial loss**: the GAN's competitive loss capturing how well the generator fools the discriminator.
- **Cycle-consistency loss**: CycleGAN's mechanism for enforcing reversible, meaningful domain translation without paired data.
- **Perceptual loss**: evaluates high-level feature similarity (not just pixels) — used in style transfer/image generation.
- **FID / Inception Score**: quantitative generative-model evaluation metrics.
- **Transfer learning (generative context)**: fine-tuning a pre-trained generative model on a related task to save training time/data.

## Mental Models
- Think of probability distributions as "the artist" shaping uncertainty and linear algebra as "the canvas" transforming latent representations into data — this is the mathematical composition view of generative modeling.
- Use "generative models sacrifice precision for diversity; discriminative models sacrifice diversity for precision" as the core trade-off when picking between (or combining) the two paradigms.
- Treat each specialized architecture (DCGAN, WGAN, StyleGAN, CycleGAN) as the base GAN plus one targeted fix: DCGAN fixes "which layers," WGAN fixes "which loss/distance metric," StyleGAN fixes "style control," CycleGAN fixes "no paired data."

## Anti-patterns
- **Using binary cross-entropy loss indiscriminately**: WGAN's Wasserstein distance exists precisely because BCE loss causes instability/mode collapse in many settings — match the loss function to the architecture's known failure modes.
- **Assuming a single "best" GAN architecture**: there is no one-size-fits-all; architecture choice must match task compatibility, data characteristics, and compute budget (see Ch3 architecture-selection checklist).
- **Skipping evaluation metrics because "the images look fine"**: subjective visual inspection misses diversity collapse and distributional mismatch that FID/Inception Score would reveal.

## Reference Tables
| Model | Core mechanism | Best suited for |
|---|---|---|
| GAN | Generator vs. discriminator, adversarial loss | Realistic image/audio synthesis |
| VAE | Probabilistic encoder/decoder, KL + reconstruction loss | Smooth latent-space exploration, molecule/music generation |
| Autoencoder | Encoder/decoder, reconstruction loss | Compression, denoising, anomaly detection |
| CycleGAN | Dual generator/discriminator + cycle-consistency loss | Unpaired image-to-image translation, style transfer |
| BERT | Bidirectional transformer, masked LM pre-training | Contextual language understanding (not generation) |
| DeepDream | CNN feature amplification via feedback loop | Artistic/creative visualization |

## Worked Example
DeepChem's VAE-based molecular generation: the encoder maps known molecular structures into a continuous latent "chemical space"; researchers sample new points in that space and the decoder reconstructs them into novel, synthesizable molecular structures with desired properties. This mirrors the general VAE recipe (encode → regularize latent space with KL divergence → decode/sample) but swaps "images" for "molecules," illustrating how the same architecture generalizes across completely different data modalities.

## Key Takeaways
1. Pick GAN when adversarial realism matters most; pick VAE when a smooth, probabilistic latent space matters most; combine both (VAE-GAN, AAE-GAN) when you need both qualities.
2. CycleGAN's cycle-consistency loss is the key innovation enabling translation without paired datasets — a recurring need in style transfer and domain adaptation.
3. Reconstruction loss + KL divergence is the VAE's dual-objective signature; mismanaging the KL term either collapses the latent space or produces blurry reconstructions.
4. BERT is not a generative model in the GAN/VAE sense — it is a contextual-understanding model, useful for conditioning or evaluating generated text, not synthesizing it from scratch.
5. Discriminative and generative models are complementary, not competing — use a generative model to create/augment data and a discriminative model to classify or filter it.

## Connects To
- **Ch1**: introduces the GAN/VAE concepts this chapter formalizes mathematically.
- **Ch7**: deep-dives into GAN architecture, training challenges (mode collapse, instability), and emerging research.
- **Ch8**: shows how to serialize, package, and deploy the exact GAN/VAE models described here.
