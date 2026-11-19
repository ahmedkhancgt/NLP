# Patterns

## GAN Generator-Discriminator Adversarial Training Loop
**When to use**: Any GAN implementation — the canonical two-network minimax training pattern.
**How**: Build independent Generator and Discriminator networks. Each batch: (1) generate fake samples from noise, (2) compute Discriminator loss on real+fake batches (BCE with real labels=1, fake labels=0), (3) compute Generator loss via Non-Saturating form (maximize log D(G(z))), (4) backprop each network's loss through its own `GradientTape`/optimizer independently.
**Trade-offs**: Simple and general, but prone to mode collapse, vanishing gradients, and oscillation if G/D capacities or learning rates are imbalanced; needs careful monitoring via loss curves + FID/Inception Score, not loss values alone.

## Non-Saturating GAN Loss Reformulation
**When to use**: Whenever training a GAN with the original minimax objective stalls early (Discriminator becomes too confident, generator gradients vanish).
**How**: Replace min_G log(1−D(G(z))) with max_G log(D(G(z))) — mathematically motivated by the fact the original form saturates to zero gradient when D(G(z))≈0.
**Trade-offs**: Fixes the saturating-gradient problem with no architectural change, but doesn't address mode collapse or instability on its own.

## Wasserstein Distance + Weight Clipping (WGAN)
**When to use**: GAN suffers simultaneous mode collapse, vanishing gradients, and instability under standard BCE loss.
**How**: Replace the Discriminator with a Critic outputting an unbounded real value approximating Wasserstein distance; clip critic weights to a fixed range to enforce Lipschitz continuity; train the critic to convergence before each generator update.
**Trade-offs**: More stable training and meaningful loss values, but weight clipping can limit critic capacity; later variants (WGAN-GP) replace clipping with a gradient penalty for better results.

## Conditional Generation via Concatenated Conditioning Vector
**When to use**: Need to control which class/attribute a GAN or VAE generates (not just "something from the dataset").
**How**: Concatenate a one-hot or embedding conditioning vector y with the noise/latent input to the Generator/decoder, and with the data fed to the Discriminator/encoder (cGAN), or condition both encoder and decoder on y (CVAE).
**Trade-offs**: Enables targeted generation but requires labeled data and increases model/training complexity.

## Cycle-Consistency Loss for Unpaired Translation
**When to use**: Image-to-image translation where paired before/after examples don't exist (e.g., horse↔zebra).
**How**: Train two generators (G: X→Y, F: Y→X) and two discriminators; add L_cycle(G,F) = ||F(G(x))−x||₁ + ||G(F(y))−y||₁ to force round-trip consistency.
**Trade-offs**: No paired data needed, but training is heavier (4 networks) and the translation may be less precise than a paired (Pix2Pix-style) approach.

## Autoencoder Bottleneck + Regularization
**When to use**: Any autoencoder, to prevent it from learning a trivial identity mapping.
**How**: Shrink the latent layer's neuron count well below the input dimensionality (bottleneck), and/or add L1/L2 penalty terms to the loss, and/or tie encoder weights to decoder weights.
**Trade-offs**: Bottleneck alone is simplest; regularization adds training-time cost but can improve sparsity/robustness even without a tight bottleneck.

## Reconstruction-Error Anomaly Detection
**When to use**: Unsupervised anomaly/outlier detection when only "normal" labeled data is available.
**How**: Train an AE only on normal data; at inference, compute reconstruction error for new inputs; flag inputs above a chosen threshold as anomalous.
**Trade-offs**: No labeled anomalies needed, but sensitive to threshold choice and to AE hyperparameters/capacity; can miss anomalies that happen to reconstruct well.

## VAE Reparameterization Trick
**When to use**: Any time a model must sample from a learned distribution inside a differentiable computation graph (the core VAE mechanism).
**How**: Encoder outputs μ(x) and log σ²(x); sample ε~N(0,I); compute z = μ(x) + σ(x)⊙ε; pass z to the decoder. Gradients flow through μ and σ because the randomness (ε) is external to the trainable parameters.
**Trade-offs**: Makes stochastic latent sampling trainable with ordinary backprop; assumes a tractable, reparameterizable distribution form (works cleanly for Gaussian, less trivially for other priors like Dirichlet).

## ELBO Training Objective (Reconstruction + KL)
**When to use**: The universal VAE loss — any VAE variant (vanilla, conditional, β-VAE, hierarchical) uses this base form.
**How**: Loss = reconstruction_loss(x, decoder(z)) + KL_divergence(qφ(z|x), prior p(z)), typically computed analytically for Gaussian q and p: KL = −0.5·Σ(1+log σ² − μ² − σ²).
**Trade-offs**: Balancing the two terms (often via a weighting constant or β) is critical — too much KL weight causes posterior collapse/blurry output, too little causes an unregularized, non-generative latent space.

## KL-Term Annealing / Warm-Up
**When to use**: VAE exhibits posterior collapse (latent space ignored, decoder relies only on the prior).
**How**: Start training with a low (or zero) weight on the KL loss term, gradually increasing it over epochs so the model first learns to reconstruct before being regularized toward the prior.
**Trade-offs**: Mitigates posterior collapse but adds a scheduling hyperparameter (anneal rate/schedule) that itself needs tuning.

## Loss-Function Selection by Data Type (AE/VAE)
**When to use**: Choosing the reconstruction loss term for any autoencoder-family model.
**How**: MSE for continuous data (images/sensor data); Binary Cross-Entropy for binary/grayscale pixel data; Categorical Cross-Entropy for categorical reconstruction; Huber loss for continuous data with outliers; KL divergence specifically for the latent-distribution regularization term (not reconstruction).
**Trade-offs**: Mismatched loss choice (e.g., MSE on outlier-heavy data) distorts training; Huber adds a δ hyperparameter to tune the MSE/MAE transition point.

## Vision Transformer Patch Embedding
**When to use**: Applying transformer self-attention to image data.
**How**: Divide image into fixed-size non-overlapping patches; flatten and linearly project each patch into an embedding vector; add positional embeddings; feed the resulting sequence into a standard transformer encoder (self-attention + FFN stack).
**Trade-offs**: Enables transformers' global-context strength on images, but has quadratic attention cost in the number of patches and typically needs large-scale pretraining to generalize well on small datasets.

## Pretrained-Backbone-as-Encoder (Transfer Learning into a VAE)
**When to use**: Building a VAE (or any generative encoder) on a dataset too small to train a ViT/CNN backbone from scratch.
**How**: Load a pretrained ViT/CNN (e.g., `vit_l32(pretrained=True)` or a pretrained Swin Transformer) with its classification head removed; feed its penultimate-layer output into Dense z_mean/z_log_var heads; keep the decoder trainable from scratch.
**Trade-offs**: Dramatically reduces overfitting versus training the encoder from scratch on small data, but ties the encoder's inductive biases to whatever the backbone was pretrained on (e.g., ImageNet).

## Shifted-Window (Swin) Attention
**When to use**: Need transformer-style global+local context on large/high-resolution images without ViT's quadratic attention cost.
**How**: Compute self-attention within small, non-overlapping local windows; shift the window partitioning between successive layers so information can flow across window boundaries over depth.
**Trade-offs**: Linear (not quadratic) complexity in image size, but implementation is more complex than plain ViT and requires careful window-size/shift configuration.

## Embedding + Vector Database Similarity Search
**When to use**: Content-based (image or text) search/retrieval at scale.
**How**: Extract embeddings with a pretrained feature-extractor model (CNN for images, language model for text); upsert embeddings with IDs/metadata into a vector database (e.g., Pinecone) with similarity indexing; query by embedding a new item and retrieving top-k nearest neighbors by cosine similarity/Euclidean distance.
**Trade-offs**: Scales to large datasets far better than brute-force comparison, but result quality is entirely bounded by the embedding model's feature quality.
