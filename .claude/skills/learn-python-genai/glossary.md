# Glossary

**AAE-GAN (Adversarial Autoencoder-GAN)** — Hybrid combining an AE's encoder with a GAN discriminator distinguishing encoded vs. generated samples. (Ch 2)

**Additive attention** — Attention scoring via learned feed-forward networks on query/key rather than a simple dot product. (Ch 8)

**Anomaly detection (via AE)** — Flagging inputs with unusually high reconstruction error after training only on "normal" data. (Ch 4)

**Autoencoder (AE)** — Neural network with encoder g and decoder f trained to reconstruct its own input via a deterministic latent code z=g(x). (Ch 1, 4, 5, 6)

**Autoregressive generation** — Generating output tokens one at a time, each conditioned on previously generated tokens. (Ch 8, 11)

**β-VAE** — VAE variant introducing hyperparameter β scaling the KL term to trade off reconstruction fidelity against latent disentanglement. (Ch 6, 7)

**Bottleneck architecture** — AE design forcing latent dimensionality q far below input dimensionality n to prevent trivial reconstruction. (Ch 4)

**Conditional GAN (cGAN)** — GAN conditioned on an additional input vector (labels/attributes) concatenated with noise. (Ch 2)

**Conditional VAE (CVAE)** — VAE whose encoder/decoder are additionally conditioned on labels/attributes y, enabling class-conditioned generation. (Ch 6, 7)

**Critic (WGAN)** — WGAN's replacement for the Discriminator; outputs an unbounded real-valued score approximating Wasserstein distance rather than a probability. (Ch 2, 3)

**Cross-attention** — Attention where queries come from a decoder and keys/values come from an encoder, used in sequence-to-sequence tasks. (Ch 8)

**CycleGAN** — GAN architecture for unpaired image-to-image translation using two generators/discriminators and a cycle-consistency loss. (Ch 2)

**Cyclic Learning Rate (CLR)** — Training schedule oscillating the learning rate between base_lr and max_lr (triangular/triangular2/exp_range patterns) instead of monotonic decay. (Ch 10)

**DCGAN (Deep Convolutional GAN)** — GAN using convolutional/deconvolutional layers with BatchNorm, no fully-connected layers, strided convolutions instead of pooling. (Ch 2, 3)

**Decoder** — Function f mapping a latent representation back to the original data space. (Ch 1, 4-7)

**Dirichlet distribution** — Multivariate distribution over proportions/compositions on a simplex; used as an alternative VAE prior for categorical/compositional data. (Ch 6, 7)

**Discriminative model** — Model learning P(y|x), the decision boundary between classes (e.g., logistic regression, SVM, CNN). (Ch 1)

**Discriminator** — GAN's binary classifier network distinguishing real from generated (fake) data. (Ch 1-3, 7)

**Disentangled representation** — Latent space where each dimension corresponds to one independent, semantically meaningful factor of variation. (Ch 5, 6)

**ELBO (Evidence Lower BOund)** — VAE training objective: E[log p(x|z)] − KL(q(z|x)‖p(z)); tractable surrogate for intractable MLE log-likelihood. (Ch 6, 7, 8)

**Embedding** — Dense vector representation capturing semantic meaning/relationships of text or image data. (Ch 11)

**Foundation model** — Large-scale, broadly pretrained model (GPT, DALL-E) adaptable to many downstream tasks with minimal fine-tuning. (Ch 11, 12)

**Generative model** — Model learning P(x) or P(x,y), capable of sampling new data resembling the training distribution. (Ch 1)

**Generator** — GAN network mapping noise z to synthetic/fake data samples. (Ch 1-3, 7)

**Global attention** — Attention mechanism assigning equal weight to all input regions. (Ch 8)

**Hard attention** — Discrete, non-differentiable attention selecting specific regions/patches. (Ch 8)

**Hierarchical VAE (HVAE)** — VAE extension introducing multiple levels/layers of latent variables for structured, expressive representations. (Ch 6)

**Huber loss** — Loss function behaving like MSE for small errors and MAE for large errors, controlled by threshold δ; robust to outliers. (Ch 5)

**Image embedding** — CNN-derived vector representation capturing an image's visual features/semantics. (Ch 11)

**KL divergence (Kullback-Leibler)** — D_KL(p‖q) = ∫p(x)log(p(x)/q(x))dx; measures information lost approximating p with q; core to VAE regularization and GAN distribution matching. (Ch 1, 6, 7)

**Latent space** — The compressed representation z between encoder and decoder; deterministic in AE, stochastic (distributional) in VAE/GAN. (Ch 1, 4-7)

**Lipschitz continuity** — WGAN constraint ensuring the critic's gradient stays bounded, enforced via weight clipping. (Ch 2)

**LLM (Large Language Model)** — Large-scale language model (e.g., GPT-3) used for analyzing/structuring text; distinguished in the book from Gen AI's generative/personalization role. (Ch 11, 12)

**Local attention** — Attention restricted to a neighborhood window around the query, reducing computation for large images. (Ch 8)

**Maximum Likelihood Estimation (MLE)** — Statistical method finding parameters θ maximizing the likelihood of observed data. (Ch 1, 6)

**Mode collapse** — GAN/VAE failure where the generator produces only a limited subset of possible outputs, lacking diversity. (Ch 1-3, 7)

**Multi-head attention** — Running several attention computations in parallel subspaces, then concatenating, to capture diverse relationship types. (Ch 8)

**Non-Saturating GAN (NS-GAN)** — GAN variant maximizing log(D(G(z))) instead of minimizing log(1−D(G(z))) to avoid the saturating gradient problem. (Ch 2)

**Oscillation (GAN training)** — Failure mode where generator/discriminator losses fluctuate erratically without converging. (Ch 3)

**Patch embedding** — ViT technique dividing an image into fixed-size patches, flattened and linearly projected into embedding vectors. (Ch 8, 9)

**PatchGAN** — Pix2Pix's discriminator design classifying local image patches rather than the whole image. (Ch 2)

**PDF (Probability Density Function)** — Function f(x) describing relative likelihood of a continuous random variable; integrates to 1. (Ch 1)

**Pix2Pix** — Conditional GAN for paired image-to-image translation using a U-Net-like generator and PatchGAN discriminator. (Ch 2)

**Positional encoding/embedding** — Injected information about token/patch order or spatial location, since self-attention is otherwise permutation-invariant. (Ch 8)

**Posterior collapse** — VAE failure where the approximate posterior collapses to the prior, producing an uninformative latent space. (Ch 6, 7)

**Progressive GAN (PGAN)** — GAN trained with resolution growing in stages for stable high-resolution image generation. (Ch 2)

**Reconstruction error/loss** — Measure of dissimilarity between AE/VAE input and reconstructed output (e.g., MSE, BCE). (Ch 1, 4, 5)

**Reparameterization trick** — VAE technique decoupling stochastic sampling (ε~N(0,1)) from trainable parameters (μ,σ) via z=μ+σ⊙ε, enabling backpropagation through sampling. (Ch 6, 7, 9, 10)

**Saturating gradient problem** — Early-GAN-training issue where D(G(z))≈0 causes log(1−D(G(z))) to produce near-zero gradients for the generator. (Ch 2)

**Scaled dot-product attention** — α = softmax(QKᵀ/√d_k); standard attention computation preventing vanishing gradients at high key dimensionality. (Ch 8)

**Self-attention** — Mechanism letting each sequence element attend to every other element, producing context-aware representations in parallel. (Ch 8)

**Soft attention** — Continuous, differentiable attention producing a weighted average over all input regions. (Ch 8)

**STL-10 dataset** — 96×96 RGB, 10-class benchmark with 5,000 labeled train / 8,000 test / 100,000 unlabeled images; harder than CIFAR-10/MNIST. (Ch 9, 10)

**StyleGAN** — GAN introducing a mapping network and style-modulation blocks for fine-grained, disentangled style control. (Ch 2)

**Swin Transformer (Shifted Window Transformer)** — Hierarchical ViT variant computing self-attention in local shifted windows, reducing complexity from quadratic to linear in image size. (Ch 10)

**Training instability (GAN)** — Failure where the loss function never converges due to inappropriate loss functions, bad hyperparameters, or noisy data. (Ch 3)

**TransVAE** — VAE whose encoder/decoder incorporate Transformer (self-attention) layers instead of pure CNN/Dense layers. (Ch 10)

**Vanishing gradients (GAN/VAE)** — Gradients shrinking to near-zero through the network, stalling learning; common with sigmoid/tanh saturation. (Ch 3, 7)

**Variational Autoencoder (VAE)** — Autoencoder with a stochastic, learned-distribution latent space (qφ(z|x)), trained via ELBO to enable true generative sampling. (Ch 1, 6, 7, 9, 10)

**Vector database** — Database storing items as numerical vectors with similarity-based indexing/retrieval (e.g., Pinecone). (Ch 11)

**Vision Transformer (ViT)** — Transformer architecture applied to images by treating fixed-size patches as a sequence of tokens. (Ch 8, 9, 10)

**Wasserstein GAN (WGAN)** — GAN replacing binary cross-entropy with Wasserstein (Earth Mover's) distance via a critic network and weight clipping for training stability. (Ch 2, 3, 7)

**Weight tying (AE)** — Regularization technique forcing encoder weights to equal (transposed) decoder weights. (Ch 4)
