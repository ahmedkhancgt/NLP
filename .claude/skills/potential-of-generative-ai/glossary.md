# Glossary

**Adversarial loss** — the GAN loss term capturing the generator's success at fooling the discriminator and the discriminator's success at telling real from fake (Ch1, Ch2, Ch3, Ch7).

**Algorithmic artistry** — art created via generative algorithms where artists define parameters/rules that autonomously produce visual or auditory output (Ch5).

**Amazon ECR (Elastic Container Registry)** — AWS's managed Docker image registry used to store inference containers before SageMaker deployment (Ch8).

**Amazon SageMaker** — AWS's fully managed machine learning platform for building, training, and deploying models at scale (Ch2, Ch3, Ch7, Ch8).

**Autoencoder** — neural network with an encoder-decoder architecture that compresses input into a latent representation and reconstructs it; basis for VAEs (Ch1, Ch3, Ch8).

**Autoregressive model** — generative model (e.g., GPT) that produces sequential output where each step conditions on all previous steps (Ch1, Ch3).

**Azure Machine Learning** — Microsoft's cloud platform for building, training, and deploying ML models, with strong responsible-AI tooling (Ch8).

**BERT (Bidirectional Encoder Representations from Transformers)** — transformer-based contextual language model using masked-language-model pre-training; used for understanding, not generation (Ch1, Ch3).

**Cycle-consistency loss** — CycleGAN's loss term ensuring an image translated to another domain and back resembles the original (Ch1, Ch3).

**CycleGAN** — GAN architecture for unpaired image-to-image translation using dual generators/discriminators and cycle-consistency loss (Ch1, Ch2, Ch3, Ch7).

**DCGAN (Deep Convolutional GAN)** — GAN architecture using convolutional/deconvolutional layers (no fully connected layers) for high-resolution image generation (Ch1, Ch2).

**Deepfake** — AI-generated (often GAN-based) synthetic video/image/audio content designed to convincingly depict a person saying or doing something they did not (Ch1, Ch2, Ch5, Ch7).

**DeepDream** — Google's CNN feature-amplification technique that transforms images into dreamlike, hallucinatory visuals; not a true generative model but a creative visualization tool (Ch1, Ch3, Ch5).

**Discriminative model** — a model that classifies or predicts labels from data (e.g., logistic regression, SVM), contrasted with generative models that create new data (Ch1, Ch3).

**Discriminator** — the GAN network trained to distinguish real data from generator-produced fake data (Ch1, Ch2, Ch3, Ch7, Ch8).

**FID (Fréchet Inception Distance)** — quantitative metric measuring the distance between real and generated data distributions, used to evaluate generative models (Ch1, Ch2, Ch3, Ch7, Ch8).

**Generative adversarial network (GAN)** — generative model consisting of a generator and discriminator trained via adversarial competition (all chapters).

**Generative AI** — AI subfield focused on creating new, realistic data instances rather than only classifying existing data (Ch1).

**Generative design** — algorithm-driven exploration of a large design space under defined constraints, used in manufacturing, architecture, and robotics (Ch2, Ch4, Ch5, Ch6).

**Generator** — the GAN network responsible for producing synthetic data samples intended to resemble real data (Ch1, Ch2, Ch3, Ch7, Ch8).

**Google AI Platform (Unified)** — Google Cloud's comprehensive ML platform combining AutoML and classic AI Platform capabilities (Ch8).

**Inception Score (IS)** — metric assessing the quality and diversity of generated samples (Ch1, Ch3, Ch7, Ch8).

**KL divergence (Kullback-Leibler divergence)** — regularization term in VAE training that smooths/constrains the latent space distribution (Ch3).

**Knowledge distillation** — technique for transferring a large model's learned behavior into a smaller model to reduce size/latency (Ch8).

**Latent space** — a lower-dimensional, often continuous representation of data learned by an encoder, from which new samples can be generated or interpolated (Ch1, Ch3, Ch5).

**Mode collapse** — GAN failure mode where the generator produces only a limited subset of possible outputs (Ch1, Ch2, Ch3, Ch7, Ch8).

**Model quantization** — reducing the numeric precision of model weights/activations to shrink model size and reduce computational cost (Ch7, Ch8).

**Normalizing flow** — generative model using invertible transformations to turn a simple distribution into a complex one (Ch8).

**Perceptual loss** — loss function comparing high-level feature similarity (not raw pixels) between generated and real samples, used in style transfer (Ch3).

**Personalized medicine** — healthcare application where generative AI analyzes patient-specific data (genetic, clinical) to tailor treatment plans (Ch2, Ch4, Ch6).

**Progressive GAN (PGAN)** — GAN architecture that trains on progressively higher resolutions to stabilize training of high-resolution image synthesis (Ch1, Ch2, Ch7).

**Reconstruction loss** — loss term in autoencoders/VAEs measuring fidelity between input and reconstructed output (Ch3).

**StyleGAN** — NVIDIA's GAN architecture introducing a style-based generator and mapping network for fine-grained control over generated image style (Ch1, Ch2, Ch3, Ch7).

**Training instability** — GAN failure mode where generator/discriminator fail to reach a stable equilibrium due to the non-convex adversarial objective (Ch1, Ch2, Ch3, Ch7, Ch8).

**Transfer learning (generative context)** — fine-tuning a pre-trained generative model on a related task/domain to reduce training time/data requirements (Ch3, Ch6).

**Variational autoencoder (VAE)** — probabilistic generative model with an encoder mapping data to a latent distribution and a decoder generating/reconstructing from samples of it (all chapters).

**Wasserstein GAN (WGAN)** — GAN variant using the Wasserstein (Earth Mover's) distance instead of binary cross-entropy loss for more stable training (Ch1, Ch2, Ch3, Ch7).
