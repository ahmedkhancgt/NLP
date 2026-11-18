# Chapter 1: Introduction to Generative AI

## Core Idea
Generative AI is the branch of AI that creates new, realistic data instances (images, text, music, molecules) rather than only classifying or predicting from existing patterns, and it has evolved from 1950s rule-based/Markov models through 1990s neural-network resurgence to the 2014 GAN breakthrough and beyond.

## Frameworks Introduced
- **Generative vs. Discriminative distinction**: generative AI focuses on creating new content that resembles training data; general AI/discriminative models classify or predict from it.
  - When to use: frame any new project first by asking whether the goal is to *produce* novel data or to *judge/label* existing data.
  - How: identify whether the model learns a data distribution to sample from (generative) or a decision boundary between classes (discriminative).
- **Generator/Discriminator pairing (GAN)**: a generator creates content; a discriminator assesses authenticity; adversarial training refines both.
  - When to use: tasks needing realistic synthetic samples (images, video, audio) where you can define a "real vs. fake" judge.
  - How: train the two networks simultaneously in a minimax game until the generator's output fools the discriminator consistently.
- **Encoder/Decoder with stochastic latent space (VAE)**: encoder maps input to a probabilistic latent representation; decoder reconstructs/generates from samples of that distribution.
  - When to use: when you need a smooth, explorable latent space (e.g., interpolating between data points) or need a probabilistic handle on uncertainty.
  - How: train jointly to minimize reconstruction error while regularizing the latent distribution.
- **Autoregressive generation (GPT-style)**: output at time *t* conditions on all previous time steps (h_t depends on x_t and all prior x's).
  - When to use: sequential data such as natural language, where context and order matter.
  - How: predict the next token/step conditioned on the full preceding sequence.

## Key Concepts
- **Generative AI**: AI subfield focused on creating new data instances that resemble or innovate beyond training data.
- **GAN (Generative Adversarial Network)**: two-network adversarial architecture (generator vs. discriminator) introduced by Ian Goodfellow et al. in 2014.
- **VAE (Variational Autoencoder)**: probabilistic generative model using encoder/decoder with a stochastic latent space.
- **Autoregressive model**: generates sequences step-by-step, each step conditioned on prior outputs (e.g., GPT).
- **Mode/architecture diversity**: DCGAN (image synthesis), StyleGAN (high-res diverse images), CLIP (cross-modal text-image alignment), BERT (bidirectional NLP context).
- **FID / Inception Score**: standard quantitative metrics for evaluating the realism/diversity of generated outputs.
- **Saturating/non-saturating loss**: training dynamics issue addressed by later GAN loss formulations (covered more deeply in Ch7).

## Mental Models
- Think of the GAN as a forger (generator) and detective (discriminator) locked in an arms race — quality emerges from the competition, not from either network alone.
- Think of a VAE's latent space as a continuous map of "concept space" — moving smoothly across it morphs one generated output into another (e.g., cat → dog).
- Use the evolutionary-trajectory lens (Markov models → RNN/CNN → GANs → VAEs → multimodal models like CLIP) to place any new technique in historical context and anticipate what problem it was built to solve.

## Anti-patterns
- **Treating generative AI as "just AI"**: conflating it with discriminative/classification AI leads to wrong tool selection and wrong evaluation metrics (accuracy vs. FID/Inception Score).
- **Ignoring the saturating-gradient problem**: naive GAN loss implementations stall generator learning early in training; use the non-saturating formulation instead.
- **Deploying without bias/ethics review**: facial-recognition and deepfake use cases show that skipping fairness and interpretability review causes real-world harm.

## Reference Tables
| Era | Key development |
|---|---|
| 1950s–1980s | Rule-based systems, Markov models, early text generation |
| 1990s | Resurgence of RNNs and CNNs (limited by compute) |
| 2014 | GANs introduced (Goodfellow et al.) |
| 2010s onward | VAEs, real-world deployment (Deepfake, drug discovery), cross-modal models (CLIP) |

## Worked Example
A GAN trained on da Vinci's paintings: the generator produces candidate "paintings"; the discriminator, trained on real da Vinci works, scores how authentic each candidate looks. Over many iterations the generator's outputs become increasingly indistinguishable from genuine da Vinci-style art — illustrating how domain-specific training data determines the kind of content a generative model can produce.

## Key Takeaways
1. Generative AI's defining trait is creation of new data instances, not classification of existing ones — pick models/metrics accordingly.
2. GANs (adversarial) and VAEs (probabilistic encoder/decoder) are the two foundational architecture families; most later variants (DCGAN, StyleGAN, CycleGAN) extend one of them.
3. Evaluate generative outputs with FID/Inception Score, not classification accuracy.
4. Real-world breakthroughs (Deepfake, drug discovery, CLIP) show generative AI's reach beyond images — into molecules, text, and cross-modal tasks.
5. Ethical risk (bias, interpretability, adversarial robustness) is inherent to the technology from its very foundations, not an afterthought.

## Connects To
- **Ch3**: expands these same GAN/VAE foundations into full mathematical and architectural detail.
- **Ch7**: revisits the saturating-gradient problem and GAN training dynamics in depth.
