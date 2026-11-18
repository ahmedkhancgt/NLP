# Cheatsheet

## GAN vs. VAE: which to pick?

| Need | Pick | Why |
|---|---|---|
| Sharpest, most photorealistic output | GAN | Adversarial training pushes toward realism |
| Smooth, interpolatable latent space | VAE | KL-regularized latent space is continuous/structured |
| Stable, predictable training | VAE | No adversarial min-max dynamics |
| Best-in-class for faces/high-res images | GAN (StyleGAN family) | Style-based generator + progressive growing |
| Molecule/music generation with uncertainty | VAE | Probabilistic sampling fits exploratory design |
| No paired training data, need domain translation | CycleGAN | Cycle-consistency loss removes pairing requirement |
| Want both realism and a structured latent space | Hybrid (VAE-GAN, AAE-GAN) | Combines encoder-driven latent control with adversarial sharpness |

## GAN failure-mode triage

| Symptom | Likely cause | First fix to try |
|---|---|---|
| Outputs look similar / lack variety | Mode collapse | Diversify loss functions, regularize generator, diversify training data |
| Losses oscillate or diverge | Training instability | Gradient clipping, batch normalization, try a different optimizer |
| Generator stops improving early | Discriminator saturation | Use non-saturating loss (maximize D(G(z)) instead of minimizing 1-D(G(z))) |
| Training/inference too slow or expensive | Computational cost | Smaller architecture, distributed training, specialized hardware (TPU/GPU) |
| Outputs are plausible but misused/deceptive | Ethical misuse risk | Transparency, auditing/monitoring, disclosure, regulation |

## Loss-function decision rule

- Default/simple case → binary cross-entropy (vanilla GAN).
- Instability or mode collapse under BCE → switch to Wasserstein distance (WGAN / WGAN-GP).
- Need style/perceptual fidelity (not just pixels) → perceptual loss.
- Need reversible domain translation → add cycle-consistency loss (CycleGAN).
- Need smooth latent space → reconstruction loss + KL divergence (VAE).

## Deployment-readiness go/no-go checklist (Ch8)

Before promoting a generative model to production, confirm all five:
1. **Compute** — do you have sustainable (not just one-off) access to adequate training/inference compute?
2. **Size** — is the model small enough for its target environment (cloud endpoint vs. mobile/edge)?
3. **Latency** — does inference time meet the application's real-time requirement?
4. **Accuracy** — has it been evaluated with FID/Inception Score or a task-specific metric (not just "looks fine")?
5. **Fairness** — has it been audited with statistical fairness metrics (demographic parity, equalized opportunity)?

A model failing any one of these is not production-ready, regardless of how good the others look.

## Cloud platform tell/smell — which one fits?

| If you value most... | Choose | Tell |
|---|---|---|
| Breadth of mature features/pre-built algorithms | AWS SageMaker | Most third-party integrations, but medium ease-of-use |
| Lowest learning curve | Google AI Platform (Unified) | AutoML-first design, simplest onboarding |
| Deep integration with existing enterprise cloud stack + responsible-AI tooling | Azure Machine Learning | Native fit with Databricks/Synapse/DevOps CI-CD |

## Deployment pipeline — universal 5 steps (map to any cloud)

1. Serialize model (SavedModel / ONNX / TorchScript)
2. Containerize (Docker)
3. Push to registry (Amazon ECR / Azure Container Registry / GCR)
4. Create platform model resource (`aws sagemaker create-model` / `az ml model register` / `gcloud ai-platform models create`)
5. Create endpoint config + endpoint, then wait for in-service

## Ethical-risk category triage (don't conflate these)

| Category | Example from the book | Distinct mitigation |
|---|---|---|
| Bias/fairness | Microsoft Tay generating racist tweets | Diverse training data, fairness-aware training, audits |
| Privacy/consent | Clearview AI scraping faces without consent | Data minimization, anonymization, opt-out controls |
| Copyright/authorship | Artist lawsuits over Imagen training data | Clear licensing/attribution policy, disclosure |
| Misuse/deepfakes/weaponization | Deepfakes, autonomous-weapons concern | Transparency, auditing, international governance |

## Quick architecture-to-fix map (Ch7)

- Mode collapse / instability at high resolution → **ProgressiveGAN**
- Need fine-grained style control → **StyleGAN**
- BCE loss causing instability → **WGAN / WGAN-GP**
- Discriminator overpowering too fast → **SNGAN (spectral normalization)**
- Need two independent realism checks → **DualDGAN**
