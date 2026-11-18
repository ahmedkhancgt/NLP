# Cheatsheet

## Architecture Selection: GAN vs. VAE vs. Transformer

| Need | Choose | Why |
|---|---|---|
| Sharpest/most realistic images | GAN | Adversarial training pushes toward photorealism |
| Stable training, interpretable latent space | VAE | Probabilistic latent space, simpler loss, easier to train |
| Controlled attribute editing (age, pose, style) | VAE or StyleGAN | Structured/mapped latent space allows targeted edits |
| Denoising / robust to messy/missing data | VAE | Reconstruction-based training tolerates noisy input |
| Open-ended text generation, chat | GPT-class Transformer | Autoregressive generation, broad pre-training |
| Deep text understanding/classification (sentiment, QA, NER) | BERT-class Transformer | Bidirectional context, fine-tunable |
| Long-range sequence dependencies | Transformer (self-attention) | Avoids RNN vanishing-gradient limits |
| Short sequential data, simple memory needs | RNN/LSTM | Lower overhead than full Transformer for simple cases |
| Image-to-image translation (unpaired) | CycleGAN | Designed for unpaired domain translation |
| Class/label-conditioned generation | Conditional GAN | Built-in conditioning mechanism |
| Anomaly detection | VAE (reconstruction error) or statistical threshold | Reconstruction error flags deviation from learned "normal" |

## Training-Failure Smells

| Symptom | Likely cause | Fix |
|---|---|---|
| Generator produces near-identical outputs | Mode collapse (GAN) | Architectural tweaks, regularization, progressive growing |
| GAN loss oscillates wildly / never converges | Training instability (generator/discriminator imbalance) | Tune learning rates, normalization, alternate update schedule |
| VAE outputs are blurry/unclear | Latent space poorly structured or data lacks clarity | Tune KL weight, improve data quality/diversity |
| Fine-tuned model memorizes training set | Overfitting on niche dataset | Dropout, regularization (L1/L2), early stopping, data augmentation |
| Validation accuracy drops after deployment | Model drift (data distribution shift) | Monitor KPIs, automate retraining above a drift threshold |
| Generated text drifts off-topic | Lost relevance to prompt | Relevance scoring + refinement/regeneration step |
| Style-transfer output has artifacts/distortions | Unbalanced content/style loss weighting | Tune `alpha`/`beta`, adjust learning rate/iterations |
| Real-time inference too slow | Model too large/complex for latency budget | Quantization, pruning, ensemble (light model for easy cases) |
| RL agent stuck in suboptimal behavior | Over-exploitation, insufficient exploration | Rebalance exploration/exploitation, reward-shaping review |
| RL agent reward exploited in unintended way | Poor reward function design | Redesign reward signal, add constraints/safety checks |

## Defaults and Thresholds the Author Commits To

| Parameter / threshold | Typical value cited | Context |
|---|---|---|
| GAN noise vector dimension | 100 | MNIST-style GAN generator input |
| VAE latent dimension | 2–10 | Simple demo VAEs (MNIST, text, recommendation) |
| Adam learning rate (style transfer) | 7.0 (pixel optimization) / 1e-4 (VAE) / 0.001 (DRL/policy gradient) | Varies by what's being optimized |
| Style transfer loss weights | alpha=10 (content), beta=40 (style) | Balances content fidelity vs. style strength |
| Discount factor (RL) | 0.99 | Standard for Q-learning / policy gradient (CartPole examples) |
| Anomaly detection threshold (demo) | 80 (on a 10–100 scale) | Simple fixed-threshold network traffic example |
| Pruning amount (model acceleration) | 0.3 (30% of weights) | Global L1 unstructured pruning example |
| Early-stopping patience | Not fixed — tune via validation monitoring | Generic guidance across fine-tuning case studies |
| Dropout / regularization | Not fixed — "increase if overfitting observed" | Generic overfitting countermeasure |

## Quick Decision Rules

- **Need generation + photorealism** → GAN family (consider StyleGAN2 for faces/high-res).
- **Need generation + stability/interpretability** → VAE.
- **Need generation + long text/chat** → Transformer (GPT-class), fine-tuned on domain data.
- **Need classification/understanding of existing text** → Transformer (BERT-class).
- **Need agent that learns from interaction** → RL; pair with a generative model if diverse training scenarios are needed.
- **Need real-time/low-latency inference** → quantize + prune + consider ensemble routing; validate accuracy loss before shipping.
- **Deploying any model to a changing environment** → add drift monitoring + automated/human-in-the-loop retraining from day one, not as an afterthought.
- **Any creative-generation product (text/image/code)** → always include a human-review or relevance/coherence-check step before output reaches the end user.
