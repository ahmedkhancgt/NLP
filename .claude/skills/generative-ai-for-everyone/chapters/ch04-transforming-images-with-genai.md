# Chapter 4: Transforming Images with GenAI

## Core Idea
Image GenAI is dominated by three competing generative-modeling families — GANs (implicit, adversarial), diffusion models (explicit, iterative denoising), and flow models (explicit, invertible transformations) — each trading off training stability, sample quality, diversity, and compute cost differently, with diffusion models (DALL-E 2, Stable Diffusion, Midjourney) currently dominant for state-of-the-art image synthesis.

## Frameworks Introduced
- **Generative model taxonomy**: implicit density (GANs — no explicit distribution, just a sampling procedure) vs. explicit density, split into tractable (autoregressive, flow models) and approximate (VAE, diffusion).
  - When to use: classifying any new generative architecture you encounter.
  - How: ask whether it defines an explicit likelihood (explicit) and whether that likelihood is exactly computable (tractable) or only approximated via a bound like ELBO (approximate).
- **GAN min-max training**: generator G vs. discriminator D in adversarial competition; `min_G max_D [log D(x) + log(1-D(G(z)))]`.
  - When to use: tasks needing sharp, realistic samples where training stability can be managed (image synthesis, super-resolution, style transfer).
  - How: alternate training — fix G, train D to maximize real/fake discrimination; fix D, train G to minimize D's ability to detect fakes. WGAN swaps JS/KL divergence for Wasserstein distance to fix vanishing-gradient/mode-collapse issues.
- **GAN variant selection by task**: cGAN (paired translation), CycleGAN (unpaired domain transfer), SRGAN (super-resolution), StyleGAN (style transfer/deepfakes), GauGAN (sketch-to-photo), VQ-GAN (+CLIP) (text-to-image via discrete codebook + transformer).
  - When to use: pick the variant matching your paired/unpaired-data and task constraints.
  - How: cGAN conditions G and D on labels; CycleGAN adds cyclic-consistency loss (`F(G(x))≈x`) to train without paired data; VQ-GAN trains a GAN to learn a codebook of visual "words," then a transformer models their composition autoregressively.
- **Denoising Diffusion Probabilistic Models (DDPM)**: forward process adds Gaussian noise over T Markov steps; reverse process uses a trained U-Net to iteratively denoise from pure noise back to data.
  - When to use: highest current image-fidelity benchmark; also conditional generation (text-to-image).
  - How: forward process is fixed/untrained (reparameterizable to sample any timestep directly); reverse process trains a U-Net (with time embeddings + self-attention) to predict the noise added, optimized via an ELBO-derived loss that reduces to MSE between true and predicted noise.
- **Conditioning strategies for diffusion**: classifier guidance, CLIP guidance, classifier-free guidance.
  - When to use: steering generation toward a label/prompt (text-to-image).
  - How: classifier guidance injects a trained classifier's gradient into the reverse process; CLIP guidance replaces the classifier with a CLIP model trained on noisy images; classifier-free guidance avoids training a separate classifier by randomly dropping the conditioning label during training and blending conditional/unconditional predictions at sampling time (weight ω).
- **Latent diffusion models (LDM)**: run the diffusion process in a VAE's compressed latent space rather than pixel space, then decode back to pixels.
  - When to use: need faster training/sampling than pixel-space diffusion (this is the Stable Diffusion approach).
  - How: encode image to latent z with VAE encoder; diffuse/denoise z with a time-conditioned, cross-attention-conditioned U-Net; decode z0 back to pixels.
- **Normalizing flow models (Real NVP, GLOW)**: build a generative model from a sequence of invertible, differentiable transformations with tractable Jacobian determinants.
  - When to use: when exact likelihood computation and lossless reconstruction are required.
  - How: Real NVP uses coupling layers that transform a masked subset of dimensions via affine functions parameterized by the unmasked subset; GLOW replaces reverse masking with invertible 1x1 convolutions plus ActNorm.

## Key Concepts
- **Generative vs. discriminative model**: generative learns p(x,y) or p(x) to create new data; discriminative learns p(y|x) to classify/predict.
- **Wasserstein (earth mover's) distance**: a smoother distance metric than KL/JS divergence, used in WGAN to fix training instability on non-overlapping distributions.
- **Cyclic consistency loss**: in CycleGAN, enforces that translating an image to another domain and back reconstructs the original (F(G(x))≈x), enabling unpaired training.
- **Codebook (VQ-GAN)**: a learned finite vocabulary of discrete visual-part vectors that a transformer then models compositionally.
- **Forward/reverse diffusion process**: forward = fixed, incremental Gaussian noising (Markov chain); reverse = learned denoising via a neural network (typically U-Net).
- **Sinusoidal time embedding**: encodes the current diffusion timestep into the U-Net's residual blocks so the model knows "how noisy" its input currently is.
- **Classifier-free guidance weight (ω)**: ω=0 fully unconditional, ω=1 standard conditional, ω>1 strongly conditional — used by DALL-E 2 and Imagen.
- **Invertible/bijective function**: a one-to-one mapping enabling lossless reconstruction between data space and latent space (core requirement for flow models).
- **Inception Score (IS) / Fréchet Inception Distance (FID)**: standard generative-image quality metrics using a pretrained Inception-v3 network; FID measures distributional distance between real and generated image embeddings (lower is better).

## Mental Models
- Think of **GAN training as a continually escalating forgery arms race**: the generator and discriminator must improve in lockstep; if either gets too far ahead, training collapses (mode collapse, vanishing gradients) — this is why WGAN's smoother distance metric matters.
- Use the **"noise-up then learn-to-undo" frame** for diffusion: training never requires running the full forward chain — the model only needs to learn, at any random timestep, how to predict and remove the noise that was added.
- Think of **latent diffusion as "diffusion on a compressed thumbnail"**: by moving the expensive iterative process into a VAE's small latent space, you get the same generative quality at a fraction of the compute — the single biggest practical lever for making diffusion models deployable.
- Use the **conditioning spectrum (unconditional ↔ weakly conditional ↔ strongly conditional)** as a dial, not a switch, when prompting or tuning diffusion models — classifier-free guidance's ω literally implements this dial.

## Anti-patterns
- **Training vanilla GANs on non-overlapping distributions with JS/KL-based loss**: leads to vanishing gradients and mode collapse — use Wasserstein distance (WGAN) instead.
- **Using CycleGAN-style unpaired translation when paired data is actually available**: pix2pix-style cGAN will generally train faster/better with paired supervision — reserve CycleGAN for genuinely unpaired domains.
- **Running diffusion directly in pixel space for large images without need**: wastes compute that latent diffusion would save; pixel-space diffusion is justified only when latent compression would lose needed detail.
- **Using classifier guidance when you can't afford to train a dedicated noisy-image classifier**: classifier-free guidance avoids this cost and is simpler to deploy at similar quality.
- **Evaluating generative image models with classification metrics (accuracy/F1)**: these don't capture sample diversity or perceptual quality — use IS/FID/PPL/LPIPS/SSIM instead.

## Code Examples
```python
# DDPM-style diffusion training step (predict and remove injected noise)
def training(self, batch_size, optimizer):
    x0 = sample_batch(batch_size, self.device)
    t = torch.randint(1, self.T + 1, (batch_size,), device=self.device, dtype=torch.long)
    epsilon = torch.randn_like(x0)
    alpha_bar_t = self.alpha_bar[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    predicted_epsilon = self.UNet_Model(
        torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon, t-1)
    loss = nn.functional.mse_loss(epsilon, predicted_epsilon)  # core DDPM loss
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()

# DDPM sampling (reverse process: noise -> image)
def sampling(self, num_samples=1, img_chn=1, img_size=(32, 32), use_tqdm=True):
    x = torch.randn((num_samples, img_chn, img_size[0], img_size[1]), device=self.device)
    for t in range(self.T, 0, -1):
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        t_tensor = torch.ones(num_samples, dtype=torch.long, device=self.device) * t
        beta_t = self.beta[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_t = self.alpha[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_bar_t = self.alpha_bar[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * self.UNet_Model(x, t_tensor-1))
        x = mean + torch.sqrt(beta_t) * z
    return x
```
```python
# Text-to-image via CLIP + VQ-GAN: cosine-similarity loss between text and image embeddings
def optimize_result(Params, prompt):
    alpha, beta = 1, 0.5
    out = generator(Params())              # generate candidate image
    out = Img_Crops(normalize(norm_data(out)))
    image_enc = clipmodel.encode_image(out)
    text_encoding_include = (weight1 * prompt + weight2 * extras_encoding)
    text_encoding_include = text_encoding_include / text_encoding_include.norm(dim=-1, keepdim=True)
    inc_loss = torch.cosine_similarity(text_encoding_include, image_enc, -1)
    exc_loss = torch.cosine_similarity(exclude_encoding, image_enc, -1)
    return -alpha * inc_loss + beta * exc_loss  # minimize to align image with prompt
```
- **What it demonstrates**: (1) the DDPM train/sample loop — training learns to predict injected noise, sampling iteratively removes predicted noise starting from pure Gaussian noise; (2) using a pretrained CLIP model's image/text embedding space to steer a VQ-GAN generator toward a text prompt via gradient-based optimization of cosine similarity.

## Reference Tables
**Generative image model families**

| Family | Density type | Training signal | Example models |
|---|---|---|---|
| GAN | Implicit | Adversarial min-max | SRGAN, CycleGAN, StyleGAN, VQ-GAN |
| VAE | Explicit (approximate) | Reconstruction + KL (ELBO) | VAE, β-VAE, VQ-VAE (Ch 3) |
| Diffusion | Explicit (approximate) | Denoising MSE (ELBO-derived) | DDPM, DALL-E 2, Stable Diffusion, Imagen |
| Flow | Explicit (tractable) | Exact negative log-likelihood | Real NVP, GLOW |

**GAN variant → application**

| Variant | Primary application |
|---|---|
| cGAN | Paired image translation (pix2pix: satellite→map, edges→photo) |
| CycleGAN | Unpaired domain transfer (photo↔painting, MRI↔CT, day↔night) |
| SRGAN | Super-resolution (low-res → high-res) |
| StyleGAN | Style transfer, synthetic faces, deepfakes |
| GauGAN | Sketch/doodle → photorealistic image |
| VQ-GAN (+CLIP) | Text-to-image generation |

**Diffusion conditioning methods**

| Method | Mechanism | Cost |
|---|---|---|
| Classifier guidance | Separate classifier trained on noisy images, gradient injected at sampling | High (extra model to train) |
| CLIP guidance | CLIP model trained on noised images replaces classifier | Moderate |
| Classifier-free guidance | Random conditioning dropout during training; blend conditional/unconditional at sampling (weight ω) | Low; used by DALL-E 2, Imagen |

**Generative image evaluation metrics**

| Metric | Measures | Direction |
|---|---|---|
| Inception Score (IS) | Diversity + sharpness via Inception-v3 classification | Higher better |
| Fréchet Inception Distance (FID) | Distributional distance (real vs. generated embeddings) | Lower better |
| Perceptual Path Length (PPL) | Smoothness/continuity of latent space | Lower better |
| LPIPS | Perceptual similarity via deep network activations | Context-dependent |
| SSIM | Structural similarity (luminance, contrast, structure) | Higher better (range -1 to 1) |
| PSNR | Reconstruction quality under lossy compression | Higher better |
| Bits per dimension (bpd) | Encoding efficiency (flow models) | Lower better |

## Worked Example
**Text-to-image with CLIP + VQ-GAN**: Load a pretrained CLIP ViT-B/16 model and a VQ-GAN (taming-transformers) generator. Encode an "include" text prompt (e.g., "Road with animals and trees") and an "exclude" text description of unwanted attributes (e.g., "Red, low resolution, blurry") into CLIP's 512-d embedding space. Generate a candidate image from the VQ-GAN latent space, crop multiple patches, embed them with CLIP, and compute a loss combining positive cosine similarity to the include-text embedding and negative cosine similarity to the exclude-text embedding. Backpropagate this loss into the VQ-GAN's latent parameters (not CLIP's weights) iteratively, progressively steering the generated image to match the prompt while avoiding the excluded attributes — producing a final RGB image from pure text input. A second worked example trains an SRGAN (generator with residual blocks + pixel-shuffle upsampling, discriminator with strided conv blocks) on the DIV2K dataset using a perceptual loss (VGG content loss + adversarial loss) to upscale low-resolution images to high-resolution ones.

## Key Takeaways
1. GANs, VAEs/diffusion, and flow models represent three distinct trade-offs (implicit vs. explicit density, tractable vs. approximate likelihood) — know which family a given generative tool belongs to before tuning it.
2. Diffusion models' core training trick is deceptively simple: learn to predict the noise that was added, not the clean image directly — this MSE-on-noise objective is what makes training stable compared to GANs.
3. Classifier-free guidance is the practical default for text-conditioned diffusion (DALL-E 2, Imagen) because it avoids training a separate noisy-image classifier.
4. Latent diffusion (Stable Diffusion's approach) is the key efficiency unlock: moving the diffusion process into a compressed VAE latent space rather than pixel space.
5. CycleGAN's cyclic-consistency loss is the general solution whenever you need image-to-image translation but lack paired training data.
6. Evaluating generative image models requires distribution/perceptual metrics (FID, IS, LPIPS, SSIM) instead of classification metrics — there is no single "accuracy" for generative quality.

## Connects To
- **Ch 3**: directly extends the U-Net architecture and VAE/VQ-VAE representation-learning concepts introduced there into GAN, diffusion, and VQ-GAN architectures.
- **Ch 5**: the CLIP text-image embedding alignment and transformer-based codebook modeling (VQ-GAN's second-stage transformer) foreshadow the transformer architecture detailed next for text.
- **Ch 6**: DALL-E/CLIP-style multimodal grounding connects to OpenAI's broader model family (DALL-E, Whisper) discussed alongside ChatGPT.
- **Ch 8**: evaluation metrics (FID, IS) and model-serving/optimization concerns here parallel the LLMOps evaluation and inference-optimization practices for text models.
