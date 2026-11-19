# Chapter 3: GenAI for Images

## Core Idea
Convolutional neural networks (CNNs) give machines the ability to extract structured visual features, and representation learning (autoencoders → VAEs → VQ-VAE) turns that visual data into a compact, navigable latent space — the shared substrate that both discriminative CV tasks (classification/segmentation/detection) and generative image synthesis build on.

## Frameworks Introduced
- **CNN architecture evolution (2012-2020)**: AlexNet → VGG-Net → GoogLeNet (inception modules) → ResNet (skip connections) → MobileNet (depth-wise separable convolutions) → DenseNet (dense connectivity) → EfficientNet (compound scaling).
  - When to use: picking a backbone by constraint — accuracy (ResNet/DenseNet), mobile/edge (MobileNet), balanced scaling (EfficientNet).
  - How: each generation solves a prior generation's bottleneck (vanishing gradients → skip connections; compute cost → depth-wise separable convolutions; manual scaling → compound coefficient).
- **CNN building blocks**: convolution layer (learnable filters, output size `Wout=(W−F+2P)/S+1`), padding (same/zero/valid/causal), pooling (max/average), fully connected layer.
  - When to use: designing any CV model from scratch.
  - How: stack conv→activation→pool blocks to extract hierarchical features, flatten into FC layers with softmax for classification.
- **U-Net encoder-decoder for segmentation**: symmetric contracting (encoder) + expansive (decoder) path with a bottleneck and skip connections.
  - When to use: pixel-wise prediction tasks (semantic/instance segmentation), especially with limited data.
  - How: encoder downsamples via conv+pool while increasing depth; decoder upsamples via transposed convolution; skip connections from encoder to decoder preserve spatial detail lost during contraction.
- **Representation learning via autoencoder/VAE**: map high-dimensional input to a low-dimensional latent space, then reconstruct.
  - When to use: unsupervised feature learning, image generation, data imputation, style transfer.
  - How: autoencoder minimizes reconstruction MSE deterministically; VAE instead learns a probability distribution (mean μ, variance σ²) over latent codes, optimized via reconstruction loss + KL-divergence regularization (ELBO), using the reparameterization trick (`z = μ + ε·σ`) to keep sampling differentiable.
- **β-VAE and VQ-VAE as disentanglement/discretization variants**: β-VAE weights the KL term by β to encourage disentangled latent factors; VQ-VAE replaces the continuous latent with a discrete learnable codebook (vector quantization) for language/speech-style data.

## Key Concepts
- **Computer vision (CV) tasks**: image classification (whole-image label), object detection (localize + classify via bounding boxes), image segmentation (semantic = per-class boundary, instance = per-object, panoptic = both), object tracking (position across frames).
- **Receptive field**: the input region that produces one value in a convolutional activation map.
- **Skip/residual connection**: a shortcut that passes information past intermediate layers, mitigating vanishing gradients (ResNet, U-Net, DenseNet).
- **Latent space**: a low-dimensional manifold where semantically similar inputs sit close together; enables navigation, sampling, and interpolation.
- **Reparameterization trick**: expressing a stochastic latent sample as a deterministic function of a learnable parameter plus independent noise, so gradients can backpropagate through sampling.
- **ELBO (evidence lower bound)**: the VAE loss = reconstruction loss + KL-divergence regularization against a prior (commonly standard Gaussian).
- **Disentanglement**: a latent-space property where changing one latent dimension affects only one factor of variation (e.g., smile) while leaving others unchanged.
- **Codebook (VQ-VAE)**: a finite set of learnable discrete vectors that encoder outputs are snapped to via nearest-Euclidean-distance lookup, enabling discrete latent representations.
- **Straight-through estimator**: a gradient-estimation trick used in VQ-VAE to backpropagate through the non-differentiable vector-quantization step.

## Mental Models
- Think of a **CNN as a feature-extraction pipeline**: early layers learn low-level features (edges, color, orientation), deeper layers compose them into higher-level semantic features — complexity increases with depth, which is why architecture evolution (skip connections, dense blocks) exists specifically to let very deep networks train at all.
- Use the **U shape as a bottleneck-then-reconstruct pattern**: anything that must preserve spatial correspondence between input and output (segmentation, later diffusion U-Nets) adopts this contract-expand-with-skip-connections shape.
- Think of the **VAE latent space as a smooth, continuous map of a visual property space**: directions in latent space correspond to semantic factors (e.g., a "smile vector"), so image edits become vector arithmetic in latent space rather than pixel manipulation.
- Use the **determinism-vs-distribution lens** to distinguish autoencoders (one fixed latent code per input, good for compression/reconstruction) from VAEs (a distribution over latent codes, good for generating novel, varied samples).

## Anti-patterns
- **Using a plain autoencoder when you need to generate new, varied samples**: autoencoders learn one fixed mapping per input and don't support meaningful random sampling of new data — use a VAE instead.
- **Ignoring skip connections in very deep networks**: without them (pre-ResNet CNNs), gradients vanish and deeper networks actually perform worse, not better.
- **Treating vector quantization as directly differentiable**: VQ-VAE's hard nearest-codebook-vector assignment blocks normal backpropagation; skipping the straight-through estimator / stop-gradient tricks breaks training.
- **Setting β too high in β-VAE without checking reconstruction quality**: over-weighting the KL term trades away reconstruction fidelity for disentanglement — it's an explicit trade-off, not a free improvement.

## Code Examples
```python
# U-Net double-convolution block (reused in encoder and decoder)
def d_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()
        self.mx_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d_conv1, self.d_conv2, self.d_conv3 = d_conv(3, 64), d_conv(64, 128), d_conv(128, 256)
        self.d_conv4, self.d_conv5 = d_conv(256, 512), d_conv(512, 1024)
        self.u_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.u_conv1 = d_conv(1024, 512)   # 1024 in_channels because of concat with skip connection
        # ... symmetric u_trans2-4 / u_conv2-4 ...
        self.out = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        d1 = self.d_conv1(x); d2 = self.mx_pool(d1)
        d3 = self.d_conv2(d2); d4 = self.mx_pool(d3)
        d5 = self.d_conv3(d4); d6 = self.mx_pool(d5)
        d7 = self.d_conv4(d6); d8 = self.mx_pool(d7)
        d9 = self.d_conv5(d8)  # bottleneck, no pooling
        u1 = self.u_trans1(d9)
        x = self.u_conv1(torch.cat([d7, u1], 1))  # skip connection concat
        # ... continue expanding with d5, d3, d1 skip connections ...
        return self.out(x)
```
```python
# VAE encoder forward pass with reparameterization trick
def reparameterize(self, mu, var):
    std = torch.exp(0.5 * var)
    eps = torch.randn_like(std)
    return mu + (eps * std)  # differentiable sample

def forward(self, x):
    x = F.relu(self.e1(x)); x = F.relu(self.e2(x)); x = F.relu(self.e3(x))
    x = F.relu(self.e4(x)); x = F.relu(self.e5(x))
    hidden = self.fc_1(F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1))
    mu, var = self.fc_mu(hidden), self.fc_var(hidden)
    z = self.reparameterize(mu, var)
    reconstruction = torch.sigmoid(self.d5(F.relu(self.d4(F.relu(self.d3(F.relu(self.d2(F.relu(self.d1(self.fc_2(z).view(-1,1024,1,1))))))))))))
    return reconstruction, mu, var

# VAE loss = reconstruction (MSE) + KL divergence against standard Gaussian prior
def final_loss(mse_loss, mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch * 784
    return mse_loss + KLD
```
- **What it demonstrates**: (1) U-Net's contracting/expanding path with skip-connection concatenation for pixel-wise segmentation; (2) a convolutional VAE's reparameterization trick and ELBO-style loss (reconstruction MSE + KL divergence) for generative face synthesis.

## Reference Tables
**CNN architecture comparison**

| Architecture | Year | Key innovation | Layers |
|---|---|---|---|
| AlexNet | 2012 | First deep CNN to win ImageNet | 5 conv + 3 FC |
| VGG-Net | 2014 | Deeper, small 3x3 filters | 16 or 19 |
| GoogLeNet | 2014 | Inception modules, batch norm | 22 |
| ResNet | 2015 | Skip connections (solves vanishing gradient) | 18-152 |
| MobileNet | 2017 | Depth-wise separable convolutions (mobile-efficient) | 28 |
| DenseNet | 2018 | Every layer connects to every other layer | 121/160/201 |
| EfficientNet | 2020 | Compound scaling (depth × width × resolution) | B0-B7 variants |

**CV task definitions**

| Task | Output | Example |
|---|---|---|
| Image classification | One label per image | Dog breed identification |
| Object detection | Bounding boxes + classes | Assembly-line damage detection |
| Semantic segmentation | Per-pixel class | Medical image diagnostics |
| Instance segmentation | Per-pixel, per-object-instance class | Counting individual objects |
| Object tracking | Position per frame + unique ID | Pedestrian tracking |

**Autoencoder vs. VAE vs. VQ-VAE**

| Model | Latent representation | Loss | Best for |
|---|---|---|---|
| Autoencoder | Fixed deterministic vector | Reconstruction MSE | Compression, denoising |
| VAE | Probability distribution (μ, σ²) | Reconstruction + KL (ELBO) | Generation, imputation, style transfer |
| β-VAE | Distribution, weighted KL (β) | Reconstruction + β·KL | Disentangled factor control |
| VQ-VAE | Discrete codebook vectors | Reconstruction + VQ loss + commitment loss | Language/speech-style discrete latents |

## Worked Example
**VAE face generation on CelebA**: Load the CelebA dataset (~200K celebrity face images), resize/center-crop to 64×64, normalize, and split 80/20 train/val. Build `ConvVAE`: a 5-layer convolutional encoder reduces the image to a 1024-d vector, two linear heads (`fc_mu`, `fc_var`) produce the latent distribution parameters, `reparameterize()` samples a 100-d latent vector `z`, and a 5-layer transposed-convolution decoder reconstructs the image with a sigmoid output. Train for 40 epochs (Adam, lr=0.0001, batch=128) minimizing MSE reconstruction loss plus KL divergence against a standard normal prior. Track train/validation loss curves and periodically save reconstructed-image grids; the result shows the VAE successfully learning and reproducing facial features (pose, hair, expression) from the latent space — demonstrating generative sampling rather than mere classification. The chapter also works a parallel U-Net example on the Oxford-IIIT Pet dataset to segment pet images into class masks, trained with cross-entropy loss and evaluated via pixel accuracy.

## Key Takeaways
1. CNN architecture choice is a trade-off: accuracy (ResNet/DenseNet) vs. compute/power budget (MobileNet) vs. balanced scaling (EfficientNet) — there's no single "best" backbone.
2. Skip/residual connections are the general fix for vanishing gradients in deep networks, reused from ResNet through U-Net and into later diffusion architectures (Ch 4).
3. VAEs generalize autoencoders by learning a *distribution* over latent codes (not a fixed vector), which is what makes them generative rather than just a compression tool.
4. The reparameterization trick is the key engineering move that makes stochastic latent sampling trainable via standard backpropagation.
5. Disentangled latent spaces (β-VAE) enable targeted, interpretable image edits as vector arithmetic (e.g., adding a "smile vector").
6. VQ-VAE's discrete codebook approach generalizes representation learning to language/speech-style data and foreshadows the discrete-token thinking used in transformer-based generation (Ch 5).

## Connects To
- **Ch 1-2**: applies the CNN/embedding concepts (image embedding via CNN hidden layers) and the deep-learning training stack (gradient descent, backpropagation, KL divergence) introduced earlier specifically to the image domain.
- **Ch 4**: continues directly into GANs and diffusion models — the next generation of image generators that build on (and in diffusion's case, structurally resemble) the U-Net encoder-decoder introduced here.
- **Ch 5**: VQ-VAE's discrete codebook/latent approach parallels the tokenization concept used for text in transformer architectures.
