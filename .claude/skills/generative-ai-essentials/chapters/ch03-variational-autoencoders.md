# Chapter 3: Variational Autoencoders

## Core Idea
A VAE learns a probabilistic latent space (mean + log-variance per data point) via an encoder/decoder pair, then uses the reparameterization trick to sample new, varied content that resembles — but is not identical to — the training data, trading GAN-style adversarial sharpness for stable, probabilistically-grounded generation.

## Frameworks Introduced
- **VAE architecture (encoder–latent space–decoder)**: encoder compresses input into distribution parameters `z_mean`, `z_log_var`; a sampled latent vector `z` is decoded back into reconstructed/new data.
  - When to use: generation tasks needing a smooth, interpolatable latent space, denoising, or probabilistic data augmentation (robust to messy/missing data).
  - How: encoder outputs `z_mean`/`z_log_var` → reparameterize to sample `z` → decoder reconstructs from `z` → loss = reconstruction loss + KL divergence.
- **Reparameterization trick**: `z = z_mean + exp(0.5 * z_log_var) * epsilon`, where `epsilon ~ N(0,1)`.
  - When to use: whenever you must backpropagate through a stochastic sampling step (core to all VAE training).
  - How: sample noise `epsilon`, scale by the learned standard deviation (`exp(0.5*logvar)`), shift by the learned mean — makes sampling differentiable.
- **VAE loss function**: `total_loss = reconstruction_loss + KL_divergence`.
  - Reconstruction loss: binary cross-entropy or MSE between input and decoder output (measures fidelity).
  - KL divergence: `-0.5 * sum(1 + log_var - mean^2 - exp(log_var))` (regularizes latent space toward a standard normal distribution).
  - When to use: always, for any VAE — balances faithful reconstruction against a well-structured, generative latent space.
- **VAE variants by use case**: denoising VAE (medical image denoising), molecule-generation VAE (drug discovery), anomaly-detection VAE (network security), text-generation VAE (NLG), recommendation VAE (user/item embeddings) — same encoder/decoder/KL skeleton, different input/output domains.

## Key Concepts
- **Latent space**: the compressed, continuous probability-distribution space where data features live; smooth/continuous so interpolating between points yields meaningful new samples.
- **Encoder**: network that maps raw input to the parameters (mean, log-variance) of a latent distribution.
- **Decoder**: network that maps a sampled latent vector back into data space (reconstruction or new generation).
- **Variational inference**: the mechanism by which VAEs learn an approximate probability distribution over the latent space rather than a single deterministic code.
- **KL (Kullback-Leibler) divergence**: statistical measure of how much one probability distribution diverges from another; used to regularize the VAE's latent distribution toward a standard normal.
- **Mode collapse (VAE context)**: the model generates limited sample diversity, failing to represent all latent-space variation.
- **Cold start problem**: in VAE-based recommendation systems, difficulty generating accurate recommendations for new users/items with little interaction history.

## Mental Models
- Think of the VAE as "chef and secret recipe": the encoder (chef 1) distills key ingredients into a secret sauce (latent space); the decoder (chef 2) uses that sauce to recreate the dish or invent new variations.
- Think of the latent space as a library of summaries, not full stories — each point is compressed but fluid/continuous, so nearby points produce plausibly similar but new "stories" (data samples).
- Use "reconstruction vs. regularization" as the core tension to reason about VAE training: too much weight on reconstruction overfits to exact inputs (less generative diversity); too much KL weight blurs reconstructions (posterior collapse).

## Anti-patterns
- **Treating the decoder output as a deterministic function of the input**: ignoring the stochastic sampling step (reparameterization) breaks backpropagation and removes the model's generative capability.
- **Using an unstructured/non-normal latent space**: without the KL regularization term pulling the latent distribution toward N(0,1), the space becomes discontinuous and new samples drawn from it are unreliable or nonsensical.
- **Over-trusting VAE-generated molecules without chemical validation**: generated molecular structures must be checked for chemical validity, not just latent-space plausibility.
- **Deploying anomaly-detection VAEs without threshold tuning**: naive thresholds cause high false-positive/false-negative rates; thresholds must be tuned against false-positive/false-negative cost tradeoffs.
- **Training content-recommendation VAEs on sparse or biased interaction data**: leads to cold-start failures and reinforces existing preference biases.
- **Ignoring training-data quality/diversity**: VAE effectiveness (denoising, generation) depends heavily on dataset quality; poor or imbalanced data causes unclear/blurry outputs and bias reproduction.

## Code Examples
```python
# Core VAE class (encoder/decoder/reparameterize/decode)
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim),
        ])
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(units=7*7*32, activation='relu'),
            layers.Reshape(target_shape=(7, 7, 32)),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'),
        ])

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        return tf.sigmoid(logits) if apply_sigmoid else logits

# Loss: reconstruction + KL divergence
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    reconstruction_loss = tf.reduce_sum(cross_entropy, axis=(1, 2, 3))
    kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    return tf.reduce_mean(reconstruction_loss + kl_divergence)
```
- **What it demonstrates**: the canonical convolutional VAE on MNIST — `Conv2D` encoder compressing to `2*latent_dim` outputs (mean + logvar), `Conv2DTranspose` decoder reconstructing images, reparameterization for differentiable sampling, and the combined reconstruction + KL loss.

```python
# Dense VAE with explicit sampling Lambda layer (denoising / NLG / recommendation variants)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

def vae_loss(y_true, y_pred):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)
```
- **What it demonstrates**: the reusable dense-layer VAE pattern (used across the X-ray denoising, NLG, and recommendation use cases in this chapter) — same `z_mean`/`z_log_var`/`sampling`/`vae_loss` skeleton applied to different `input_dim`/`output_dim` shapes.

## Reference Tables
| VAE use case | Encoder input | Decoder output | Key loss components |
|---|---|---|---|
| Medical image denoising | Noisy image | Denoised/reconstructed image | Reconstruction + KL |
| Drug discovery | Molecular structure | Novel molecular structure | Reconstruction + KL + property objective |
| Network anomaly detection | Network traffic features | Reconstructed "normal" traffic | Reconstruction + KL, threshold on reconstruction error |
| Natural language generation | Tokenized text sequence | Generated text/next-word distribution | Categorical cross-entropy + KL |
| Personalized recommendation | User ID + item ID embeddings | Recommendation scores | Binary cross-entropy + KL |

| Challenge | Description |
|---|---|
| Mode collapse | Limited diversity in generated samples |
| Unclear outputs | Blurry/ambiguous samples from poorly structured latent space |
| Hyperparameter tuning | Latent dimensionality, learning rate, architecture balance |
| Imbalanced data | Outliers/imbalance poorly represented in latent space |
| Interpreting latent space | Hard to map latent regions to meaningful features |
| Training stability | Less stable than some other generative models |
| Fairness/bias | Reproduces biases present in training data |
| Evaluation metrics | Subjective; no single agreed-upon quality metric |

## Worked Example
Network anomaly detection with a VAE: build an `encoder` (Dense(64) → `z_mean`/`z_log_var` of `latent_dim=10`) and `decoder` (Dense(64) → sigmoid output of `input_dim`); wrap both in a custom `VAE(models.Model)` subclass with an overridden `train_step` that computes `reconstruction_loss = mean(square(data - reconstruction))` plus `kl_loss`, backprops via `tape.gradient`, and applies Adam updates. Train on preprocessed normal network traffic (`X_train`) for `epochs`/`batch_size`. At inference, traffic that reconstructs poorly (high reconstruction error relative to a tuned threshold) is flagged as anomalous — balancing false positives against false negatives per the security team's risk tolerance.

## Key Takeaways
1. VAEs generate via a probabilistic latent space (mean + log-variance) rather than adversarial competition — the reparameterization trick is what makes that latent sampling trainable with backprop.
2. The loss is always a two-term balance: reconstruction fidelity vs. KL-divergence regularization toward a standard normal latent distribution; over-weighting either term degrades output quality.
3. The exact same encoder→latent→decoder skeleton (with `sampling` Lambda + `vae_loss`) generalizes across images, text, molecules, network traffic, and recommendations — only input/output layer shapes and the reconstruction loss type change.
4. VAEs excel where GANs struggle: handling missing/noisy/uncertain data (denoising) and providing an interpretable, continuous latent space for controlled variation.
5. Persistent challenges — mode collapse, blurry outputs, hyperparameter sensitivity, bias reproduction — require careful data curation, architecture tuning, and domain-specific validation (e.g., chemical validity for drug discovery, threshold tuning for anomaly detection).
6. VAE-based anomaly detection works by flagging high reconstruction error against a learned "normal" baseline, not by directly classifying attacks.

## Connects To
- **Ch2**: Contrasts directly with GAN's adversarial training approach — VAEs trade GAN sharpness for stable, probabilistic generation and an interpretable latent space.
- **Ch4**: Transformer-based language models next replace the simple VAE text-generation approach shown here (Use case 4) with far more capable architectures for text.
- **Ch5**: Image denoising/enhancement use cases here connect to the deeper image-generation and style-transfer techniques covered next.
- **Ch10**: The GCP VAE deployment workflow (Docker containerization, AI Platform Training/Prediction) previews the full custom-model-building pipeline detailed later.
