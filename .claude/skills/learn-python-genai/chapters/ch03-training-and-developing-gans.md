# Chapter 3: Training and Developing Generative Adversarial Networks

## Core Idea
Training a GAN is a two-network zero-sum game with no explicit accuracy metric; success depends on monitoring generator/discriminator loss curves for characteristic failure signatures (mode collapse, vanishing gradients, oscillation, instability) and applying the matching mitigation.

## Frameworks Introduced
- **GAN training loop (TensorFlow)**: `train_step` computes `gen_loss`/`disc_loss` inside a `tf.GradientTape`, applies gradients to generator and discriminator optimizers separately each batch.
  - When to use: canonical implementation pattern for any custom GAN in TF/Keras.
  - How: generate fake images from noise → get discriminator outputs on real and fake batches → compute both losses → backprop each network's own loss through its own optimizer.
- **GAN failure-mode diagnostics framework**: classify training pathology by its loss-curve signature — mode collapse (D loss drops sharply to near-zero, G loss plateaus/oscillates), vanishing gradients (both losses plateau/stagnate), oscillation (both losses fluctuate erratically without convergence), instability (loss never converges at all).
  - When to use: as the first diagnostic step whenever a GAN "isn't working" — read the shape of the loss curves before changing architecture.
- **WGAN / WGAN-LP as a structural fix**: replacing BCE-based GAN loss with Wasserstein distance + Lipschitz enforcement directly targets mode collapse, vanishing gradients, and instability simultaneously.
  - When to use: when a standard GAN exhibits multiple simultaneous failure modes and incremental fixes (normalization, LR tuning) aren't enough.

## Key Concepts
- **Mode collapse**: generator produces a narrow subset of outputs because the discriminator is easily fooled by a few "safe" samples; visible as discriminator loss crashing toward 0 while generator loss plateaus/oscillates.
- **Vanishing gradients**: gradients shrink to near-zero through the network (often from sigmoid/tanh saturation), causing both losses to plateau with no learning progress.
- **Oscillation**: generator and discriminator losses swing back and forth without converging, caused by capacity imbalance, bad hyperparameters, or mismatched learning rates between G and D.
- **Training instability**: the overall failure of the loss function to converge at all, from inappropriate loss functions, bad hyperparameters, or noisy data.
- **Qualitative vs. quantitative GAN evaluation**: qualitative = human visual judgment of samples (biased, can miss mode collapse); quantitative = Inception Score, Fréchet Inception Distance (FID), LPIPS — the recommended basis for evaluation, not raw loss values.
- **Early stopping by FID/perceptual similarity**: halting training when a quantitative quality metric degrades, rather than relying on loss curves.

## Mental Models
- Never trust GAN loss values alone as a proxy for sample quality — "the reliability of loss values obtained during training of GANs is often questionable"; always pair with visual inspection and a quantitative metric (FID/Inception Score).
- Treat each failure mode's loss-curve shape as a diagnostic signature, the way a doctor reads symptoms: sharp D-loss crash + G-loss plateau → suspect mode collapse; both losses flatlining early → suspect vanishing gradients; both losses jittering with no trend → suspect oscillation/instability.
- The data distribution a GAN can produce is strictly bounded by the training set's distribution — "the examples the Generator will be able to produce are determined by the training dataset," so class imbalance in data becomes class imbalance (or total absence) in generated samples unless corrected.

## Anti-patterns
- **Judging GAN quality from loss values alone**: loss can look fine while mode collapse silently destroys sample diversity — always combine with FID/Inception Score and visual sampling.
- **Ignoring mode collapse risk with small/imbalanced datasets or overly simple G/D architectures**: both factors measurably increase the likelihood of mode collapse per the chapter's analysis.
- **Running excessively long training (e.g., 1000 epochs) without checkpointing or quantitative monitoring**: computationally wasteful and risks training past the point of degradation without noticing.

## Code Examples
```python
# Generator model
def make_generator_model():
    model = Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model
# Discriminator model
def make_discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1))
    return model
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
```
- **What it demonstrates**: The complete, canonical DCGAN generator/discriminator architecture and training step on MNIST — two independent GradientTapes, two independent optimizer updates, the structural template for every GAN in the book.

## Reference Tables
| Failure Mode | Loss-Curve Signature | Key Mitigations |
|---|---|---|
| Mode collapse | D loss crashes near 0; G loss plateaus/fluctuates | Varied loss functions, generator regularization, diverse training data |
| Vanishing gradients | Both losses plateau early, slow/no convergence | LeakyReLU/ReLU over sigmoid/tanh, weight norm, gradient clipping, batch norm |
| Oscillation | Both losses fluctuate without convergence | Balance G/D capacity, tune LR per-network, spectral norm, gradient penalty |
| Instability | Loss never converges | Architecture mods (skip connections, norm layers), regularization, LR scheduling, progressive growing |

## Worked Example
Case study: balancing the imbalanced CranexD dental X-ray dataset with a GAN. The classes of dental cavities are highly imbalanced; a GAN generator/discriminator pair (Conv2D + LeakyReLU + MaxPooling2D stacks) is trained adversarially on the minority classes, producing synthetic cavity images that are combined with the original dataset to rebalance it. The augmented, balanced dataset is then used to train a downstream cavity-detection classifier, evaluated on precision/recall/F1, showing measurable improvement in detecting underrepresented cavity types versus training on the original imbalanced data alone.

## Key Takeaways
1. GANs lack a native accuracy metric — always monitor generator and discriminator loss jointly, never G or D loss in isolation.
2. Learn to read loss-curve shapes as diagnostic signatures for mode collapse, vanishing gradients, oscillation, and instability.
3. Quantitative metrics (FID, Inception Score, LPIPS) are the recommended basis for GAN evaluation, not raw loss values.
4. WGAN/WGAN-LP structurally address several training pathologies at once via the Wasserstein distance and Lipschitz constraint.
5. GANs are a legitimate, practical tool for correcting class imbalance in downstream classification datasets (as shown in the dental X-ray case study).

## Connects To
- **Ch 2**: Applies the architectures and loss-function theory from Chapter 2 into actual trainable TensorFlow code.
- **Ch 4-5**: Many of the same failure-mode vocabulary (overfitting, mode collapse, instability) reappears when the book turns to training Autoencoders and VAEs, inviting direct comparison of GAN vs. AE/VAE training dynamics.
- **Ch 7**: Revisits mode collapse, training instability, and computational cost as "ongoing research challenges," picking up from the practical fixes introduced here.
