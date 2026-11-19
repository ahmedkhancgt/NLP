# Chapter 9: Implementing Generative Vision Transformer

## Core Idea
Fusing a VAE's probabilistic latent space with a Vision Transformer's self-attention encoder (ViT-as-encoder VAE) produces richer contextual image representations than either a plain CNN-VAE or a plain VAE alone, but only when ViT weights are pretrained — a from-scratch ViT-VAE overfits badly on small datasets like STL-10.

## Frameworks Introduced
- **VAE on STL-10 (step-by-step build)**: preprocess (torchvision) → CNN encoder → latent dim choice (e.g. 32/64/128) → VAE loss (reconstruction + KL) → train with Adam → sample/generate from decoder.
  - When to use: template for scaling the MNIST/CIFAR VAE pattern up to larger (96×96×3), more diverse image data.
- **VAE-TensorFlow-vs-PyTorch comparison framework**: implement the identical VAE architecture in both frameworks to isolate whether overfitting is a framework artifact or an architecture/data problem.
  - When to use: diagnosing whether a training pathology (e.g., overfitting) is caused by the library/defaults or the model design itself.
- **VAE with ViT-from-scratch encoder**: replace CNN encoder with `vit.vit_l32(..., pretrained=False)`, reshape patch sequence, GlobalAveragePooling1D, Dense → latent. Decoder stays CNN-based (Conv2DTranspose).
  - When to use: exploratory baseline before committing to pretrained weights — the chapter shows this configuration still overfits badly.
- **VAE with pretrained ViT encoder**: same structure but `vit.vit_l32(..., pretrained=False)` loaded with ImageNet-pretrained weights plus `LayerNormalization`, L2-regularized latent Dense layers.
  - When to use: the recommended default — leverages transfer learning to avoid the data-hungry failure mode of training a ViT encoder from scratch on a small dataset.
- **Pretrained-vs-scratch ViT decision framework**: pretrained = faster convergence, less data needed, but features may not be task-specific; from-scratch = fully task-specific features and full architectural control, but needs much more data/compute and risks overfitting.
  - When to use pretrained: limited data/compute, target task overlaps with ImageNet-style features. When to use from-scratch: abundant task-relevant data and need for fully customized representations.

## Key Concepts
- **STL-10 dataset**: 96×96 RGB images, 10 classes, 5,000 labeled train / 8,000 test images + 100,000 unlabeled images; higher resolution and more diverse than CIFAR-10/MNIST, making it a harder generative-modeling benchmark.
- **Overfitting signature on a loss curve**: training loss keeps decreasing while validation loss starts increasing, with a widening gap between the two — the chapter's repeated diagnostic across all STL-10 VAE variants until the pretrained-ViT fix is applied.
- **GlobalAveragePooling1D (ViT→VAE bridge)**: reduces the sequence-of-patch-embeddings output of a ViT backbone into a single fixed-length vector suitable for a Dense latent layer.
- **vit-keras library**: third-party Keras implementation of ViT (`vit_b16`, `vit_l32`) supporting pretrained ImageNet weights, fine-tuning, and attention-map visualization.
- **Loss-curve-driven iteration**: the chapter's core methodology is "train → plot loss curve → diagnose overfitting/underfitting → change one architectural element → retrain" — a reinforcement of the diagnostic discipline from Chapters 3 and 7.

## Mental Models
- Treat "VAE on STL-10" as a controlled experiment series, not a single recipe: CNN-encoder (overfits) → same model in PyTorch (still overfits — rules out framework) → ViT-from-scratch encoder (still overfits — rules out "just add transformer") → pretrained-ViT encoder (finally starts learning properly) — the chapter is teaching the *debugging process*, not just a final architecture.
- Pretrained ViT weights act as a strong prior on "what natural images look like," which a small dataset like STL-10's 5,000 labeled images cannot teach a from-scratch ViT on its own — this is why transfer learning specifically fixes the overfitting here, when more epochs or a different library did not.

## Anti-patterns
- **Switching ML frameworks (TensorFlow → PyTorch) expecting it to fix an architecture/data-scale problem**: the chapter explicitly tests this and the same overfitting persists — the library was never the cause.
- **Training a ViT encoder from scratch on a small dataset (STL-10's 5,000 labeled images) and expecting good generalization**: ViTs are known to need large pretraining corpora; without pretrained weights, a from-scratch ViT-VAE overfits just as badly as the CNN baseline.
- **Judging an image-generation model purely by training loss**: every STL-10 experiment in this chapter looked fine on raw training loss until the validation curve was plotted — always plot both.

## Code Examples
```python
# VAE with pretrained ViT encoder (the chapter's working configuration)
inputs = Input(shape=(96, 96, 3))
x = vit.vit_l32(image_size=96, activation='relu', pretrained=False, include_top=False, pretrained_top=False)(inputs)
x = LayerNormalization(epsilon=1e-6)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
decoder_inputs = Input(shape=(latent_dim,))
x = Dense(6 * 6 * 32, activation='relu')(decoder_inputs)
x = Reshape((6, 6, 32))(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
```
- **What it demonstrates**: The fusion point between a ViT backbone and a VAE's probabilistic latent layer — LayerNormalization right after the ViT output stabilizes the features before they hit the Dense mean/log-var heads, and the reparameterization trick proceeds exactly as in earlier chapters.

```python
# vit-keras: loading a pretrained classifier directly
from vit_keras import vit, utils
model = vit.vit_b16(image_size=384, pretrained=True, include_top=True, pretrained_top=True)
X = vit.preprocess_inputs(image).reshape(1, 384, 384, 3)
y = model.predict(X)
print(classes[y[0].argmax()])
```
- **What it demonstrates**: The simplest possible pretrained-ViT usage pattern (classification), establishing the library API before it's repurposed as a VAE encoder.

## Reference Tables
| Variant | Encoder | Pretrained? | Outcome (per chapter's figures) |
|---|---|---|---|
| TF CNN-VAE on STL-10 | CNN (Conv2D) | n/a | Overfits, no real reconstruction |
| PyTorch CNN-VAE on STL-10 | CNN (Conv2d) | n/a | Still overfits (rules out framework) |
| ViT-VAE from scratch | vit_l32 | No | Overfits, no real reconstruction |
| ViT-VAE pretrained | vit_l32 | Yes (ImageNet) | Model starts learning / best reconstructions |

| Approach | Advantages | Disadvantages |
|---|---|---|
| Pretrained ViT | Faster convergence, less data needed, leverages prior knowledge | May not be perfectly task-specific, limited to learned features |
| ViT from scratch | Fully task-specific features, full architectural control | Needs much more data/compute, slower convergence, higher overfitting risk |

## Worked Example
The chapter's central worked example is itself a four-stage ablation on STL-10: (1) a TensorFlow CNN-encoder VAE overfits (Figure 9.1); (2) re-implementing the identical idea in PyTorch still overfits (Figure 9.2-9.3), eliminating "framework choice" as the cause; (3) swapping the CNN encoder for a from-scratch `vit_l32` ViT backbone still overfits (Figure 9.4-9.5), eliminating "just add a transformer" as a fix; (4) finally loading `vit_l32` with pretrained ImageNet weights plus LayerNormalization and L2-regularized latent layers produces a model that visibly starts learning real reconstructions (Figure 9.6-9.7). This systematic elimination of hypotheses is the chapter's central teaching method.

## Key Takeaways
1. Overfitting on a small, high-resolution dataset (STL-10) is not fixed by switching deep-learning frameworks.
2. Replacing a CNN encoder with a from-scratch ViT encoder does not by itself fix overfitting — ViTs need large-scale pretraining to be useful on small datasets.
3. Pretrained ViT weights (transfer learning) are the specific fix that worked in this chapter's experiments — LayerNormalization and L2 regularization on the latent heads further stabilized training.
4. Always evaluate via the train/validation loss-curve gap, not training loss alone, to detect overfitting.
5. The pretrained-vs-scratch ViT trade-off (speed/data-efficiency vs. task-specificity/control) generalizes beyond VAEs to any ViT-based pipeline.

## Connects To
- **Ch 6-7**: Directly reuses the VAE loss function (reconstruction + KL), reparameterization trick, and architecture patterns built there, now applied to a harder dataset.
- **Ch 8**: Applies the ViT patch-embedding/self-attention theory from Chapter 8 concretely as a drop-in VAE encoder.
- **Ch 10**: The overfitting-mitigation techniques only hinted at here (regularization, LayerNormalization) are formalized into Cyclic Learning Rate, EarlyStopping, and L2-regularization frameworks, then extended to the SWIN Transformer.
