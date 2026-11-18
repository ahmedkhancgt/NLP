# Chapter 10: Architectural Refactoring for Generative Modeling

## Core Idea
Systematically refactoring a generative model — basic VAE → TransVAE (transformer-based VAE) → Swin TransVAE (hierarchical, shifted-window transformer VAE) — combined with disciplined training techniques (cyclic learning rates, early stopping, L2 regularization) is the practical path from an overfitting prototype to a converging, deployable generative model.

## Frameworks Introduced
- **Encoder-decoder-transformer fusion pattern**: `encoder(input_shape)` (CNN feature extractor) → `transformer(...)` (attention-based contextual processor) → `decoder(latent_dim, output_dim)` (Dense reconstruction/classification head), composed via functional-API `Model` chaining.
  - When to use: as the conceptual template whenever you want to inject transformer-level contextual reasoning into an existing CNN encoder-decoder pipeline.
- **Cyclic Learning Rate (CLR) schedule**: learning rate oscillates between `base_lr` and `max_lr` following a `triangular`/`triangular2`/`exp_range` pattern instead of monotonically decaying.
  - When to use: when training is stuck in a local minimum or converging too slowly with a fixed/monotonic schedule — CLR trades manual LR tuning for automatic exploration/exploitation cycling.
- **Swin (Shifted Window) Transformer**: hierarchical ViT variant that computes self-attention within local, non-overlapping windows, then shifts the windows across layers to enable cross-window connections — reduces attention complexity from quadratic to linear in image size.
  - When to use: high-resolution images or any setting where plain ViT's O(n²) attention cost is prohibitive, while still needing both local and global context.
- **Swin-Transformer-encoder VAE**: Swin backbone (classification head removed) feeds a Dense z_mean/z_log_var latent pair; Conv2DTranspose decoder reconstructs; combined with L2 regularization on latent Dense layers and EarlyStopping.
  - When to use: scaling the Ch.9 pretrained-ViT-VAE pattern to larger/higher-resolution images more efficiently.
- **Systematic hyperparameter-tuning workflow**: understand problem → establish baseline → define hyperparameter space → choose tuning strategy (grid/random/Bayesian) → pick metric → iterate → validate (cross-validation) → monitor → refine → automate (Hyperopt/Optuna/Ray Tune).
  - When to use: as the general-purpose process for improving any deep generative model once a working baseline exists.

## Key Concepts
- **TransVAE**: a VAE whose encoder/decoder incorporate Transformer layers (self-attention) instead of pure CNN/Dense layers, for capturing long-range dependencies in image data.
- **Swin TransVAE**: a TransVAE whose Transformer layers are specifically Swin Transformer blocks (hierarchical, windowed attention) rather than plain ViT blocks.
- **LearningRateScheduler**: callback that decreases learning rate over time to prevent overshooting optimal weights late in training, complementing early stopping.
- **EarlyStopping**: callback that halts training once validation loss stops improving, directly preventing the overfitting pattern diagnosed throughout Chapter 9.
- **Weight decay (L2 regularization)**: penalty term proportional to the sum of squared weights, shrinking weights toward zero and discouraging overly complex (overfitting-prone) solutions.
- **Shifted-window attention**: Swin's core innovation — attention computed within fixed local windows, with the window partitioning shifted between successive layers so information can flow across window boundaries without quadratic global attention cost.

## Mental Models
- Treat this chapter's three-stage model lineage (basic VAE → TransVAE → Swin TransVAE) as a literal refactoring ladder: each rung swaps one architectural component (CNN layers → Transformer layers → Swin Transformer layers) while keeping the VAE's probabilistic latent-space contract (z_mean, z_log_var, reparameterization, ELBO loss) fixed — architecture evolves, training objective doesn't.
- CLR's "triangular" and "triangular2" modes are best understood as literal shapes on a learning-rate-vs-iteration plot: triangular bounces fully between base_lr and max_lr every cycle; triangular2 does the same but with the peak amplitude halving each cycle, favoring more exploration early and more fine-tuning later.
- Swin's linear-vs-quadratic complexity trade-off is the direct payoff of windowed attention: trade a small amount of "every patch sees every other patch" global context for large computational savings, recovered partially via shifting windows across layers.

## Anti-patterns
- **Bolting a Transformer onto a VAE without addressing the underlying overfitting/regularization problem first**: the chapter shows the naive encoder-decoder-transformer outline is "simplified" and placeholder-only — real integration needs the L2 regularization, CLR, and EarlyStopping machinery introduced alongside it.
- **Using a fixed, monotonically-decaying learning rate by default**: works, but the chapter frames CLR as a way to reduce manual tuning burden and escape local minima more reliably — a fixed schedule leaves this benefit on the table.
- **Scaling up model complexity (more transformer blocks, higher latent dim) before exhausting simpler fixes (regularization strength, learning rate schedule, data augmentation)**: the chapter's own "Improving the models" checklist orders cheap wins (hyperparameters, regularization) before expensive ones (model complexity, ensembling).

## Code Examples
```python
# Cyclic Learning Rate as a Keras Callback
class CyclicLR(Callback):
    def __init__(self, base_lr=1e-4, max_lr=3e-4, step_size=2000., mode='triangular2'):
        super(CyclicLR, self).__init__()
        self.base_lr, self.max_lr, self.step_size, self.mode = base_lr, max_lr, step_size, mode
        self.clr_iterations = 0.
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        elif self.mode == 'triangular2':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
    def on_batch_end(self, batch, logs=None):
        self.clr_iterations += 1
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = autoencoder.fit(x_train, epochs=100, batch_size=128,
                           validation_data=(x_test, None), callbacks=[clr, early_stopping])
```
- **What it demonstrates**: Combining a custom Cyclic Learning Rate callback with EarlyStopping in a single `.fit()` call — the chapter's concrete recipe for stabilizing the pretrained-ViT-VAE's training from Chapter 9.

```python
# Swin Transformer as a VAE encoder (classification head removed)
swin_model = SwinTransformer(input_shape=(96, 96, 3), num_classes=10, num_heads=4, num_transformer_blocks=2)
swin_model = tf.keras.Model(inputs=swin_model.inputs, outputs=swin_model.layers[-2].output)
inputs = Input(shape=(96, 96, 3))
x = swin_model(inputs)
z_mean = Dense(latent_dim, kernel_regularizer=l2(0.01))(x)
z_log_var = Dense(latent_dim, kernel_regularizer=l2(0.01))(x)
```
- **What it demonstrates**: Reusing a pretrained/backbone classification model as a VAE encoder by stripping its final classification layer and feeding its penultimate-layer features into the usual z_mean/z_log_var heads — the same pattern as the ViT-VAE in Chapter 9, now with Swin's hierarchical windowed attention.

## Reference Tables
| Model Stage | Encoder | Key Addition |
|---|---|---|
| Basic VAE | CNN | Reparameterization + KL loss |
| TransVAE | Transformer (ViT) | Self-attention for long-range dependencies |
| Swin TransVAE | Swin Transformer | Hierarchical, windowed (linear-complexity) attention |

| Technique | Purpose |
|---|---|
| Cyclic Learning Rate | Faster exploration, escape local minima, less manual tuning |
| LearningRateScheduler | Prevent overshoot, fine-tune convergence |
| EarlyStopping | Halt before overfitting worsens |
| Weight decay (L2) | Discourage overly complex weight solutions |

## Worked Example
Training the Swin-Transformer-encoder VAE on STL-10 with L2-regularized latent Dense layers, Adam at a conservative learning rate (1e-4), and EarlyStopping(patience=5, restore_best_weights=True): the chapter reports (Figure 10.2) that "the training and validation curve shows convergence however can be improved" — an explicitly honest intermediate result, followed by the "Improving the models" section's full menu of further refinements (architecture swaps, different loss functions, data augmentation, β-VAE variants, latent-space grid traversal for visualization).

## Key Takeaways
1. Architectural refactoring (CNN → ViT → Swin) should be paired with training-discipline upgrades (CLR, EarlyStopping, L2) at each step — architecture alone doesn't fix overfitting.
2. Swin Transformer's windowed, shifted attention gives near-ViT expressiveness at linear (not quadratic) computational cost — the practical fix for scaling ViT-VAEs to larger images.
3. Convergence that "can be improved" is a legitimate, reportable research outcome — the chapter models honest iterative reporting rather than only showing success.
4. The book's own systematic hyperparameter-tuning workflow (baseline → space definition → tuning strategy → validation → automation) is reusable for any deep generative model, not just VAEs.
5. The progression basic VAE → TransVAE → Swin TransVAE exemplifies a general principle: each new architecture addresses a specific, previously-diagnosed limitation of its predecessor.

## Connects To
- **Ch 9**: Directly continues the ViT-VAE experiments, now generalizing "replace the encoder with a transformer" into a full refactoring methodology and introducing Swin as the efficiency upgrade over plain ViT.
- **Ch 11**: This chapter's closing summary explicitly previews Chapter 11's technical-roadblocks discussion (limited expressiveness, training/sampling inefficiency, interpretability, generalization/robustness).
- **Ch 6-7**: The VAE loss function (reconstruction + KL) and reparameterization trick remain unchanged across every architectural refactoring step in this chapter, reinforcing that they are the stable "contract" underneath any encoder/decoder swap.
