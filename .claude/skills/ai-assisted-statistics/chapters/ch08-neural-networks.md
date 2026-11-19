# Chapter 8: Neural Networks

## Core Idea
Neural networks learn flexible, nonlinear mappings from inputs to outputs via layers of weighted sums and activation functions, trained end-to-end by backpropagation/gradient descent — but their flexibility makes them highly prone to overfitting, which must be actively controlled via regularization (L1/L2, dropout) and validation-based monitoring, and on plain structured (rectangular) data they typically offer no advantage over random forests/boosted trees.

## Frameworks Introduced
- **Feedforward Neural Network**: input layer → one or more hidden layers → output layer, each node computing a weighted sum of inputs plus a bias, passed through an activation function.
  - When to use: complex nonlinear relationships, especially for unstructured data (image/text/sequence) covered in later chapters; for plain tabular data, compare against tree ensembles first.
  - How: `keras_model_sequential()` + `layer_dense()` (R) / `keras.Sequential([keras.layers.Dense(...)])` (Python); choose activation per layer and compile with a loss function and optimizer.
- **Backpropagation + (Stochastic) Gradient Descent**: compute the loss on a batch, propagate the error gradient backward through the network, and update weights/biases in the direction that reduces loss.
  - When to use: the standard training algorithm for virtually all neural networks.
  - How: forward pass → compute loss → backpropagate gradients → optimizer (e.g., Adam) updates weights; repeated over many batches (an epoch = one full pass through training data) and many epochs.
- **Regularization for Neural Nets (L1, L2, Dropout)**: constrain model complexity to prevent overfitting.
  - When to use: whenever validation accuracy/loss diverges from training accuracy/loss (the classic overfitting signature).
  - How: L1 (`regularizer_l1`) penalizes Σ|weight| (sparsity, like lasso); L2 (`regularizer_l2`/weight decay) penalizes Σweight² (like ridge); Dropout randomly zeroes a fraction of nodes during training only.
- **SHAP (SHapley Additive exPlanations)**: model-agnostic method from game theory that decomposes a prediction into additive per-feature contributions.
  - When to use: interpreting/explaining individual neural-net predictions (or any black-box model) since neural nets don't expose built-in variable importance like trees do.
  - How: `fastshap::explain()` (R) / `shap.DeepExplainer` (Python); visualize with a waterfall plot (per-record) or mean-absolute-SHAP bar chart (global importance).
- **Three-way data split (train / validation / test)**: separate the data used to fit weights, the data used to tune hyperparameters, and a final untouched holdout to get an unbiased performance estimate.
  - When to use: always for neural nets, since hyperparameter tuning on a validation set can itself leak information and bias the validation metric.

## Key Concepts
- **Weight / Bias**: weight = multiplier on an input signal (analogous to a regression coefficient); bias = constant offset added to the weighted sum (analogous to an intercept).
- **Activation function**: nonlinear function applied to a node's weighted sum; ReLU (0 for negative input, linear for positive) is the standard hidden-layer choice today; sigmoid (0–1 squash) is used for binary output; softmax generalizes sigmoid to multiclass probability outputs that sum to 1; tanh is an older squashing function, still used in some architectures (e.g., LSTMs).
- **Loss (cost) function**: scalar penalty for prediction error that training minimizes — MSE for regression, cross-entropy (log-loss) for classification.
- **Learning rate**: hyperparameter controlling the step size of each gradient-descent weight update; too high overshoots, too low trains too slowly.
- **Momentum**: hyperparameter that carries forward some of the previous update's direction, helping escape local minima and damp oscillation.
- **Epoch / batch**: an epoch = one full pass through the training set; a batch (mini-batch) = the subset of cases processed before one weight update.
- **Vanishing / exploding gradients**: in deep networks, gradients can shrink toward zero (stalling learning) or blow up (destabilizing training) as they propagate backward through many layers; ReLU and better weight initialization mitigate this.
- **Logits**: the raw weighted-sum outputs of the final layer before the sigmoid/softmax activation is applied; used internally for numerical stability in loss computation.

## Mental Models
- Think of weights and biases as the neural-net analog of regression coefficients and intercepts — each node is literally a tiny (possibly nonlinear) regression unit, and a network is many such units chained and combined.
- Picture gradient descent as repeatedly "feeling the slope" of the error surface and taking a small step downhill — exactly the iterative version of what least squares does analytically in one step for linear regression.
- Treat the train/validation accuracy divergence plot as the primary overfitting diagnostic for neural nets: training accuracy climbing while validation accuracy plateaus or degrades is the tell that regularization is needed.
- Use the "neural net ≈ tree model with step-function activations" analogy to connect this chapter back to Chapter 6: both can represent arbitrary nonlinear/interaction relationships, but trees are interpretable and fast to train, while neural nets are flexible and continuous but opaque and expensive to train — the complexity trade-off explains why neural nets only decisively outperform trees once given large enough data and the specialized architectures covered in Chapter 9.

## Anti-patterns
- **Training a flexible neural net without a validation set to monitor overfitting**: the chapter's own example shows training accuracy improving steadily while validation accuracy degrades — invisible without tracking both.
- **Feeding unscaled predictors into a neural net**: predictors with very different magnitudes can destabilize gradient descent optimization — always rescale (e.g., 0–1 min-max scaling) before training.
- **Using a basic feedforward neural network on plain structured/tabular data by default**: it typically offers no advantage over random forest/boosted trees, while costing more in complexity, training time, and interpretability.
- **Treating default hyperparameters as sufficient**: neural nets have "many more knobs" (architecture, learning rate, regularization strength, dropout rate) than tree ensembles, and tuning is both more necessary and more consequential.
- **Reusing the validation set for both hyperparameter tuning and final performance reporting**: biases the reported metric upward — reserve a separate, untouched test set.
- **Over-trusting individual SHAP values from a single sample run**: SHAP for neural nets uses Monte Carlo sampling (`nsim`) and is computationally expensive — results are approximate and sample-size dependent.

## Code Examples
```r
reg_model <- keras_model_sequential(input_shape = c(15)) %>%
  layer_dense(units = 64, activation = "relu",
    kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 32, activation = "relu",
    kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = "sigmoid")
```
- **What it demonstrates**: a regularized binary-classification network combining L2 weight penalty and 30% dropout on two hidden layers — directly fixes the overfitting seen in the unregularized version.

```r
simple_model <- keras_model_sequential(input_shape = c(4)) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 2, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
simple_model %>% compile(loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.01), metrics = c("accuracy"))
```
- **What it demonstrates**: the three-step neural net recipe (define structure → compile with loss/optimizer → fit) for binary classification on loan data.

```r
shap_values <- fastshap::explain(object = reg_model, X = train_x[1:floor(0.9*n), ],
  pred_wrapper = predict_function, nsim = 50, newdata = newdata)
sv_waterfall(shapviz(shap_values, X = newdata), row_id = 24)
```
- **What it demonstrates**: computing and visualizing SHAP values to explain why one specific loan record was predicted to default (borrower score, loan term, grade, payment-to-income ratio were the main drivers).

## Reference Tables
| Activation function | Range | Typical use |
|---|---|---|
| ReLU | [0, ∞) | Standard hidden-layer activation |
| Sigmoid | (0, 1) | Binary classification output; older hidden layers |
| Tanh | (−1, 1) | Older hidden layers; still used in LSTMs |
| Softmax | (0,1), sums to 1 | Multiclass classification output |
| Linear | (−∞, ∞) | Regression output (when negative values are valid) |

| Regularization | Mechanism | Regression analog |
|---|---|---|
| L1 | Penalizes Σ\|weight\| → sparsity | Lasso |
| L2 | Penalizes Σweight² → weight decay | Ridge |
| Dropout | Randomly disables nodes during training | — (ensemble-like effect) |

| Task | Loss function |
|---|---|
| Regression | Mean squared error (MSE) |
| Binary classification | Cross-entropy / log-loss (sigmoid output) |
| Multiclass classification | Cross-entropy on softmax output |

## Worked Example
LendingClub loan data (`loan_data.csv.gz`, 15 predictors): a simple 2-hidden-layer network (4 predictors, ReLU → ReLU → sigmoid, Adam optimizer lr=0.01) is fit as a minimal illustration. A larger network (64 → 32 → 1 units, all 15 predictors, 500 epochs, batch size 256, 10% validation split) shows the classic overfitting signature: training accuracy keeps rising while validation accuracy degrades over epochs (Figure 8-5). Adding L2 regularization (λ=0.001) to both hidden layers plus 30% dropout after each stabilizes validation accuracy across the same 500 epochs (Figure 8-6) — a direct before/after demonstration of regularization's effect. SHAP analysis of a specific defaulted loan record finds a predicted log-odds of default ≈0.262 (≈56.5% probability vs. a 50% baseline), attributed mainly to borrower score, loan term, grade, and payment-to-income ratio (Figure 8-7); aggregating mean absolute SHAP values across a 100-record sample ranks borrower score as the single most important feature overall (Figure 8-8).

## Key Takeaways
1. Neural nets are regression/classification generalized to arbitrary nonlinear mappings — weights/biases/activations are the direct analogs of regression coefficients/intercepts/link functions.
2. Always monitor training vs. validation performance during training; a widening gap is the clearest overfitting signal and the trigger to add regularization (L1/L2/dropout) or stop training early.
3. Rescale predictors (e.g., 0–1 min-max) before training — unscaled features destabilize gradient descent.
4. Use a three-way split (train/validation/test) when hyperparameter tuning is involved, so the final reported performance isn't biased by validation-set leakage.
5. Use SHAP values (not built-in importance, which neural nets lack) to interpret individual predictions and estimate overall feature importance — but expect it to be computationally expensive and sample-based.
6. On plain structured/tabular data, don't default to neural networks — random forests and boosted trees are typically competitive or better with far less tuning complexity; neural nets earn their advantage mainly on image/text/sequence data (Chapter 9) and with very large datasets.

## Connects To
- **Ch 5 (Classification)**: the sigmoid activation and cross-entropy loss used here are identical in spirit to logistic regression's logit/log-odds machinery; softmax generalizes this to multinomial logistic regression.
- **Ch 6 (Statistical Machine Learning)**: gradient descent parallels boosted trees' iterative error-correction; L1/L2 regularization here directly mirrors lasso/ridge regression and XGBoost's `alpha`/`lambda` penalties.
- **Ch 9 (Deep Learning)**: this chapter's basic feedforward network is the foundation extended by CNNs, RNNs, and transformers to handle image, sequence, and text data where neural networks truly outperform tree ensembles.
