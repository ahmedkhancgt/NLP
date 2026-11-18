# Chapter 2: GenAI Foundation

## Core Idea
Deep learning is the mathematical machinery (neurons, activation functions, cost functions, gradient descent, backpropagation, regularization) that makes GenAI possible, and GenAI itself is specifically the *generative* branch of deep learning — models that produce new content (text/image/audio) rather than just classify or predict a number.

## Frameworks Introduced
- **Discriminative vs. generative models**: discriminative models predict a label/category; generative models produce new data from a learned probability distribution.
  - When to use: deciding if a use case even needs GenAI — "if the output is a number or classification, it is not a GenAI model."
  - How: check whether the desired output resembles natural language/image/audio (GenAI) vs. a score/class (discriminative ANN).
- **Deep learning training pipeline**: forward pass → cost function → gradient descent → backpropagation (chain rule) → regularization → evaluation metrics.
  - When to use: building/debugging any supervised neural network.
  - How: compute output via weighted sums + activation functions; measure error via a cost function; use the gradient (via chain rule/autodiff) to update weights via gradient descent; apply regularization (L1/L2, dropout, batch norm) to control generalization error.
- **Foundation model framework**: a transformer-based model (self-attention) pre-trained at massive scale (e.g., GPT-3: 175B parameters, ~50s of GB data, 60% from Common Crawl) that is then tuned/prompted for many downstream tasks.
  - When to use: any GenAI project — start from a foundation model rather than training from scratch.
  - How: select a foundation model matched to modality/task, then prompt-engineer or fine-tune.
- **GenAI use-case selection framework**: categorize by domain (customer journey, business process) or by type (content creation, virtual agent), then rate by capability/risk/demand (1-5 scales) before committing.
  - When to use: scoping an enterprise GenAI initiative.
  - How: start with low blast-radius use cases (low consequence of error), mitigate hallucination via creativity-level tuning, grounding with internal data, guardrail libraries, moderation models, and disclaimers.

## Key Concepts
- **Activation function**: a non-linear function (sigmoid, Tanh, ReLU, Leaky ReLU, softmax, GELU) applied to neuron outputs so the network can model non-linear relationships; ReLU is the hidden-layer industry standard, GELU is standard in GPT-3/BERT-class LLMs.
- **Cost/loss function**: quantifies prediction error (MSE/MAE/Huber for regression; cross-entropy/log loss for classification); must be continuous and differentiable to be optimized by gradient descent.
- **Gradient descent**: iterative optimization that steps weights in the negative-gradient direction; variants are batch, stochastic (SGD), and mini-batch, often combined with momentum or adaptive schemes (AdaGrad, AdaDelta, RMSProp).
- **Backpropagation**: recursive application of the chain rule to compute ∂Cost/∂weight across all network layers, enabling gradient descent to update every parameter.
- **Regularization**: techniques (L1/L2 weight decay, dropout, batch normalization, gradient centralization) that reduce generalization error by constraining model complexity.
- **Foundation model**: a large pre-trained model (e.g., GPT series) built on transformer architecture and self-attention, adaptable to many downstream tasks.
- **Prompt engineering vs. prompt design**: prompt design tunes language models via APIs (for builders); prompt engineering is end-user crafting of text prompts (0-shot/1-shot/few-shot) to get desired outputs.
- **Hallucination**: GenAI's tendency to generate plausible-sounding but incorrect content because it "lacks the capacity to decline" to answer.
- **Confusion matrix / AUC-ROC**: classification evaluation tools (true/false positive/negative; ROC curve plots TPR vs. FPR across thresholds).

## Mental Models
- Think of **AI ⊃ ML ⊃ Deep Learning ⊃ GenAI** as nested subject domains, each adding more specific machinery — useful for correcting the common conflation of "AI" with "GenAI."
- Use the **"output shape" test** for GenAI applicability: a number/class = discriminative task; natural language/image/audio = candidate GenAI task.
- Think of **learning rate as a dial between speed and precision**: too large overshoots the minimum; too small risks getting stuck or taking too long — hence learning-rate schedules/decay and adaptive methods.
- Use the **bias-variance-regularization triangle** from Ch 1 together with concrete regularization levers (L1/L2, dropout, batch norm) as the practical toolkit for generalization.

## Anti-patterns
- **Zero/constant weight initialization**: makes all neurons learn identical features (symmetric updates), defeating the purpose of multiple neurons — use Xavier (sigmoid/Tanh) or He/Kaiming (ReLU) initialization instead.
- **Using sigmoid/Tanh in very deep networks**: causes vanishing gradients and slow, zig-zag convergence; ReLU-family activations are preferred for hidden layers.
- **Over-sized ReLU negative bias / high learning rate**: causes "dying ReLU" where neurons permanently output 0 and stop learning; use Leaky ReLU/PReLU/ELU/GELU to mitigate.
- **Using a full foundation model for a narrow use case when a smaller tuned model would do**: wastes compute/budget and causes avoidable environmental impact (sustainability concern flagged here, expanded in Ch 10).
- **Deploying GenAI into high-consequence-of-error use cases early**: hallucination risk makes this unsafe before the organization has built mitigation strategies.

## Code Examples
```python
# Load libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Fetch California housing dataset (regression target example)
from sklearn.datasets import fetch_california_housing
ann_data = fetch_california_housing()

# Train/test split and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = ann_data.data, ann_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# Build an ANN regression model with Keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=X_train.shape[1:]))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))  # single output neuron for regression

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_squared_error'])
history = model.fit(X_train, y_train, batch_size=128, validation_split=0.1, epochs=500, verbose=1)
score, mse_test = model.evaluate(X_test, y_test, batch_size=100)
```
- **What it demonstrates**: the end-to-end supervised ANN workflow (load → split → scale → build Sequential Keras model → compile with optimizer/loss → train → evaluate) used as the template for the chapter's regression (California housing) and classification (bank churn) walkthroughs.

## Reference Tables
**Activation function choice by output type**

| Problem type | Output layer activation |
|---|---|
| Binary classification | Sigmoid |
| Multi-class classification | Softmax |
| Multi-label classification | Sigmoid (per-label threshold) |
| Regression | Linear/identity |

**Gradient descent variants**

| Type | Update granularity | Pros | Cons |
|---|---|---|---|
| Batch | Entire dataset | Stable gradient | Slow, memory-heavy, no online updates |
| Stochastic (SGD) | Single sample | Fast, can escape local minima, online updates | High variance, slower convergence |
| Mini-batch | n samples | Balances cost and variance | n is a hyperparameter to tune |

**Regularization techniques**

| Technique | Mechanism |
|---|---|
| L1 (Lasso) | Penalty = λ·‖θ‖, forces sparsity (feature selection) |
| L2 (Ridge/weight decay) | Penalty = λ·‖θ‖², shrinks weights toward zero |
| Dropout | Randomly deactivates neurons per iteration (probability p) |
| Batch normalization | Normalizes layer inputs to zero mean/unit variance |
| Gradient centralization | Centers gradient vectors to zero mean directly |

## Worked Example
**California housing regression**: Load the California housing dataset (8 features, 20,640 rows, target = median house value in $100k). Explore with `df.describe()` and a longitude/latitude scatterplot colored by house age. Split 80/20 train/test, standard-scale features. Build a 4-hidden-layer (16 neurons each, ReLU) Keras `Sequential` ANN with a single linear output neuron. Compile with the Adam optimizer and MAE loss, train for 500 epochs with a 10% validation split, then evaluate with RMSE/MSE/MAE/adjusted R² and a predicted-vs-actual scatter plot. The same template (import → fetch data → split/scale → build Sequential model → compile/fit → evaluate with a confusion matrix instead of regression metrics) is reused for the chapter's second worked example: a bank customer-churn **classification** problem predicting whether a customer will close their account.

## Key Takeaways
1. Deep learning's full training stack (activations, cost functions, gradient descent, backpropagation, regularization) is the substrate every later GenAI architecture (transformers, GANs, diffusion models) is built on top of.
2. GenAI is distinguished from ordinary ANN/ML by its *output* — natural language, images, or audio generated from a learned distribution, not a score or class label.
3. Foundation models (GPT-1 through GPT-4, scaling from 11B to ~1T parameters) are pre-trained once at massive scale and then adapted via prompting or fine-tuning rather than retrained per use case.
4. Prompt engineering (0-shot/1-shot/few-shot) is the primary lever end-users have over foundation-model behavior without retraining.
5. Hallucination, token limits, and real-time knowledge gaps are GenAI's core present-day limitations; mitigate with grounding, guardrails, moderation, and disclaimers rather than assuming they'll disappear on their own.
6. When scoping a GenAI initiative, categorize use cases by domain/type and rate them on capability, risk, and demand; start with low-consequence-of-error use cases.
7. Sustainability (compute/resource cost of training and running foundation models) must be weighed against using a full foundation model when a smaller, use-case-tailored model would suffice.

## Connects To
- **Ch 1**: operationalizes the supervised-learning math (loss functions, gradient descent) and the bias-variance/regularization concepts introduced there into full neural-network training mechanics.
- **Ch 3-4**: apply this chapter's deep-learning foundations (CNNs via representation learning, generative modeling) specifically to the image domain.
- **Ch 5**: extends the activation-function/attention groundwork here directly into transformer architecture for text.
- **Ch 9**: the GenAI use-case categorization and risk/demand framework introduced here is expanded into a full enterprise adoption strategy.
- **Ch 10**: the sustainability concern flagged here (resource cost of foundation models) becomes a dedicated chapter topic.
