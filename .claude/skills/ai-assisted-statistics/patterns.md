# Patterns: Statistical and Machine Learning Techniques

## Bootstrap Resampling
**When to use**: Need a standard error or confidence interval for a statistic with no known formula (median, correlation, model coefficients), or want a distribution-free alternative to classical formulas.
**How**: Draw samples of size n with replacement from the observed data, compute the statistic, repeat R times (e.g., 1,000); use the spread/percentiles of the R results as the sampling distribution.
**Trade-offs**: Computationally intensive but general-purpose; does not create new information or fix a too-small sample — it only reveals how variable a statistic would be across resamples.

## Permutation Test
**When to use**: Testing whether an observed group difference (A/B test, multi-group) exceeds what random chance could produce, for any statistic/metric, with any sample sizes.
**How**: Combine all groups' data, repeatedly shuffle and redeal into original-size groups, recompute the statistic; p-value = proportion of shuffled results at least as extreme as observed.
**Trade-offs**: Assumption-free and intuitive, but computationally heavier than formula-based tests; for very small/sparse counts use an exhaustive (Fisher's exact) test instead of random shuffling.

## Trimmed Mean / Median / MAD (Robust Estimation)
**When to use**: Data has outliers or skew that would distort the mean/standard deviation.
**How**: Trimmed mean drops p% from each tail before averaging; median takes the middle value; MAD takes the median of absolute deviations from the median.
**Trade-offs**: More robust but discard some information relative to the mean; most beneficial on small/moderate datasets where outliers have outsized leverage.

## Multiple Linear Regression
**When to use**: Predicting/explaining a numeric outcome from several numeric/categorical predictors with an approximately linear relationship.
**How**: Fit by least squares (`lm()`/`LinearRegression`); check RMSE, R², and coefficient significance (t-statistics); encode categoricals with P−1 reference dummies.
**Trade-offs**: Interpretable and fast but assumes linearity; vulnerable to confounding, multicollinearity, and extrapolation errors; needs diagnostics (residual plots, Cook's distance) especially for explanatory use.

## Logistic Regression
**When to use**: Binary classification needing fast scoring and interpretable, business-readable odds-ratio coefficients.
**How**: Fit via maximum likelihood (`glm(family=binomial)`/`LogisticRegression`); interpret exp(coefficient) as an odds ratio.
**Trade-offs**: Fast, interpretable, good baseline; less flexible than trees for capturing nonlinear interactions unless you add splines/interaction terms manually.

## Stepwise / Penalized Regression (Model Selection)
**When to use**: Many candidate predictors; need a parsimonious model balancing fit against complexity.
**How**: Forward/backward/both stepwise selection minimizing AIC or maximizing adjusted R² (`stepAIC`), or penalized regression (ridge/lasso) shrinking coefficients toward zero.
**Trade-offs**: In-sample selection risks overfitting; validate with cross-validation. Penalized regression retains all variables (shrunk) vs. stepwise's all-or-nothing inclusion.

## K-Nearest Neighbors (KNN)
**When to use**: Simple baseline classification/regression, or as a "feature engine" feeding a derived local-similarity score into a stronger second-stage model.
**How**: Standardize predictors to z-scores; pick K via cross-validation (typical range 1–20); classify/predict via majority vote or mean of K nearest neighbors.
**Trade-offs**: No training step, intuitive, but scales poorly with large data and is sensitive to irrelevant/unscaled features; rarely competitive alone against tree ensembles.

## Decision Tree (CART)
**When to use**: Need interpretable if-then rules, or as the base learner for ensembles.
**How**: Recursive partitioning on the predictor/threshold minimizing Gini impurity/entropy at each split; prune via `cp`/`ccp_alpha`, `max_depth`, or `min_samples_leaf`, tuned by cross-validation.
**Trade-offs**: Highly interpretable but prone to overfitting if grown fully; unstable (small data changes can produce very different trees).

## Random Forest
**When to use**: Strong off-the-shelf predictive accuracy with minimal tuning; need variable-importance ranking.
**How**: Bag (bootstrap-resample) many trees, each restricted to a random predictor subset per split; average/vote predictions; monitor OOB error to pick number of trees.
**Trade-offs**: Robust and low-maintenance but loses single-tree interpretability ("black box"); importance via Gini-decrease (fast, less reliable) or permutation/accuracy-decrease (slower, more reliable).

## Gradient Boosting (XGBoost)
**When to use**: Maximum predictive accuracy when willing to carefully tune hyperparameters.
**How**: Fit a sequence of shallow trees, each correcting prior errors (reweighted observations or pseudo-residuals); tune `eta`/`learning_rate`, `subsample`, `max_depth`, `lambda`/`alpha` regularization.
**Trade-offs**: Often best-in-class accuracy but easily overfits without careful regularization and cross-validated hyperparameter tuning; more fragile/sensitive than random forest.

## Principal Components Analysis (PCA)
**When to use**: Reduce dimensionality of correlated numeric variables before modeling; explore which variables move together.
**How**: Compute orthogonal, variance-maximizing linear combinations (components); inspect loadings and a scree plot to decide how many to retain.
**Trade-offs**: Only for numeric data; components can be dominated by a few high-variance variables unless scaled; components often lack intuitive interpretation.

## K-Means Clustering
**When to use**: Scalable segmentation of numeric data; cold-start problems with no labels yet.
**How**: Standardize variables; choose K (elbow method or managerial constraint); run with multiple random starts, keep the lowest within-cluster SS solution.
**Trade-offs**: Scales well, simple, but requires pre-specifying K and assumes roughly spherical, similarly-sized clusters; sensitive to variable scaling and outliers.

## Hierarchical Clustering
**When to use**: Smaller datasets; want to visualize cluster structure at multiple granularities without pre-specifying K; mixed numeric/categorical data (with Gower's distance).
**How**: Agglomerative merging of least-dissimilar clusters (complete/single/average/Ward's linkage); cut the resulting dendrogram at a chosen height/cluster count.
**Trade-offs**: Flexible and visual but doesn't scale to large n; different linkage methods can produce very different results.

## Neural Network (Feedforward)
**When to use**: Complex nonlinear relationships; compare against tree ensembles first for plain tabular data (often no advantage there).
**How**: Layers of weighted sums + activation functions, trained via backpropagation/gradient descent; regularize with L1/L2 and dropout; monitor train/validation gap for overfitting.
**Trade-offs**: Extremely flexible but opaque (needs SHAP for interpretation), data- and compute-hungry, many hyperparameters to tune.

## RNN / LSTM
**When to use**: Sequential/time-ordered data (time series, text) where order and context matter.
**How**: Carry a hidden state forward across sequence steps (RNN); LSTM adds gated cell state for longer-range memory; train via backpropagation through time.
**Trade-offs**: Captures temporal dependency but sequential (not parallelizable) and limited by vanishing gradients (RNN) even with LSTM's mitigation; largely supplanted by transformers for large-scale tasks.

## CNN (Convolutional Neural Network)
**When to use**: Image (grid-structured, spatially correlated) data.
**How**: Stack convolution (shared filters) + activation + pooling/stride layers to build hierarchical features; flatten or global-average-pool into a final classifier; use transfer learning from a pretrained model when possible.
**Trade-offs**: Dominant for vision tasks; training from scratch is expensive — transfer learning is almost always preferable unless you have very large task-specific data.

## Transformer (Self-Attention)
**When to use**: Large-scale sequence/language modeling needing long-range context and parallel training (the basis of LLMs).
**How**: Project tokens into Query/Key/Value vectors; compute attention = softmax(QK^T/√dk)·V; stack multi-head attention + feedforward + residual/layer-norm blocks; decoder-only architectures use causal masking.
**Trade-offs**: Captures long-range dependencies and parallelizes well, but quadratic compute cost in sequence length and requires massive data/compute to train from scratch — fine-tuning or RAG on a foundation model is the practical path for most applications.

## Multi-Arm Bandit
**When to use**: Optimizing conversion/click-through across multiple treatments in production, where speed/efficiency matters more than formal statistical proof.
**How**: Adaptive allocation (epsilon-greedy, epsilon-first, Upper Confidence Bound, Thompson sampling) shifting traffic toward better-performing arms as data accumulates.
**Trade-offs**: Faster and more efficient than fixed-sample A/B testing for pure optimization, but less suited when formal, auditable significance proof is required (e.g., regulated clinical trials).

## Fine-Tuning vs. RAG (LLM Adaptation)
**When to use**: Adapting a foundation LLM to a specific domain.
**How**: Fine-tuning further trains model weights on domain data (deep, stable knowledge); RAG retrieves relevant records at query time and has the (unchanged) LLM synthesize a response (fresh, frequently-changing data).
**Trade-offs**: Fine-tuning is costlier/static but supports deep reasoning; RAG is cheaper/current but weaker for tasks needing deep built-in domain reasoning.
