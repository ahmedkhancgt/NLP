# Glossary

**Accuracy** — Percent of predictions classified correctly (Ch 5).
**Activation function** — Nonlinear function applied to a node's weighted sum (ReLU, sigmoid, tanh, softmax) (Ch 8).
**AGI (Artificial General Intelligence)** — Hypothetical future AI with human-like general cognitive capability (Ch 9, Ch 11).
**AIC (Akaike's Information Criterion)** — Metric penalizing model complexity for model selection (Ch 4).
**Alignment** — Training process making LLM output match human preferences (helpful, honest, harmless) (Ch 10).
**Alpha** — Pre-specified significance threshold (e.g., 0.05) for hypothesis tests (Ch 3).
**Alpha inflation** — Increased Type 1 error rate from running multiple tests (Ch 3).
**Attention mechanism** — Transformer component weighing relevance of all tokens to each other (Ch 9, Ch 10).
**AUC (Area Under the ROC Curve)** — Overall classifier discriminative power metric (Ch 5).
**Backpropagation** — Algorithm computing gradients backward through a network's layers (Ch 8).
**Bagging** — Bootstrap aggregating; averaging models fit to bootstrap resamples (Ch 2, Ch 6).
**Bias (statistical)** — Systematic error in measurement/sampling (Ch 2).
**Bias-variance trade-off** — Tension between underfitting (bias) and overfitting (variance) (Ch 6).
**Binomial distribution** — Distribution of number of successes in n trials (Ch 2).
**Bootstrap** — Resampling with replacement from observed data to estimate a sampling distribution (Ch 2).
**Boosting** — Ensemble method fitting sequential models that focus on prior errors (Ch 6).
**CART / Decision Tree** — If-then rule model via recursive partitioning (Ch 6).
**Central Limit Theorem (CLT)** — Sample statistics tend toward normal distribution as n grows (Ch 2).
**Chi-square statistic** — Measure of departure of observed counts from expected (Ch 2, Ch 3).
**CNN (Convolutional Neural Network)** — Architecture using shared filters for grid/image data (Ch 9).
**Coefficient of determination (R²)** — Proportion of variance explained by a regression model (Ch 4).
**Confidence interval** — Range expected to contain a statistic a specified % of the time under repeated sampling (Ch 2).
**Confounding variable** — Omitted important predictor causing spurious regression relationships (Ch 4).
**Confusion matrix** — Table of predicted vs. actual classification counts (Ch 5).
**Correlation coefficient** — Standardized (-1 to 1) measure of linear association (Ch 1).
**Covariance matrix** — Matrix of variances and covariances among variables (Ch 5, Ch 7).
**Cross-entropy loss** — Classification loss function based on predicted probability divergence (Ch 8).
**Cross-validation** — Repeated train/holdout partitioning to assess out-of-sample performance (Ch 4).
**Data dictionary** — Document describing a dataset's variables, source, and meaning (Ch 1).
**Degrees of freedom** — Number of values free to vary in a statistic's calculation (Ch 2, Ch 3).
**Dendrogram** — Tree diagram of hierarchical clustering merges (Ch 7).
**Diffusion model** — Image generation model that iteratively denoises toward a target (Ch 10).
**Discriminant analysis (LDA)** — Classification method maximizing between/within-group variance ratio (Ch 5).
**Double descent** — Pattern where validation error falls, rises, then falls again as model capacity grows (Ch 9).
**Embedding** — Dense vector representation of discrete items (words, tokens) capturing semantic similarity (Ch 9, Ch 10).
**Ensemble** — Prediction formed by combining multiple models (Ch 6).
**EDA (Exploratory Data Analysis)** — Tukey's discipline of summarizing/visualizing data before modeling (Ch 1).
**False discovery rate** — Rate of false "discoveries" (Type 1 errors) across multiple tests (Ch 3).
**Fine-tuning** — Further training a pretrained foundation model on domain-specific data (Ch 10).
**Gini impurity** — Measure of class heterogeneity used to select tree splits (Ch 6).
**Gradient descent** — Iterative optimization adjusting parameters to reduce a loss function (Ch 8).
**Hallucination** — LLM generating fluent but false/fabricated content (Ch 10, Ch 11).
**Hierarchical clustering** — Agglomerative clustering building a dendrogram without pre-specifying K (Ch 7).
**Hyperparameter** — Model setting chosen before training, not learned from data (Ch 6, Ch 8).
**Interquartile range (IQR)** — 75th minus 25th percentile; robust variability measure (Ch 1).
**K-means clustering** — Partitioning data into K clusters minimizing within-cluster sum of squares (Ch 7).
**KNN (K-Nearest Neighbors)** — Classify/predict using the majority/average of K nearest records (Ch 6).
**Lasso / L1 regularization** — Penalty on sum of absolute coefficient values, encouraging sparsity (Ch 4, Ch 6, Ch 8).
**LLM (Large Language Model)** — Model generating lengthy text via next-token prediction at scale (Ch 10).
**Logistic regression** — Regression modeling log-odds of a binary outcome (Ch 5).
**Loss function** — Function quantifying prediction error minimized during training (Ch 8).
**LSTM (Long Short-Term Memory)** — RNN variant with gated cell state for long-range sequence memory (Ch 9).
**MAD (Median Absolute Deviation)** — Robust variability estimate based on median of absolute deviations (Ch 1).
**Mahalanobis distance** — Correlation-adjusted distance metric (Ch 6, Ch 7).
**Mean absolute deviation** — Average of absolute deviations from the mean (Ch 1).
**Model collapse** — Degradation from training increasingly on AI-generated data (Ch 10, Ch 11).
**Multi-arm bandit** — Adaptive experiment allocating trials toward better-performing treatments (Ch 3, Ch 9).
**Multicollinearity** — Near-perfect correlation among predictors causing unstable regression estimates (Ch 4).
**Naive Bayes** — Classifier estimating class probability via conditional-independence assumption (Ch 5).
**Normal distribution** — Bell-shaped distribution; central to sampling-distribution theory (Ch 2).
**Null hypothesis** — Baseline assumption that observed differences are due to chance (Ch 3).
**Odds ratio** — exp(logistic regression coefficient); ratio of odds between two groups (Ch 5).
**One-hot encoding** — Representing categorical variables as binary dummy columns (Ch 4, Ch 6).
**Outlier** — Data value far from the rest of the data (Ch 1, Ch 4).
**Overfitting** — Model fitting noise in training data, degrading generalization (Ch 6, Ch 8).
**p-value** — Probability, under the null model, of a result as extreme as observed (Ch 3).
**PCA (Principal Components Analysis)** — Dimension reduction via orthogonal variance-maximizing linear combinations (Ch 7).
**Permutation test** — Resampling test shuffling combined group data to assess chance variation (Ch 3).
**Poisson distribution** — Distribution of event counts per unit time/space at constant rate λ (Ch 2).
**Population** — The larger (often theoretical) dataset from which a sample is drawn (Ch 2).
**Precision** — Proportion of predicted positives that are true positives (Ch 5).
**Prediction interval** — Uncertainty interval around a single predicted value (Ch 4).
**Propensity score** — Model's estimated probability a record belongs to a class of interest (Ch 5).
**p-value controversy** — Debate over p-value misinterpretation; 2016 ASA statement (Ch 3).
**R² (R-squared)** — See coefficient of determination.
**RAG (Retrieval-Augmented Generation)** — Directing an LLM to reference retrieved records without retraining (Ch 10).
**Random forest** — Bagged decision trees with random predictor subsampling at each split (Ch 6).
**Recall / Sensitivity** — Proportion of actual positives correctly identified (Ch 5).
**Regression to the mean** — Extreme observations tend to be followed by more average ones (Ch 2, Ch 11).
**Regularization** — Techniques (L1/L2, dropout) constraining model complexity to prevent overfitting (Ch 4, Ch 6, Ch 8).
**Reinforcement learning (RL)** — Learning via agent-environment interaction and reward feedback (Ch 9).
**Reinforcement learning from human feedback (RLHF)** — Using human preference ratings as RL reward signal for LLMs (Ch 10).
**Residual** — Difference between observed and fitted/predicted values (Ch 4).
**RMSE (Root Mean Squared Error)** — Square root of average squared prediction error (Ch 4).
**RNN (Recurrent Neural Network)** — Network architecture with feedback loop carrying hidden state across sequence steps (Ch 9).
**ROC curve** — Plot of recall vs. specificity across classification cutoffs (Ch 5).
**Sampling distribution** — Distribution of a sample statistic across repeated samples (Ch 2).
**Selection bias** — Bias from how data/observations are selected, including the vast search effect (Ch 2).
**SHAP values** — Game-theoretic method decomposing a prediction into additive feature contributions (Ch 8).
**Significance test** — Procedure testing if an observed effect exceeds chance variation (Ch 3).
**SMOTE** — Synthetic Minority Oversampling Technique; generates synthetic minority-class records (Ch 5).
**Specificity** — Proportion of actual negatives correctly identified (Ch 5).
**Standard error** — Standard deviation of a sample statistic's sampling distribution (Ch 2).
**Standardization (z-score)** — (x − mean)/SD; essential before KNN/PCA/clustering (Ch 6, Ch 7).
**Stepwise regression** — Iterative addition/removal of predictors to optimize model fit criteria (Ch 4).
**t-distribution** — Normal-like distribution with heavier tails, used for small-sample inference (Ch 2, Ch 3).
**Temperature** — LLM parameter controlling next-token sampling randomness (Ch 10).
**Token** — Discrete unit of a text sequence (word, subword, character) (Ch 9, Ch 10).
**Transformer** — Attention-based architecture replacing recurrence, basis of modern LLMs (Ch 9).
**Type 1 / Type 2 error** — False positive (concluding real effect when chance) / false negative (Ch 3).
**Variable importance** — Ranking of predictors by contribution to model accuracy/impurity reduction (Ch 6).
**Variance (statistical)** — Average squared deviation from the mean (Ch 1).
**Weighted regression** — Regression giving different observations different weights (Ch 4).
**XGBoost** — Popular stochastic gradient boosting implementation (Ch 6).
**z-score** — Standardized value: (x − mean)/SD (Ch 2, Ch 6).
