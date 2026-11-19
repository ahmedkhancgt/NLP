---
name: ai-assisted-statistics
description: "Knowledge base from \"AI-Assisted Statistics for Data Scientists\" (3rd ed.) by Peter Bruce, Andrew Bruce, and Peter Gedeck. Use when applying statistical methods for data science — exploratory data analysis, sampling distributions, significance testing, regression/classification, statistical machine learning, unsupervised learning, neural/deep learning, or generative AI and LLM statistical caveats."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# AI-Assisted Statistics for Data Scientists
**Authors**: Peter Bruce, Andrew Bruce, Peter Gedeck | **Chapters**: 11 | **Generated**: 2026-11-19

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch05`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**EDA first, always**: Before modeling, summarize (mean/median/variance/MAD) and visualize (histogram, boxplot, scatterplot, hexbin) the data. Use robust estimators (median, trimmed mean, MAD, IQR) alongside classical ones (mean, SD) — large divergence signals skew/outliers worth investigating.

**Resampling over formulas**: The bootstrap (resample with replacement, recompute statistic, repeat) and permutation tests (shuffle combined groups, recompute, repeat) are distribution-free, intuitive alternatives to classical formula-based inference (t-test, chi-square). Use them whenever you want to understand what a p-value or confidence interval actually means, or when no formula exists for your statistic.

**p-value discipline**: A p-value is P(result this extreme | null/chance model) — NOT the probability a hypothesis is true. Pick your test statistic and alpha before looking at results. Guard against the "vast search effect" (torturing data until something looks significant) with holdout sets or permutation tests. Statistical significance ≠ practical significance.

**Regression diagnostics before trusting coefficients**: Check for confounding variables (important omitted predictors causing spurious coefficient signs) and multicollinearity (correlated predictors causing unstable estimates). Use partial residual plots to catch nonlinearity. Never extrapolate beyond the training data's predictor range.

**Classification beyond accuracy**: For imbalanced/rare-class problems (fraud, churn, conversion), accuracy is misleading — use precision, recall, specificity, ROC/AUC, and lift. Rebalance training data (undersample/oversample/weight/SMOTE) when the rare class is swamped. Choose the classification cutoff based on actual cost/benefit, not a default 0.5.

**Bias-variance trade-off governs every flexible model**: KNN's K, tree depth/pruning, random forest's leaf size, boosting's learning rate/regularization, and neural net regularization (L1/L2/dropout) are all knobs on this same trade-off. Always monitor train vs. validation/test performance — a widening gap is the overfitting signal that triggers more regularization or simpler models.

**Standardize before distance-based/unsupervised methods**: KNN, PCA, K-means, and hierarchical clustering all require z-score standardization first — unscaled large-magnitude variables silently dominate distance calculations and loadings.

**Match architecture to data structure (deep learning)**: RNN/LSTM for sequences needing memory; CNN for spatially-correlated grid (image) data; transformers (self-attention) for large-scale sequence/language tasks needing long-range context and parallelization. Prefer transfer learning/fine-tuning over training from scratch unless you have very large task-specific data.

**AI-assisted analysis needs the same skepticism as any junior analyst's draft**: LLMs can misread chart density regions, fabricate unsupported qualifiers, botch simple algebra, and hallucinate fluently. Use self-criticism (ask the AI "what would make this wrong?"), cross-check with a second AI, and always verify against the actual numbers. Model collapse (training on increasingly AI-generated data) narrows output diversity over time — curated, proprietary data remains a durable advantage.

**Fine-tuning vs. RAG for domain LLM adaptation**: Fine-tuning changes the model's weights (deep, stable domain knowledge, costlier); RAG retrieves fresh reference records at query time without retraining (cheaper, current, weaker for deep reasoning).

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | Exploratory Data Analysis | Location/variability estimators, boxplots/histograms/hexbin, correlation, AI-assisted data dictionaries & visualization interpretation |
| ch02 | Data and Sampling Distributions | Random/stratified sampling, bootstrap, central limit theorem, standard error, confidence intervals, binomial/Poisson/exponential/Weibull distributions |
| ch03 | Statistical Experiments and Significance Testing | A/B testing, permutation tests, p-values, t-tests, ANOVA, chi-square test, multi-arm bandits, power/sample size |
| ch04 | Regression and Prediction | Linear/multiple regression, stepwise selection, factor variables, confounding/multicollinearity, regression diagnostics, polynomial/spline/GAM |
| ch05 | Classification | Naive Bayes, LDA, logistic regression, confusion matrix/ROC/AUC/lift, imbalanced-data strategies (SMOTE, weighting) |
| ch06 | Statistical Machine Learning | KNN, decision trees, random forest, boosting/XGBoost, regularization, hyperparameter tuning |
| ch07 | Unsupervised Learning | PCA, correspondence analysis, K-means, hierarchical clustering, model-based (Gaussian mixture) clustering, Gower's distance |
| ch08 | Neural Networks | Feedforward networks, backpropagation/gradient descent, activation functions, L1/L2/dropout regularization, SHAP values |
| ch09 | Deep Learning | RNN/LSTM, CNN, transformers/self-attention, embeddings, reinforcement learning (bandits, MDPs, deep RL) |
| ch10 | Generative AI and Large Language Models | Next-token prediction, temperature, diffusion models, fine-tuning/RAG, prompt engineering, RLHF/DPO/RLAIF, agentic AI |
| ch11 | Caveats and Concerns | Model collapse, hallucination, AI self-criticism, vibe coding, data curation pipeline, AI societal/legal risks |

## Topic Index
- **A/B testing** → ch03
- **Activation functions (ReLU/sigmoid/softmax)** → ch08
- **Agentic AI** → ch10
- **AI-assisted visualization interpretation** → ch01, ch11
- **ANOVA** → ch03
- **Attention mechanism** → ch09, ch10
- **Bagging** → ch02, ch06
- **Bias-variance trade-off** → ch06
- **Bootstrap** → ch02
- **Boosting / XGBoost** → ch06
- **Chi-square test** → ch02, ch03
- **Classification evaluation (ROC/AUC/lift)** → ch05
- **Clustering (K-means, hierarchical, model-based)** → ch07
- **Confounding variables** → ch04
- **Confidence intervals** → ch02
- **CNN** → ch09
- **Correlation** → ch01
- **Cross-validation** → ch04
- **Data curation pipeline (LLM)** → ch11
- **Diffusion models** → ch10
- **Discriminant analysis (LDA/QDA)** → ch05
- **Embeddings** → ch09, ch10
- **Fine-tuning vs. RAG** → ch10
- **Gradient descent / backpropagation** → ch08
- **Hallucination / model collapse** → ch10, ch11
- **Hyperparameter tuning** → ch06, ch08
- **KNN** → ch06
- **LDA (linear discriminant analysis)** → ch05
- **Logistic regression** → ch05
- **LSTM / RNN** → ch09
- **Multi-arm bandits** → ch03, ch09
- **Multicollinearity** → ch04
- **Multiple testing** → ch03
- **Naive Bayes** → ch05
- **Neural networks (basics)** → ch08
- **Normal distribution** → ch02
- **One-hot encoding** → ch04, ch06
- **Outliers / influential points** → ch01, ch04
- **Overfitting / regularization** → ch06, ch08
- **p-values** → ch03
- **PCA** → ch07
- **Permutation tests** → ch03
- **Power and sample size** → ch03
- **Prompt engineering** → ch10
- **Random forest** → ch06
- **Regression (linear/multiple)** → ch04
- **Reinforcement learning (RLHF/MDP/bandits)** → ch09, ch10
- **Rare class / imbalanced data** → ch05
- **SHAP values** → ch08
- **Standardization (z-scores)** → ch06, ch07
- **Statistical significance** → ch03
- **Transformers** → ch09, ch10
- **Vibe coding** → ch11

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules and quick reference

---

## Scope & Limits
This skill covers the book's 11 chapters, synthesized as structure, not verbatim text. The book's own Index is excluded as non-substantive.
