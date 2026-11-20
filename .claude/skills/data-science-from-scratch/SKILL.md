---
name: data-science-from-scratch
description: "Knowledge base from \"Data Science from Scratch\" (2nd ed.) by Joel Grus. Use when learning or implementing core data science/ML techniques from first principles — linear algebra, statistics, probability, hypothesis testing, gradient descent, machine learning fundamentals (bias-variance, precision/recall), k-NN, Naive Bayes, linear/logistic regression, decision trees, neural networks, deep learning, clustering, NLP (n-grams, LDA, word vectors, RNNs), network analysis (centrality, PageRank), recommender systems, databases/SQL, or MapReduce, all built from plain Python without external ML libraries."
---

<!-- argument-hint: [topic, framework name, or chapter number] -->

# Data Science from Scratch (2nd Edition)
**Author**: Joel Grus | **Chapters**: 27 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch14`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**"From Scratch" as Pedagogy, Not Practice** (Ch1, Ch27): Every technique in this book is implemented in plain Python with no NumPy/pandas/scikit-learn, specifically so you understand the mechanics before reaching for the real libraries in production. The book's own closing argument: once you understand what a library is doing internally, switching to it is an informed choice, not a leap of faith.

**Vector/Matrix Foundation** (Ch4): `Vector = List[float]`, `Matrix = List[List[float]]`, plus `add`/`subtract`/`dot`/`magnitude`/`distance` — the atomic operations every later algorithm (regression, gradient descent, k-NN, clustering, neural nets) is built on.

**Gradient Descent as the Universal Optimizer** (Ch8): Derive a loss function's gradient, step against it repeatedly, choose batch/minibatch/stochastic based on dataset size — the single fitting mechanism reused for linear regression, logistic regression, PCA, matrix factorization, and neural networks.

**Train/Test/Validation + Bias-Variance Diagnosis** (Ch11): Always evaluate on held-out data; diagnose underperformance as high bias (too simple, add complexity) or high variance (too complex, add data/regularize/remove features) before reaching for a fix.

**Precision/Recall/F1 over Raw Accuracy** (Ch11): Accuracy is misleading under class imbalance (the "Luke -> leukemia" joke test is 98% "accurate" and completely useless); always check precision, recall, and F1 for binary classifiers.

**Bayes's Theorem as the Probabilistic Backbone** (Ch6, Ch13): `P(E|F) = P(F|E)P(E)/P(F)` underlies hypothesis testing's Bayesian alternative (Ch7), Naive Bayes spam classification (Ch13), and LDA topic modeling's Gibbs sampling (Ch21).

**Regression Family Progression** (Ch14-16): Simple linear regression (closed form) -> multiple regression (gradient descent, watch collinearity/omitted-variable bias) -> logistic regression (squash through sigmoid for 0/1 targets) — each extends the prior chapter's machinery to a harder problem shape.

**The Curse of Dimensionality** (Ch12, Ch20, Ch23): Any distance-based technique (k-NN, clustering, user-based collaborative filtering) degrades in high dimensions because nearest-neighbor distance approaches average distance — reduce dimensionality (PCA, Ch10) first.

**Neural Network Progression** (Ch18-19): Perceptron (linearly separable only) -> hidden layers + backpropagation (solves XOR) -> reusable Layer/Sequential/Loss/Optimizer framework (Tensor, Dropout, softmax+cross-entropy) — each chapter generalizes the prior one's hand-rolled mechanics into composable abstractions.

**MapReduce's Central Design Decision** (Ch25): The mapper's choice of key is where the real algorithm design happens — it determines what information ends up grouped together for the reducer to see.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | Introduction | Degree centrality, FOAF recommendation, indexing vs. scanning |
| ch02 | A Crash Course in Python | Type annotations, assert-based testing, defaultdict/Counter, comprehensions, generators |
| ch03 | Visualizing Data | matplotlib line/bar/scatter charts, misleading-axis anti-patterns |
| ch04 | Linear Algebra | Vector/Matrix type aliases, dot product, magnitude, distance, adjacency matrix |
| ch05 | Statistics | Mean/median/mode, variance/stdev/IQR, covariance/correlation, Simpson's paradox |
| ch06 | Probability | Conditional probability, Bayes's theorem, normal distribution, Central Limit Theorem |
| ch07 | Hypothesis and Inference | Null/alternative hypothesis, p-values, confidence intervals, A/B testing, Bayesian inference |
| ch08 | Gradient Descent | Batch/minibatch/stochastic gradient descent, analytic vs. numerical gradients |
| ch09 | Getting Data | stdin/stdout pipelines, csv module, BeautifulSoup scraping, JSON APIs, OAuth streaming |
| ch10 | Working with Data | EDA ladder, NamedTuple records, fail-soft parsing, rescaling, PCA |
| ch11 | Machine Learning | Train/test/validation split, precision/recall/F1, bias-variance tradeoff, feature selection |
| ch12 | k-Nearest Neighbors | k-NN classification, tie-breaking, curse of dimensionality |
| ch13 | Naive Bayes | Conditional-independence assumption, Laplace smoothing, log-probability summation |
| ch14 | Simple Linear Regression | Closed-form least squares, R-squared, MLE justification |
| ch15 | Multiple Regression | Vector regression via gradient descent, bootstrap standard errors, ridge/lasso |
| ch16 | Logistic Regression | Logistic function, log-likelihood maximization, SVM/kernel trick (conceptual) |
| ch17 | Decision Trees | Entropy, partition entropy, ID3 algorithm, random forests |
| ch18 | Neural Networks | Perceptron, feed-forward networks, backpropagation, input/output encoding |
| ch19 | Deep Learning | Tensor/Layer/Sequential, Loss/Optimizer abstractions, softmax+cross-entropy, dropout |
| ch20 | Clustering | k-means, elbow method, bottom-up hierarchical clustering |
| ch21 | Natural Language Processing | n-gram models, formal grammars, Gibbs sampling, LDA, word embeddings, RNNs |
| ch22 | Network Analysis | Betweenness/closeness/eigenvector centrality, power iteration, PageRank |
| ch23 | Recommender Systems | Popularity-based, user/item-based collaborative filtering, matrix factorization |
| ch24 | Databases and SQL | NotQuiteABase Table, CRUD, select/where/group_by/join, query optimization |
| ch25 | MapReduce | map_reduce model, combiners, distributed sparse matrix multiplication |
| ch26 | Data Ethics | Accuracy/fairness tradeoffs, bias in training data, data protection tensions |
| ch27 | Go Forth and Do Data Science | IPython, NumPy/pandas/scikit-learn, deep learning frameworks, project-driven learning |

## Topic Index

- **A/B testing** → ch07
- **Backpropagation** → ch18, ch19
- **Bayes's theorem** → ch06, ch13, ch21
- **Bias-variance tradeoff** → ch11, ch17, ch19
- **Bootstrap resampling** → ch15, ch17
- **Centrality (betweenness/closeness/eigenvector/PageRank)** → ch01, ch22
- **Clustering (k-means/hierarchical)** → ch20
- **Confidence intervals** → ch07
- **Correlation / covariance** → ch05, ch14
- **Cross-entropy / softmax** → ch19
- **Curse of dimensionality** → ch12, ch20, ch23
- **Decision trees / random forests** → ch17
- **Deep learning framework (Layer/Sequential/Loss/Optimizer)** → ch19
- **Dimensionality reduction (PCA)** → ch10
- **Entropy** → ch17
- **Gradient descent (all variants)** → ch08, ch14, ch15, ch16, ch18, ch19, ch23
- **Hypothesis testing / p-values** → ch07
- **k-Nearest Neighbors** → ch12
- **LDA / topic modeling** → ch21
- **Linear algebra primitives (Vector/Matrix/dot)** → ch04
- **Logistic regression / SVM** → ch16
- **MapReduce** → ch25
- **Matrix factorization / recommenders** → ch23
- **Naive Bayes** → ch13
- **Neural networks** → ch18, ch19
- **Normal distribution / CLT** → ch06
- **NotQuiteABase / SQL** → ch24
- **Overfitting / underfitting** → ch11
- **PageRank** → ch22
- **Precision / recall / F1** → ch11, ch12, ch16
- **Probability (conditional, Bayes)** → ch06
- **Regression (simple/multiple/logistic)** → ch14, ch15, ch16
- **Regularization (ridge/lasso)** → ch15
- **Rescaling / standardization** → ch10, ch15, ch16
- **RNNs / word embeddings** → ch21
- **Scraping / APIs** → ch09
- **Statistics (mean/variance/IQR)** → ch05
- **Train/test/validation split** → ch11

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — reusable techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules, formulas, and a quick workflow checklist

---

## Scope & Limits
This skill covers the book's 27 chapters, synthesized as structure, not verbatim text. All code examples are from-scratch pure-Python implementations as presented in the book; the book itself (Ch27) explicitly recommends switching to NumPy/pandas/scikit-learn for production use once the underlying mechanics are understood. The book's own Index is excluded as non-substantive.
