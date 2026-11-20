---
name: python-data-science-handbook
description: "Knowledge base from \"Python Data Science Handbook\" (2nd ed.) by Jake VanderPlas. Use when working with the core Python data science stack — IPython/Jupyter productivity, NumPy arrays (ufuncs, broadcasting, fancy indexing, structured arrays), Pandas (indexing, missing data, MultiIndex, merge/join, groupby, pivot tables, string/time-series operations, eval/query), Matplotlib/Seaborn visualization, or machine learning with Scikit-Learn (the Estimator API, model validation, feature engineering, Naive Bayes, linear regression, SVM, decision trees/random forests, PCA, manifold learning, k-means, Gaussian mixture models, kernel density estimation, or a face-detection pipeline)."
---

<!-- argument-hint: [topic, technique name, or chapter number] -->

# Python Data Science Handbook
**Author**: Jake VanderPlas | **Edition**: 2nd | **Chapters**: 50 | **Generated**: 2026-11-20

## How to Use This Skill
- **Without arguments** — load core frameworks below for reference
- **With a topic** — ask about a concept or technique; I find and read the relevant chapter
- **With a chapter** — ask for `ch14`; I load that specific chapter file
- **Browse** — ask "what chapters do you have?" to see the full index

When you ask about a topic not covered in Core Frameworks below, I will read the relevant chapter file before answering.

---

## Core Frameworks & Mental Models

**The NumPy foundation** — Python's dynamic typing makes every value a full object with type/refcount overhead; NumPy trades that flexibility for fixed-type, contiguous storage. Nearly everything later in the book (Pandas, Matplotlib, Scikit-Learn) is built on NumPy arrays underneath.

**Vectorization over looping** — the book's single most-repeated principle: push per-element work into compiled code (ufuncs, broadcasting, `.str` accessor methods, Pandas vectorized ops) instead of explicit Python `for` loops, for orders-of-magnitude speedups.

**Broadcasting rules** — NumPy's mechanism for applying binary ufuncs to arrays of different (but compatible) shapes, by conceptually "stretching" the smaller array without copying memory. Shapes are compared trailing-dimension-first; dimensions of size 1 stretch to match.

**loc/iloc split** — Pandas resolves the ambiguity between label-based and position-based access with explicit indexers: `.loc[]` (label-based, slices inclusive) vs. `.iloc[]` (position-based, slices exclusive). Plain brackets follow separate conventions for Series (array/dict-like) vs. DataFrame (column-first, but row-first for slices/masks).

**Split-apply-combine (GroupBy)** — the conceptual decomposition behind `groupby`: split data by a key, apply a function per group, combine results back together — implemented as a single lazily-evaluated pass, not a naive loop.

**The Scikit-Learn Estimator API** — every model follows the same five-step pattern: choose a model class → choose hyperparameters → arrange data into a features matrix `X` [n_samples, n_features] + optional target array `y` [n_samples] → `model.fit(X, y)` → `model.predict(X_new)` / `model.transform(X_new)`. This consistency is what lets `Pipeline`, `GridSearchCV`, and `cross_val_score` work uniformly across every model type.

**Bias-variance trade-off** — the central diagnostic lens for model selection: underfitting = high bias (model too simple, training and validation scores both low and close); overfitting = high variance (model too flexible, training score ≫ validation score). Validation curves (score vs. complexity) and learning curves (score vs. training-set size) are the book's standard tools for diagnosing which regime you're in and whether more data or a different model would help.

**Generative vs. discriminative classification** — generative models (Naive Bayes, GMM-based Bayes) model each class's full data distribution and compare likelihoods; discriminative models (SVM, logistic regression) find a boundary between classes directly without modeling the distributions. Generative models are faster and give probability estimates "for free"; discriminative models are often more accurate when the generative assumptions don't hold.

**Ensemble averaging** — combining many independently-fit, individually overfit/noisy estimators (bagging, random forests) to cancel out noise and reduce variance — "the sum can be greater than the parts." The randomization (data subsampling, feature-selection-per-split) is what makes the averaging effective, not just the averaging itself.

**Dimensionality reduction spectrum** — PCA (linear, fast, robust, interpretable) should be the first tool reached for; manifold learning (MDS/LLE/Isomap — nonlinear, better for visualization, finicky in practice) is a complement used only when PCA visibly fails to capture structure, not a default replacement.

**Real-world ML pipeline shape** (Ch 50 synthesizes the whole book) — raw data → domain-specific feature extraction (e.g., HOG) → standard classifier (e.g., linear SVM) → inference strategy (e.g., sliding window) → post-processing (deduplication/non-max suppression) → honest accounting of what separates a prototype from production.

---

## Chapter Index

| # | Title | Key Frameworks |
|---|-------|----------------|
| ch01 | Getting Started in IPython and Jupyter | Tab completion; `?`/`??` help & source introspection; namespace exploration |
| ch02 | Enhanced Interactive Features | Magic commands; input/output history (`Out`, `_`); shell integration |
| ch03 | Debugging and Profiling | Exception verbosity (`%xmode`); `%debug`/`ipdb`; `%timeit`/`%prun`/`%lprun`/`%memit` |
| ch04 | Understanding Data Types in Python | Python object overhead; fixed-type arrays; `dtype` |
| ch05 | The Basics of NumPy Arrays | Attributes, indexing, slicing, reshaping, concatenation/splitting; views vs. copies |
| ch06 | Computation on NumPy Arrays: Universal Functions | Ufuncs; vectorization; `np.add.at` for unbuffered operations |
| ch07 | Aggregations: min, max, and Everything in Between | `sum`/`min`/`max`/`mean`/`std`/`percentile`; axis-wise reduction |
| ch08 | Computation on Arrays: Broadcasting | Broadcasting rules; shape-stretching without copying |
| ch09 | Comparisons, Masks, and Boolean Logic | Comparison ufuncs; Boolean masks; `&`/`\|`/`~` vs. `and`/`or` |
| ch10 | Fancy Indexing | Index arrays; broadcast-shaped results; combined indexing |
| ch11 | Sorting Arrays | `np.sort`/`argsort`; `np.partition`/`argpartition`; vectorized k-NN |
| ch12 | Structured Data: NumPy's Structured Arrays | Compound dtypes; named heterogeneous fields; record arrays |
| ch13 | Introducing Pandas Objects | `Series`/`DataFrame`/`Index`; array-like + dict-like access |
| ch14 | Data Indexing and Selection | `.loc`/`.iloc`; Series vs. DataFrame bracket conventions |
| ch15 | Operating on Data in Pandas | Ufunc label preservation; automatic index alignment |
| ch16 | Handling Missing Data | `None`/`NaN`/`pd.NA`; `isnull`/`notnull`/`dropna`/`fillna` |
| ch17 | Hierarchical Indexing | `MultiIndex`; `stack`/`unstack` |
| ch18 | Combining Datasets: concat and append | `pd.concat`; index preservation/duplication; `join='outer'/'inner'` |
| ch19 | Combining Datasets: merge and join | `pd.merge`; one-to-one/many-to-one/many-to-many; `how=` |
| ch20 | Aggregation and Grouping | `groupby`; split-apply-combine; `agg`/`filter`/`transform`/`apply` |
| ch21 | Pivot Tables | `pivot_table`; `pd.cut`/`qcut` for binning |
| ch22 | Vectorized String Operations | `.str` accessor; regex methods; missing-value-aware |
| ch23 | Working with Time Series | `Timestamp`/`DatetimeIndex`/`datetime64`; resampling, shifting, rolling |
| ch24 | High-Performance Pandas: eval and query | `pd.eval`/`df.eval`/`df.query`; NumExpr; memory vs. speed trade-off |
| ch25 | General Matplotlib Tips | `plt` (stateful) vs. `Figure`/`Axes` (object-oriented) APIs |
| ch26 | Simple Line Plots | `Figure`+`Axes`; color/style/limits/labels/legend keywords |
| ch27 | Simple Scatter Plots | `plt.plot` vs. `plt.scatter`; `errorbar`; `fill_between` |
| ch28 | Density and Contour Plots | `contour`/`contourf`/`imshow`; 1D/2D histograms and KDE |
| ch29 | Customizing Plot Legends | `legend()` placement/styling; proxy artists; multiple legends |
| ch30 | Customizing Colorbars | Sequential/divergent/qualitative colormaps; perceptually uniform maps (viridis) |
| ch31 | Multiple Subplots | `plt.axes`/`subplot`/`subplots`/`GridSpec` |
| ch32 | Text and Annotation | `ax.text`/`ax.annotate`; data/axes-fraction/figure-fraction transforms |
| ch33 | Customizing Ticks | Locator (placement) / Formatter (label text) |
| ch34 | Customizing Matplotlib: Configurations and Stylesheets | `plt.rc`/`rcParams`; `plt.style.use`/`context` |
| ch35 | Three-Dimensional Plotting in Matplotlib | `mplot3d`; `projection='3d'`; wireframe/surface/triangulation |
| ch36 | Visualization with Seaborn | DataFrame-native plotting; `jointplot`/`pairplot`/`catplot` |
| ch37 | What Is Machine Learning? | Supervised (classification/regression) vs. unsupervised (clustering/dim. reduction) |
| ch38 | Introducing Scikit-Learn | Features matrix `X`/target `y`; five-step Estimator API |
| ch39 | Hyperparameters and Model Validation | Holdout/cross-validation; bias-variance; validation/learning curves; `GridSearchCV` |
| ch40 | Feature Engineering | One-hot encoding; text (TF-IDF); derived features; `Pipeline`/`make_pipeline` |
| ch41 | In Depth: Naive Bayes Classification | Gaussian/multinomial NB; fast generative baseline |
| ch42 | In Depth: Linear Regression | Basis functions (polynomial/Gaussian); ridge (L2)/lasso (L1) regularization |
| ch43 | In Depth: Support Vector Machines | Maximum-margin boundary; support vectors; kernel trick (`kernel='rbf'`) |
| ch44 | In Depth: Decision Trees and Random Forests | Axis-aligned splits; overfitting; bagging; `RandomForestClassifier`/`Regressor` |
| ch45 | In Depth: Principal Component Analysis | Principal axes; explained variance; noise filtering; eigenfaces |
| ch46 | In Depth: Manifold Learning | MDS/LLE/Isomap; nonlinear structure PCA misses |
| ch47 | In Depth: k-Means Clustering | Expectation-maximization; linear cluster boundaries; `n_init` |
| ch48 | In Depth: Gaussian Mixture Models | Soft/elliptical clustering; density estimation; AIC/BIC component selection |
| ch49 | In Depth: Kernel Density Estimation | Per-point kernels; bandwidth; KDE-based Bayesian generative classifier |
| ch50 | Application: A Face Detection Pipeline | HOG features; linear SVM; sliding window; non-max suppression |

## Topic Index
- **Array fundamentals (dtypes, indexing, slicing)** → ch04, ch05, ch10, ch12
- **Broadcasting** → ch08
- **Boolean masks / Boolean logic** → ch09, ch14
- **Ufuncs / vectorization** → ch06, ch07, ch22
- **Sorting / nearest-neighbors** → ch11
- **Pandas core objects (Series/DataFrame/Index)** → ch13, ch14
- **Missing data** → ch16
- **MultiIndex / hierarchical data** → ch17
- **Combining data (concat/merge/join)** → ch18, ch19
- **GroupBy / pivot tables** → ch20, ch21
- **String operations** → ch22
- **Time series** → ch23
- **Performance (eval/query/NumExpr)** → ch24
- **Matplotlib basics & API choice** → ch25, ch26
- **Plot types (line, scatter, density, contour, 3D)** → ch26, ch27, ch28, ch35
- **Legends / colorbars / annotation / ticks / styling** → ch29, ch30, ch32, ch33, ch34
- **Subplot layout** → ch31
- **Seaborn** → ch36
- **ML fundamentals / taxonomy** → ch37
- **Scikit-Learn API** → ch38
- **Model validation / hyperparameter tuning** → ch39
- **Feature engineering / pipelines** → ch40
- **Classification algorithms** → ch41, ch43, ch44
- **Regression** → ch42
- **Clustering** → ch47, ch48
- **Density estimation** → ch48, ch49
- **Dimensionality reduction** → ch45, ch46
- **End-to-end applied project** → ch50

## Supporting Files
- [glossary.md](glossary.md) — key terms with chapter cross-references
- [patterns.md](patterns.md) — reusable techniques with when-to-use/how/trade-offs
- [cheatsheet.md](cheatsheet.md) — decision rules, defaults, and common-mistake ("tells & smells") reference

---

## Scope & Limits
This skill covers the book's 50 chapters, synthesized as structure and technique, not verbatim text. The book's own front matter, preface, and index are excluded as non-substantive.
