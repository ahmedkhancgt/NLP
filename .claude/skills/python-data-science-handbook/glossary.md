# Glossary

**Aggregation** — reducing an array/column/group to summary statistic(s) (sum, mean, min, max, etc.) (Ch 7, 20)

**AIC/BIC** — Akaike/Bayesian information criterion; likelihood-based model-complexity penalties used to choose component count in density estimators like GMM (Ch 48)

**Axes (Matplotlib)** — a bounding box with ticks/labels/grid holding plot elements; contained within a Figure (Ch 25, 26)

**Bagging** — ensemble method averaging many overfit estimators trained on random data subsets to reduce variance (Ch 44)

**Bandwidth (KDE)** — the kernel width parameter controlling the bias-variance trade-off in kernel density estimation (Ch 49)

**Basis function** — a transform of input features (e.g., polynomial, Gaussian) used so a linear model can fit nonlinear relationships (Ch 42)

**Bias-variance trade-off** — the balance between underfitting (high bias) and overfitting (high variance) that determines optimal model complexity (Ch 39)

**Boolean mask** — a same-shaped Boolean array used to index/filter another array or DataFrame (Ch 9, 14)

**Broadcasting** — NumPy's rule-based mechanism for applying operations between arrays of different shapes (Ch 8)

**Classification** — supervised learning predicting discrete category labels (Ch 37)

**Clustering** — unsupervised learning that groups data into discrete clusters without labels (Ch 37, 47, 48)

**Colorbar** — a continuous-data legend showing what plotted colors represent (Ch 30)

**Confusion matrix** — a table showing predicted vs. true labels, revealing specific misclassification patterns (Ch 38, 41, 44)

**Cross-validation** — repeatedly splitting data into train/validation folds to robustly estimate model performance (Ch 39)

**Datetime64** — NumPy's compact 64-bit date/time dtype, basis for Pandas' Timestamp/DatetimeIndex (Ch 23)

**Decision tree** — a model that recursively splits data via axis-aligned feature cutoffs; prone to overfitting alone, powerful in ensembles (Ch 44)

**Density estimator** — an algorithm estimating the probability distribution that generated a dataset (histograms, KDE, GMM) (Ch 48, 49)

**DictVectorizer** — Scikit-Learn tool for one-hot encoding categorical data from lists of dicts (Ch 40)

**Discriminative classification** — finding a boundary between classes directly (e.g., SVM), vs. modeling each class's distribution (generative) (Ch 43)

**Dimensionality reduction** — unsupervised learning finding lower-dimensional representations preserving structure (PCA, manifold learning) (Ch 37, 45, 46)

**DataFrame** — Pandas' 2D labeled data structure; aligned collection of Series sharing an index (Ch 13)

**Estimator API** — Scikit-Learn's consistent five-step interface (choose class → hyperparameters → arrange data → fit → predict/transform) (Ch 38)

**Eval/query** — Pandas' NumExpr-backed tools for memory-efficient compound expression evaluation (Ch 24)

**Expectation-Maximization (E-M)** — iterative algorithm alternating assignment (E-step) and parameter update (M-step); basis of k-means and GMM (Ch 47, 48)

**Explained variance (ratio)** — the fraction of total data variance captured by a PCA component (Ch 45)

**Face recognition / eigenfaces** — PCA components of face images, used as compact features for classification (Ch 43, 45)

**Feature engineering** — converting raw/real-world data into numerical feature matrices (Ch 40)

**Feature matrix (X)** — Scikit-Learn's 2D [n_samples, n_features] data representation (Ch 38)

**Figure (Matplotlib)** — the top-level container for all plot elements (Ch 25, 26)

**Fancy indexing** — indexing an array with an array of indices rather than a scalar or slice (Ch 10)

**GridSearchCV** — Scikit-Learn tool automating hyperparameter search across a grid, using cross-validation (Ch 39)

**GroupBy object** — Pandas' lazy view enabling split-apply-combine operations (Ch 20)

**HOG (Histogram of Oriented Gradients)** — an image feature-extraction technique sensitive to edges/contours, robust to illumination (Ch 50)

**Holdout set** — data withheld from training, used to evaluate generalization (Ch 39)

**Hyperparameter** — a model configuration value set before fitting (e.g., `n_neighbors`, `C`, `alpha`) (Ch 38, 39)

**Index (Pandas)** — the immutable, set-like array of labels for a Series/DataFrame (Ch 13)

**Isomap** — a manifold learning algorithm recovering nonlinear low-dimensional structure via geodesic distances (Ch 46)

**Kernel density estimation (KDE)** — nonparametric density estimation placing one kernel per data point (Ch 49)

**Kernel trick** — implicitly computing a high-dimensional projection (as in kernel SVM) without materializing it, via pairwise similarity functions (Ch 43)

**k-means** — clustering algorithm finding k cluster centers via iterative E-M, assuming circular/linear boundaries (Ch 47)

**Learning curve** — plot of train/validation score vs. training-set size, diagnosing whether more data would help (Ch 39)

**Linear regression** — fitting a line/hyperplane (`y = a·x + b`) to predict continuous labels (Ch 42)

**Locally linear embedding (LLE)** — manifold learning preserving only local (neighbor) distances, enabling nonlinear unrolling (Ch 46)

**Locator/Formatter** — Matplotlib objects controlling tick placement (Locator) and tick label text (Formatter) (Ch 33)

**Manifold learning** — unsupervised methods (MDS, LLE, Isomap) recovering low-dimensional structure from nonlinearly embedded high-dimensional data (Ch 46)

**MultiIndex** — Pandas' hierarchical index, representing N-dimensional data within 1D/2D structures (Ch 17)

**NaN** — IEEE floating-point "Not a Number," Pandas' default missing-value sentinel (Ch 16)

**Naive Bayes** — fast generative classifier assuming a simplified per-class distribution (Gaussian or multinomial) (Ch 41)

**NumExpr** — library evaluating compound array expressions without allocating full intermediate arrays; backs Pandas eval/query (Ch 24)

**One-hot encoding** — representing a categorical variable as multiple binary indicator columns (Ch 40)

**Overfitting** — a model too flexible, fitting noise rather than signal (high variance) (Ch 39)

**Pipeline (Scikit-Learn)** — `make_pipeline`-chained sequence of transformers + estimator, usable as a single fit/predict object (Ch 40)

**Pivot table** — a convenience wrapper for 2D+ groupby aggregation producing a spreadsheet-style summary grid (Ch 21)

**Principal component analysis (PCA)** — linear dimensionality reduction finding directions (principal axes) of maximal variance (Ch 45)

**Random forest** — ensemble of randomized decision trees, averaging predictions to reduce overfitting (Ch 44)

**Regularization** — penalizing large model coefficients to control overfitting; ridge (L2) shrinks smoothly, lasso (L1) induces sparsity (Ch 42)

**Series** — Pandas' 1D labeled array, combining array-like and dict-like access (Ch 13)

**Split-apply-combine** — the conceptual decomposition of groupby operations: split by key, apply a function, combine results (Ch 20)

**Structured array** — NumPy array with a compound dtype holding named heterogeneous fields (Ch 12)

**Support vector machine (SVM)** — discriminative classifier maximizing the margin between classes; kernel SVM handles nonlinear boundaries (Ch 43)

**Target array (y)** — Scikit-Learn's 1D array of labels to predict (Ch 38)

**TF-IDF** — term frequency-inverse document frequency; text feature weighting downweighting common words (Ch 40)

**Timestamp/DatetimeIndex** — Pandas' efficient, convenient date/time types built on NumPy datetime64 (Ch 23)

**Ufunc (universal function)** — NumPy's vectorized, compiled element-wise function, avoiding slow Python loops (Ch 6)

**Validation curve** — plot of train/validation score vs. model complexity, revealing the bias-variance optimal point (Ch 39)

**Vectorization** — expressing an operation over whole arrays (or strings, via `.str`) rather than via explicit loops (Ch 6, 22)

**View (NumPy)** — a slice sharing the same underlying memory as the original array (vs. a copy) (Ch 5)
