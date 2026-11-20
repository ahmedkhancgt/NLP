# Glossary

**alpha / beta** — Intercept and slope of a simple linear model `y = beta*x + alpha + error` (Ch14).

**Adjacency list / matrix** — Dict mapping node -> neighbor list (sparse-friendly) vs. an NxN 0/1 matrix (dense, O(1) lookup) representation of a graph (Ch1, Ch4).

**Backpropagation** — Training neural network weights via gradient descent by propagating output-layer error backward through hidden layers using the chain rule (Ch18).

**Bagging (bootstrap aggregating)** — Training each ensemble member (e.g., a random forest tree) on an independent bootstrap resample of the data (Ch17).

**Bayes's theorem** — `P(E|F) = P(F|E)P(E)/P(F)`; reverses a known conditional probability into the one you actually want (Ch6).

**Betweenness centrality** — A node's importance measured by the fraction of all shortest paths between other node pairs that pass through it (Ch22).

**Bias-variance tradeoff** — High bias = underfitting (too simple, fix with more features); high variance = overfitting (too complex, fix with fewer features/regularization/more data) (Ch11).

**Bootstrap (resampling)** — Resampling a dataset with replacement many times to estimate a statistic's standard error without an analytic formula (Ch15).

**Collinearity** — When one regression predictor is a linear function of others, making beta unidentifiable (Ch15).

**Confusion matrix** — Table of true/false positive/negative counts underlying precision, recall, and F1 (Ch11, Ch12).

**Cosine similarity** — `dot(v1,v2)/(|v1||v2|)`, in [-1,1]; measures vector direction similarity regardless of magnitude (Ch21, Ch23).

**Counter / defaultdict** — `collections` classes used throughout for from-scratch tallying and auto-initialized key/value aggregation (Ch1, Ch2).

**Curse of dimensionality** — In high dimensions, nearest-neighbor distance approaches average distance, degrading any distance-based method (Ch12).

**DataSciencester** — The book's running fictional "social network for data scientists," the dataset source for nearly every chapter's worked example (Ch1).

**Dot product** — `sum(v_i*w_i)`; the core linear-algebra primitive underlying magnitude, distance, projection, and every weighted-sum computation in the book (Ch4).

**Entropy** — `sum(-p*log2(p))`; measures classification uncertainty; ID3 picks splits minimizing weighted (partition) entropy (Ch17).

**Gradient descent** — Repeatedly stepping parameters against the loss function's gradient to minimize it; the book's universal optimization tool (Ch8).

**Gradient descent variants (batch/minibatch/stochastic)** — Full-dataset, sampled sub-batch, or single-example gradient computation per update step, trading stability for update frequency (Ch8).

**ID3 algorithm** — Greedy recursive decision-tree construction: pick the lowest-partition-entropy attribute at each step, split, recurse (Ch17).

**k-means** — Unsupervised clustering: iteratively assign points to nearest of k means, recompute means, repeat until stable (Ch20).

**k-Nearest Neighbors (k-NN)** — Classify a point by majority vote among its k nearest labeled neighbors (Ch12).

**Latent Dirichlet Allocation (LDA)** — Topic-modeling technique assuming documents are topic mixtures and topics are word distributions, fit via Gibbs sampling (Ch21).

**Least squares** — Fitting a model by minimizing sum of squared errors; has a closed form for simple regression, requires gradient descent for multiple regression (Ch14, Ch15).

**Logistic function** — `1/(1+exp(-x))`; squashes a real number into (0,1), used as both a classification output and a neural-network activation (Ch16, Ch18).

**MapReduce** — Programming model: mapper emits key/value pairs, group by key, reducer collapses each group — a model for distributed parallel data processing (Ch25).

**Matrix factorization** — Learning latent user/item embedding vectors such that `dot(user_vec, item_vec)` predicts a rating, fit via gradient descent (Ch23).

**Maximum Likelihood Estimation (MLE)** — Choosing parameters that maximize the probability of observed data; justifies least squares (under normal errors) and cross-entropy (under softmax outputs) as principled loss functions (Ch14, Ch19).

**Naive Bayes** — Text classifier assuming word presence is conditionally independent given class, letting Bayes's theorem reduce to a per-word probability product (Ch13).

**NamedTuple** — Typed, immutable tuple subclass used throughout for structured records (e.g., `LabeledPoint`, `Message`, `StockPrice`), safer than dict-as-struct (Ch10).

**Normal distribution / CLT** — Bell-curve distribution; the Central Limit Theorem explains why sums/averages of many independent variables converge to it regardless of underlying distribution (Ch6).

**NotQuiteABase** — The book's from-scratch pure-Python relational `Table` class mirroring SQL operations (select/where/group_by/join/etc.) (Ch24).

**Null hypothesis (H0) / p-value** — The default assumption tested in hypothesis testing; p-value is the probability of data at least as extreme as observed, assuming H0 true (Ch7).

**One-hot encoding** — Representing a categorical label as a vector of 0s with a single 1 at the correct class index (Ch18, Ch19).

**Overfitting / underfitting** — A model that fits training data too specifically vs. one too simple to fit it at all; diagnosed via train/test performance gap (Ch11).

**PageRank** — Iterative rank-propagation algorithm for directed graphs, damping a node's influence along outgoing edges plus a uniform baseline; resists gaming unlike raw in-degree (Ch22).

**PCA (Principal Component Analysis)** — Dimensionality reduction by finding directions of maximal variance via gradient ascent on `directional_variance`, after de-meaning the data (Ch10).

**Perceptron** — A single artificial neuron (`step_function(dot(weights,x)+bias)`); can represent AND/OR/NOT but not XOR (linearly separable functions only) (Ch18).

**Power iteration** — Repeated matrix-vector multiply + rescale procedure for finding a matrix's dominant eigenvector (used for eigenvector centrality) (Ch22).

**Precision / Recall / F1** — Precision = tp/(tp+fp); Recall = tp/(tp+fn); F1 = their harmonic mean — binary classification metrics that avoid accuracy's class-imbalance blind spot (Ch11).

**R-squared** — `1 - SSE/SST`; fraction of outcome variance a regression model explains, bounded [0,1] (Ch14, Ch15).

**Random forest** — Ensemble of decision trees, each trained on a bootstrap sample with random attribute subsampling at each split, combined by majority vote (Ch17).

**Regularization (ridge / lasso)** — Adding an L2 (`sum(beta^2)`) or L1 (`sum(|beta|)`) penalty to a regression loss to shrink or sparsify coefficients and reduce overfitting (Ch15).

**Sigmoid / Tanh / ReLU** — Neural network activation functions with different ranges/saturation behavior; sigmoid=(0,1), tanh=(-1,1), ReLU=max(x,0) (Ch18, Ch19).

**Simpson's paradox** — A correlation/relationship that reverses direction once a confounding variable is controlled for (Ch5).

**Softmax + cross-entropy** — Turns raw outputs into a probability distribution and scores them via negative log-likelihood; trains faster than sigmoid+SSE for multi-class classification (Ch19).

**Supervised / unsupervised learning** — Learning from labeled data with known correct answers vs. finding structure in unlabeled data (clustering) (Ch11, Ch20).

**Tensor / Layer / Sequential** — The book's from-scratch deep-learning abstractions: nested-list n-dimensional arrays, a forward/backward/params/grads component interface, and layer-chaining composition (Ch19).

**Tokenize** — Extracting a lowercase, deduplicated set of word tokens from text via regex, the first step of every NLP technique in the book (Ch13, Ch21).

**Train/test/validation split** — Dividing data to fit a model (train), evaluate it (test), and choose among candidate models (validation) — using the test set for model selection is a common mistake (Ch11).

**Type annotation** — Python `typing`-module hints (`List[float]`, `Optional[int]`) used as documentation (not runtime-enforced) throughout the book's code, checked via `mypy` (Ch2).

**Variance / standard deviation / covariance / correlation** — Core dispersion and co-movement statistics; correlation normalizes covariance to [-1,1] (Ch5).

**Vector / Matrix (type aliases)** — `Vector = List[float]`, `Matrix = List[List[float]]`; the book's from-scratch numeric representation underlying nearly every later algorithm (Ch4).
