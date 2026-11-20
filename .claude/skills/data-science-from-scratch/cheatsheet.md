# Cheatsheet

## Which Model/Technique for Which Problem

| Need | Reach for | Chapter |
|---|---|---|
| Predict a continuous number from one predictor | Simple linear regression (closed-form) | Ch14 |
| Predict a continuous number from many predictors | Multiple regression (gradient descent) | Ch15 |
| Predict a 0/1 label, want calibrated probability | Logistic regression | Ch16 |
| Predict a label, want full interpretability | Decision tree (ID3) | Ch17 |
| Predict a label, decision tree overfits | Random forest | Ch17 |
| Classify text (spam/ham) | Naive Bayes | Ch13 |
| Classify via similarity to known examples | k-Nearest Neighbors | Ch12 |
| Need a nonlinear decision boundary, no interpretability required | Neural network | Ch18/Ch19 |
| Group unlabeled points, know desired group count | k-means | Ch20 |
| Group unlabeled points, want a full hierarchy / unknown k | Hierarchical clustering | Ch20 |
| Discover latent themes across documents | LDA topic modeling | Ch21 |
| Find "important" nodes in an undirected network | Eigenvector/betweenness/closeness centrality | Ch22 |
| Rank nodes in a directed endorsement network | PageRank | Ch22 |
| Recommend items from binary interest data | User-based or item-based collaborative filtering | Ch23 |
| Recommend/predict ratings from sparse numeric data | Matrix factorization | Ch23 |
| Reduce dimensionality / denoise correlated features | PCA | Ch10 |
| Process data across many machines | MapReduce | Ch25 |

## Decision Rules

- **Mean vs. median**: outliers present or suspected -> median/IQR; otherwise mean/stdev is fine and uses all the data.
- **One-sided vs. two-sided hypothesis test**: alternative is directional ("p > 0.5") -> one-sided (more powerful); alternative is just "different" -> two-sided.
- **Ridge vs. lasso**: want smooth shrinkage, keep all variables -> ridge (L2); want automatic variable elimination (sparse model) -> lasso (L1), but it needs more than plain gradient descent.
- **User-based vs. item-based collaborative filtering**: small/medium item catalog -> either; large item catalog -> item-based (more robust to the curse of dimensionality).
- **Batch vs. minibatch vs. stochastic gradient descent**: small dataset, want stable convergence -> batch; large dataset -> minibatch or stochastic (faster per-epoch wall-clock, noisier steps).
- **Min vs. max linkage (hierarchical clustering)**: want loosely-connected chain-like clusters -> min; want tight, compact, k-means-like clusters -> max.
- **Sigmoid vs. softmax output (multi-class)**: always prefer softmax + cross-entropy over sigmoid + SSE for classification — trains faster, true probability interpretation.

## Thresholds & Tells

| Signal | Meaning |
|---|---|
| High train accuracy, low test accuracy | Overfitting (high variance) -> remove features, regularize, get more data |
| Low train AND test accuracy | Underfitting (high bias) -> add features/complexity, don't just add data |
| p-value < significance level (e.g. 0.05) | Reject H0; but check for p-hacking if many tests were run |
| R-squared near 0 | Model barely better than predicting the mean |
| R-squared always rises as you add predictors | Expected mechanically — check standard errors/p-values, not R-squared alone, to judge if a predictor matters |
| Coefficient p-value > 0.4ish, shrinks fastest under ridge | Likely noise, not real signal |
| Partition entropy near 0 for a split | Near-pure classes after split — but check if it's a high-cardinality "cheat" attribute (e.g. ID) |
| Nearest-neighbor distance ~ average distance | Curse of dimensionality — reduce dimensions (PCA) before k-NN/clustering |
| Accuracy high (>95%) but precision/recall near 0 | Classic class-imbalance trap — always check precision/recall/F1 |

## Common Formulas

| Quantity | Formula |
|---|---|
| Dot product | `sum(v_i * w_i)` |
| Euclidean distance | `sqrt(sum((v_i-w_i)^2))` |
| Variance (sample) | `sum((x_i-mean)^2) / (n-1)` |
| Correlation | `covariance(x,y) / (stdev_x * stdev_y)` |
| R-squared | `1 - SSE/SST` |
| Precision | `tp / (tp+fp)` |
| Recall | `tp / (tp+fn)` |
| F1 | `2*precision*recall / (precision+recall)` |
| Entropy | `sum(-p*log2(p) for p in class_probs)` |
| Sigmoid/logistic | `1 / (1 + exp(-x))` |
| Cosine similarity | `dot(v,w) / (magnitude(v)*magnitude(w))` |
| Ridge penalty | `alpha * sum(beta_i^2 for i>0)` |
| Lasso penalty | `alpha * sum(abs(beta_i) for i>0)` |

## Quick Workflow Checklist

1. Explore first: histogram (1D), scatterplot (2D), correlation matrix (many-D) — never trust summary stats alone (Ch3, Ch5, Ch10).
2. Clean/parse defensively: fail-soft parsing returning `Optional`, use typed `NamedTuple` records, not raw dicts (Ch9, Ch10).
3. Split data: train/test (always), add validation if comparing models (Ch11).
4. Rescale if doing anything distance-based or regularized (Ch10, Ch15).
5. Pick the model from the table above based on task shape (label type, interpretability need, data size).
6. Fit via closed form if one exists (Ch14), else gradient descent (Ch8, Ch15, Ch16, Ch18-19).
7. Evaluate with the right metric: R-squared for regression, precision/recall/F1 for classification, never raw accuracy under class imbalance (Ch11).
8. Diagnose bias vs. variance if results are poor; apply the matching fix, not just "more data" (Ch11).
9. In production, replace from-scratch code with NumPy/pandas/scikit-learn — this book's implementations are for understanding mechanics, not for shipping (Ch27).
