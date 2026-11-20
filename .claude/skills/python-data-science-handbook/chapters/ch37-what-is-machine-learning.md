# Chapter 37: What Is Machine Learning?

## Core Idea
Machine learning builds tunable models of data that "learn" from observed examples, splitting into supervised learning (predict labels from labeled training data: classification=discrete, regression=continuous) and unsupervised learning (find structure in unlabeled data: clustering=discrete groups, dimensionality reduction=lower-dimensional representation).

## Frameworks Introduced
- **Supervised vs. unsupervised learning taxonomy**: the primary axis for categorizing ML problems.
  - Supervised: model the relationship between features and labels to predict labels for new data.
    - Classification: discrete category labels (e.g., spam/not-spam).
    - Regression: continuous labels (e.g., distance/redshift).
  - Unsupervised: model structure in features without any label reference.
    - Clustering: assign data to discrete groups based on intrinsic structure (e.g., k-means).
    - Dimensionality reduction: find a lower-dimensional representation preserving relevant structure (e.g., Isomap manifold learning).
  - Semi-supervised: between the two, useful when only incomplete labels are available.
- **Train → predict workflow**: fit model parameters to observed ("training") data, then generalize ("predict") on new, unseen data — the core machine-learning process pattern, independent of algorithm.

## Key Concepts
- **Model**: a parameterized mathematical structure whose tunable parameters are fit ("learned") from data.
- **Training**: the process of determining optimal parameter values from data.
- **Prediction**: applying a trained model to new, unlabeled data.
- **Feature**: a measured/quantitative attribute of a data point (e.g., (x,y) position).
- **Label**: the target quantity a supervised model predicts (discrete for classification, continuous for regression).
- **k-means**: fits k cluster centers minimizing each point's distance to its assigned center (Ch 47).
- **Manifold learning (e.g., Isomap)**: detects nonlinear lower-dimensional ("intrinsic") structure embedded in higher-dimensional space (Ch 46).

## Mental Models
- "Machine learning is a means of building models of data," not necessarily AI in the philosophical sense — this reframing emphasizes practical parameter-fitting over anthropomorphic "learning."
- A 2D/3D toy example (points on a plane, cluster blobs, a spiral) is "trivial to solve by eye" precisely so that the reader can verify the algorithm's output against intuition — the real value of these methods emerges only at higher dimensionality/larger scale, where human visual inspection fails.
- Regression generalizes "fit a line to 2D data" into "fit a hyperplane to N-D data" — same conceptual operation, more dimensions.

## Anti-patterns
- **Treating "machine learning" and "artificial intelligence" as interchangeable in a data-science context**: the author explicitly cautions this framing "can be misleading" — in data science, ML is better understood as data-model-fitting, not a general AI capability claim.
- **Judging a toy low-dimensional example as proof that an algorithm is unnecessary ("I could just do this by eye")**: misses the point — these methods' value is in generalizing to high-dimensional/large-scale data where manual inspection is impossible.
- **Expecting Scikit-Learn-style models to explain *why* a parameter has a particular value**: that's a statistical-modeling question, not primarily what ML models are built for (foreshadows Ch 38's point about `model.coef_` interpretation).

## Code Examples
```python
# No runnable code snippets in this conceptual chapter;
# illustrative pseudocode of the general ML workflow pattern:

# Supervised (classification/regression):
# 1. have features X and labels y
# 2. train/fit a model on (X, y) -> learns parameters (e.g., a separating line, a regression plane)
# 3. predict labels for new X_new using the trained model

# Unsupervised (clustering/dimensionality reduction):
# 1. have only features X (no labels)
# 2. fit a model that discovers structure (e.g., k cluster centers, or a lower-dim embedding)
# 3. use the discovered structure (cluster assignment, or reduced coordinates) for downstream analysis/visualization
```
- **What it demonstrates**: this is a conceptual/vocabulary-setting chapter (all figures generated from actual ML computations, detailed code deferred to later chapters) — illustrates the four canonical problem types (classification, regression, clustering, dimensionality reduction) using toy 2D datasets.

## Reference Tables
| Category | Subtype | Label type | Example algorithms (covered later) |
|---|---|---|---|
| Supervised | Classification | Discrete | Gaussian naive Bayes (Ch 41), SVM (Ch 43), random forest (Ch 44) |
| Supervised | Regression | Continuous | Linear regression (Ch 42), SVM (Ch 43), random forest (Ch 44) |
| Unsupervised | Clustering | None (assigns groups) | k-means (Ch 47), Gaussian mixture models (Ch 48) |
| Unsupervised | Dimensionality reduction | None (finds structure) | PCA (Ch 45), Isomap/manifold learning (Ch 46) |

## Worked Example
Spiral-data dimensionality reduction: a 2D dataset visually forms a 1D line embedded as a spiral in 2D space — "intrinsically" one-dimensional despite its 2D coordinates. The Isomap manifold-learning algorithm (Ch 46) extracts a 1D latent variable that changes uniformly along the spiral, confirming it correctly recovered the underlying structure purely from unlabeled data — illustrating unsupervised dimensionality reduction's goal of finding a lower-dimensional representation that preserves relevant structure, which becomes essential (not just illustrative) once data has 100s or 1000s of features that can't be visualized directly.

## Key Takeaways
1. ML is best understood (in data science) as building tunable models of data, not strictly as "AI."
2. Supervised learning predicts labels from labeled training data: classification (discrete) vs. regression (continuous).
3. Unsupervised learning finds structure without labels: clustering (discrete groups) vs. dimensionality reduction (lower-dimensional structure).
4. Semi-supervised learning sits between the two, for partially labeled data.
5. Toy low-dimensional examples exist to build intuition/verify correctness — the real power of these methods is at scale and in high dimensions where manual inspection is impossible.
6. Real-world analogs: spam detection (classification), photometric redshift estimation (regression), automatic grouping (clustering), visualizing high-dimensional data (dimensionality reduction).

## Connects To
- **Ch 38**: introduces the Scikit-Learn API that implements all four problem types with a consistent interface.
- **Ch 41/42/43/44**: classification and regression algorithms covered in depth.
- **Ch 45/46/47/48**: dimensionality reduction and clustering algorithms covered in depth.
