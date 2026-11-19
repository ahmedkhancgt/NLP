# Chapter 7: Unsupervised Learning

## Core Idea
Unsupervised learning (PCA, K-means, hierarchical, model-based clustering) extracts structure from unlabeled data for dimension reduction, segmentation, and exploratory insight — but every method depends critically on proper variable scaling, and different clustering approaches can yield strikingly different results on noisy real-world data, with no universal rule for choosing among them.

## Frameworks Introduced
- **Principal Components Analysis (PCA)**: finds linear combinations (components) of numeric predictors that successively maximize explained variance while being mutually uncorrelated (orthogonal).
  - When to use: reduce dimensionality of correlated numeric variables before modeling; exploratory structure discovery (e.g., which stocks move together).
  - How: `princomp()` / `sklearn.decomposition.PCA`; inspect loadings (weights) and a scree plot to decide how many components to keep (common rule: cumulative variance > ~80%, or visual "elbow").
- **Correspondence Analysis**: PCA's categorical-data analog; finds associations between categories via a biplot of scaled row/column coordinates.
  - When to use: visualizing relationships in small categorical contingency tables (not for big-data dimension reduction).
- **K-Means Clustering**: partitions records into K groups by iteratively minimizing within-cluster sum of squares (distance of each point to its cluster mean).
  - When to use: scalable clustering of numeric data; cold-start problems (no labels yet, e.g., new marketing segments or fraud types).
  - How: `k_means()`/`KMeans()`; always standardize features first; use multiple random starts (`nstart`/`n_init`) since the algorithm only finds a local optimum; select K via the elbow method or managerial/practical constraints.
- **Hierarchical Clustering (agglomerative)**: starts with each record as its own cluster and iteratively merges the least dissimilar pair, producing a dendrogram.
  - When to use: smaller datasets, need to visualize cluster structure at multiple granularities without pre-specifying K, or mixed numeric/categorical data (with Gower's distance).
  - How: `hclust()`/`scipy.cluster.hierarchy.linkage`; choose a dissimilarity measure (complete, single, average, or minimum-variance/Ward's linkage); cut the dendrogram at a chosen height or cluster count.
- **Model-Based Clustering (Gaussian Mixture Models)**: assumes data is drawn from a mixture of K multivariate normal distributions, each with its own mean and covariance.
  - When to use: want a statistically principled method and automatic selection of cluster count; suspect clusters differ in shape/spread, not just location.
  - How: `mclust::Mclust()` / `sklearn.mixture.GaussianMixture`; select number of clusters/components via BIC (Bayesian Information Criterion).
- **Gower's Distance**: a mixed numeric/categorical dissimilarity metric — Manhattan distance (scaled 0-1) for numeric/ordinal variables, 0/1 match/mismatch for categorical variables, averaged across variables.
  - When to use: clustering data with both continuous and categorical features (e.g., loan `dti` + `home_ownership` type).

## Key Concepts
- **Loadings**: the weights that transform original variables into a principal component; sign is arbitrary (flipping all signs gives an equivalent component).
- **Scree plot**: bar/line chart of component variances, used to visually select how many components to retain.
- **Cluster mean**: the vector of variable means for all records in a cluster (the "centroid").
- **Within-cluster sum of squares (SS)**: K-means' objective function — sum of squared distances from each point to its cluster's mean; minimized by the algorithm.
- **Dendrogram**: tree diagram showing the hierarchy and merge-distances of hierarchical clustering; branch length encodes dissimilarity.
- **Dissimilarity / linkage methods**: complete (max pairwise distance — compact clusters), single (min pairwise distance — "greedy," can chain disparate points), average, and minimum-variance/Ward's (most similar to K-means).
- **BIC (Bayesian Information Criterion)**: penalizes model complexity (more clusters/parameters) against fit quality; used by model-based clustering to auto-select cluster count.
- **Standardization (z-scores)**: essential before PCA/K-means/hierarchical clustering so large-magnitude variables (e.g., dollar amounts) don't dominate distance/variance calculations.
- **Dominant variables**: a few high-variance variables (e.g., GOOGL/AMZN stock swings) can swamp PCA/clustering results even on a "common" scale; address by rescaling or separate analysis.

## Mental Models
- Think of PCA as finding the "long axis" and successively orthogonal "short axes" of the data's variability ellipse — the first component is literally the direction of greatest spread.
- Treat clustering method choice as a trade-off triangle: K-means (scalable, simple, needs K upfront), hierarchical (flexible, visual, doesn't scale), model-based (statistically rigorous, auto-selects K, computationally heavy, assumes normal-ish clusters).
- Use the elbow-plot/BIC-plot mental model consistently: as you add clusters/components, fit always improves; look for where the *marginal* improvement flattens, not where it's maximized.
- View cluster analysis and PCA as related but distinct lenses: PCA finds *directions* of shared variation (sign is not meaningful on its own), while clustering finds *groups* of records that are physically close together (cluster mean signs are meaningful, representing actual group characteristics).

## Anti-patterns
- **Running PCA/K-means/hierarchical clustering on unscaled variables**: a dollar-magnitude variable (e.g., `annual_inc`) will dominate cluster formation and loadings, producing lopsided, uninterpretable clusters (shown directly in the chapter's unscaled-vs-scaled loan default example: cluster sizes 52/1192/13902/7525 vs. a far more balanced split after standardizing).
- **One-hot encoding categorical variables and feeding them into K-means/PCA without care**: binary 0/1 variables can dominate clustering because they minimize within-cluster SS trivially by grouping all 0s or all 1s together — use Gower's distance or separate categorical-stratified analysis instead.
- **Treating K-means/hierarchical clustering results as ground truth when clusters aren't well-separated**: on noisy data (stock returns, loan data), different methods produce dramatically different clusters — there is no universally "correct" answer, so always sanity-check robustness (e.g., cross-validate or compare multiple methods).
- **Picking K purely by maximizing variance explained / minimizing BIC without considering interpretability or practical/managerial constraints**: statistical optimality and business usability can diverge.
- **Forgetting that PCA component sign is arbitrary**: don't over-interpret whether loadings are positive or negative in absolute terms — only relative signs within a component carry meaning.

## Code Examples
```r
pca <- princomp(oil_px)
pca$loadings
# CVX -0.747  0.665
# XOM -0.665 -0.747
```
- **What it demonstrates**: PC1 ≈ average of CVX/XOM (co-movement), PC2 ≈ their divergence — a simple two-stock PCA.

```r
kmeans_wf <- workflow() %>%
  add_recipe(recipe(~ XOM + CVX, data = df)) %>%
  add_model(k_means(num_clusters = 4))
model <- kmeans_wf %>% fit(df)
```
- **What it demonstrates**: K-means on standardized-by-construction stock returns, producing 4 clusters corresponding to "up market"/"down market" regimes.

```r
d <- dist(df)   # df transposed so companies are rows
hcl <- hclust(d)
cutree(hcl, k = 4)
```
- **What it demonstrates**: hierarchical clustering of companies (not days) by return similarity; oil stocks (SLB, CVX, XOM, COP) form their own cluster, GOOGL and AMZN are each singleton clusters.

```r
mcl <- Mclust(df)
summary(mcl)
# Mclust VEE model with 2 components; BIC selects cluster count automatically
```
- **What it demonstrates**: Gaussian-mixture model-based clustering automatically choosing 2 components via BIC, revealing a "core" normal cluster plus a higher-variance outlier cluster (reflecting stock returns' long tails).

## Reference Tables
| Method | Scales to big data? | Needs K upfront? | Handles mixed data? | Statistical rigor |
|---|---|---|---|---|
| K-means | Yes | Yes | Poorly (needs one-hot + careful scaling) | Heuristic |
| Hierarchical | No (small/medium only) | No (choose via dendrogram cut) | Yes (via Gower's distance) | Heuristic |
| Model-based (GMM) | No (computationally heavy) | No (BIC auto-selects) | Numeric only | Statistical (assumes normal mixture) |

| Linkage method | Produces | Similar to |
|---|---|---|
| Complete | Compact, similar clusters | — |
| Single | "Greedy," can chain disparate records | — |
| Average | Compromise | — |
| Minimum variance (Ward's) | Minimizes within-cluster SS | K-means |

## Worked Example
S&P 500 stock price data (`sp500_px.csv.gz`): PCA on just Chevron (CVX) and ExxonMobil (XOM) daily returns shows PC1 (loadings ≈ −0.747/−0.665) captures their shared co-movement, PC2 captures their divergence. Extending PCA to 16 top S&P 500 stocks reveals PC1 as a broad market factor, PC2 as energy-vs-other sector rotation, PC3 as Apple-vs-retail, etc. — with ChatGPT asked to interpret the loadings and producing a coherent "PC1 = market/cyclical factor, PC2 = consumer/tech vs. energy rotation..." narrative. K-means with K=4 on CVX/XOM returns finds clusters corresponding to "up market"/"down market" regimes; extending to all 16 stocks with K=5 finds clusters like "broad bull rally," "energy-specific downturn," "sector rotation," interpreted again via an LLM prompt. Hierarchical clustering on 18 stocks (adding GOOGL, AMZN) shows oil stocks clustering together, while GOOGL and AMZN are each distinct outlier singleton clusters — illustrating hierarchical clustering's sensitivity to outlying groups. Model-based clustering (`mclust`) on CVX/XOM finds 2 components via BIC: one tight "normal" cluster (963 points) and one high-variance "outlier" cluster (168 points), reflecting that stock returns are long-tailed rather than truly normal. Separately, LendingClub loan default data clustering without scaling produces lopsided clusters (dominated by `annual_inc`/`revol_bal`); after z-score standardization, cluster sizes balance out and reveal more genuine structure. Gower's distance applied to mixed loan variables (`dti`, `payment_inc_ratio`, `home_`, `purpose_`) correctly groups together records sharing the same categorical values (e.g., all RENT/"other"-purpose loans) in a hierarchical dendrogram.

## Key Takeaways
1. Always standardize (z-score) numeric variables before PCA, K-means, or hierarchical clustering — unscaled large-magnitude variables will dominate the result.
2. Use PCA for dimension reduction of correlated numeric data; pick the number of components via a scree plot or cumulative-variance threshold, and always check if a few dominant variables are hijacking the top components.
3. Choose a clustering method based on your constraints: K-means for scale and simplicity, hierarchical for visual/exploratory flexibility and mixed data (via Gower's distance), model-based for statistical rigor and automatic K-selection.
4. On noisy, poorly-separated real-world data, expect different clustering methods to disagree substantially — there's no universal rule for picking the "right" one; validate against practical interpretability and (if possible) cross-validation.
5. Handle mixed categorical/numeric data carefully — one-hot encoding plus naive z-scoring can let binary variables dominate; Gower's distance or stratified analysis are safer alternatives.
6. LLMs can meaningfully assist in interpreting PCA loadings and cluster centers by narrating patterns, but their narrative should be checked against the actual numbers and domain knowledge.

## Connects To
- **Ch 5 (Classification)**: PCA is described as "the unsupervised version of LDA" — both rely on covariance-matrix-based linear transformations.
- **Ch 6 (Statistical Machine Learning)**: the same standardization and distance-metric concepts (Euclidean, Manhattan, Mahalanobis) from KNN reappear here as the backbone of clustering.
- **Ch 1 (Exploratory Data Analysis)**: unsupervised learning is framed as an extension of EDA to many-variable, many-record settings — clustering as a more formal way to "fully explore the data."
