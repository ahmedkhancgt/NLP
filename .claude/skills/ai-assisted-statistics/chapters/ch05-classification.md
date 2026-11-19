# Chapter 5: Classification

## Core Idea
Classification predicts a propensity (probability) that a record belongs to a class of interest, and choosing the right model (naive Bayes, discriminant analysis, logistic regression) is only half the job — correctly evaluating performance (beyond raw accuracy) and handling class imbalance are essential, especially since the rare class is usually the one that matters most.

## Frameworks Introduced
- **Naive Bayes**: estimates P(Y=i | X1,...,Xp) by assuming conditional independence of predictors given the class, multiplying individual conditional probabilities P(Xj|Y=i).
  - When to use: categorical predictors (e.g., spam text features); when unbiased probability isn't needed, only correct ranking.
  - How: for numeric predictors, either assume a normal distribution per class or bin into categories; use Laplace smoothing to avoid zero-probability from unseen categories.
- **Linear Discriminant Analysis (LDA)**: maximizes the ratio of between-group to within-group sum of squares (SSbetween/SSwithin) using the covariance matrix to find a linear combination separating classes.
  - When to use: continuous predictors, feature importance/selection (after normalizing predictors, discriminant weights rank importance); less common now vs. logistic regression/trees.
  - How: `discrim_linear()` (R, MASS engine) / `LinearDiscriminantAnalysis()` (sklearn); QDA relaxes the equal-covariance assumption across groups.
- **Logistic Regression**: models log-odds of a binary outcome as a linear function of predictors, via the logit (log-odds) link and logistic (sigmoid) response function; fit by maximum likelihood estimation (MLE), not least squares.
  - When to use: binary classification needing fast scoring and interpretable coefficients (odds ratios).
  - How: `glm(Y ~ X, family=binomial)` / `LogisticRegression()`; coefficient βj → odds ratio = exp(βj).
- **Evaluating Classification Models (confusion matrix, precision/recall/specificity, ROC/AUC, lift)**: beyond-accuracy metrics essential when classes are imbalanced.
  - When to use: always, but critical whenever the class of interest is rare.
  - How: build confusion matrix; precision = TP/(TP+FP); recall(sensitivity) = TP/(TP+FN); specificity = TN/(TN+FP); ROC plots recall vs. specificity across cutoffs; AUC = area under ROC; lift = ratio of a model's top-decile capture rate to random selection's.
- **Strategies for Imbalanced Data (undersample/oversample/weighting/SMOTE)**: rebalance training so the rare class isn't swamped.
  - When to use: rare-class problems (fraud, churn, conversion) where naive full-data training predicts almost everything as the majority class.
  - How: undersample majority class, oversample (bootstrap) minority class, apply case weights (`sample_weight`), or generate synthetic minority examples via SMOTE (interpolate between a record and its nearest neighbor).
- **Cost-based classification**: incorporate the actual dollar cost/benefit of each outcome type rather than optimizing a generic accuracy/AUC metric.
  - When to use: decisions where false positive and false negative costs are asymmetric and quantifiable (e.g., loan default risk vs. return).

## Key Concepts
- **Propensity score**: the model's estimated probability a record belongs to the class of interest; converted to a decision via a cutoff (default 0.5, but often tuned).
- **Odds / log-odds (logit)**: odds = p/(1−p); logit = log(odds), maps probability (0,1) to (−∞,+∞), enabling a linear model.
- **Confusion matrix**: cross-tab of predicted vs. actual class counts (TP, TN, FP, FN) underlying all classification metrics.
- **Rare class problem**: when the class of interest (fraud, churn, conversion) is a small minority, accuracy is misleading (a model predicting "always 0" can still score >99% accuracy but be useless).
- **ROC curve / AUC**: plots recall vs. specificity across all cutoffs; AUC of 1.0 = perfect classifier, 0.5 = no better than chance.
- **Lift (gains)**: how much better than random the model performs at capturing 1s within the top-ranked fraction of records (e.g., top decile).
- **SMOTE**: synthetic minority oversampling — creates new minority-class records as weighted averages of a real record and its nearest neighbor.
- **Discriminant weights / covariance matrix**: LDA's learned separating direction; related to Mahalanobis distance.

## Mental Models
- Think of every classifier as producing a continuous propensity score first, with the "classification" itself being a downstream choice of cutoff — never conflate the two steps, since the right cutoff depends on costs and class balance, not a fixed 0.5 default.
- Treat accuracy as a trap for imbalanced problems: a model that is 99.9% "accurate" by always predicting the majority class is informative only when framed against precision/recall/lift for the rare class you actually care about.
- Use the ROC/AUC lens to think about a classifier's entire operating range (every possible cutoff) rather than a single point estimate of performance — useful when you don't yet know the right cutoff.
- View logistic regression's "odds ratio = exp(coefficient)" as the direct business-readable translation of an abstract log-odds coefficient (e.g., "small business loans have ~3.4x the default odds of credit-card loans").

## Anti-patterns
- **Using 0.5 as an automatic cutoff for rare-class problems**: with a strongly imbalanced class, this systematically under-predicts the class of interest — tune the cutoff to the actual decision cost/benefit instead.
- **Reporting only accuracy on imbalanced data**: hides that the model may never correctly flag the class that matters.
- **Training on raw imbalanced data without any rebalancing or weighting**: e.g., the chapter's full-loan-data example predicts only 0.39% defaults (true rate ~19%) because non-defaulting loans "drown out" defaulters.
- **Confusing false positive rate with 1 − specificity in different contexts**: the chapter flags that "false positive rate" is inconsistently defined across tools/literature — always check which denominator is used.
- **Treating naive Bayes probability estimates as well-calibrated**: they're biased but still useful for ranking records by relative likelihood.
- **Using one-hot (P-column) encoding in logistic regression**: causes the same multicollinearity problem as in linear regression — use P−1 reference coding.

## Code Examples
```r
logistic_model <- logistic_reg(mode = "classification") %>%
  fit(outcome ~ payment_inc_ratio + purpose_ + home_ + emp_len_ + borrower_score,
      loan_data)
# borrower_score coefficient: -4.61264 -> odds ratio exp(-4.61264) ≈ 0.01
```
- **What it demonstrates**: fitting logistic regression and interpreting a coefficient as an odds ratio — best borrowers have ~100x lower default odds than worst.

```r
conf_mat(predictions, truth = outcome, estimate = .pred_class)
#           Truth
# Prediction paid off default
#   paid off   14620    8378
#   default     8051   14293
```
- **What it demonstrates**: building a confusion matrix as the foundation for precision/recall/specificity.

```python
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y, logit_reg.predict_proba(X)[:, 0], pos_label="default")
roc_auc_score([1 if yi == "default" else 0 for yi in y], logit_reg.predict_proba(X)[:, 0])
# AUC ≈ 0.69
```
- **What it demonstrates**: computing the ROC curve and AUC metric; 0.69 indicates a relatively weak classifier.

```python
default_wt = 1 / np.mean(full_train_set.outcome == "default")
wt = [default_wt if outcome == "default" else 1 for outcome in full_train_set.outcome]
full_model.fit(X, y, sample_weight=wt)
# predicted default rate jumps from 0.39% to ~58%
```
- **What it demonstrates**: case weighting as an alternative to under/oversampling for imbalanced data.

## Reference Tables
| Metric | Formula | Answers |
|---|---|---|
| Accuracy | (TP+TN)/Total | Overall correctness (misleading if imbalanced) |
| Precision | TP/(TP+FP) | Of predicted 1s, how many are actually 1s |
| Recall / Sensitivity | TP/(TP+FN) | Of actual 1s, how many were caught |
| Specificity | TN/(TN+FP) | Of actual 0s, how many were correctly identified |
| AUC | area under ROC curve | Overall discriminative ability across all cutoffs |
| Lift | model top-decile capture rate ÷ random rate | Efficiency of ranking rare-class records highly |

| Imbalance strategy | Mechanism |
|---|---|
| Undersample | Drop majority-class rows |
| Oversample | Bootstrap minority-class rows |
| Case weighting | Up-weight minority class in the loss function |
| SMOTE | Generate synthetic minority rows via nearest-neighbor interpolation |

## Worked Example
LendingClub loan data (`loan_data.csv.gz`, balanced 50/50 default vs. paid-off subset `loan3000.csv` for illustration): naive Bayes on categorical predictors (`purpose_`, `home_`, `emp_len_`) predicts "default" for a small-business/mortgage/>1yr-employment loan with posterior P(default) ≈ 0.65. LDA on two continuous predictors (`borrower_score`, `payment_inc_ratio`) draws a linear decision boundary visualized in Figure 5-2, splitting the predictor space by predicted default probability. Logistic regression on the same plus more predictors gives `borrower_score` coefficient −4.613 (odds ratio ≈ 0.01: best borrowers have ~100x lower default odds) and `purpose_small_business` coefficient 1.215 (odds ratio ≈ 3.4: small-business loans are far riskier than credit-card consolidation loans). Evaluated against the full unbalanced dataset (true default rate ≈ 18.9%), the model trained on balanced data over-predicts defaults (confusion matrix: 14,293 TP, 8,378 FN, 8,051 FP, 14,620 TN; precision ≈ 0.636, recall ≈ 0.645); AUC ≈ 0.69 indicates a modest classifier. Training instead on the full imbalanced set without correction predicts only 0.39% defaults vs. the true ~19% rate — demonstrating the rare-class problem directly — and applying inverse-probability case weights restores the predicted default rate to ~58%, overcorrecting but illustrating the mechanism.

## Key Takeaways
1. Always separate "estimate the propensity" from "choose the cutoff" — the right cutoff depends on class balance and the business cost of false positives vs. false negatives, not a default 0.5.
2. For imbalanced/rare-class problems, evaluate with precision/recall/specificity/AUC/lift, never accuracy alone.
3. Rebalance training data (undersample, oversample, weight, or SMOTE) when the rare class is swamped by the majority class during model fitting.
4. Logistic regression coefficients translate directly to odds ratios via exp(β) — a practical, business-interpretable output that drives its popularity over black-box alternatives.
5. Use lift/gains charts when resource-constrained decisions (e.g., which loans to audit) require ranking records rather than a global classification threshold.
6. Cost-based classification (using actual expected return/cost) is more decision-relevant than optimizing AUC or accuracy in isolation.

## Connects To
- **Ch 4 (Regression and Prediction)**: logistic regression directly extends linear regression's machinery (GLM framework, factor coding, diagnostics) to binary outcomes.
- **Ch 6 (Statistical Machine Learning)**: tree models, bagging, and boosting (mentioned here for SMOTE-adjacent ensemble intuition) are developed fully in the next chapter and compared against LDA/logistic regression's decision boundaries (Figure 5-9).
- **External concept**: the ROC curve's WWII radar-detection origin and the SMOTE algorithm (Chawla et al. 2002) are the standard external references for this chapter's evaluation and imbalance-handling techniques.
