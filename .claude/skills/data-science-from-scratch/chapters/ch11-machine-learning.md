# Chapter 11: Machine Learning

## Core Idea
Machine learning means creating and using models learned from data; this chapter lays the conceptual groundwork — overfitting/underfitting, train/test/validation splits, correctness metrics beyond accuracy (precision/recall/F1), the bias-variance tradeoff, and feature extraction/selection — before the book dives into specific model families.

## Frameworks Introduced
- **Train/test/validation split**: divide data into a training set (fit the model), a test set (judge it), and, when comparing multiple models, a separate validation set (choose among them) — because using the test set to pick a model turns it into a second training set.
  - When to use: every supervised model, to detect overfitting; use three-way splits whenever model selection (not just training) happens.
  - How: `split_data(data, prob)` shuffles and cuts; `train_test_split(xs, ys, test_pct)` keeps paired `(x,y)` correspondence intact.
- **Confusion-matrix metrics (precision, recall, F1)**: for binary classification, compute accuracy, precision (`tp/(tp+fp)`), recall (`tp/(tp+fn)`), and F1 (harmonic mean of precision and recall) instead of relying on raw accuracy.
  - When to use: any binary classifier, especially with class imbalance, where accuracy is misleadingly high for a useless model.
- **Bias-variance tradeoff diagnosis**: classify a struggling model as high-bias (underfits, performs poorly even on training data — fix by adding features/complexity) or high-variance (overfits, very sensitive to which training set is used — fix by removing features, adding data, or regularizing).
  - When to use: deciding *what to try next* when a model isn't performing well.
- **Feature extraction/selection**: convert raw data (e.g., email text) into binary, numeric, or categorical features appropriate for a given model family; or reduce an already-large feature set via dimensionality reduction/regularization to combat overfitting.
  - When to use: always — model family choice constrains feature type (Naive Bayes wants binary features, regression wants numeric, decision trees handle either).

## Key Concepts
- **Model**: a specification of a mathematical/probabilistic relationship between variables (a business spreadsheet, a recipe, a poker win-probability calculator are all "models").
- **Supervised vs. unsupervised learning**: supervised has labeled correct-answer data to learn from; unsupervised does not. (Semisupervised, online, and reinforcement learning are named but not covered.)
- **Overfitting**: a model that performs well on training data but generalizes poorly — often from learning noise or memorizing specific inputs rather than true predictive factors.
- **Underfitting**: a model that performs poorly even on training data — typically means "find a better model."
- **Confusion matrix categories**: true positive, false positive (type 1 error), false negative (type 2 error), true negative.
- **Precision**: fraction of positive *predictions* that were correct (`tp/(tp+fp)`).
- **Recall**: fraction of actual positives that were correctly identified (`tp/(tp+fn)`).
- **F1 score**: harmonic mean of precision and recall, always between them.
- **Bias**: error from a model too simple to capture the true relationship (high bias = underfitting); low variance across retrainings.
- **Variance**: sensitivity of a model's fit to which particular training set was used (high variance = overfitting); low bias since it fits each training set well.
- **Feature**: any input provided to a model — binary (yes/no), numeric, or categorical.

## Mental Models
- **The "Luke -> leukemia" joke test**: a deliberately absurd 98%+ "accurate" classifier (predict leukemia iff baby is named Luke) that has essentially zero precision (1.4%) and recall (0.5%) — a vivid demonstration that accuracy alone is meaningless, especially under class imbalance.
- **Precision/recall as a dial, not a fixed property**: raising a decision threshold (e.g., "predict leukemia if >= k risk factors") trades recall for precision — there's rarely a model that maximizes both simultaneously; the right operating point depends on the cost of false positives vs. false negatives for your specific problem.
- **More data fixes variance, not bias**: holding model complexity fixed, more training data shrinks overfitting (variance) but does nothing for a model that's fundamentally too simple (bias) — know which problem you have before reaching for more data.
- **Feature type constrains model choice**: binary features suit Naive Bayes (Ch13), numeric features suit regression (Ch14/Ch16), decision trees (Ch17) tolerate either — picking a model family is partly picking what feature representation you're committing to.

## Anti-patterns
- **Using raw accuracy to judge a classifier under class imbalance**: the leukemia joke test's 98.1% accuracy masks a completely useless model (precision 1.4%, recall 0.5%).
- **Using the test set to choose among multiple models**: this makes the test set function as a second training set — "the model that performed best on the test set is going to perform well on the test set" is circular; use a separate validation set for model selection.
- **Letting common non-generalizing patterns leak between train and test splits**: e.g., per-user weekly activity data where the same users appear in both splits risks the model learning to identify users rather than real relationships.
- **Throwing more data at a high-bias (underfit) model**: won't help — more data only reduces variance, not bias; a too-simple model needs more/better features instead.
- **Treating feature engineering as an afterthought**: raw data (e.g., email text) must be deliberately converted into binary/numeric/categorical features before most models can use it — this choice directly shapes which model families are viable.

## Code Examples
```python
def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]

def train_test_split(xs, ys, test_pct):
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)
    return ([xs[i] for i in train_idxs], [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs], [ys[i] for i in test_idxs])
```
- **What it demonstrates**: shuffling and splitting paired `(x, y)` data into train/test sets while preserving correspondence.

```python
def precision(tp, fp, fn, tn): return tp / (tp + fp)
def recall(tp, fp, fn, tn): return tp / (tp + fn)
def f1_score(tp, fp, fn, tn):
    p, r = precision(tp, fp, fn, tn), recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)
```
- **What it demonstrates**: the standard binary-classification metrics built from confusion-matrix counts.

## Reference Tables
| Confusion matrix | Predicted spam | Predicted not spam |
|---|---|---|
| Actually spam | True Positive | False Negative (Type 2) |
| Actually not spam | False Positive (Type 1) | True Negative |

| Model problem | Symptom | Fix |
|---|---|---|
| High bias (underfitting) | poor performance even on training data | add features/complexity |
| High variance (overfitting) | great on training data, poor on new data | remove features, regularize, get more data |

| Metric | Formula | Answers |
|---|---|---|
| Accuracy | `(tp+tn)/(tp+fp+fn+tn)` | overall fraction correct (misleading under imbalance) |
| Precision | `tp/(tp+fp)` | of predicted positives, how many were right? |
| Recall | `tp/(tp+fn)` | of actual positives, how many did we catch? |
| F1 | `2pr/(p+r)` | harmonic mean balancing precision and recall |

## Worked Example
The "Luke predicts leukemia" test applied to 1,000,000 people (5/1000 named Luke, 14/1000 lifetime leukemia prevalence, assumed independent) produces confusion matrix: TP=70, FP=4930, FN=13930, TN=981070. Accuracy = 0.98114 (looks great); but precision = 70/5000 = 0.014 and recall = 70/14000 = 0.005 (both terrible) — correctly exposing the test as useless despite its high accuracy. Separately, polynomial curve-fitting examples (degree 0, 1, 9) on the same dataset show degree 0 underfitting (high bias, low variance), degree 9 overfitting (low bias, high variance), and degree 1 striking the right balance; retraining the degree-9 model on 10 vs. 100 vs. 1,000 data points shows variance shrinking as data grows while bias stays unaffected.

## Key Takeaways
1. Always validate on held-out data; use a three-way train/validation/test split whenever comparing multiple models, not just one.
2. Never judge a binary classifier by accuracy alone — check precision, recall, and F1, especially under class imbalance.
3. Diagnose underperforming models via the bias-variance lens: high bias needs more features/complexity, high variance needs fewer features, regularization, or more data.
4. More data only helps with variance (overfitting), never with bias (underfitting) — know which problem you have.
5. Feature type (binary/numeric/categorical) constrains viable model families — choose features and model family together.

## Connects To
- **Ch12 (k-NN)**, **Ch13 (Naive Bayes)**, **Ch14-16 (Regression)**, **Ch17 (Decision Trees)**: all are specific model families this chapter's concepts (overfitting, bias-variance, feature types) apply to.
- **Ch10 (Working with Data)**: PCA/dimensionality reduction referenced here as a way to combat high-variance overfitting from too many features.
- **Regularization (referenced, covered implicitly in Ch15/16)**: another high-variance fix mentioned here and expanded later.
- **The Elements of Statistical Learning**: external canonical textbook the chapter recommends for deeper theory.
