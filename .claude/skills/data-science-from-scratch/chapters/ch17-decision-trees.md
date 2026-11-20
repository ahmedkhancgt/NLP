# Chapter 17: Decision Trees

## Core Idea
Decision trees predict by walking a sequence of attribute-based questions to a leaf; the chapter builds the ID3 algorithm from scratch using entropy to greedily pick the lowest-uncertainty split at each step, then extends to random forests (bagging + random attribute subsets) to combat the overfitting tendency of single trees.

## Frameworks Introduced
- **Entropy as uncertainty measure**: `entropy(probs) = sum(-p * log2(p) for p in probs if p > 0)`; low when one class dominates, high when classes are evenly spread.
  - When to use: scoring how "informative" a potential split is, for any classification-tree algorithm.
- **Partition entropy**: weighted average of each subset's entropy, `H = sum(q_i * H(S_i))`, where `q_i` is the subset's fraction of the data.
  - When to use: comparing candidate attributes to split on — pick whichever attribute's partition has the lowest weighted entropy.
- **ID3 algorithm (greedy decision tree construction)**: recursively (1) if all labels match, leaf with that label; (2) if no attributes left, leaf with majority label; (3) else pick the attribute with lowest partition entropy, split, and recurse on each subset with that attribute removed.
  - When to use: building a from-scratch classification tree for binary (or multi-class) categorical/mixed-type targets.
  - Known limitation: greedy — picks the locally best split at each step, which can miss a globally better tree with a worse-looking first move.
- **Random forests (bagging + feature subsampling)**: build many trees, each trained on a `bootstrap_sample` of the data and, at each split, choosing the best attribute from only a random subset of remaining attributes; combine via majority vote (classification) or averaging (regression).
  - When to use: whenever a single decision tree overfits — an instance of the broader "ensemble learning" pattern of combining many weak/high-variance learners into a stronger aggregate.

## Key Concepts
- **Classification tree vs. regression tree**: categorical output vs. numeric output; the chapter focuses on binary classification trees.
- **Decision node / leaf node**: internal nodes ask a question and branch; leaf nodes return a prediction.
- **Leaf / Split representation**: `Leaf(value)` for a terminal prediction; `Split(attribute, subtrees: dict, default_value=None)` for an internal decision node, with `DecisionTree = Union[Leaf, Split]`.
- **Greedy algorithm**: makes the locally optimal choice at each step without backtracking — simple and fast, not guaranteed globally optimal.
- **Overfitting via high-cardinality attributes**: splitting on an attribute with many distinct values (e.g., Social Security number) can drive partition entropy to zero while producing a model that generalizes terribly — avoid or bucket such attributes.
- **Bagging (bootstrap aggregating)**: training each ensemble member on an independent bootstrap resample of the data, so each tree differs and out-of-sample data can double as a validation set.
- **Ensemble learning**: combining multiple weak learners (e.g., many high-variance, low-bias trees) to produce a stronger overall model.

## Mental Models
- **Entropy as "how surprised would I be"**: a near-pure partition (one class dominating) has near-zero entropy because there's little uncertainty left; a 50/50 split has maximum entropy (1 bit) because either outcome is equally likely.
- **Greedy-but-good-enough trade-off**: finding the provably optimal decision tree is computationally hard, so ID3 accepts a locally-greedy, "good enough" tree instead — a recurring theme (also seen in gradient descent's local-minimum risk).
- **Watch for attributes that "cheat" via memorization**: an attribute with as many distinct values as there are data points (SSN) will always produce zero-entropy single-item partitions — a red flag for overfitting disguised as a great split.
- **Variance reduction through diversity, not individual accuracy**: random forests work because bagging + feature-subsampling make each tree different from the others — the point isn't to make any single tree better, but to make the ensemble's errors less correlated so they average out.

## Anti-patterns
- **Splitting on high-cardinality identifier-like attributes** (e.g., SSN, email, row ID): produces deceptively low partition entropy from memorization, not real predictive signal — avoid or bucket such attributes.
- **Treating ID3's greedy choice as globally optimal**: a different first split might lead to a better overall tree; ID3 won't find it by design.
- **Building a single decision tree on all data and trusting it to generalize**: decision trees "have a tendency to overfit" almost by construction (they can perfectly fit training data) — use random forests or proper train/validation/test evaluation instead of a single deterministic tree for real use.
- **Ignoring unknown/missing attribute values at prediction time**: without a `default_value` fallback (populated with the most common label), the classifier has no sensible behavior for inputs it hasn't seen before (e.g., an "Intern" level not in training data).

## Code Examples
```python
def entropy(class_probabilities: List[float]) -> float:
    return sum(-p * math.log(p, 2) for p in class_probabilities if p > 0)

def partition_entropy(subsets: List[List[Any]]) -> float:
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)
```
- **What it demonstrates**: entropy and partition entropy, the core scoring functions ID3 uses to pick splits.

```python
class Leaf(NamedTuple):
    value: Any

class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None

DecisionTree = Union[Leaf, Split]

def classify(tree: DecisionTree, input: Any) -> Any:
    if isinstance(tree, Leaf):
        return tree.value
    subtree_key = getattr(input, tree.attribute)
    if subtree_key not in tree.subtrees:
        return tree.default_value
    return classify(tree.subtrees[subtree_key], input)
```
- **What it demonstrates**: the lightweight recursive tree representation and classification walk, with graceful handling of unseen attribute values.

```python
def build_tree_id3(inputs, split_attributes, target_attribute) -> DecisionTree:
    label_counts = Counter(getattr(i, target_attribute) for i in inputs)
    most_common_label = label_counts.most_common(1)[0][0]
    if len(label_counts) == 1 or not split_attributes:
        return Leaf(most_common_label)
    best_attribute = min(split_attributes, key=lambda a: partition_entropy_by(inputs, a, target_attribute))
    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]
    subtrees = {v: build_tree_id3(subset, new_attributes, target_attribute) for v, subset in partitions.items()}
    return Split(best_attribute, subtrees, default_value=most_common_label)
```
- **What it demonstrates**: the complete recursive ID3 tree-building algorithm.

## Reference Tables
| Attribute | Partition entropy (interviewee data) |
|---|---|
| level | ~0.69 (lowest -> chosen first) |
| lang | ~0.86 |
| tweets | ~0.78 |
| phd | ~0.89 |

| Entropy value | Meaning |
|---|---|
| 0 | all data in one class, no uncertainty |
| 1 (binary, equal split) | maximum uncertainty |

## Worked Example
14 job candidates (`Candidate(level, lang, tweets, phd, did_well)`) are used to build a hiring decision tree. Splitting on `level` gives the lowest partition entropy (~0.69) of the four attributes, so it's chosen first. All `Mid` candidates are `True` -> immediate leaf. Among `Senior` candidates, splitting on `tweets` gives zero entropy (tweets=True -> always did_well; tweets=False -> always not) -> next split. Among `Junior` candidates, splitting on `phd` gives zero entropy similarly. The resulting tree (`hiring_tree = Split('level', {...})`) perfectly classifies the training data and also handles an unseen `level='Intern'` value by falling back to `default_value` (the most common label).

## Key Takeaways
1. Entropy quantifies uncertainty; ID3 greedily picks the attribute whose partition minimizes weighted entropy at each step.
2. Decision trees are transparent and handle mixed numeric/categorical data naturally, but are prone to overfitting and can be fooled by high-cardinality attributes.
3. ID3's greediness means it may miss a globally better tree — accept this as the practical trade-off for tractability.
4. Random forests (bagging + random attribute subsampling at each split) reduce overfitting by combining many diverse trees — an instance of the general ensemble-learning principle.
5. Always provide a default/fallback prediction for attribute values not seen during training.

## Connects To
- **Ch11 (Machine Learning)**: directly illustrates the bias-variance tradeoff — single trees are low-bias/high-variance; random forests trade some bias for much lower variance.
- **Ch15's bootstrap**: `bootstrap_sample` is reused verbatim to build each random forest tree's training set (bagging).
- **Ch13 (Naive Bayes)**: another classification approach with a very different bias profile, useful to contrast against decision trees' feature-type flexibility.
- **scikit-learn / XGBoost**: external libraries with production decision tree, random forest, and gradient-boosted tree implementations.
