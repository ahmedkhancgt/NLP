# Chapter 9: Modeling Data

## Core Idea
Modeling (OSEMN step 4) abstracts data points (features, optionally labels) into numeric models that predict or structure new data; the command line supports this via specialized tools for three canonical task types — unsupervised dimensionality reduction (tapkee), supervised regression (Vowpal Wabbit), and supervised classification (SKLL/SciKit-Learn) — each reachable without writing Python/R model code directly.

## Frameworks Introduced
- **Data-point/feature/label vocabulary**: the shared ML vocabulary used throughout.
  - When to use: translating a CSV into modeling terms.
  - How: each non-header row = a data point; each column (except an optional outcome column) = a feature; an optional outcome/judgment column = the label. Unsupervised algorithms (dimensionality reduction) use only features; supervised algorithms (regression, classification) use features + labels.
- **Three Modeling Task Types**: dimensionality reduction (unsupervised), regression (supervised, continuous label), classification (supervised, categorical label).
  - When to use: choosing a modeling approach based on your label's nature (none, continuous, or categorical).
  - How: dimensionality reduction → `tapkee` (PCA for linear structure, t-SNE for nonlinear/separation); regression → `vw` (Vowpal Wabbit) when the label is numeric; classification → `skll`/`run_experiment` when the label is categorical.
- **Train/test split discipline**: hold out data to evaluate generalization.
  - When to use: before training any supervised model.
  - How: shuffle (`shuf`) then split (`split -d -n r/N`) into roughly 80% train / 20% test (or balanced classes via `csvstack` + per-class sampling for classification); never evaluate on data used for training.

## Key Concepts
- **csvstack -g <labels> -n <col>**: stacks multiple CSVs into one, adding a new column (`-n`) populated with per-file group labels (`-g`) — used to combine red/white wine files with a `type` column.
- **Standardization/scaling**: transforming features to comparable scales (via R's `scale()` through `rush run --tidyverse`) before dimensionality reduction/modeling so no feature dominates due to its raw magnitude.
- **tapkee**: C++ CLI for dimensionality reduction; `--method pca` (linear) or `--method t-sne` (nonlinear); reads unlabeled, headerless numeric data from stdin and outputs lower-dimensional coordinates.
- **PCA vs. t-SNE**: PCA finds linear combinations of features maximizing variance (fast, interpretable, may not separate classes well); t-SNE models nonlinear neighbor structure (often better visual separation of classes, more computationally expensive, less interpretable axes).
- **Vowpal Wabbit (vw) data format**: `<label> | <feature>:<value> <feature>:<value> ...` per line — more verbose than CSV but supports weights, tags, namespaces, and sparse representations; `csv2vw --label <col>` converts CSV to this format.
- **vw training/testing flags**: `--data` (input), `--final_regressor`/`-f` (save trained model), `--passes` (multiple epochs, requires `--cache_file`), `--nn` (neural network hidden units), `--quadratic` (interaction features), `--l2` (regularization), `--bit_precision` (hash space size) for training; `--initial_regressor`/`-i` (load model), `--testonly`/`-t`, `--predictions`/`-p`, `--quiet` for testing.
- **MAE (Mean Absolute Error)**: `mean(|predicted - observed|)` — a regression accuracy metric computed here via `awk` after pasting predictions against true labels.
- **SKLL (SciKit-Learn Laboratory) / `run_experiment`**: config-file-driven (not flag-driven) classification tool wrapping scikit-learn; config sections define `[General]` (task), `[Input]` (train/test directories, feature sets, label/id columns, learners list), `[Tuning]` (grid search, objectives), `[Output]` (where logs/results/predictions/models go).
- **Balanced dataset construction**: when classes are imbalanced (1599 red vs. 4898 white wines), undersample the majority class to match the minority count before training a classifier, to avoid biased accuracy.
- **Confusion matrix**: table of true label (columns) vs. predicted label (rows); diagonal = correct predictions; off-diagonal = misclassifications; paired with Precision/Recall/F-measure per class — essential when per-class misclassification costs differ.
- **Config-file vs. flag-driven tool design tradeoff**: vw's many CLI flags enable ad hoc usage and are easy to script/Makefile-ify; skll's config file is easier to version/reproduce but less convenient for quick one-offs.

## Mental Models
- Treat dimensionality reduction as a diagnostic/exploratory step before committing to supervised modeling: if a 2D projection (PCA/t-SNE) already separates classes visually, that's evidence the features carry real signal for a later classifier.
- Visualizing a model's predictions vs. observed values (e.g., `rush plot --geom jitter`) is as important as a single aggregate metric (MAE) — it reveals patterns a single number hides.
- A confusion matrix is more informative than a single accuracy number whenever there are more than two classes or asymmetric misclassification costs.

## Anti-patterns
- **Training and evaluating on the same data**: gives an optimistic, meaningless accuracy estimate — always hold out a test set.
- **Training a classifier on heavily imbalanced classes without correction**: a naive high "accuracy" can just reflect predicting the majority class every time — balance the dataset (undersample majority) first.
- **Feeding unscaled features (wildly different magnitudes) into PCA/t-SNE or other distance-based methods**: features with larger raw ranges dominate the mapping — standardize/scale first.
- **Reporting only a single aggregate metric (accuracy or MAE) without inspecting the confusion matrix or prediction-vs-observed plot**: hides which specific cases/classes are being misclassified.
- **Skipping a proper ML background before applying these tools**: the author explicitly warns this chapter is not an ML tutorial — understand an algorithm before applying it to real data.

## Code Examples
```bash
$ csvstack -g red,white -n type wine-{red,white}-clean.csv | xsv select 2-,1 > wine.csv
```
- **What it demonstrates**: combining two CSVs into one labeled dataset (adding a `type` column) and reordering so the label ends up last, as several algorithms expect.

```bash
$ xsv select '!type' wine-scaled.csv | header -d | tapkee --method pca | header -a pc1,pc2 | \
    paste -d, - <(xsv select type wine-scaled.csv) > wine-pca.csv
```
- **What it demonstrates**: the full dimensionality-reduction pipeline — strip the label, strip the header, run PCA, relabel the output columns, and reattach the original label column for visualization/coloring.

```bash
$ csv2vw wine-white-clean.csv --label quality | shuf | split -d -n r/5 - wine-part-
$ mv wine-part-00 wine-test.vw && cat wine-part-* > wine-train.vw
$ vw --data wine-train.vw --final_regressor wine.model --passes 10 --cache_file wine.cache --nn 3 --quadratic :: --l2 0.000005 --bit_precision 25
$ vw --data wine-test.vw --initial_regressor wine.model --testonly --predictions predictions --quiet
$ paste -d, predictions <(cut -d '|' -f 1 wine-test.vw) | awk -F, '{E+=sqrt(($1-$2)^2)} END {print "MAE: " E/NR}'
```
- **What it demonstrates**: the complete regression workflow — convert CSV to vw format, shuffle/split 80/20, train with tuned hyperparameters, test, and compute MAE by pasting predictions against observed values.

```ini
[General]
experiment_name = wine
task = evaluate

[Input]
train_directory = train
test_directory = test
featuresets = [["features"]]
label_col = type
id_col = id
learners = ["KNeighborsClassifier", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"]
```
- **What it demonstrates**: SKLL's declarative config-file approach to specifying multiple classifiers to train/evaluate in a single experiment run (`skll -l classify.cfg`).

## Reference Tables
| Task type | Supervision | Tool | Label needed? |
|---|---|---|---|
| Dimensionality reduction | Unsupervised | `tapkee` (`--method pca`/`t-sne`) | No |
| Regression | Supervised | `vw` | Yes (continuous) |
| Classification | Supervised | `skll`/`run_experiment` | Yes (categorical) |

| Metric | Meaning | Tool used here |
|---|---|---|
| MAE | Average absolute prediction error (regression) | `awk` after `paste` |
| Accuracy | % correctly classified | SKLL summary TSV + `csvsql` |
| Confusion matrix | True vs. predicted label breakdown | SKLL results JSON + `jq` |
| Precision/Recall/F-measure | Per-class correctness quality | SKLL results JSON |

## Worked Example
Full classification pipeline for red-vs-white wine type prediction:
```bash
$ NUM_RED="$(< wine-red-clean.csv wc -l)"
$ csvstack -n type -g red,white wine-red-clean.csv \
    <(< wine-white-clean.csv body shuf | head -n $NUM_RED) | body shuf | \
    nl -s, -w1 -v0 | sed '1s/0,/id,/' > wine-balanced.csv     # balance classes + add id
$ < wine-balanced.csv header -d | shuf | split -d -n r/5 - wine-part-   # 80/20 split
$ cat wine-part-00 | header -a "$HEADER" > test/features.csv
$ cat wine-part-{01,02,03,04} | header -a "$HEADER" > train/features.csv
$ skll -l classify.cfg 2>/dev/null                              # train+evaluate 4 classifiers
$ < output/wine_summary.tsv csvsql --query "SELECT learner_name, accuracy FROM stdin ORDER BY accuracy DESC" | csvlook -I
$ jq -r '.[] | "\(.learner_name):\n\(.result_table)\n"' output/*.json   # confusion matrices
```
This produces a ranked comparison of four classifiers (RandomForest, KNeighbors, LogisticRegression, DecisionTree) by accuracy, plus per-classifier confusion matrices with precision/recall/F-measure, entirely driven by a declarative config file and CSV-aware command-line tools.

## Key Takeaways
1. Translate your dataset into features/labels and pick the modeling tool by label type: none → `tapkee`; continuous → `vw`; categorical → `skll`.
2. Always scale/standardize features before distance-based algorithms (PCA, t-SNE) when features have different units/magnitudes.
3. Always hold out a test set (commonly 80/20, shuffled first) before evaluating a supervised model; balance classes for classification when they're skewed.
4. Use dimensionality reduction (especially t-SNE) as an exploratory check for class separability before committing to full supervised modeling.
5. Evaluate regression with MAE (or similar) and visualize predicted-vs-observed; evaluate classification with accuracy plus a full confusion matrix (precision/recall/F-measure), especially with >2 classes or asymmetric costs.
6. Recognize the flag-driven (vw) vs. config-file-driven (skll) tool design tradeoff: flags suit ad hoc scripting/Makefile integration, config files suit reproducible, shareable experiment specs.
7. This chapter is deliberately not a machine-learning tutorial — build real ML understanding before applying these tools to consequential decisions.

## Connects To
- **Ch 7 (Exploring Data)**: reuses `rush plot`/`rush run` for scaling, visualization of model structure (density plots, scatter plots of PCA/t-SNE output, prediction-vs-observed jitter plots).
- **Ch 5 (Scrubbing Data)**: reuses `csvstack`, `header`, `xsv`, `csvsql` for dataset combination/splitting/aggregation.
- **Ch 8 (Parallel Pipelines)**: the chapter opens by using `parallel` to fetch the two wine CSVs concurrently.
- **Ch 10 (Polyglot Data Science)**: natural follow-on — invoking R (`rush`) from the shell here foreshadows deeper R/Python/Spark integration next chapter.
- **scikit-learn, Vowpal Wabbit, Tapkee**: external ML libraries/tools this chapter wraps at the command-line level.
