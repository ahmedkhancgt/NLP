# Chapter 14: Supervised 3D Machine Learning Fundamentals

## Core Idea
Supervised learning assigns real semantic meaning (ground/vegetation/building) to 3D data by training a classifier on labeled features, but a model that scores 98% on same-distribution test data can collapse to 54% on a geographically different validation set — generalization, not training accuracy, is the true test, and better feature selection (not a fancier model) is usually the fix.

## Frameworks Introduced
- **Training vs. Inference (2-Phase Supervised Learning)**: Phase 1 (training) fits a model's internal logic to labeled input-output pairs; Phase 2 (inference) applies that fixed logic to predict labels on unseen data.
  - When to use: as the mental split for any supervised 3D ML project — know which phase you're in and what data is appropriate for each.
  - How: prepare features+labels → fit model on training split → evaluate on held-out split → only then deploy to genuinely novel data.
- **Three-Way Data Split (Training / Validation / Test)**: training data fits the model; validation data gives an impartial(ish) check used to tune hyperparameters/features (so it's slightly biased); test data is the truly uncorrelated, held-back sample for final unbiased assessment.
  - When to use: whenever overfitting risk is non-trivial — which is always in 3D ML given limited labeled data.
  - How: never let feature/hyperparameter tuning touch the final test set; use a geographically/distributionally different validation set (not just a random split of the same scene) to truly test generalization.
- **3D Machine Learning Development Cycle Checklist**: data acquisition/preprocessing → feature engineering (non-DL only) → model selection → evaluation metrics → deployment environment.
  - When to use: as a comprehensive pre-flight checklist before starting any 3D supervised learning project.
  - How: walk each category's sub-bullets (cleaning, augmentation, normalization; handcrafted vs. learned features; algorithm trade-offs; domain-specific metrics like IoU; hardware/scalability) before training.

## Key Concepts
- **Classification vs. regression**: classification predicts a discrete categorical label (car vs. tree); regression predicts a continuous value (best-fit curve parameters) — pick based on whether the target is categorical or numeric.
- **3D semantic segmentation**: classifying every data unit (point, voxel, triangle) in a scene, giving holistic scene comprehension — contrast with instance segmentation/object detection, which focus on individual object boundaries.
- **Confusion-matrix quantities**: True Positive (TP), False Negative (FN), True Negative (TN), False Positive (FP) — the atoms every classification metric is built from.
- **Precision / Recall / F1-score / Overall accuracy**: precision = avoid false positives; recall = find all true positives; F1 = harmonic mean of the two; overall accuracy = fraction correct across all classes — F1 is preferred under class imbalance (common in indoor/outdoor 3D scenes).
- **Feature scaling methods**: min-max scaling (linear map to [0,1], preserves distribution shape, good default for spatial data); z-score/standardization (centers at 0, unit variance, good for normally-distributed/sensor-noise-affected data); robust scaling (median + IQR, resistant to outliers — good for noisy terrestrial/mobile scans).
- **Overfitting vs. generalization**: a model can achieve near-perfect metrics on data from the same distribution as training (98% in the book's example) yet collapse (54%) on a geographically distinct validation set — the gap is the generalization failure signature.
- **Model drift monitoring (production)**: data drift (distribution changes over time), model degradation (metric decay), retraining (scheduled refresh), explainability (understanding predictions) — the four components of a production feedback loop.

## Mental Models
- Treat the training/validation/test split as a trust hierarchy: training data the model sees fully; validation data it's indirectly tuned against (slightly compromised); test data it never touches until the very end (the only truly honest signal).
- When a model's metrics look great on held-out data from the *same* scan/scene but you need real-world deployment confidence, assume that split is optimistic — validate against a genuinely different geography/sensor/time period before trusting the number.
- Treat "add more/better features" as the first lever to pull when generalization fails, before reaching for a more complex model — the book's case shows feature re-selection recovering accuracy from 54% to 97%, far more effective than switching algorithms.
- Think of class imbalance as a silent accuracy inflator — overall accuracy can look good purely because one dominant class (e.g., ground) is easy, while rarer classes (buildings) are poorly served; always check per-class precision/recall/F1, not just the aggregate.

## Anti-patterns
- **Judging a model solely by accuracy on a random split of the same dataset it was trained on**: this doesn't test generalization — a geographically/distributionally different validation set is required to catch overfitting (book's drop from 98% to 54% illustrates this starkly).
- **Using only X, Y, Z, R, G, B as features without geometric descriptors**: raw spatial+color features generalize poorly across different urban contexts; adding covariance-derived features (planarity, verticality, omnivariance, normal change rate, number of returns) recovered validation accuracy from 54% to 97% in the book's case.
- **Skipping feature normalization before training spatially-heavy models**: unnormalized X/Y coordinates tie the model to the training data's absolute coordinate range, causing catastrophic failure on data with a different spatial extent (even if it's the same scene, just translated).
- **Treating overall accuracy as sufficient under class imbalance**: a dominant class skews the headline number — always inspect per-class precision/recall/F1 (the book highlights the MLP's poor "buildings" class performance as a clear overfitting signal hidden by a misleadingly okay-looking overall number).
- **Blending test data into training to chase better validation numbers without acknowledging the trade-off**: injecting some validation-distribution data into training boosts validation metrics but by definition erodes the test set's statistical independence — the book is explicit that this trade-off (minimal feature noise for a ~5% validation gain while losing only 1% on the original test set) must be a deliberate, disclosed choice.

## Code Examples
```python
labels = pcd['Classification']
features = pcd[['X', 'Y', 'Z', 'R', 'G', 'B']]
features_scaled = MinMaxScaler().fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.4)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
print(classification_report(y_test, rf_predictions, target_names=['ground', 'vegetation', 'buildings']))
```
- **What it demonstrates**: the complete baseline supervised pipeline — feature selection, scaling, train/test split, random forest fit, prediction, classification report. (Baseline result: 98% overall accuracy on same-distribution test data.)

```python
# Generalization check on a geographically different validation city
val_features = val_pcd[['X', 'Y', 'Z', 'R', 'G', 'B']]
val_features_scaled = MinMaxScaler().fit_transform(val_features)
val_predictions = rf_classifier.predict(val_features_scaled)
print(classification_report(val_labels, val_predictions, target_names=['ground', 'vegetation', 'buildings']))
# Result: accuracy collapses from 98% to 54%
```
- **What it demonstrates**: the overfitting/generalization-failure exposure — a model that looks excellent on in-distribution test data can fail badly on a different city's data.

```python
# Fix: richer, geometry-aware feature set
features = pcd[['Z','R','G','B','omnivariance_2','normal_cr_2','NumberOfReturns','planarity_2','omnivariance_1','verticality_1']]
val_features = val_pcd[['Z','R','G','B','omnivariance_2','normal_cr_2','NumberOfReturns','planarity_2','omnivariance_1','verticality_1']]
# Retrain and re-evaluate -> validation accuracy jumps to 97%
```
- **What it demonstrates**: feature engineering (covariance-derived geometric descriptors from Ch. 6/7) is far more effective at fixing generalization than algorithm selection alone.

```python
import pickle
pickle.dump(rf_classifier, open(result_folder + "urban_classifier.poux", 'wb'))
# Later:
loaded_model = pickle.load(open(result_folder + "urban_classifier.poux", 'rb'))
predictions = loaded_model.predict(data_to_predict)
```
- **What it demonstrates**: persisting a trained model with pickle for later reuse/deployment without retraining.

## Reference Tables
*Model comparison on the same aerial LiDAR dataset (ground/vegetation/buildings, X,Y,Z,R,G,B features, normalized, 60/40 split)*

| Model | Overall accuracy | Notes |
|---|---|---|
| Random Forest | 98% | Ground recall=1.00, precision=0.99; errors spread fairly evenly |
| k-NN | 91% | Homogeneous precision/recall across classes (less overfitting risk, but lower peak accuracy) |
| MLP (neural net) | 64% | Poor, uneven inter-class performance; building class especially weak — classic overfitting signature |

*Generalization improvement sequence (random forest, validation city)*

| Stage | Features | Validation accuracy |
|---|---|---|
| 1. Baseline | X,Y,Z,R,G,B | 54% |
| 2. Richer features | Z,R,G,B + omnivariance, normal_cr, NumberOfReturns, planarity, verticality | 97% |
| 3. + partial validation-data injection into training | Same richer set + 10% validation data | +~5% further, -1% on original test |

*Confusion matrix terms*: TP/FN/TN/FP → Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean, Accuracy = (TP+TN)/all.

## Worked Example
The chapter's central worked example is 3D point cloud semantic segmentation of aerial LiDAR data from Louhans, France (IGN LiDAR HD, classes: ground/vegetation/buildings): (1) load `3DML_urban_point_cloud.xyz` with pandas, drop NaN rows (<1% loss); (2) select initial features (X,Y,Z,R,G,B), scale with MinMaxScaler, split 60/40 train/test; (3) train and compare three classifiers — RandomForestClassifier (98% accuracy), KNeighborsClassifier (91%), MLPClassifier (64%, clear overfitting on the "buildings" class) — selecting random forest as the winner; (4) critically, test generalization against `3DML_validation.xyz`, a LiDAR scan of a different city (Manosque) with different topology/urban context — accuracy collapses to 54%, exposing the earlier "98%" as an illusion of same-distribution testing; (5) diagnose the cause as insufficiently discriminative features (not insufficient model complexity) and re-select a richer geometry-aware feature set (Z, R, G, B, omnivariance at two scales, normal change rate, number of returns, planarity, verticality) computed via the Ch. 6/7 PCA-based feature extraction pipeline — retraining with this set lifts validation accuracy to 97%; (6) as a further refinement, inject 10% of the validation city's data into training (keeping 90% as a true held-out test), trading a small amount of statistical purity for a further ~5% validation accuracy gain; (7) export the final labeled point cloud (`.xyz`) and persist the trained random forest model with `pickle` for later reuse without retraining.

## Key Takeaways
1. Training/test accuracy from the same distribution is not evidence of real-world generalization — always validate against a genuinely different dataset (different city, sensor, or time period).
2. When generalization fails, improve the feature set (geometry-aware descriptors like planarity/verticality/omnivariance/normal change rate) before switching to a more complex model.
3. Random forests are a strong, interpretable baseline for 3D semantic segmentation and outperformed both k-NN and a basic MLP in the book's benchmark.
4. Always inspect per-class precision/recall/F1, not just overall accuracy, especially under class imbalance common in real 3D scenes.
5. Normalize/scale spatial features (min-max, z-score, or robust scaling) before training — unscaled coordinates tie the model to the training range and break on shifted data.
6. Deliberately injecting a small, disclosed amount of target-distribution data into training can meaningfully improve deployment performance, at a quantifiable cost to test-set purity.
7. Persist trained models (pickle) for deployment, and in production, monitor data drift, model degradation, and schedule retraining.

## Connects To
- **Ch 6, 7**: the geometry-aware features (planarity, verticality, omnivariance, normal change rate) that rescue generalization here were extracted using the PCA/covariance techniques from these chapters.
- **Ch 12, 13**: unsupervised clustering/segmentation (previous chapters) is positioned as a complementary, cheaper way to bootstrap labeled datasets that feed this chapter's supervised pipeline.
- **Ch 15, 16**: the next chapters move from feature-engineering-dependent classical ML (this chapter) to 3D deep learning, which can learn features automatically.
- **pandas, scikit-learn (RandomForestClassifier, KNeighborsClassifier, MLPClassifier, train_test_split, MinMaxScaler, classification_report), Pickle**: the core library stack for this chapter's supervised learning pipeline.
- **IGN LiDAR HD (France)**: external open-data source for the Louhans/Manosque aerial LiDAR datasets used throughout.
