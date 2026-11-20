# Chapter 50: Application: A Face Detection Pipeline

## Core Idea
A complete, realistic face-detection pipeline combines domain-specific feature extraction (HOG, histogram of oriented gradients) with a standard ML classifier (linear SVM) and a sliding-window scan, illustrating the full "real-world data → feature extraction → model → inference" workflow — while exposing the real limitations (incomplete negative data, single-scale search, overlapping detections) that separate a prototype from production.

## Frameworks Introduced
- **HOG (histogram of oriented gradients)**: a feature-extraction procedure for images, sensitive to broadly informative structure (edges, contours, texture) while resisting illumination variation.
  - Steps: (1) optionally prenormalize image; (2) convolve with horizontal/vertical gradient filters; (3) subdivide into cells, histogram gradient orientations per cell; (4) normalize each cell's histogram against neighboring-cell blocks; (5) flatten into one feature vector.
  - `skimage.feature.hog(image, visualize=True)` — built into Scikit-Image.
- **Face detection pipeline recipe**: (1) gather positive samples (face thumbnails); (2) gather negative samples (non-face thumbnails, extracted at varied scales from generic images via `PatchExtractor`); (3) extract HOG features from all samples; (4) train a binary classifier (`LinearSVC`, tuned via `GridSearchCV`); (5) slide a window across a new image, extracting HOG features and classifying each patch; (6) (ideally) merge overlapping positive detections into single detections.

## Key Concepts
- **Positive vs. negative training samples**: faces (`fetch_lfw_people`) vs. non-face patches extracted at multiple scales from unrelated images (`camera`, `text`, `coins`, etc. via `skimage.data` + `PatchExtractor`).
- **`LinearSVC` over `SVC`**: chosen for better scaling to a large number of samples (43,000+ training patches here) in high-dimensional (1,215-feature) binary classification.
- **Sliding window**: a function that steps a fixed-size (or multi-scale) window across an image, extracting a patch at each position for classification — the mechanism that turns a single-patch classifier into a full-image detector.
- **Baseline-then-upgrade methodology**: quickly check `GaussianNB` (~95% CV accuracy) as a fast sanity-check baseline before investing in a tuned `LinearSVC` (~99% CV accuracy via `GridSearchCV` over `C`) — mirrors Ch 41's advice to use naive Bayes as a first baseline.
- **Known pipeline limitations** (explicitly enumerated): incomplete/non-representative negative training set (causes false positives — addressable via more diverse negatives or *hard negative mining*: explicitly adding false-positive patches found on unseen images back into training); single-scale search (misses faces of other sizes — addressable via multi-scale sliding windows); unmerged overlapping detections (addressable via mean-shift clustering or non-maximum suppression); unstreamlined code (needs packaging into a clean object-oriented API for real use).
- **Deep learning as an alternative**: modern object-detection pipelines often replace hand-engineered features like HOG with neural networks that learn optimal feature extraction directly from data — conceptually continuous with (not fundamentally different from) the "build models from data" goal of the methods in this book, differing mainly in computational scale and training-data volume.

## Mental Models
- A real-world ML application is "glue work plus known pieces": the hard, unglamorous part isn't the classifier (a standard `LinearSVC`) but gathering representative positive/negative samples and turning raw pixels into usable features (HOG) — exactly the chapter's own framing: "there is no formula for how to do this... this is where you... must exercise your own intuition and expertise."
- "Found 48 detections out of ~2000 patches, all clustered around the true face" is a success *and* a visible symptom of an incomplete pipeline (no deduplication/merging yet) — recognizing both the win and the remaining rough edges is the mark of an honest practitioner, not overclaiming a "production-ready" result from a prototype.
- Hard negative mining is a general pattern beyond face detection: whenever a classifier's false positives reveal a gap in the negative training distribution, feed those exact false positives back in as new negative examples and retrain.

## Anti-patterns
- **Treating a successfully-detecting prototype as "done"**: the chapter explicitly lists multiple unresolved issues (incomplete negatives, single scale, unmerged overlaps, unstreamlined code) — a working demo on one test image is not evidence of production readiness.
- **Using `SVC` instead of `LinearSVC` for large-sample, high-dimensional linear classification tasks**: `LinearSVC` scales better; reserve kernelized `SVC` for cases needing nonlinear boundaries (Ch 43).
- **Expecting a single-scale sliding window to detect faces of varying sizes**: by construction, it will miss anything not close to the trained patch size — multi-scale search (resizing patches before classification) is required for robustness.
- **Not deduplicating overlapping positive detections in a real pipeline**: leaves many redundant boxes around the same face — apply clustering (mean-shift) or non-maximum suppression before presenting final results.

## Code Examples
```python
from skimage import data, color, feature
import skimage.data

image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualize=True)

# 1. Positive samples
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people()
positive_patches = faces.images                  # (13233, 62, 47)

# 2. Negative samples (varied scales, from non-face images)
from skimage import data, transform
imgs_to_use = ['camera', 'text', 'coins', 'moon', 'page', 'clock',
                'immunohistochemistry', 'chelsea', 'coffee', 'hubble_deep_field']
raw_images = (getattr(data, name)() for name in imgs_to_use)
images = [color.rgb2gray(image) if image.ndim == 3 else image for image in raw_images]

from sklearn.feature_extraction.image import PatchExtractor
def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size) for patch in patches])
    return patches

negative_patches = np.vstack([extract_patches(im, 1000, scale)
                               for im in images for scale in [0.5, 1.0, 2.0]])  # (30000, 62, 47)

# 3. Extract HOG features
from itertools import chain
X_train = np.array([feature.hog(im) for im in chain(positive_patches, negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1            # (43233, 1215)

# 4. Baseline then tuned classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(), X_train, y_train)      # ~95% baseline

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)
grid.best_score_                                      # ~98.9%
model = grid.best_estimator_
model.fit(X_train, y_train)

# 5. Sliding window detection on new image
test_image = skimage.color.rgb2gray(skimage.data.astronaut())
test_image = skimage.transform.rescale(test_image, 0.5)[:160, 40:180]

def sliding_window(img, patch_size=positive_patches[0].shape, istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i+Ni, j:j+Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch

indices, patches = zip(*sliding_window(test_image))
patches_hog = np.array([feature.hog(patch) for patch in patches])
labels = model.predict(patches_hog)                    # 48 positive detections out of ~1900 patches
```
- **What it demonstrates**: a complete, honest, real-world ML pipeline — gathering/engineering positive and negative training data, extracting domain-appropriate (HOG) features, baseline-then-tuned classifier selection, and inference via sliding-window search on a new test image, plus a candid discussion of the gap between this prototype and a production system.

## Reference Tables
| Pipeline step | Tool |
|---|---|
| Feature extraction | `skimage.feature.hog` |
| Positive samples | `fetch_lfw_people` |
| Negative samples | `PatchExtractor` on non-face images, multiple scales |
| Classifier (baseline) | `GaussianNB` |
| Classifier (tuned) | `LinearSVC` + `GridSearchCV` |
| Inference | Sliding window + per-patch HOG + `model.predict` |

| Known limitation | Fix |
|---|---|
| Incomplete negative set | More diverse negatives, or hard negative mining |
| Single-scale search | Multi-scale sliding windows (resize patches) |
| Overlapping detections | Mean-shift clustering, or non-maximum suppression |
| Unstreamlined code | Package into an object-oriented pipeline |

## Worked Example
Finding a face in a cropped astronaut image: a sliding window scans ~1,911 patches across the 160×140 test image; HOG features computed per patch are classified by the tuned `LinearSVC`, yielding 48 positive detections, all clustered tightly around the actual face location — drawing red rectangles at each detection visually confirms success, while also illustrating (via the sheer count of overlapping boxes) the "combine overlapped detections" limitation discussed afterward.

## Key Takeaways
1. Real-world ML applications require domain-specific feature engineering (here, HOG) before any standard classifier can be applied — "there is no formula... this is where you... must exercise your own intuition."
2. A full pipeline recipe: gather positive/negative samples → extract features → train and tune a classifier (baseline first, then tuned) → sliding-window inference on new data.
3. `LinearSVC` is preferred over kernelized `SVC` for large-sample, high-dimensional linear classification due to better scaling.
4. A working prototype (correctly finding a face) still has real, enumerable gaps versus production-readiness: incomplete negative training data, single-scale search, unmerged overlapping detections, and unstreamlined code.
5. Hard negative mining (feeding false positives from unseen data back into training as negatives) is a general technique for improving classifier robustness.
6. Deep learning is presented as a natural extension of these same principles — learning feature extraction from data at larger scale — not a conceptually separate paradigm.

## Connects To
- **Ch 40**: feature engineering principles (turning raw data into usable features) are the foundation HOG builds on for images specifically.
- **Ch 41**: the baseline-with-naive-Bayes-first methodology directly reuses that chapter's advice.
- **Ch 43**: `LinearSVC` is the same support-vector-machine family covered in depth there.
- **Ch 39**: `GridSearchCV` hyperparameter tuning (`C` parameter) directly reuses the validation chapter's tools.
- **Ch 47**: mean-shift clustering, mentioned as a candidate for merging overlapping detections, connects to the clustering chapter's broader algorithm family.
