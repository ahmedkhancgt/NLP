# Cheatsheet: Decision Rules, Trade-offs, Tells & Smells

## Which data structure/tool?
| Need | Use |
|---|---|
| Homogeneous numeric array | NumPy `ndarray` |
| Labeled 1D data, mixed access patterns | Pandas `Series` |
| Labeled 2D tabular data | Pandas `DataFrame` |
| 3+ dimensional labeled data | `MultiIndex` on Series/DataFrame |
| Heterogeneous fixed-schema records, C interop | NumPy structured array (else prefer DataFrame) |

## Which indexing method?
| Situation | Use |
|---|---|
| Integer-labeled index, need explicit label access | `.loc[]` |
| Integer-labeled index, need explicit position access | `.iloc[]` |
| Filtering rows by condition | Boolean mask: `df[mask]` or `df.loc[mask, cols]` |
| Selecting scattered elements by position | Fancy indexing: `arr[index_array]` |
| Column access on DataFrame | `df['col']` or `df.col` (not `df[0]`!) |

## Which join/combine operation?
| Situation | Use |
|---|---|
| Stacking same-structure data | `pd.concat` (axis=0 rows, axis='columns' cols) |
| Combining by shared key column(s) | `pd.merge(..., on=)` |
| Combining by index | `pd.merge(..., left_index=True, ...)` or `.join()` |
| Need union of all rows regardless of match | `how='outer'` |
| Need only matching keys | `how='inner'` (default) |

## Which aggregation tool?
| Situation | Use |
|---|---|
| Single summary stat over whole array | `.sum()`, `.mean()`, etc. |
| Stat per category | `groupby(key).agg(...)` |
| Stat across two categorical dimensions | `pivot_table(values, index, columns, aggfunc)` |
| Need same-shape transformed output per group | `groupby(key).transform(func)` |
| Need to drop whole groups by a condition | `groupby(key).filter(func)` |

## eval/query threshold
Use `pd.eval`/`df.eval`/`df.query` instead of plain Pandas syntax when:
- the DataFrame's `.values.nbytes` approaches your CPU's L1/L2 cache size or system RAM, AND
- the expression is a compound one (multiple chained ops) that would otherwise create several temporary arrays.
Below that scale, plain syntax is usually just as fast and more readable.

## Classifier/regressor first choice
| Situation | Try first |
|---|---|
| Any classification, want a fast baseline | `GaussianNB` / `MultinomialNB` (text) |
| Any regression, want a fast baseline | `LinearRegression` |
| Nonlinear relationship suspected | Basis-expanded linear model, or kernel SVM |
| High accuracy needed, interpretability not required | `RandomForestClassifier`/`Regressor` |
| Clean margin-based separation, moderate data size | `SVC` (kernel='rbf' if nonlinear) |
| Large dataset, SVM too slow | Avoid SVM (O(N²)-O(N³)); use random forest or linear model |

## Tells & smells (red flags to watch for)
- **Accuracy = 1.0 on training data** → you evaluated on training data, not a holdout; redo with `train_test_split`/cross-validation.
- **Training score ≫ validation score** → overfitting (high variance); simplify model, regularize, or get more data (check learning curve first).
- **Training score ≈ validation score, both low** → underfitting (high bias); increase model complexity/flexibility, add features.
- **Learning curve already converged (train/val curves close together, flat)** → more data won't help; need a different/more complex model.
- **`x[i] += 1` with repeated indices in `i` not accumulating as expected** → use `np.add.at(x, i, 1)` instead.
- **DataFrame slice/mask behaving "row-wise" when you expected column-wise (or vice versa)** → remember `df['col']` is column-first but `df[slice]`/`df[mask]` are row-first; use `.loc`/`.iloc` for clarity.
- **`ValueError: truth value of an array... is ambiguous`** → used `and`/`or`/`if arr:` on a multi-element array; use `&`/`|`/`.any()`/`.all()`.
- **NaN propagating silently through a whole computation** → use `np.nansum`/`nanmean`/etc., or `dropna`/`fillna` first.
- **Qualitative colormap (jet/rainbow) on quantitative data** → uneven grayscale luminance misleads the eye; use `viridis` or another perceptually uniform map.
- **Manually looping to build a histogram/distance matrix/pairwise comparison** → there's almost always a vectorized/broadcasting one-liner (`np.histogram`, `pairwise_distances`, `X[:,None]-X[None,:]`).
- **k-means on crescent-shaped/non-convex clusters giving nonsensical splits** → k-means boundaries are always linear; use `SpectralClustering` or GMM.
- **Very confident-looking decision tree with oddly-shaped, "spiky" regions** → classic overfitting; use a random forest instead of a single deep tree.

## Key numeric thresholds / defaults worth remembering
| Thing | Default/typical value |
|---|---|
| `train_test_split` default split | 75/25 |
| `GridSearchCV`/`cross_val_score` default folds | `cv=5` |
| KMeans default random restarts | `n_init=10` |
| `datetime64[ns]` | practical default resolution for modern dates |
| PCA "good" cumulative variance threshold | commonly 90-95% as a rule of thumb (always check the curve, don't assume) |

## When to reach for Seaborn instead of raw Matplotlib
- Plotting directly from a DataFrame by column name (vs. manually extracting Series).
- Need a statistical plot type in one line: joint distribution (`jointplot`), pairwise relationships (`pairplot`), categorical comparison (`catplot`/`violinplot`).
- Want sane default styling without manual rcParams tweaking.
Otherwise (fine axis/tick/annotation control, custom layouts via GridSpec, 3D plots) — stay in raw Matplotlib.
