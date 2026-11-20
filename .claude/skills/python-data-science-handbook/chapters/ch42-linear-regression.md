# Chapter 42: In Depth: Linear Regression

## Core Idea
Linear regression fits a line/hyperplane to data (`LinearRegression`), generalizes to nonlinear relationships via basis-function transforms (polynomial, Gaussian) of the input, and basis expansion's tendency to overfit is controlled via regularization (ridge/L2, lasso/L1) — making it a fast, interpretable starting point for regression tasks.

## Frameworks Introduced
- **Simple/multidimensional linear regression**: `LinearRegression(fit_intercept=True)` fits `y = a·x + b` (1D) or `y = a0 + a1·x1 + a2·x2 + ...` (multidimensional, a hyperplane) — learned params in `.coef_`/`.intercept_` (trailing underscore = fit result).
- **Basis function regression**: transform a single input `x` into multiple derived features `f1(x), f2(x), ...` before fitting a (still linear-in-coefficients) model — "linearity" refers to coefficients never multiplying/dividing each other, not to the shape of the fit curve.
  - Polynomial basis: `PolynomialFeatures(degree)` + `LinearRegression` via `make_pipeline` — recovers polynomial regression.
  - Gaussian basis: custom `BaseEstimator`/`TransformerMixin` subclass creating a sum of Gaussian "bumps" centered at fixed locations — demonstrates that any domain-appropriate basis can be used, not just polynomials.
- **Regularization**: penalizing large coefficient magnitudes to control overfitting from flexible basis expansions.
  - Ridge (`Ridge(alpha=)`, L2): penalty `α·Σθₙ²` — shrinks coefficients smoothly; α→0 recovers standard regression, α→∞ suppresses all response; computationally cheap.
  - Lasso (`Lasso(alpha=)`, L1): penalty `α·Σ|θₙ|` — favors *sparse* models, driving many coefficients to exactly zero.

## Key Concepts
- **"Linear" in linear regression**: refers to linearity in the *coefficients*, not the fitted curve's shape — this is why polynomial/Gaussian basis expansion can produce curved fits while still being "linear regression" underneath.
- **Custom Scikit-Learn transformers**: subclass `BaseEstimator` + `TransformerMixin`, implement `.fit()` and `.transform()` — makes any custom feature-engineering step pipeline-compatible.
- **Overfitting signature in basis regression**: when basis functions overlap too much (too many, too densely packed), adjacent coefficients "blow up and cancel each other out" — visible by plotting coefficient magnitude vs. basis location.
- **Ridge vs. lasso trade-off**: both shrink coefficients, but ridge shrinks smoothly toward zero (rarely exactly zero) while lasso actively zeroes out coefficients — lasso effectively performs feature selection.
- **Statistical-modeling use of ML tools**: unlike typical "focus on prediction" ML usage (Ch 38), interpreting `model.coef_` as the real-world effect size of each feature is a legitimate (if less common) use of Scikit-Learn — but requires uncertainty estimates (e.g., via bootstrap resampling) to be meaningful.
- **Bootstrap resampling for coefficient uncertainty**: `sklearn.utils.resample` + repeated refitting gives an empirical standard deviation for each coefficient, enabling effect-size-with-uncertainty statements.

## Mental Models
- Think of basis-function regression as "linear regression in a transformed feature space" — the model stays simple (linear) while the *features* become arbitrarily rich, echoing Ch 40's "transform the input, not the model" principle.
- Regularization is "a complexity dial, separate from the model class" — you don't need to change your basis functions to control overfitting; you add a penalty term that discourages the optimizer from using extreme coefficient values.
- When interpreting a fitted linear model's coefficients as "the effect of X holding everything else constant," always pair the coefficient estimate with an uncertainty measure — an unqualified point estimate can be misleading.

## Anti-patterns
- **Using many overlapping, finely-spaced basis functions without regularization**: leads to wildly oscillating fits where adjacent coefficients blow up and cancel — always pair flexible basis expansions with ridge/lasso regularization (tuned via cross-validation, Ch 39).
- **Reporting fitted coefficients without uncertainty estimates**: a bare `model.coef_` value invites overconfident claims about feature effects — use bootstrap resampling (or statsmodels) to quantify uncertainty before interpreting.
- **Assuming a linear model with a modest fit captures all relevant structure**: the bicycle-traffic example explicitly notes that nonlinear effects (e.g., disinclination to ride at extreme temperatures) and cross-day correlations are missed by this simple linear approach — a so-so fit is a cue to look for missing features or nonlinear structure, not necessarily a dead end.

## Code Examples
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# simple 1D linear regression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
model.coef_[0]; model.intercept_

# multidimensional
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])
model.fit(X, y)
model.intercept_; model.coef_               # recovers [1.5, -2., 1.]

# polynomial basis via pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
poly_model.fit(x[:, np.newaxis], y)        # fits sine wave with noise well

# custom Gaussian basis transformer
from sklearn.base import BaseEstimator, TransformerMixin
class GaussianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, N, width_factor=2.0):
        self.N = N; self.width_factor = width_factor
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
    def fit(self, X, y=None):
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)

gauss_model = make_pipeline(GaussianFeatures(20), LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)

# regularization
from sklearn.linear_model import Ridge, Lasso
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))         # smooth shrinkage
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001, max_iter=2000))  # sparse

# bicycle traffic example (feature engineering + interpretation)
import pandas as pd
counts = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('SeattleWeather.csv', index_col='DATE', parse_dates=True)
daily = counts.resample('d').sum()
daily['Total'] = daily.sum(axis=1); daily = daily[['Total']]

days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
for i in range(7): daily[days[i]] = (daily.index.dayofweek == i).astype(float)

from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2020')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'].fillna(0, inplace=True)

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
weather['Temp (F)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])
weather['Rainfall (in)'] = weather['PRCP']
weather['dry day'] = (weather['PRCP'] == 0).astype(int)
daily = daily.join(weather[['Rainfall (in)', 'Temp (F)', 'dry day']])
daily['annual'] = (daily.index - daily.index[0]).days / 365.

daily.dropna(axis=0, how='any', inplace=True)
X = daily[days + ['holiday', 'daylight_hrs', 'Rainfall (in)', 'dry day', 'Temp (F)', 'annual']]
y = daily['Total']
model = LinearRegression(fit_intercept=False)    # day flags act as per-day intercepts
model.fit(X, y)
daily['predicted'] = model.predict(X)

params = pd.Series(model.coef_, index=X.columns)    # effect size per feature
from sklearn.utils import resample
err = np.std([model.fit(*resample(X, y)).coef_ for i in range(1000)], 0)  # bootstrap uncertainty
```
- **What it demonstrates**: simple/multidimensional linear regression, polynomial and custom Gaussian basis function expansion, ridge vs. lasso regularization behavior contrast, and a complete real-world statistical-modeling workflow (Seattle bicycle traffic) with feature engineering and bootstrap-based coefficient uncertainty.

## Reference Tables
| Regularization | Penalty | Effect |
|---|---|---|
| Ridge (L2) | `α·Σθₙ²` | Smooth coefficient shrinkage |
| Lasso (L1) | `α·Σ|θₙ|` | Sparse: many coefficients exactly zero |

| α limit | Ridge/Lasso behavior |
|---|---|
| α → 0 | Recovers standard (unregularized) regression |
| α → ∞ | All model response suppressed |

## Worked Example
Seattle Fremont Bridge bicycle traffic model: after engineering day-of-week flags, holiday indicator, daylight hours, temperature, rainfall, dry-day flag, and an annual trend counter, `LinearRegression(fit_intercept=False)` fits total daily ridership; bootstrap resampling (1000 refits) gives coefficient ± uncertainty, revealing: ~10,400-10,800 fewer riders on weekends vs. weekdays, +409±26 riders per extra daylight hour, +179±7 riders per °F, +2,111±101 riders on dry days, -2,790±186 riders per inch of rain, and a modest +324±22 new riders per year — demonstrating linear regression used not just for prediction but for interpretable effect-size estimation with honest uncertainty.

## Key Takeaways
1. "Linear" regression means linear in the coefficients — basis-function transforms (polynomial, Gaussian, or custom) let a linear model fit arbitrarily nonlinear relationships between the original input and output.
2. Custom Scikit-Learn transformers (`BaseEstimator` + `TransformerMixin`, implementing `fit`/`transform`) make any feature-engineering step pipeline-compatible.
3. Rich basis expansions overfit easily (overlapping basis coefficients blow up and cancel) — regularization (ridge/L2 for smooth shrinkage, lasso/L1 for sparsity) controls this.
4. The regularization strength `α` is itself a hyperparameter to tune via cross-validation (Ch 39).
5. Interpreting fitted coefficients as real-world effect sizes is a legitimate statistical-modeling use of Scikit-Learn, but requires uncertainty estimates (e.g., bootstrap resampling) to be meaningful — Scikit-Learn itself doesn't provide these natively.
6. A so-so model fit is informative — it signals missing features or unmodeled nonlinear structure, not necessarily model failure.

## Connects To
- **Ch 23**: the Fremont Bridge bicycle dataset is reused directly from the time series chapter.
- **Ch 39**: overfitting/regularization strength tuning builds directly on bias-variance and cross-validation concepts.
- **Ch 40**: basis-function regression is a direct continuation of the "derived features" principle introduced there.
- **Ch 43**: kernel methods (SVM) generalize the basis-function idea further.
