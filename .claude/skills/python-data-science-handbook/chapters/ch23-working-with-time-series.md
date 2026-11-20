# Chapter 23: Working with Time Series

## Core Idea
Pandas' `Timestamp`/`DatetimeIndex` combine `datetime`'s ease of use with NumPy's `datetime64` efficient vectorized storage, enabling intuitive date-based indexing, regular sequence generation, and time-series-specific operations (resampling, shifting, rolling windows).

## Frameworks Introduced
- **Three date/time flavors**: timestamps (a moment), periods/intervals (a span, uniform-length "Period"), time deltas/durations (an exact length) — map to Pandas' `Timestamp`/`DatetimeIndex`, `Period`/`PeriodIndex`, `Timedelta`/`TimedeltaIndex`.
- **`pd.to_datetime`**: parses flexible date formats into a `Timestamp` (single date) or `DatetimeIndex` (sequence); `.to_period(freq)` converts to `PeriodIndex`; subtracting dates yields a `TimedeltaIndex`.
- **Regular sequence generators**: `pd.date_range(start, end/periods, freq=)`, `pd.period_range`, `pd.timedelta_range` — analogous to `range`/`np.arange` but for dates.
- **Time-series-specific operations**: `resample()` (aggregation-based frequency conversion) vs. `asfreq()` (selection-based frequency conversion); `.shift(n)` (shift values in time, e.g. for ROI calculations); `.rolling(window, center=)` (rolling-window view supporting `.mean()`, `.median()`, `.sum()`, custom `aggregate`/`apply`, like groupby).

## Key Concepts
- **`datetime`/`dateutil`**: flexible, easy-to-use native Python date tools, but inefficient for large arrays.
- **NumPy `datetime64`**: compact 64-bit vectorized date encoding; trades time resolution against total encodable span (e.g., nanosecond resolution → ~600-year range); format codes (`Y`,`M`,`W`,`D`,`h`,`m`,`s`,`ms`,`us`,`ns`, etc.) set the fundamental unit; `datetime64[ns]` is the practical default for modern data.
- **Pandas `Timestamp`/`DatetimeIndex`**: built on `datetime64` for efficiency, with `datetime`/`dateutil`-style convenience methods (e.g., `.strftime('%A')`).
- **Date-based indexing**: slicing a time-indexed Series with date strings (`data['2020-07-04':'2021-07-04']`) or even just a year (`data['2021']`) for a special "all data from that year" shorthand.
- **Frequency codes**: `D` (day), `B` (business day), `W` (week), `M`/`BM` (month/business-month end), `Q`/`BQ` (quarter end), `A`/`BA` (year end), `H` (hour), `T` (minute), `S` (second), etc.; adding `S` suffix (e.g., `MS`) marks the *start* instead of the end of the period; suffixing a 3-letter month/weekday code customizes the anchor (`Q-JAN`, `W-SUN`); codes combine with numbers (`'2H30T'`).
- **`resample` vs. `asfreq`**: `resample` aggregates (e.g., yearly average of all points in that year); `asfreq` selects (the value exactly at that point), with upsampling by default leaving gaps as NA (fillable via `method='bfill'/'ffill'`, mirroring Ch 16's `fillna`).
- **`.shift(n)`**: shifts the *data* by n periods along a regularly-sampled index — the basis for lag/lead calculations like period-over-period return.
- **`.rolling(window, center=)`**: lazy view (like GroupBy) supporting aggregate methods, with optional `win_type='gaussian'` (smoothing window) and custom `std=` parameter.

## Mental Models
- Think of the Python date-tooling landscape as a ladder: `datetime`/`dateutil` (flexible, slow, for one-off dates) → NumPy `datetime64` (compact, fast, vectorized, but inconvenient) → Pandas `Timestamp`/`DatetimeIndex` ("best of both worlds" — convenience methods on top of NumPy's efficient storage).
- `resample` = "groupby by time bucket, then aggregate" (parallel to Ch 20's groupby); `.rolling()` = "a sliding groupby window" — both return a lazy view object before an aggregate is applied.
- `asfreq` answers "what was the value exactly at this instant?" while `resample` answers "what's the typical/aggregate value during this period?" — a crucial distinction when downsampling.

## Anti-patterns
- **Using Python's native `datetime` objects in large lists/arrays for numerical time-series work**: slow compared to vectorized `datetime64`/`DatetimeIndex` — use Pandas' time types for anything beyond small one-off date math.
- **Confusing `resample` and `asfreq` when downsampling**: `asfreq` just picks the value at the period boundary (ignoring intermediate data), while `resample` aggregates all data within the period — using the wrong one silently changes the meaning of your summary.
- **Assuming `.shift()` with repeated values of `+=1` in a loop accomplishes a true accumulation across repeated indices (unrelated anti-pattern, not applicable here)** — N/A for this chapter; instead: **forgetting that upsampling with `asfreq` leaves NA gaps by default**: always specify a `method=` (or follow with `fillna`) when the gaps should be filled rather than left missing.

## Code Examples
```python
from datetime import datetime
datetime(year=2021, month=7, day=4)

from dateutil import parser
date = parser.parse("4th of July, 2021")
date.strftime('%A')                       # 'Sunday'

import numpy as np
date = np.array('2021-07-04', dtype=np.datetime64)
date + np.arange(12)                        # vectorized date arithmetic
np.datetime64('2021-07-04 12:59:59.50', 'ns')   # force nanosecond precision

import pandas as pd
date = pd.to_datetime("4th of July, 2021")   # Timestamp
date.strftime('%A')
date + pd.to_timedelta(np.arange(12), 'D')     # DatetimeIndex result

index = pd.DatetimeIndex(['2020-07-04','2020-08-04','2021-07-04','2021-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)
data['2020-07-04':'2021-07-04']               # date-string slicing
data['2021']                                     # year shorthand

dates = pd.to_datetime([datetime(2021,7,3), '4th of July, 2021',
                         '2021-Jul-6', '07-07-2021', '20210708'])
dates.to_period('D')                              # DatetimeIndex -> PeriodIndex
dates - dates[0]                                    # -> TimedeltaIndex

pd.date_range('2015-07-03', '2015-07-10')             # start/end
pd.date_range('2015-07-03', periods=8)                  # start/periods
pd.date_range('2015-07-03', periods=8, freq='H')          # hourly
pd.period_range('2015-07', periods=8, freq='M')             # monthly periods
pd.timedelta_range(0, periods=6, freq='H')                   # hourly deltas
pd.timedelta_range(0, periods=6, freq="2H30T")                 # combined codes

from pandas.tseries.offsets import BDay
pd.date_range('2015-07-01', periods=6, freq=BDay())

# resample vs asfreq
sp500.resample('BA').mean()      # aggregate: avg of each business year
sp500.asfreq('BA')                 # select: value at year end

data.asfreq('D').plot(marker='o')                 # upsample, gaps = NaN
data.asfreq('D', method='bfill')                     # backward fill
data.asfreq('D', method='ffill')                      # forward fill

# shift for ROI
sp500 = sp500.asfreq('D', method='pad')
ROI = 100 * (sp500.shift(-365) - sp500) / sp500

# rolling windows
rolling = sp500.rolling(365, center=True)
rolling.mean(); rolling.median()

# Fremont Bridge bicycle example
data = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
data.columns = ['Total', 'East', 'West']
weekly = data.resample('W').sum()
daily = data.resample('D').sum()
daily.rolling(30, center=True).sum()
daily.rolling(50, center=True, win_type='gaussian').sum(std=10)

by_time = data.groupby(data.index.time).mean()        # average by time-of-day
by_weekday = data.groupby(data.index.dayofweek).mean()  # average by weekday
weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
by_time = data.groupby([weekend, data.index.time]).mean()  # compound groupby
```
- **What it demonstrates**: the datetime/dateutil → datetime64 → Pandas Timestamp progression, flexible date parsing and sequence generation, frequency codes, resample vs. asfreq, shift-based ROI calculation, rolling-window statistics with window types, and a full real-world exploratory workflow (Fremont Bridge bicycle counts) combining resampling, rolling means, and groupby-by-time-component.

## Reference Tables
| datetime64 code | Unit | Approx. relative span |
|---|---|---|
| Y | Year | ±9.2e18 years |
| M | Month | ±7.6e17 years |
| D | Day | ±2.5e16 years |
| h | Hour | ±1.0e15 years |
| s | Second | ±2.9e12 years |
| ms | Millisecond | ±2.9e9 years |
| us | Microsecond | ±2.9e6 years |
| ns | Nanosecond | ±292 years (practical default) |

| Frequency code | Meaning |
|---|---|
| `D`/`B` | Calendar day / business day |
| `W` | Weekly |
| `M`/`BM` | Month end / business month end |
| `Q`/`BQ` | Quarter end / business quarter end |
| `A`/`BA` | Year end / business year end |
| `H`/`T`/`S` | Hours / minutes / seconds |
| `MS`/`QS`/`AS` | Start-anchored variants (month/quarter/year start) |

## Worked Example
Fremont Bridge bicycle counts: load ~150K hourly rows with `parse_dates=True`, resample weekly (`data.resample('W').sum()`) to reveal seasonal trends and the COVID-19 commuting drop in early 2020; apply a 30-day centered rolling sum (jagged due to hard window cutoff) vs. a Gaussian-windowed rolling sum (`win_type='gaussian', std=10`, smoother); then use `groupby(data.index.time)` and `groupby(data.index.dayofweek)` to reveal a bimodal AM/PM commuter pattern on weekdays vs. a unimodal recreational pattern on weekends, including a compound `groupby([weekend_flag, data.index.time])` to split both dimensions at once.

## Key Takeaways
1. Pandas' `Timestamp`/`DatetimeIndex` combine `datetime`'s convenience with `datetime64`'s vectorized efficiency — the practical default dtype is `datetime64[ns]`.
2. `pd.to_datetime` parses flexible formats; `pd.date_range`/`period_range`/`timedelta_range` generate regular sequences with a frequency code.
3. Time-indexed Series/DataFrames support intuitive date-string slicing, including year-only shorthand.
4. `resample` aggregates data within each period (data reduction); `asfreq` selects the value at period boundaries (data selection) — know which one the task needs.
5. `.shift(n)` enables lag/lead-style calculations (e.g., period-over-period return); `.rolling(window, center=)` provides a lazy, groupby-like view for rolling statistics, including smoothing window types like Gaussian.
6. Combining resample/rolling with `groupby` on index components (`.index.time`, `.index.dayofweek`) is a powerful pattern for revealing cyclical structure in time series.

## Connects To
- **Ch 6**: `datetime64` vectorization is framed the same way as NumPy ufunc vectorization.
- **Ch 16**: `asfreq`'s fill methods (`bfill`/`ffill`) directly parallel `fillna`.
- **Ch 20**: `.resample()` and `.rolling()` are explicitly compared to `groupby`'s lazy-view pattern.
- **Ch 42**: the Fremont Bridge dataset is revisited for modeling (linear regression with weather/seasonal features).
