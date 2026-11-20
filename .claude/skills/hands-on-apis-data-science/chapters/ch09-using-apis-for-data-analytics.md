# Chapter 9: Using APIs for Data Analytics

## Core Idea
Build a custom analytics metric (the "Shark League Score") end-to-end from an API data source, using a hand-rolled API client, Jupyter Notebooks for interactive computing, and pandas for data wrangling — demonstrating the full custom-metric workflow data scientists use in sports analytics and beyond.

## Frameworks Introduced
- **Custom Metric Design Checklist**: five required components for any credible custom metric.
  - When to use: before building any new analytics metric (sports or otherwise).
  - How: define (1) the **Question** it answers (broad enough to matter, specific enough to add knowledge); (2) the **Theory** — which subcomponents matter and how they're weighted (a value judgment); (3) a **Valid approach** — do the calculations actually support the stated purpose; (4) a **Data source** — can you actually get the data at a usable frequency (adjust theory to data availability if needed); (5) a memorable **Name**.
- **API-vs-Web-Scraping for Recurring Data Collection**: choosing a data acquisition method for repeatable analytics.
  - When to use: building any automated, recurring metric pipeline from a web data source.
  - How: prefer an API when available — it's stable across site redesigns; web scraping (parsing HTML) is powerful but brittle, breaking whenever site structure changes and requiring rework.
- **Standalone API Client for Notebooks**: separate API-calling logic from notebook analysis code (SoC, continued from Ch. 8).
  - When to use: any Jupyter Notebook project that calls an API more than once or across multiple notebooks.
  - How: write a plain `.py` module (not a full SDK) with endpoint constants, a `logging` logger, and one `backoff`-decorated `call_api_endpoint()` function using `httpx.Client` as a context manager; `import` it into notebooks like any other library.
- **Nested-JSON-to-Tidy-DataFrame Pattern**: flattening API responses with nested structures into analysis-ready rows.
  - When to use: an API returns a list of objects where one field is itself a list/array of sub-records (e.g., each team has a `weekly_scores` list).
  - How: use `pandas.json_normalize(data, record_path, meta)` to explode the nested list into one row per sub-record while carrying along specified parent (`meta`) columns.

## Key Concepts
- **Coefficient of Variation (CV)**: standard deviation divided by the mean; a dimensionless relative-variability measure that allows fair comparison across datasets with different scales (here, different fantasy scoring systems) — used as the basis for the League Balance Score.
- **Interactive computing (Jupyter Notebooks)**: a programming mode mixing code cells, Markdown cells, and persisted output; cells can run out of order, and variables persist across cells within a session but not between sessions — "Execute Above Cells" is needed to rebuild state when resuming work.
- **pandas DataFrame**: the two-dimensional, spreadsheet-like data structure central to Python data analysis; supports `groupby().agg()`/`groupby().sum()` (SQL `GROUP BY`-like aggregation), `str.slice()` (substring extraction), `drop()` (column removal), `merge()` (joins), `apply()` with `lambda` (row-wise custom calculations), and `sort_values()` (display-only sorting).
- **lambda functions in aggregation**: inline anonymous functions used inside `.agg()` or `.apply()` to compute custom per-group or per-row values (e.g., the CV-based balance score formula) without writing a named function.
- **Exponential backoff with jitter (recap)**: reapplied here via the `backoff` library decorating the notebook's own `call_api_endpoint()` function, same technique as Chapter 7's SDK.
- **Logging to a file in a notebook**: clearing any pre-existing logging handlers (`logging.root.handlers`) and reconfiguring via `logging.basicConfig(filename=..., level=...)` so notebook API calls log to a dedicated file excluded from version control via `.gitignore`.

## Mental Models
- **"Your eyes see the game much better than the numbers. But the numbers see all the games."** (Dean Oliver epigraph): custom metrics exist to aggregate signal across volume that human observation can't track exhaustively.
- **APIs as the stable backbone for recurring analytics**: once scoring/roster data changes weekly, only an API (not a fragile scraper) can be depended on for an automated, repeatable metric pipeline.
- **Scaling multiple sub-metrics onto a common numeric range**: both the League Balance Score and League Juice Score are deliberately transformed (×100, inverted, etc.) so they land on comparable 0–100-ish scales before being combined into the composite Shark League Score.
- **Separating data retrieval, data shaping, and metric calculation into distinct notebook sections**: each Markdown-headed section (Get Max Scores → Get League Scoring Type → Get Team Totals → Calculate Balance → Calculate Juice → Combine into Shark Score) is a discrete, inspectable stage — mirroring good pipeline design even inside a single notebook.

## Anti-patterns
- **Web scraping for recurring/automated metrics**: brittle — any site redesign silently breaks the pipeline, requiring ongoing maintenance that an API avoids.
- **Comparing raw standard deviations across leagues with different scoring systems**: misleading, since higher-scoring systems naturally have higher absolute variability; CV corrects for this by normalizing against the mean.
- **Embedding API call logic directly inline throughout notebook cells**: makes retry/logging/error-handling inconsistent and hard to reuse; the chapter instead isolates it into `swc_simple_client.py`.
- **Letting notebook logging pollute version control**: log files are explicitly `.gitignore`d since they're local session artifacts, not code.

## Code Examples
```python
# swc_simple_client.py — standalone API client with backoff
@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(httpx.RequestError, httpx.HTTPStatusError),
    max_time=5,
    jitter=backoff.random_jitter
)
def call_api_endpoint(base_url: str, api_endpoint: str, api_params: dict = None) -> httpx.Response:
    try:
        with httpx.Client(base_url=base_url) as client:
            response = client.get(api_endpoint, params=api_params)
            response.raise_for_status()
            return response
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP status error occurred: {e.response.text}")
        return httpx.Response(status_code=e.response.status_code, content=b"API error")
    except httpx.RequestError as e:
        logger.error(f"Request error occurred: {str(e)}")
        return httpx.Response(status_code=500, content=b"Network error")
```
- **What it demonstrates**: a reusable, backoff-protected API call function returning a response object even on failure, rather than crashing — designed to be imported into any notebook.

```python
# Flattening nested JSON and summing regular-season team totals
teams_parsed_df = pd.json_normalize(team_api_response.json(), 'weekly_scores',
                                     ['league_id', 'team_id', 'team_name'])
teams_parsed_df['year'] = teams_parsed_df['week_number'].str.slice(0, 4).astype(int)
teams_parsed_df['week'] = teams_parsed_df['week_number'].str.slice(4, 6).astype(int)
teams_regular_season_df = teams_parsed_df.query('week <= 14')
team_totals_df = teams_regular_season_df.groupby(
    ['league_id', 'team_id', 'year'], as_index=False)['fantasy_points'].sum()
```
- **What it demonstrates**: `json_normalize` exploding a nested `weekly_scores` array into one row per week while retaining parent IDs, then filtering to regular-season weeks and aggregating totals per team/year.

```python
# League Balance Score via Coefficient of Variation
league_stats_df = team_totals_df.groupby(['league_id', 'year']).agg(
    league_points_sum=('fantasy_points', 'sum'),
    league_points_mean=('fantasy_points', 'mean'),
    league_points_stdev=('fantasy_points', 'std'),
    league_balance_score=('fantasy_points', lambda x: (100 - (x.std() / x.mean()) * 100))
).reset_index()
```
- **What it demonstrates**: a single `.agg()` call computing sum, mean, std, and a custom lambda-based derived metric (CV-based balance score, scaled 0–100, higher=better) in one grouped operation.

```python
# League Juice Score via merged DataFrames and conditional row-wise calculation
combined_metrics_df['league_juice_score'] = combined_metrics_df.apply(
    lambda row: (
        100 * (row['league_points_sum'] / row['ppr_12_max_points'])
        if (row['scoring_type'] == 'PPR' and row['league_size'] == 12)
        else (100 * (row['league_points_sum'] / row['half_ppr_8_max_points'])
              if (row['scoring_type'] == 'Half-PPR' and row['league_size'] == 8)
              else None)
    ), axis=1)

# Final Shark League Score
combined_metrics_df['shark_league_score'] = combined_metrics_df.apply(
    lambda league: (2 * league['league_juice_score']) + league['league_balance_score'], axis=1)
```
- **What it demonstrates**: merging three DataFrames (team totals, max-potential-points, league metadata) and using conditional row-wise `apply`/`lambda` logic to compute a percentage-of-max-potential score, then combining two normalized sub-metrics into a final weighted composite score.

## Reference Tables
**Table 9-1. Key tools used in this chapter**

| Software | Purpose |
|---|---|
| `backoff` | Backoff/retry for web calls |
| `httpx` | Making web calls |
| Jupyter Notebooks | Interactive data science environment |
| `pandas` | Data analysis and formatting |

**Custom metric design components**

| Component | Question it answers |
|---|---|
| Question | What broad-but-specific question does this metric answer? |
| Theory | What subcomponents matter, and how are they weighted (value judgment)? |
| Valid approach | Do the calculations actually support the stated purpose? |
| Data source | Can the needed data be obtained at a usable frequency? |
| Name | Memorable branding for adoption/impact |

**Shark League Score formula stack**

| Metric | Formula | Scale |
|---|---|---|
| League Balance Score | `100 - (stdev/mean) * 100` (inverted CV) | ~0–100, higher = more balanced |
| League Juice Score | `100 * LeagueTotalPoints / MaxPotentialPoints` | 0–100, higher = more efficient lineups |
| Shark League Score | `(2 * League Juice Score) + League Balance Score` | Composite, Juice weighted double |

## Worked Example
Using SWC API v0.2, the chapter builds the **Shark League Score** — a metric answering "how tough is our fantasy league?" — through: (1) `GET /v0/weeks/` → sum max possible points (PPR-12 and Half-PPR-8) per year; (2) `GET /v0/leagues/` → scoring type and league size per league; (3) `GET /v0/teams/` → nested weekly scores, flattened via `json_normalize`, filtered to weeks 1–14 (regular season), summed per team/year; (4) **League Balance Score** from the coefficient of variation of each league's team totals (lower variability → higher score, scaled to ~0–100); (5) **League Juice Score** by merging team totals against the correct max-potential-points column (conditional on scoring type/league size) and expressing actual points as a percentage of the max; (6) the final **Shark League Score** = `2×Juice + Balance`, sorted descending to rank leagues. Each stage is isolated in its own notebook section (Markdown header + code cell), building naturally from raw API JSON to a ranked analytics output.

## Key Takeaways
1. Any credible custom metric needs a clear question, an explicit (value-laden) theory, a logically valid calculation, an available data source, and ideally a memorable name.
2. Prefer APIs over web scraping for any metric that must be recalculated on a recurring schedule — scraping breaks on every site redesign.
3. Isolate API-calling code (with backoff/retry and logging) into a standalone client module, imported into notebooks rather than inlined — the SoC principle from Chapter 8 applied concretely.
4. Use the coefficient of variation, not raw standard deviation, when comparing variability across groups with different underlying scales.
5. `pandas.json_normalize()` is the standard tool for flattening nested JSON API responses into tidy, analyzable rows.
6. Scale sub-metrics onto comparable numeric ranges (e.g., 0–100) before combining them into a composite score, and document weighting choices explicitly (here, Juice weighted 2x Balance).
7. Jupyter Notebook state (variables/imports) does not persist between sessions — use "Execute Above Cells" to rebuild it, and keep imports/config at the top for clarity.

## Connects To
- **Ch 1**: Directly realizes the "data scientists create analytics/dashboards/metrics" use case identified as a core API consumer need.
- **Ch 7/Ch 8**: Reapplies the backoff-with-jitter and SoC/API-client patterns built/discussed there, in a lighter-weight, notebook-specific form.
- **Ch 11**: The same `swc_simple_client.py`-style client (or its logic) is reused in the Streamlit app built next.
- **External**: *Python for Data Analysis* (Wes McKinney); *Football Analytics with Python & R* (Eager and Erickson); Leeger Python library (Joey Greco); Big Data Bowl (NFL analytics contest); KenPom ranking; "Coefficient of Variation: Meaning and How to Use It."
