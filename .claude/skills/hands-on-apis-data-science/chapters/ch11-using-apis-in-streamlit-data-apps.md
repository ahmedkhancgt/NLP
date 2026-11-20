# Chapter 11: Using APIs in Streamlit Data Apps

## Core Idea
Build a multi-page, interactive Streamlit data app that consumes the SWC API (via the reused Chapter 9 client), enriches it with an external data source (`nfl_data_py`) joined on the GSIS ID, and visualizes it — demonstrating how an API-backed data app is both a portfolio showcase and a practical pattern for combining internal and external data.

## Frameworks Introduced
- **Streamlit Multi-Page App Structure**: entrypoint file + per-page files + shared session state.
  - When to use: any Streamlit app with more than one view/screen.
  - How: an entrypoint file (`streamlit_football_app.py`) calls `st.set_page_config()`, defines `st.Page(...)` objects for each subpage, builds a navigation bar with `st.navigation([...])`, and runs the selected page with `pg.run()`; each page file is a plain Python script that Streamlit executes top-to-bottom when selected.
- **st.session_state for Cross-Page Data Sharing**: avoid redundant API calls across pages.
  - When to use: a multi-page app where one page fetches data another page also needs.
  - How: on first fetch, store the DataFrame/list in `st.session_state['key']` (guarded by `if 'key' not in st.session_state`); subsequent pages read it back from session state instead of re-calling the API.
- **API Enrichment via External Data Join**: combine an organization's own API data with a public dataset using a shared identifier.
  - When to use: your API data lacks some dimension (e.g., touchdown stats) that an external library/dataset provides.
  - How: ensure your API includes a standard external identifier (Ch. 1/3's GSIS ID recommendation); load the external dataset (`nfl_data_py.import_seasonal_data`) into a DataFrame; `pandas.merge()` the two DataFrames on the shared key (`gsis_id` ↔ `player_id`); aggregate and visualize the combined result.

## Key Concepts
- **Streamlit**: open source Python library turning a script into an interactive web app (`streamlit run app.py`); handles UI rendering so developers focus on data/backend logic; supports "run on save" live reload during development.
- **Widget**: a built-in interactive control (slider, select box, etc.) letting users filter/interact with displayed data in real time, e.g. `st.sidebar.selectbox('Pick league ID', unique_leagues)`.
- **Tabular reports vs. visualizations**: two main analytics product types — tabular reports show raw rows/columns (detailed view); visualizations (charts, graphs, maps) communicate patterns hard to see in a spreadsheet; Streamlit supports both interactively.
- **nfl_data_py / nflverse**: open source Python library and data ecosystem providing NFL statistics, used here to enrich SWC data with real touchdown stats via `import_seasonal_data([2023], 'REG')`.
- **GSIS ID as a join key**: the external identifier added to the API back in Chapter 3 specifically "for data scientists" now pays off directly — it's the join key letting SWC's internal player data merge cleanly with `nfl_data_py`'s external dataset.
- **Reused API client**: the chapter explicitly copies `swc_simple_client.py` from Chapter 9 into the Streamlit app folder rather than rewriting it — a direct demonstration of the Separation-of-Concerns payoff described in Chapter 8.
- **Streamlit Community Cloud**: free hosting for Streamlit apps deployed from a GitHub repo; apps "sleep" periodically and must be woken by a user, unlike always-on paid hosts.

## Mental Models
- **"A simple app today is better than an over-designed app three months late"** (Thiago Teixeira epigraph): Streamlit's whole design philosophy favors fast, simple-default apps over polished-but-slow development — matching data scientists' need to ship quickly.
- **A data app as a portfolio/communication artifact**: beyond its technical content, a working, link-shareable data app is framed as a strong signal to recruiters/clients — "it's hard to beat a data app" for demonstrating skills.
- **Three-layer page design**: each page follows (1) call API/retrieve session state → (2) shape/filter with pandas → (3) render with Streamlit widgets/charts — the same ETL-like shape seen in the pipeline chapter, just rendered interactively instead of stored.
- **Pay-it-forward data design**: a design decision made early (including GSIS ID "just in case," per Ch. 1/3) turns out to directly enable a feature built much later (joining to nfl_data_py) — a concrete illustration of why data-scientist-friendly API design (Ch. 1) matters beyond the immediate use case.

## Anti-patterns
- **Re-fetching the same API data on every page**: wastes calls and slows the app; `st.session_state` exists specifically to avoid this.
- **Embedding raw API-calling logic inside page scripts**: the chapter instead reuses the standalone `swc_simple_client.py` module built in Chapter 9, keeping API logic separated and testable.
- **No error handling around API calls in a user-facing app**: every page wraps its logic in `try...except`, checks `status_code == 200` explicitly, and logs/display errors instead of crashing the app silently.
- **Hardcoding API URLs inside each page file**: the base URL is set once in session state in the entrypoint file and read from there by every page.

## Code Examples
```python
# streamlit_football_app.py — entrypoint: session state + navigation
import streamlit as st

if 'base_url' not in st.session_state:
    st.session_state['base_url'] = "[insert your API base URL]"

st.set_page_config(page_title="Football App", page_icon=":material/sports_football:")

page_1 = st.Page("page1.py", title="Team Rosters", icon=":material/trophy:")
page_2 = st.Page("page2.py", title="Team Stats", icon=":material/star_border:")

pg = st.navigation([page_1, page_2])
pg.run()
```
- **What it demonstrates**: initializing shared state once, then wiring up a two-page navigation structure that Streamlit renders as a sidebar.

```python
# page1.py — API call, error handling, flatten+filter, display
try:
    team_api_response = swc.call_api_endpoint(base_url, swc.LIST_TEAMS_ENDPOINT)
    if team_api_response.status_code == 200:
        team_data = team_api_response.json()
        teams_df = pd.DataFrame.from_dict(team_data)
        unique_leagues = sorted(teams_df['league_id'].unique().astype(str))
        if 'unique_leagues' not in st.session_state:
            st.session_state['unique_leagues'] = unique_leagues

        selected_league = st.sidebar.selectbox('Pick league ID', unique_leagues)

        flat_team_df = pd.json_normalize(team_data, 'players', ['team_id', 'team_name', 'league_id'])
        display_df = flat_team_df.drop(columns=['team_id', 'player_id'])
        display_df['league_id'] = display_df['league_id'].astype(str)
        display_df = display_df[display_df['league_id'] == selected_league]
        st.dataframe(display_df, hide_index=True)
    else:
        logger.error(f"Error encountered: {team_api_response.status_code} {team_api_response.text}")
        st.write("Error encountered while accessing data source.")
except Exception as e:
    logger.error(f"Exception encountered: {str(e)}")
    st.write("An unexpected error occurred.")
```
- **What it demonstrates**: the full page pattern — API call, status check, flatten nested JSON, store sharable state, filter by widget selection, render, and layered error handling (API-level and exception-level).

```python
# page2.py — joining API data with nfl_data_py and plotting
nfl_data_2023_df = nfl.import_seasonal_data([2023], 'REG')
nfl_data_2023_subset_df = nfl_data_2023_df[['player_id', 'passing_tds', 'rushing_tds', 'receiving_tds']].copy()
nfl_data_2023_subset_df['total_tds'] = (
    nfl_data_2023_subset_df['passing_tds'] +
    nfl_data_2023_subset_df['rushing_tds'] +
    nfl_data_2023_subset_df['receiving_tds'])

merged_df = pd.merge(flat_team_df_ordered, nfl_data_2023_subset_df,
                      how='left', left_on='gsis_id', right_on='player_id')
grouped_df = merged_df.groupby('team_name')['total_tds'].sum()

fig, ax = plt.subplots()
grouped_df.plot(kind="barh", xlabel='Total TDs', ylabel="Team Name", title="Total TDs - 2023", ax=ax)
st.pyplot(fig)
```
- **What it demonstrates**: loading an external dataset, deriving a computed column, merging on the GSIS ID/player_id key, aggregating by team, and rendering a matplotlib bar chart inside Streamlit via `st.pyplot()`.

## Reference Tables
**Table 11-1. Key tools used in this chapter**

| Software | Purpose |
|---|---|
| `nfl_data_py` | NFL data library (external enrichment source) |
| Streamlit | Web-based interactive analytics app framework |

**App structure (final)**

| File | Role |
|---|---|
| `streamlit_football_app.py` | Entrypoint — session state, page config, navigation |
| `page1.py` | Team Rosters — API call, flatten/filter, table display |
| `page2.py` | Team Stats — session-state reuse, nfl_data_py join, bar chart |
| `swc_simple_client.py` | Reused Chapter 9 API client (backoff, logging) |

**Analytics product types**

| Type | Format | Best for |
|---|---|---|
| Tabular report | Rows/columns, spreadsheet-like | Detailed inspection of exact values |
| Visualization | Charts, graphs, maps | Communicating patterns/context hard to see in raw data |

## Worked Example
After installing `streamlit`, `nfl_data_py`, and `matplotlib`, and launching the SWC API v0.2 in a separate terminal, the chapter copies `swc_simple_client.py` from Chapter 9 into a new `streamlit/` folder. It builds the entrypoint (`streamlit_football_app.py`) wiring up `st.session_state['base_url']` and a two-page navigation. `page1.py` ("Team Rosters") calls `LIST_TEAMS_ENDPOINT`, builds a DataFrame, exposes a league-ID select box, flattens nested `players` data via `json_normalize`, and displays a filtered roster table — storing the flattened DataFrame and unique league list in session state for reuse. `page2.py` ("Team Stats") reads that session state (no new API call), loads 2023 regular-season stats from `nfl_data_py`, computes a `total_tds` column, merges on `gsis_id`/`player_id`, groups by team, and renders a horizontal bar chart of total touchdowns per team via matplotlib + `st.pyplot()`. The chapter closes Part II by branching off a `learning-branch` backup, then flattening the repo structure (`notebooks/`, `airflow/`, `streamlit/`, `api/` at the root) and updating the README for the finished Part II portfolio project.

## Key Takeaways
1. Streamlit turns a plain Python script into an interactive, shareable web app with minimal boilerplate — ideal for showcasing data science work.
2. Structure multi-page Streamlit apps around one entrypoint (config + navigation) and separate page files, sharing state via `st.session_state` to avoid redundant API calls.
3. Reuse a previously built, standalone API client (from Chapter 9) rather than re-implementing API-calling logic inside the app — a direct payoff of separation of concerns.
4. Wrap every API call in explicit status-code checks and exception handling so the app degrades gracefully and logs errors instead of crashing.
5. A thoughtfully included external identifier (GSIS ID) in your API pays dividends later by enabling clean joins with outside datasets like `nfl_data_py`.
6. Visualizations (charts) and tabular displays serve different communication purposes — Streamlit supports both, and widgets make either interactive.
7. Free hosting (Streamlit Community Cloud) is available for sharing portfolio apps, with the trade-off that the app periodically sleeps.

## Connects To
- **Ch 1 / Ch 3**: The GSIS ID design decision ("standard external identifiers for data scientists") pays off directly as the join key enabling enrichment with `nfl_data_py`.
- **Ch 8**: Demonstrates the Separation of Concerns principle in practice by reusing a dedicated API client module across projects.
- **Ch 9**: Directly reuses the `swc_simple_client.py` file built in that chapter.
- **Ch 10**: Completes Part II's three consumption patterns (notebook analytics → pipeline → interactive app), all against the same SWC API v0.2.
- **External**: Streamlit official documentation and Cheat Sheet; *Streamlit for Data Science, 2nd Edition* (Tyler Richards); nflverse/`nfl_data_py` project.
