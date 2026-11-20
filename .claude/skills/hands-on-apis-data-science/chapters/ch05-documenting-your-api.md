# Chapter 5: Documenting Your API

## Core Idea
Great API documentation signals trust and improves developer experience (DX); FastAPI generates most of it for free via the OpenAPI Specification (OAS), and small code additions (titles, tags, summaries, descriptions, parameter docs) dramatically improve both human-facing docs (Swagger UI, Redoc) and machine/AI consumption of the API.

## Frameworks Introduced
- **Core + Extra Documentation Features Checklist**: a two-tier model for what API docs need.
  - When to use: scoping documentation effort for any new or maturing API.
  - How: ship the **core** features first (getting started, endpoint definitions, OAS file, terms of service, feedback mechanism, SDKs) since an API won't get adopted without them; add **extra** features (sample code, interactive docs, sandbox environment, Postman collection) as resources allow, to reduce "time to hello world" (TTHW).
- **Trust Signals for Repeat Integrations**: documentation as a trust-building surface, not just a reference.
  - When to use: whenever consumers will build a long-lived integration (scheduled ETL, embedded product feature) rather than a one-off call.
  - How: keep release history/versioning current, respond promptly to feedback, and for data-oriented consumers specifically state data quality and update-frequency commitments (distinct from uptime/support commitments for integration-builders).
- **OAS-Driven Documentation Enrichment**: incrementally improve the auto-generated OpenAPI spec via FastAPI code annotations rather than hand-writing docs.
  - When to use: once a baseline FastAPI app exists and default Swagger UI/Redoc output is too generic.
  - How: add `description`/`title`/`version` to the `FastAPI()` constructor (OAS `info`); add `tags=[...]` to each route decorator to group endpoints; add `summary`, `description`, `response_description`, and a custom `operation_id` to each route; wrap each function parameter in `Query(default, description="...")` to document query parameters — all of this flows automatically into `/openapi.json`, `/docs` (Swagger UI), and `/redoc`.

## Key Concepts
- **Developer Experience (DX)**: the overall ease and pleasantness of consuming an API/SDK — documentation, SDKs, and interactive tooling are all levers for DX.
- **Time to Hello World (TTHW)**: a metric for how quickly a new consumer can get a basic working call against the API; lower is better, and extra documentation features exist largely to reduce it.
- **OpenAPI Specification (OAS)**: a machine-readable (JSON/YAML) standard describing an API's paths, parameters, and schemas; FastAPI auto-generates it at `/openapi.json`; it is a general-purpose artifact consumed by far more tools than just documentation generators (SDK generators, mock servers, validators, and later in the book, a ChatGPT custom action).
- **Swagger UI**: interactive documentation UI (SmartBear-led open source project) bundled into FastAPI at `/docs`; renders the OAS and lets users execute real test calls ("Try it out" → "Execute") against the live API, showing the constructed curl command, request URL, response body, and headers.
- **Redoc**: a second documentation UI (Redocly-led) bundled into FastAPI at `/redoc`; renders the same OAS as a three-column reference layout but is not interactive (no request execution).
- **Operation object / path object**: in the OAS JSON structure, `paths` maps URL paths to HTTP-verb keyed operation objects (`get`, `post`, ...), each with `summary`, `operationId`, `parameters`, and `responses`.
- **Tags**: free-form labels on an OAS operation used by Swagger UI to group related endpoints into visual sections (e.g., "player," "scoring," "membership," "analytics").
- **operationId**: a unique machine-readable identifier for an operation, used by tooling (SDK generators, AI agents) beyond what's shown in the human UI.
- **Sandbox environment vs. interactive documentation**: interactive docs let you fire test calls but don't persist state between them; a sandbox environment persists state across calls (e.g., create an order, then retrieve it).
- **Arazzo Specification**: an emerging OpenAPI Initiative standard (mentioned as context) for describing workflows/sequences of API calls, relevant to LLM-based applications that need to chain calls.

## Mental Models
- **Documentation is generated, not hand-written, wherever possible**: because FastAPI/Pydantic derive the OAS directly from your typed code, the best "documentation investment" is often just better code annotations (descriptions, tags, response models) rather than a separate hand-maintained doc site.
- **Different consumers ask different trust questions**: integration-builders ask "will this API still work for me in a year?"; data scientists doing one-time loads ask "can I trust your data's quality and definitions?" — documentation should answer both.
- **Docs as an AI-consumption surface**: detailed `description`, `response_description` (kept under ~300 characters "to make ChatGPT happy"), and parameter `Query()` descriptions are written not just for humans but explicitly to help LLM/AI tools correctly call the API (foreshadowing Part III).

## Anti-patterns
- **Stale, "ghost town" documentation**: outdated blog posts, unanswered message boards, or out-of-date roadmaps signal abandonment and drive users away.
- **Generic auto-generated info/titles left unedited**: a default `"title": "FastAPI", "version": "0.1.0"` tells consumers nothing about the API's purpose.
- **Undocumented or undescribed parameters**: omitting `Query(..., description=...)` leaves both humans and AI agents guessing about what a parameter means or expects.
- **Treating interactive docs as a sandbox**: assuming test calls persist state between calls when using plain interactive documentation (they don't — that requires a true sandbox).

## Code Examples
```python
# main.py — enriching the OAS info object
api_description = """
This API provides read-only access to info from the SportsWorldCentral
(SWC) Fantasy Football API.
The endpoints are grouped into the following categories:

## Analytics
Get information about the health of the API and counts of leagues, teams, and players.
"""

app = FastAPI(
    description=api_description,
    title="Sports World Central (SWC) Fantasy Football API",
    version="0.1"
)
```
- **What it demonstrates**: populating the OAS `info` object (title, description, version) from the FastAPI constructor — the top-level metadata shown by both Swagger UI and Redoc.

```python
# Tagging and documenting an individual endpoint
@app.get("/v0/players/{player_id}",
        response_model=schemas.Player,
        summary="Get one player using the Player ID, which is internal to SWC",
        description="If you have an SWC Player ID of a player from another API "
                     "call such as v0_get_players, you can call this API "
                     "using the player ID",
        response_description="One NFL player",
        operation_id="v0_get_players_by_player_id",
        tags=["player"])
```
- **What it demonstrates**: adding `summary`, `description`, `response_description`, a custom `operation_id`, and a `tags` grouping to a route decorator, all of which flow into the generated OAS and the Swagger UI display.

```python
def read_players(
    skip: int = Query(0, description="The number of items to skip at the beginning of API call."),
    limit: int = Query(100, description="The number of records to return after the skipped records."),
    minimum_last_changed_date: date = Query(None, description="The minimum date of change that you want to return records. Exclude any records changed before this."),
    first_name: str = Query(None, description="The first name of the players to return"),
    last_name: str = Query(None, description="The last name of the players to return"),
    db: Session = Depends(get_db)):
```
- **What it demonstrates**: replacing bare default values with `Query(default, description=...)` so every query parameter carries documentation visible in the OAS/Swagger UI and usable by AI tooling.

```json
{
    "openapi": "3.1.0",
    "info": {"title": "FastAPI", "version": "0.1.0"},
    "paths": {
        "/": {},
        "/v0/players/": {},
        "/v0/players/{player_id}": {}
    },
    "components": {
        "schemas": {"Counts": {}, "League": {}, "Player": {}, "PlayerBase": {}}
    }
}
```
- **What it demonstrates**: the top-level structure of the auto-generated `openapi.json` — `info` (metadata), `paths` (endpoints), `components.schemas` (reusable Pydantic-derived data shapes referenced via `$ref`).

## Reference Tables
**Table 5-1. Updates made to the OAS file via main.py**

| Update | Change in main.py | OAS field affected |
|---|---|---|
| Add API title, version, description | Elements in `FastAPI()` constructor | `info` |
| Add path summaries | Parameter on path decorator | `paths` |
| Add detailed path descriptions | Parameter on path decorator | `paths` |
| Add path tags to group endpoints | Parameter on path decorator | `paths` |
| Add unique path operation IDs | Update built-in operation IDs | `paths` |
| Add description to query parameters | `Query()` wrapper on function params | `parameters` |

**Table 5-2. Tags for endpoints in main.py**

| Endpoint | Tag |
|---|---|
| / | analytics |
| /v0/players/ | player |
| /v0/players/{player_id} | player |
| /v0/performances/ | scoring |
| /v0/leagues/ | membership |
| /v0/leagues/{league_id} | membership |
| /v0/teams/ | membership |
| /v0/counts/ | analytics |

**Core vs. Extra documentation features**

| Core (required) | Extra (improves DX) |
|---|---|
| Getting started | Sample program code |
| Endpoint definitions | Interactive documentation |
| OAS file | Sandbox environment |
| Terms of service | Postman collection |
| Feedback mechanism | — |
| SDKs | — |

## Worked Example
Starting from the Chapter 4 FastAPI app, the chapter tours real fantasy-sports API docs (Sleeper's getting-started/endpoint pages, MyFantasyLeague's versioned welcome page and interactive docs, Yahoo!'s sample PHP code) to illustrate core/extra features in the wild. It then opens the book's own API's built-in `/docs` (Swagger UI) and `/redoc`, walks through testing the `GET /v0/players/{player_id}` endpoint interactively (missing `player_id` → 422 validation error; `player_id=1385` → 200 with full `Player` JSON including nested `performances`), and inspects the raw `/openapi.json`. It then modifies `main.py` to add an `api_description`, title, and version to the `FastAPI()` constructor; adds `tags` per Table 5-2; adds `summary`/`description`/`response_description`/`operation_id` to each endpoint; and wraps each query parameter in `Query(default, description=...)`. After regression-testing with `pytest test_main.py` (still 11/11 passing), it updates `README.md` with a getting-started section, terms of service (2,000 requests/day, no warranty), and example Python/HTTPX code for the health-check call.

## Key Takeaways
1. Documentation's job is partly functional (how to call the API) and partly trust-building (will this API still work for me later; can I trust this data).
2. Prioritize core documentation features (getting started, endpoint definitions, OAS, terms of service, feedback channel, SDK) before investing in extras (sandbox, Postman collections).
3. FastAPI/Pydantic generate the OpenAPI spec automatically — the highest-leverage documentation work is annotating your code (`FastAPI()` constructor fields, route decorator fields, `Query()` parameter descriptions), not hand-writing separate docs.
4. Swagger UI (`/docs`) is interactive (can execute real calls); Redoc (`/redoc`) is a non-interactive but often more attractive three-column reference — both are generated from the same `/openapi.json`.
5. Tags, summaries, descriptions, and operation IDs all serve dual audiences: human developers reading Swagger UI/Redoc, and tools/AI agents parsing the OAS programmatically.
6. Keep response/parameter descriptions concise (e.g., under ~300 characters) since downstream consumers like ChatGPT-based tools have practical limits.
7. Regression-test (`pytest`) after any documentation-driving code change to confirm functional behavior wasn't broken.

## Connects To
- **Ch 1**: Directly serves the "data scientists should be able to trust and use your API" design goals via explicit trust signals and data-quality statements.
- **Ch 4**: All documentation enrichment happens inside the same `main.py` FastAPI routes and `schemas.py` Pydantic classes built in that chapter.
- **Ch 7**: README's "Example Code"/SDK placeholder sections point directly at the Python SDK built in Chapter 7.
- **Ch 15 (Part III)**: The OAS file produced/enriched here is later used as the custom action definition for a ChatGPT integration.
- **External**: *Designing Web APIs* (Jin, Sahni, Shevat); *Principles of Web API Design* (James Higginbotham); Speakeasy's OpenAPI Guide; OpenAPI Initiative's Arazzo specification; OpenApi.Tools directory.
