# Chapter 4: Developing the FastAPI Code

## Core Idea
Wrap the Chapter 3 database layer with Pydantic DTO schemas and a FastAPI controller to produce a working, versioned, validated, testable REST API with minimal code, then verify it with pytest's `TestClient` and a live Uvicorn server.

## Frameworks Introduced
- **DTO (Data Transfer Object) pattern via Pydantic schemas**: decouple the API's wire format from the database's internal model.
  - When to use: whenever an API's response shape should be controlled independently of the ORM/database schema (hide fields, reshape relationships, add computed fields).
  - How: define `pydantic.BaseModel` subclasses (`schemas.py`) separate from SQLAlchemy models (`models.py`); use `ConfigDict(from_attributes=True)` so Pydantic can read directly from ORM object attributes; FastAPI auto-serializes the Pydantic object to JSON.
- **Primary/Secondary (Base) Schema Split**: avoid over-fetching nested data.
  - When to use: an entity (e.g., Player) is sometimes needed in full (with nested children) and sometimes just as a lightweight reference inside a parent object.
  - How: define a `XBase` schema with the entity's own fields only, then a full `X(XBase)` subclass that adds the nested list of children; use `XBase` wherever the entity appears as a secondary/nested field, and `X` wherever it's the primary resource returned.
- **FastAPI Dependency Injection for DB sessions**: manage resource lifecycle per-request.
  - When to use: any endpoint needing a database (or other resource) handle that must open before and close after the request.
  - How: define `get_db()` as a generator function (`yield db` inside try/finally), and declare route parameters as `db: Session = Depends(get_db)`; FastAPI calls it, hands the session to the route, and ensures cleanup.
- **Route = decorator + typed function signature**: FastAPI's core convention for defining endpoints.
  - How: `@app.get(url, response_model=...)` above a function; path parameters appear as `{name}` in the URL and as same-named function arguments; query parameters are any other typed function arguments with defaults; FastAPI auto-validates types, builds the OpenAPI spec, and serializes the return value per `response_model`.

## Key Concepts
- **FastAPI**: Python web framework (Sebastián Ramírez Montaño, 2018) purpose-built for APIs — handles HTTP plumbing, auto-generates an OpenAPI spec, provides interactive docs, supports versioning/security; still pre-1.0 (0.x) so breaking changes are possible per semantic versioning.
- **Pydantic**: data validation library (v2 core implemented in Rust for speed) used by FastAPI both to validate inbound/outbound data and to generate JSON Schema (feeding the OpenAPI spec); distinct from "SQLAlchemy models" — the book calls these "Pydantic schemas" to avoid the overloaded term "model."
- **Uvicorn**: ASGI web server that actually runs the FastAPI app and handles sync/async request processing.
- **HTTPX**: async-capable HTTP client, used by pytest to exercise FastAPI apps and later (Ch. 7) to build the SDK; contrasted with the synchronous-only `requests` library.
- **Serialization / deserialization**: converting Python objects to JSON (serialization, done by Pydantic on the way out) and JSON back to validated Python objects (deserialization, on the way in).
- **Query parameter vs. path parameter**: path parameters are embedded in the URL path (`/v0/players/{player_id}`) and mapped to same-named function args; query parameters are appended after `?` and `&` (`?skip=0&limit=100`) and correspond to other typed function args with defaults.
- **HTTPException / HTTP status codes**: FastAPI's standard mechanism for returning RESTful error responses (e.g., `404 Not Found`) instead of ad hoc error payloads.
- **TestClient**: FastAPI/Starlette test utility that lets pytest call API routes in-process without a running web server.
- **API versioning via URL prefix**: endpoints are namespaced `/v0/...` to signal pre-1.0 instability to consumers, per Chapter 2's `api/version/endpoint` model.

## Mental Models
- **Three-layer separation**: SQLAlchemy models (storage) → Pydantic schemas (wire contract/DTO) → FastAPI routes (orchestration). Each layer has a single responsibility, and the "model" naming overload between SQLAlchemy and Pydantic is explicitly called out as a source of confusion to avoid.
- **"A lot with a few lines"**: FastAPI + Pydantic push most plumbing (validation, serialization, docs generation, dependency injection) into the framework, leaving developer code focused on business logic.
- **AI-friendly design carried into endpoints**: the name-based player search (`first_name`/`last_name` without needing a key) and the `/v0/counts/` endpoint exist specifically to make the API easier for LLM/AI agents to query without needing IDs or large payloads — a direct callback to Chapter 1's AI-oriented design guidance.

## Anti-patterns
- **Conflating SQLAlchemy models with Pydantic schemas (terminology and code)**: both ecosystems use the word "model," but mixing them causes confusion — the book deliberately separates `models.py` (SQLAlchemy) from `schemas.py` (Pydantic).
- **Returning full nested graphs everywhere**: using the full `Player` schema (with all its `Performance` history) as a nested field inside `Team` would over-fetch; the `PlayerBase`/`Player` split avoids this.
- **Using `=` instead of `:` in Pydantic field definitions**: a noted common mistake — Pydantic schema fields use `field_name: type`, not SQLAlchemy's `Column(...)` assignment style.
- **Returning raw `None`/crashing on missing records**: instead, explicit `HTTPException(status_code=404, ...)` gives consumers a predictable, standard RESTful error contract.

## Code Examples
```python
# schemas.py — primary/secondary schema split
class PlayerBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    player_id: int
    gsis_id: str
    first_name: str
    last_name: str
    position: str
    last_changed_date: date

class Player(PlayerBase):
    model_config = ConfigDict(from_attributes=True)
    performances: List[Performance] = []

class Counts(BaseModel):
    league_count: int
    team_count: int
    player_count: int
```
- **What it demonstrates**: `PlayerBase` (lightweight, used nested inside `Team`) vs. `Player` (full, used as primary endpoint response with nested `Performance` list); `Counts` is a schema with no backing table, used purely for the analytics endpoint.

```python
# main.py — FastAPI app, DB dependency, and representative routes
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    return {"message": "API health check successful"}

@app.get("/v0/players/{player_id}", response_model=schemas.Player)
def read_player(player_id: int, db: Session = Depends(get_db)):
    player = crud.get_player(db, player_id=player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return player

@app.get("/v0/counts/", response_model=schemas.Counts)
def get_count(db: Session = Depends(get_db)):
    counts = schemas.Counts(
        league_count=crud.get_league_count(db),
        team_count=crud.get_team_count(db),
        player_count=crud.get_player_count(db))
    return counts
```
- **What it demonstrates**: health-check route, dependency-injected DB session, a path-parameter route with a 404 `HTTPException`, and a schema-only analytics endpoint with no SQLAlchemy backing.

```python
# test_main.py
from fastapi.testclient import TestClient
from main import app
client = TestClient(app)

def test_read_players_by_name():
    response = client.get("/v0/players/?first_name=Bryce&last_name=Young")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0].get("player_id") == 2009
```
- **What it demonstrates**: testing a running FastAPI app in-process via `TestClient`, asserting HTTP status code and JSON body shape/content without a live server.

```bash
$ fastapi run main.py
# ...
$ curl .../v0/performances/?skip=0&limit=1
[{"performance_id":2501,"player_id":1001,"week_number":"202301","fantasy_points":20.0,"last_changed_date":"2024-03-01"}]
```
- **What it demonstrates**: launching the API with Uvicorn via the FastAPI CLI and confirming a live, paginated JSON response.

## Reference Tables
**Table 4-1. Endpoints for the SWC Fantasy Football API**

| Description | Verb | URL |
|---|---|---|
| API health check | GET | / |
| Read player list | GET | /v0/players/ |
| Read individual player | GET | /v0/players/{player_id}/ |
| Read performance list | GET | /v0/performances/ |
| Read league list | GET | /v0/leagues/ |
| Read individual league | GET | /v0/leagues/{league_id}/ |
| Read team list | GET | /v0/teams/ |
| Read counts | GET | /v0/counts/ |

**Table 4-4. Mapping of schemas to endpoints**

| Endpoint | Primary schema | Secondary schema |
|---|---|---|
| / | None | None |
| /v0/players/ | Player | Performance |
| /v0/players/{player_id}/ | Player | Performance |
| /v0/performances/ | Performance | None |
| /v0/leagues/ | League | TeamBase |
| /v0/leagues/{league_id} | League | TeamBase |
| /v0/teams/ | Team | PlayerBase |
| /v0/counts/ | Counts | None |

## Worked Example
Building on Chapter 3's database layer, `schemas.py` defines six Pydantic classes (`Performance`, `PlayerBase`/`Player`, `TeamBase`/`Team`, `League`, `Counts`), each wired to SQLAlchemy via `ConfigDict(from_attributes=True)` except the DB-less `Counts`. `main.py` creates a `FastAPI()` app, a `get_db()` dependency, and eight routes matching Table 4-1, each calling the matching `crud.py` helper from Chapter 3 and declaring a `response_model` for automatic validation/serialization. `test_main.py` uses `TestClient` to run 11 tests asserting exact row counts (1018 players, 17306 performances, 5 leagues, 20 teams, 2711 records after a last-changed-date filter) — mirroring the SQL/pytest checks from Chapter 3 but now through the live HTTP-shaped API surface. Finally, `fastapi run main.py` launches the app on Uvicorn; in GitHub Codespaces the forwarded port opens a browser tab confirming the health check (`{"message":"API health check successful"}`) and a sample data call.

## Key Takeaways
1. Keep ORM models (storage) and Pydantic schemas (wire format) as separate layers — this is the DTO pattern, and it buys independence between your database structure and your API contract.
2. Split entities into a lightweight `XBase` schema and a full `X` schema to control how much nested data is returned depending on context (primary resource vs. nested reference).
3. Use FastAPI's `Depends()` for per-request resource lifecycle management (DB sessions open/close cleanly even on errors, via try/finally with yield).
4. Prefer path parameters for identifying a specific resource and query parameters for filtering/pagination options; let FastAPI's type-annotated function signatures drive both validation and OpenAPI generation automatically.
5. Return standard HTTP status codes (e.g., `404` via `HTTPException`) rather than ad hoc error shapes, for predictable RESTful behavior.
6. Test FastAPI apps in-process with `TestClient` before ever running a live server — faster and more reliable for CI/regression testing.
7. Name unstable/early APIs with a `v0` prefix to signal to consumers that breaking changes may still occur.

## Connects To
- **Ch 3**: Every FastAPI route is a thin wrapper around a `crud.py` function and a SQLAlchemy model defined in that chapter.
- **Ch 2**: Implements the `api/version/endpoint` structure and REST choice decided there.
- **Ch 5**: FastAPI's auto-generated OpenAPI spec (enabled by these Pydantic schemas) is the documentation artifact built out next.
- **Ch 7**: HTTPX, used here for testing, becomes the HTTP client underlying the Python SDK.
- **External**: *FastAPI: Modern Python Web Development* (Bill Lubanovic); official FastAPI/Pydantic/Uvicorn documentation; Marcelo Trylesinski's FastAPI Tips.
