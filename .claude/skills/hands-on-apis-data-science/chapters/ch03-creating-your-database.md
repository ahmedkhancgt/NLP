# Chapter 3: Creating Your Database

## Core Idea
Build the data layer for the API: design a relational schema in SQLite, map it to Python with SQLAlchemy ORM models and CRUD helper functions, then unit-test everything with pytest — establishing a read-only ("R" of CRUD) data layer that will be wrapped by FastAPI in Chapter 4.

## Frameworks Introduced
- **ORM Mapping Pattern (SQLAlchemy)**: represent database tables as Python classes to avoid raw SQL and manual type conversion.
  - When to use: whenever a Python app needs to query a relational database repeatedly and safely.
  - How: define a `Base` declarative class (`database.py`), subclass it per table (`models.py`) with `Column()` fields and `relationship()` links, then query through `Session` objects using helper (CRUD) functions rather than inline SQL strings.
- **Parent/Child/Association Table Pattern**: model one-to-many and many-to-many relationships.
  - When to use: entities naturally nest (league→team) or cross-reference (team↔player).
  - How: one-to-many via a `FOREIGN KEY` on the child + `relationship(..., back_populates=...)` on both sides; many-to-many via an association table with two foreign keys as a composite primary key, referenced with `relationship(..., secondary="assoc_table", back_populates=...)`.
- **Pagination via skip/limit**: standard way to return data in page-sized chunks.
  - When to use: any query that could return large result sets.
  - How: `query.offset(skip).limit(limit).all()`, with `skip` defaulting to 0 and `limit` to 100; caller increments `skip` by `limit` for successive pages.
- **Delta-query filtering**: implement the Chapter 1 "query by last-changed date" recommendation.
  - How: add a `last_changed_date` column to every table; helper functions accept an optional `min_last_changed_date` parameter and apply `.filter(Model.last_changed_date >= min_last_changed_date)`.
- **Count/analytics endpoints as a separate query type**: expose record counts distinctly from paginated row data.
  - When to use: supporting dashboards, pagination planning, or (as flagged here) AI/LLM agents that need to know dataset size without pulling full data.
  - How: simple `query.count()` helper functions (`get_player_count`, `get_team_count`, `get_league_count`).

## Key Concepts
- **SQLite**: lightweight, file-based relational database; good for prototypes/learning projects since the whole DB can live in a Git repo; supports full SQL and SQLAlchemy.
- **ORM (Object-Relational Mapping)**: library (SQLAlchemy) that maps DB rows to Python objects, handles type conversion, supports multiple backend DBs, and uses prepared statements to prevent SQL injection.
- **SQL injection**: a vulnerability where malicious input is interpreted as SQL code; mitigated by prepared/parameterized statements (which ORMs generate automatically) instead of raw string-built SQL.
- **Primary key / foreign key**: primary key uniquely identifies a row and enables joins; foreign key in a child table references a parent table's primary key, enforcing referential integrity (`PRAGMA foreign_keys = ON` in SQLite).
- **Association table**: a table with no business data of its own (or minimal), existing purely to implement a many-to-many relationship (here, `team_player`), modeled as a composite primary key of both foreign keys.
- **GSIS ID**: the NFL's Game Statistics and Information System identifier, included as a nullable external identifier so data scientists can join SWC data with other NFL data sources (direct application of Ch. 1's "standard external identifiers" tip).
- **Session (SQLAlchemy)**: object that manages the live conversation with the database; created via `sessionmaker(bind=engine)` and used/closed per request or test.
- **Eager loading (`joinedload`)**: SQLAlchemy option to fetch related rows (e.g., a league's teams) in the same query instead of lazily on access.
- **pytest fixture**: reusable setup/teardown logic (`@pytest.fixture`) — here, a `db_session` fixture that opens a session, `yield`s it to the test, then closes it afterward.
- **Regression testing**: rerunning the test suite after a library/version upgrade to confirm nothing broke; pairs with virtual environments (e.g., Codespaces) to safely test upgrades.

## Mental Models
- **CRUD, but only "R" so far**: the file is named `crud.py` by convention, but since all user stories are read-only, only Read functions exist — a placeholder structure for future Create/Update/Delete if ever needed.
- **requirements.txt as a reproducibility contract**: pinning library versions (`SQLAlchemy>=2.0.0`) ensures consistent environments and gives a clear point for regression-testing upgrades.
- **Tests as a safety net for schema + library changes**: pytest assertions that check known row counts act as a tripwire for data-loading or library-upgrade regressions.

## Anti-patterns
- **Hand-rolled raw SQL string queries from user input**: exposes SQL injection risk and forces manual type mapping — avoided entirely by using SQLAlchemy's parameterized query builder.
- **No pagination on large tables**: returning all rows unconditionally risks timeouts/memory issues (ties back to Ch. 1's bulk-download/pagination guidance).
- **Skipping `last_changed_date` population**: without it, delta/incremental pipeline queries (Ch. 1, Ch. 10) become impossible.
- **Importing an entire library wholesale** (`import sqlalchemy` instead of specific names): increases risk of naming conflicts across libraries; the book explicitly imports only needed names.

## Code Examples
```sql
CREATE TABLE player (
        player_id INTEGER NOT NULL,
        gsis_id VARCHAR,
        first_name VARCHAR NOT NULL,
        last_name VARCHAR NOT NULL,
        position VARCHAR NOT NULL,
        last_changed_date DATE NOT NULL,
        PRIMARY KEY (player_id)
);

CREATE TABLE team_player (
        team_id INTEGER NOT NULL,
        player_id INTEGER NOT NULL,
        last_changed_date DATE NOT NULL,
        PRIMARY KEY (team_id, player_id),
        FOREIGN KEY(team_id) REFERENCES team (team_id),
        FOREIGN KEY(player_id) REFERENCES player (player_id)
);
```
- **What it demonstrates**: a nullable external-identifier column (`gsis_id`) for joining with outside data, and an association table implementing a many-to-many relationship via a composite primary key.

```python
# models.py (SQLAlchemy ORM classes)
class Player(Base):
    __tablename__ = "player"
    player_id = Column(Integer, primary_key=True, index=True)
    gsis_id = Column(String, nullable=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    position = Column(String, nullable=False)
    last_changed_date = Column(Date, nullable=False)

    performances = relationship("Performance", back_populates="player")
    teams = relationship("Team", secondary="team_player", back_populates="players")

class Performance(Base):
    __tablename__ = "performance"
    performance_id = Column(Integer, primary_key=True, index=True)
    week_number = Column(String, nullable=False)
    fantasy_points = Column(Float, nullable=False)
    last_changed_date = Column(Date, nullable=False)
    player_id = Column(Integer, ForeignKey("player.player_id"))
    player = relationship("Player", back_populates="performances")
```
- **What it demonstrates**: declaring an ORM model with typed columns, a one-to-many relationship (`Performance.player`/`Player.performances`), and a many-to-many relationship through an association table.

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./fantasy_data.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```
- **What it demonstrates**: wiring SQLAlchemy to a SQLite file and creating the reusable `Base`/`SessionLocal` objects models and CRUD functions depend on.

```python
# crud.py
def get_players(db: Session, skip: int = 0, limit: int = 100,
               min_last_changed_date: date = None,
               last_name: str = None, first_name: str = None):
   query = db.query(models.Player)
   if min_last_changed_date:
       query = query.filter(models.Player.last_changed_date >= min_last_changed_date)
   if first_name:
       query = query.filter(models.Player.first_name == first_name)
   if last_name:
       query = query.filter(models.Player.last_name == last_name)
   return query.offset(skip).limit(limit).all()

def get_leagues(db: Session, skip: int = 0, limit: int = 100,
                min_last_changed_date: date = None, league_name: str = None):
   query = db.query(models.League).options(joinedload(models.League.teams))
   ...
   return query.offset(skip).limit(limit).all()

def get_player_count(db: Session):
   return db.query(models.Player).count()
```
- **What it demonstrates**: optional-filter query construction, pagination, eager-loading a related collection, and a lightweight count-only "analytics" query for AI/agent consumers.

```python
# test_crud.py
@pytest.fixture(scope="function")
def db_session():
    session = SessionLocal()
    yield session
    session.close()

def test_get_players(db_session):
    players = crud.get_players(db_session, skip=0, limit=10000, min_last_changed_date=test_date)
    assert len(players) == 1018
```
- **What it demonstrates**: a pytest fixture providing setup/teardown for a DB session, and an assertion-based unit test validating a CRUD helper's row count against a known value.

## Reference Tables
**Table 3-2. Purpose of the Chapter 3 files**

| File | Purpose |
|---|---|
| `crud.py` | Helper functions to query the database |
| `database.py` | Configures SQLAlchemy to use the SQLite database |
| `models.py` | Defines the SQLAlchemy classes related to the database tables |
| `requirements.txt` | Pins library versions for pip install |
| `test_crud.py` | pytest unit tests for the SQLAlchemy/CRUD layer |

**Table structure**: `player` (parent) → `performance` (child, FK to player); `league` (parent) → `team` (child, FK to league) → `team_player` (association table, FK to both `team` and `player`, composite PK).

## Worked Example
Five SQLite tables (`player`, `performance`, `league`, `team`, `team_player`) are created via raw SQL, loaded from CSV with `.import --skip 1`, and verified with row counts (1,018 players; 17,306 performances; 5 leagues; 20 teams; 140 team_player rows). Foreign-key enforcement is turned on with `PRAGMA foreign_keys = ON`. The Python layer mirrors this exactly: `models.py` defines `Player`, `Performance`, `League`, `Team`, `TeamPlayer` ORM classes with matching relationships; `database.py` wires up the SQLite engine and session factory; `crud.py` provides parameterized read functions (`get_player`, `get_players`, `get_performances`, `get_league`, `get_leagues`, `get_teams`, plus count functions). `test_crud.py` then unit-tests these functions against the known row counts, confirming e.g. that filtering `performance` by `last_changed_date > 2024-04-01` returns exactly 2,711 of the 17,306 total rows — validating both the delta-query logic and the ORM mapping end to end.

## Key Takeaways
1. Use an ORM (SQLAlchemy) rather than raw SQL strings to get automatic type mapping, multi-database portability, and built-in protection against SQL injection via prepared statements.
2. Model one-to-many relationships with a foreign key + `relationship(back_populates=...)`, and many-to-many relationships with an association table plus `secondary=` relationships.
3. Bake pagination (`skip`/`limit`) and delta filtering (`min_last_changed_date`) into every list-returning query from the start — these are the data-layer implementation of Chapter 1's API design tips.
4. Include nullable external identifiers (like GSIS ID) specifically so downstream data scientists can join your data with other sources.
5. Expose lightweight count-only queries separately from full-row queries — useful for pagination planning and essential for AI/LLM agents that shouldn't pull large payloads just to learn a size.
6. Pin dependency versions in `requirements.txt` and validate upgrades with pytest regression tests run inside a fresh virtual environment (e.g., a new Codespace).

## Connects To
- **Ch 1**: Implements the "bulk download," "standard external identifiers," "data type conformance," and "query by last-changed date" API design tips directly in the schema and CRUD layer.
- **Ch 2**: Realizes the read-replica/SQLite component of the technology architecture diagram (Figure 3-1 maps directly onto Figure 2-2's components).
- **Ch 4**: FastAPI endpoints in the next chapter call these exact `crud.py` functions and serialize the SQLAlchemy models via Pydantic.
- **Ch 10**: The `last_changed_date` delta pattern is the foundation for incremental data-pipeline loads.
- **External**: *Learning SQL* (Alan Beaulieu); *SQL Pocket Guide* (Alice Zhao); OWASP's SQL injection reference; SQLAlchemy 2 documentation.
