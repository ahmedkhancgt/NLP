# Chapter 7: Implementing Data Contracts

## Core Idea
A working data contract implementation is just four wired-together pieces — a data asset (Postgres + migrations), a contract spec (JSON Schema file), a detection layer (metadata comparison scripts), and a prevention layer (unit tests run in CI/CD) — and the chapter proves this end-to-end with a museum-app mock scenario, including a full violation-and-resolution cycle.

## Frameworks Introduced
- **Reference Implementation Directory Structure**: `data_contract_components/{data_assets, contract_definition, detection, prevention}/` mapped 1:1 onto the four architecture components.
  - When to use: scaffolding your own first implementation.
  - How: data_assets holds DB migrations (Alembic) and seed scripts; contract_definition holds the JSON Schema spec files; detection holds scripts that build a "data catalog" from `information_schema` and compare it to specs; prevention holds the unit tests that CI/CD runs.
- **Metadata-Only Detection Principle**: contract enforcement should only need metadata (schema/column info), never raw data values.
  - When to use: designing detection scripts, especially where cross-team/security sensitivity matters.
  - How: query `information_schema` (or a real data catalog) for table/column metadata; never read actual row values to check a contract.
- **Violation Resolution Loop**: trigger violation → read owner contact from spec → consult (not just notify) downstream team → negotiate a transition plan → create a *new* contract for the new asset rather than mutating the old one destructively → fix migration until tests pass → push and merge.
  - When to use: handling any real contract violation surfaced in CI/CD.
  - How: follow the mock scenario below step by step — it is the canonical walkthrough.

## Key Concepts
- **Data catalog (DIY version in this chapter)**: a SQL script wrapper (`_get_data_catalog.py`) reading Postgres's `information_schema` table — a minimal stand-in for a real data catalog tool.
- **Contract coverage detector**: checks whether every asset *in* the contract spec directory is also *present* in the data catalog (left-join, flags assets in contract but missing from catalog).
- **Contract violation detector**: joins contract spec metadata to catalog metadata on `table_catalog`, `table_schema`, `table_name`, and `column_name` to find per-column mismatches.
- **Alembic**: the database migration tool used to version-control and apply schema changes to Postgres (`alembic downgrade base` / `alembic upgrade head`).
- **Open Data Contract Standard (ODCS)**: a Linux Foundation open source data contract spec (with a JSON Schema adaptation) recommended over building your own unless requirements are simple or company-specific.
- **GitHub Actions `test` job**: the CI/CD job that spins up Docker Compose (Postgres + devcontainer), runs migrations, runs the unit test suite, and posts failures as a PR comment.

## Mental Models
- Think of the data contract spec as *metadata-only by design* — this is what makes contracts cross-team-safe (backend, data platform, and data science can all participate without raw-data access concerns).
- Treat "current state becomes contract baseline" as the default bootstrapping approach, but recognize the alternative: use the contract to define a *desired* future state and let it drive the migration, as happens later in the chapter with `object_images_normalized`.
- Use the violation message as the entry point into conversation, not just a blocker — the real fix in the worked example wasn't a database change, it was discovering the data science team's dependency and negotiating a transition window.
- View building your own contract spec / data catalog as a *learning* exercise; for production, prefer existing open source specs (ODCS) and real data catalog tools — the DIY versions here trade robustness for pedagogical simplicity.

## Anti-patterns
- **Dropping/renaming a table because "no one downstream is connected yet"**: the engineer in the scenario almost did this, unaware the data science team had already started model training against `object_images` — exactly the kind of silent-breakage data contracts are meant to catch.
- **Treating a contract violation as purely a code-fix task**: the correct response was to consult the owner listed in the spec and negotiate (keep the old table, add a new one, give a one-month transition window) — not just patch the migration to satisfy the test.
- **Building bespoke data catalog / spec tooling for production use**: reasonable for learning, but the chapter explicitly recommends real data catalog tools and the Open Data Contract Standard once you move past a handful of contracts, since cross-asset comparison logic (e.g., `transform_contract_specs_to_catalog_format()`) gets complex fast.
- **Comparing raw data instead of metadata**: increases security/access scope unnecessarily; the whole design keeps detection at the metadata layer.

## Code Examples
```json
{
    "spec-version": "1.0.0",
    "name": "object-images-normalized-contract-spec",
    "namespace": "met-museum-data",
    "dataAssetResourceName": "postgresql://postgres:5432/postgres.object_images_normalized",
    "doc": "Data contract for the object_images_normalized table...",
    "owner": { "name": "Data Engineering Team", "email": "data-eng@museum.org" },
    "schema": {
      "table_name": "object_images_normalized",
      "properties": {
        "object_id": { "constraints": { "primaryKey": true, "data_type": "integer", "is_nullable": false } },
        "image": { "constraints": { "data_type": "text", "is_nullable": true } },
        "is_primary_image": { "constraints": { "data_type": "boolean", "is_nullable": true } },
        "created_at": { "constraints": { "data_type": "timestamp without time zone", "is_nullable": false } }
      }
    }
}
```
- **What it demonstrates**: a new contract spec created for a *new, normalized* table (unnesting `additional_images`, renaming `primary_image`→`image`, adding `is_primary_image` boolean) rather than mutating the old contract destructively.

```python
# contract_coverage_detector.py
def detect_coverage_in_data_catalog(self) -> List[str]:
    coverage = self.get_contract_spec_coverage()
    coverage_df = pd.DataFrame(coverage)
    catalog_df = get_data_catalog()
    merged = coverage_df.merge(
        catalog_df, on=['table_catalog', 'table_schema', 'table_name'],
        how='left', indicator=True
    )
    missing_assets_df = merged[merged['_merge'] == 'left_only']
    return missing_assets_df['table_name'].tolist()
```
- **What it demonstrates**: detecting assets declared in a contract spec but absent from the live database (left-join on table identity columns).

```bash
# Local reproduction of the CI/CD violation check
cd ~/../workspace
python -m unittest data_contract_components/prevention/test_data_contract_violations.py -v
# FAIL: Column 'primary_image' is defined in contract but missing from data catalog
# FAIL: Column 'additional_images' is defined in contract but missing from data catalog
```
- **What it demonstrates**: running the exact same unit test CI/CD runs, locally, to reproduce a contract violation before even pushing — the `test_data_contracts_against_data_catalog` test fails per-column with a clear diff-style message.

## Reference Tables
| Component | Key files |
|---|---|
| Data assets | `data_assets/db_migrations/versions/*.py` (Alembic), `seed_db.py` |
| Contract definition | `contract_definition/*_contract_spec.json` |
| Detection | `_get_data_catalog.py`, `_get_data_contract_specs.py`, `contract_coverage_detector.py`, `contract_violation_detector.py` |
| Prevention | `prevention/test_data_contract_violations.py`, `.github/workflows/docker-image.yml` |

| CI/CD step | What happens |
|---|---|
| 1–3 | GitHub Actions triggers Docker Compose `test` build |
| 4–6 | Postgres container boots, signals healthy, devcontainer installs deps |
| 7 | Alembic runs migrations against Postgres |
| 8–11 | Unit test runs detectors, compares catalog metadata vs. contract spec |
| 12–13 | On failure, log saved and posted as a PR comment |

## Worked Example
**Full violation-and-resolution cycle (the chapter's running scenario)**: A backend engineer is asked to normalize `object_images` (unnest `additional_images` array, rename `primary_image`→`image`, add `is_primary_image` boolean) and initially plans to just drop and replace the table, assuming no one downstream depends on it yet. They branch (`update-objects-images-schema`), edit the Alembic migration, reset the DB (`alembic downgrade base` / `upgrade head`), and run the local unit test — which unexpectedly fails because the data platform team had already implemented a contract on `object_images`. The failure names the exact two missing columns. After pushing (to log the failure on the PR) and reverting, the engineer looks up the contract's `owner` field, contacts the Data Engineering Team, and learns the data science team started model training on `object_images` last week and needs time to migrate its preprocessing pipeline. The resolution: keep `object_images` as-is for a one-month transition, create a *new* `object_images_normalized` table with its own new contract spec, and update the migration to add (not replace) the table. Running the tests again first fails with "contract exists but asset missing from catalog" (expected, since the migration hadn't created the table yet), then passes once the migration is updated — demonstrating how contracts can also validate a desired future state before it's even built, not just protect an existing one.

## Key Takeaways
1. A minimal, real data contract system needs only four wired pieces: versioned data asset (Alembic), spec (JSON Schema), metadata-comparison detection, and CI/CD-embedded prevention (unit tests + PR comments).
2. Keep contract enforcement metadata-only — this is what lets it safely span teams (backend, platform, data science) without raw-data access concerns.
3. A contract violation is an entry point to a conversation (consult the owner, learn the real dependency), not just a red CI check to silence.
4. When a breaking change is genuinely needed, prefer additive migration (new table + new contract) with a transition window over destructive replacement, especially once consumers exist.
5. Contracts can validate a *desired* future schema before the migration that creates it is even written — not only protect the current state.
6. For real production use, prefer existing specs (Open Data Contract Standard) and real data catalog tools over the DIY versions built here for teaching purposes — the comparison/transform logic gets complex fast at scale.
7. Having runnable code (not just theory) is itself an adoption lever — it directly counters the "too theoretical" objection common when pitching data contracts to engineering teams.

## Connects To
- **Ch 5/6**: Directly implements the four components (data assets, contract definition, detection, prevention) defined conceptually there.
- **Ch 4**: The violation workflow (request contract → confirm viability → PR → CI/CD check → notify/update) is the exact 7-step workflow from Ch. 4, now executed in code.
- **Ch 8**: Promises real-world case studies (including Glassdoor's SCA-based implementation) as the next step beyond this sandbox.
- **Ch 9–12**: This chapter's repo is explicitly framed as an adoption/buy-in tool for the cultural-change chapters that follow.
