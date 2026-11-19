# Chapter 4: An Introduction to Data Contracts

## Core Idea
A data contract is an API-like agreement between data producers and consumers that moves data quality management from reactive (detect-after-break) to proactive (prevent-before-break) by embedding automated, CI/CD-enforced checks into the producer's deployment workflow — effectively giving data the same pull-request-style collaboration software engineering already has.

## Frameworks Introduced
- **The Data Contract Workflow** (7 steps): (1) data constraint identified by consumer, (2) consumer requests a contract for the asset, (3) producer confirms viability, (4) contract confirmed as code, (5) producer opens a PR to change the data asset, (6) CI/CD automatically checks for contract violation, (7a) on failure, owners notified and failure protocol followed / (7b) on pass, asset updates flow downstream.
  - When to use: as the canonical operating loop for any data contract implementation.
  - How: treat the contract as version-controlled code (typically YAML) wired into the producer's existing CI/CD, not a side-channel document.
- **Data Contract Maturity Curve**: Awareness → Collaboration → Ownership.
  - When to use: sequencing a rollout; don't jump straight to enforcement.
  - How: Awareness = make producers/consumers visible to each other around tier-one data products; Collaboration = producers pre-notify consumers of changes (PR-like feedback); Ownership = producers proactively define and maintain contracts themselves.
- **Data Contract Architecture — 4 Components**: Data assets, Contract definition, Detection, Prevention.
  - When to use: structuring any contract implementation (expanded fully in Ch. 5–6).
  - How: assets define scope, contract definition codifies expectations, detection finds violations, prevention blocks them pre-deployment.
- **Data Contracts vs. Data Observability**: contracts prevent known issues pre-change (targeted, CI/CD-embedded, informed by business logic); observability detects unknown issues post-change (broad, complements CI/CD, reflects business logic in the data).
  - When to use: deciding which tool/approach fits a given problem — not an either/or, they compose.
  - How: use observability broadly across the whole stack to surface unknown issues; use contracts narrowly on tier-one assets to prevent known, specified violations before merge.

## Key Concepts
- **Data producer**: typically a software engineer who collects/stores data as a by-product of building an application; their accountability stops once data is stored.
- **Data consumer**: technical (data engineers, scientists, analysts, analytics engineers, platform engineers) or nontechnical (business users/PMs) employees who use produced data.
- **Transactional events**: individual state-change records (e.g., an order), stored in RDBMS or streamed (Kafka); a "by-product" of the application.
- **Clickstream events**: user-interaction records captured by third-party SDKs (Amplitude, Segment, Google Analytics) for product analytics.
- **Lineage**: the web of connections tying data assets to each other; the denser/older the environment, the harder it is to trace.
- **Tier-one data products**: the subset of downstream data assets important enough to warrant contract coverage first — scoping tool to avoid trying to contract everything at once.
- **Data product**: the sum of data components (pipeline + interface, e.g. dashboard = queries + pipeline + visualization) organized to serve a business need — contracts serve the product, not vice versa.
- **Hard failure vs. soft failure**: a contract violation can either block the change outright (hard) or merely alert impacted downstream teams while allowing the change through (soft) — the contract spec should state which applies and under what exception conditions.
- **Institutional knowledge loss**: query logic/meaning leaves with departing employees, making "going to the source" (upstream data) preferable to reusing undocumented downstream queries.

## Mental Models
- Think of the data contract workflow as recreating the pull-request review loop for data: just as engineers request code review before merging, consumers request a "data contract review" before a producer's change ships.
- Use the flashlight vs. laser pointer analogy: data observability is a flashlight (broad, illuminates the whole system, used after something happens); data contracts are a laser pointer (narrow, targeted, used to prevent specific known issues before they happen).
- Think of maturity progression (awareness → collaboration → ownership) as a trust-building ladder — you cannot skip to producer-owned contracts before producers even know who depends on their data.
- Use "go upstream first" as the default placement heuristic: contracts closer to the source (transactional → analytical database boundary) prevent more issues than contracts placed near dashboards.

## Anti-patterns
- **Treating data quality fixes as purely technical**: the chapter's 11-step manual resolution process (surface → triage → scope → replicate → profile → downstream investigate → upstream investigate → consult stakeholders → pre-deploy fix → deploy → communicate) is mostly communication/coordination work, not code.
- **Accepting data issue requests via unstructured DMs**: should go to a shared, visible channel (e.g., Jira) so triage and prioritization are possible.
- **Jumping to a fix before scoping**: skipping requirements scoping and issue replication risks solving the wrong problem or masking a human/process error as a "data quality issue."
- **Placing contracts too far downstream first** (e.g., directly on dashboards): diminishes prevention value; issues are already baked in by the time data reaches that stage.
- **Treating every contract violation as a hard block**: ignores legitimate urgent hotfix scenarios; the contract spec should define when soft failures (alert only) are appropriate.
- **Full observability-style visibility applied to every asset as if it were a contract**: causes alert fatigue; contracts should be targeted only at tier-one assets.

## Code Examples
```python
# product_operations.py example code
import db_helper_functions as db_helper

class ProductSold:
    def __init__(self) -> None:
        # Connect to the database
        self.db_connection = db_helper.connect_to_database()
        pass

    def product_sold_count(self, product_id, sold_count):
        # Update the product_table in the database with new count
        <python code implementing logic>
        # Update database
        db_connection.commit()
        db_connection.close()
```
- **What it demonstrates**: how upstream investigation of a data quality issue must trace beyond tables into the actual application code/function (`product_sold_count`) generating the data — "hidden" data quality issues often live in code outside the data team's visibility.

## Reference Tables
| | Data producer | Data consumer |
|---|---|---|
| Problem | Update software to align with changing technical requirements | Ensure underlying data is trustworthy for business insight |

| | Data contracts | Data observability |
|---|---|---|
| Function | Prevents specific known data quality issues | Highlights data quality trends |
| Workflow | Included in CI/CD | Complements CI/CD |
| Input | Informed by business logic | Reflects how data captures business logic |
| Visibility | Targeted (tier-one assets only) | Broad (whole stack) |
| Timing | Alerts before change | Alerts after change |

## Worked Example
**Mark's manual data-quality resolution process (early-stage org)**: a stakeholder reports "these numbers look off" → issue triaged in a shared channel (not DM) → requirements scoped with the downstream stakeholder to weigh effort vs. impact → issue replicated via the data product or source SQL → data profiled (timelines, nulls, count spikes, lineage review) to narrow hypotheses → downstream pipeline investigated (common causes: bad JOIN, missing WHERE edge case, wrong table) → upstream pipeline investigated down to the actual application function (e.g., `product_sold_count()`) → technical stakeholders consulted (not just requested) to validate the fix and get it prioritized → fix implemented pre-deploy using the profiling baseline → deployed and backfilled → stakeholders communicated to throughout, since silent resolution erodes trust even if the fix works. This process is almost entirely communication-driven, illustrating why the data contract workflow (proactive, automated) is proposed as the scalable alternative to repeating this manual loop every time.

**HR tech "manager status" asset (Mark's data scientist experience)**: this data asset constantly broke as enterprise customers changed employee-management systems (e.g., switching from daily to monthly batch ingestion, misaligned with the semantics of "manager status"), forcing the same exploration/validation cycle on every new project. A data contract would shortlist this asset as trustworthy, document its quirks/expectations once, and version-control changes to those assumptions — eliminating repeated rediscovery.

## Key Takeaways
1. Data contracts exist to import software engineering's missing "pull request for data" — pre-deployment, CI/CD-embedded visibility between producers and consumers.
2. The contract itself (as version-controlled code) is the most important artifact — it forces explicit negotiation, not just automated enforcement.
3. Place contracts as far upstream as feasible; the transactional-to-analytical database boundary is the recommended starting point for most orgs.
4. Not every violation should hard-block a deploy — define hard vs. soft failure policy explicitly in the contract spec, accounting for legitimate emergency hotfixes.
5. Roll out contracts along a maturity curve (awareness → collaboration → ownership); skipping stages (e.g., expecting producer ownership before awareness exists) will fail.
6. Scope contract coverage to tier-one data products to avoid alert fatigue — contracts are a laser pointer, not a flashlight.
7. Data contracts and data observability are complementary, not competing: observability informs which constraints belong in a contract; contracts prevent the issues observability would otherwise just report on.

## Connects To
- **Ch 3**: Directly answers the "design surface" requirements (prevention-first, communicative, contextual, timely, inclusive) defined there.
- **Ch 5/6**: Will detail the four architecture components (data assets, contract definition, detection, prevention) in depth.
- **Ch 9**: The maturity curve and upstream-placement principle here is the seed of the full "shift left data" cultural framework.
- **Ch 11**: References back to this chapter's outcome metrics (faster iteration, improved communication, fewer DQ issues) for measuring contract success.
