# Patterns

## Write-Audit-Publish (WAP)
**When to use**: Protecting streaming/live data or any batch load before it reaches production, especially when you can't gate a producer's deployment directly (e.g., third-party platform replication).
**How**: Write new data to a staging database/table → run data quality/semantic checks from the contract spec against it → if it passes, promote to production; if it fails, route to a dead-letter queue for review.
**Trade-offs**: Adds latency and infrastructure (staging layer, audit step) but prevents bad data from ever reaching consumers, rather than detecting it after the fact.

## Data Quarantining ("Dead Letter Queue")
**When to use**: When you're a consumer with no leverage over a producer (e.g., third-party data broker), or when "no data" is safer than "wrong data" (e.g., health insurance eligibility files).
**How**: Monitor incoming data against contract constraints; tag/separate violating records instead of passing them through; route to manual review or automated remediation.
**Trade-offs**: Not universally correct — for some use cases (e.g., telemetry hinting at hardware failure) bad data passed through is actually useful signal; decide per use case based on which data quality dimension matters most.

## Static Code Analysis (SCA) for Data Contract Detection
**When to use**: When you need the earliest possible (code-level) detection of a data-impacting change, before it ever touches a database.
**How**: Embed SCA into the CI/CD code-review step; compare determined code behavior on main vs. the proposed branch; flag changes that affect fields/events covered by a contract (e.g., Meta's Pysa/Infer, Glassdoor's implementation).
**Trade-offs**: Most preventative and highest-leverage detection method, but hard to build (requires specialized software engineering expertise) and language/framework-specific; start with schema/data-in-motion detection if SCA isn't feasible yet.

## End-to-End Lineage (Source Code Flow Graph + Data Lineage)
**When to use**: Tracing a data quality issue all the way from a downstream dashboard back to the exact code/commit that caused it.
**How**: Use data lineage (table/asset graph) for the database-side trace; use source code flow graphs (function-call graph) for the application-code-side trace; combine both, since most lineage tools stop at the analytical database.
**Trade-offs**: Manual tracing (as in Mark's 10-step process) takes days to weeks; automating this via connected metadata capture is high-effort but dramatically reduces resolution time once built.

## Vertical-by-Vertical Rollout
**When to use**: Any organization-wide data contract adoption effort.
**How**: Attack one business vertical, team, or asset category at a time rather than putting everything under contract simultaneously (Convoy's explicit advice: "you should attack it vertical by vertical, or only put contracts where they really make sense").
**Trade-offs**: Slower initial coverage, but dramatically higher adoption success rate; avoids overwhelming teams and diluting focus.

## Steel Thread Selection via Weighted Decision Matrix
**When to use**: Choosing your first (or next) data contract pilot candidate among multiple compelling options.
**How**: Define criteria (business criticality, producer/consumer engagement, data volatility, existing DQ issues, dependency complexity, ease of implementation, organizational visibility); weight each 1–10; score each candidate 1–5; multiply and sum; pick the highest total.
**Trade-offs**: Takes upfront effort and stakeholder conversation to populate accurately, but prevents picking a pilot based on convenience or politics rather than actual pilot-worthiness.

## Minimum Viable Metric Tree (MVMT)
**When to use**: Connecting any data contract's technical implementation to a visible business outcome for leadership communication.
**How**: Start with a company-visible North Star metric; decompose into contributing metrics/levers with stakeholder input; repeat until reaching the node your protected data product directly drives.
**Trade-offs**: Deliberately lightweight compared to full DuPont/KPI-tree analysis — favors speed-to-insight over analytical completeness; if you can't trace the chain, that's a signal the asset may not be as business-critical as assumed.

## Monomyth-Structured Measurement Storytelling
**When to use**: Presenting data contract metrics/progress to stakeholders who need persuasion, not just information.
**How**: Structure the pitch as Departure (unacceptable pain, who feels it) → Transformation (enabling action and expected resistance) → Return (tangible proof metrics of the new state).
**Trade-offs**: Requires more narrative crafting effort than a raw metrics dump, but metrics alone — per Hubbard — cannot overcome human bias or resistance to change.

## Data Contract Maturity Curve (Awareness → Collaboration → Ownership)
**When to use**: Sequencing any contract rollout within a team or organization.
**How**: Awareness = make producers/consumers visible to each other on tier-one products; Collaboration = producers pre-notify consumers of changes (PR-like review); Ownership = producers define and maintain contracts themselves.
**Trade-offs**: Skipping stages (e.g., expecting producer-owned contracts before awareness exists) reliably fails; the curve takes real calendar time to traverse per team.

## Push Architecture for Data Ownership ("You Produce It, You Own It")
**When to use**: Moving an organization away from a pull model where consumers passively accept whatever data arrives.
**How**: Require producers to explicitly define what they expose and to whom (Adevinta Spain's shift), rather than letting consumers silently take undocumented dependencies on raw event streams.
**Trade-offs**: More upfront producer effort (schema definition, contract authoring) but dramatically reduces downstream surprise breakage and clarifies accountability.

## First-Principles "Going Deeper" Problem Diagnosis
**When to use**: When an adoption or implementation effort stalls despite seeming technically sound.
**How**: Iteratively restate the problem ("What must be true to solve X?"), peeling back assumptions layer by layer until you reach a level where you find actual traction; the authors' own chain went 10 levels deep before landing on "effective change management workflow" as the real root problem.
**Trade-offs**: Time-consuming and can feel like analysis paralysis if taken too far; stop "going deeper" once you find traction, not before.
