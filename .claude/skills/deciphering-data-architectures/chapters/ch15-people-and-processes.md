# Chapter 15: People and Processes

## Core Idea
Technology choices determine capability, but people and process failures determine whether data architecture projects actually succeed — most big data project failures (87% of data science projects reportedly never reach production) trace back to a specific, recurring set of organizational and process pitfalls, all of which are preventable with known countermeasures.

## Frameworks Introduced
- **Team Role Taxonomy (MDW/Fabric/Lakehouse)**: the standard role set for centralized architectures.
  - When to use: staffing any MDW, data fabric, or lakehouse project.
  - How: Data architect (high-level design, tech/governance decisions), Data engineer (build/test/maintain, ELT), Data steward (data quality/governance rules, access), DBA (operations, performance, backup/recovery), Data analyst (descriptive/diagnostic analysis, reports/dashboards), Data scientist (predictive/prescriptive modeling, ML, A/B tests), Business analyst (bridges business↔technical requirements), Project manager/scrum master (coordination, blockers, stakeholder communication), Data privacy officer (regulatory compliance), Data governance manager (org-wide data management policy), Data quality manager (completeness/consistency/accuracy oversight). In smaller companies, one person may hold multiple roles; in larger ones, many people may share one role.
- **Team Role Taxonomy (Data Mesh)**: an expanded, three-tier role structure unique to decentralized architectures.
  - When to use: staffing a data mesh implementation.
  - How: (1) **Domain teams** — mirror MDW/fabric/lakehouse roles but scoped to one domain, plus a new **data product owner** role (manages a domain's data product lifecycle, quality, accessibility, business alignment). (2) **Self-service data infrastructure platform team** (principle #3) — data mesh architect, platform architect, platform engineers/developers, data engineer, DevOps engineer, DataOps engineer, security/compliance officer, product owner/manager, UX designer, support/training staff. (3) **Federated computational governance platform team** (principle #4) — data governance lead, data governance specialist, data privacy/compliance officer, data security analyst, data quality specialist, domain representative, data architect, DataOps engineer.
- **Project Pitfall Catalog + Prevention**: an enumerated list of the most common causes of data project failure, each paired with a specific countermeasure (see Anti-patterns below — this chapter's core content IS the anti-pattern catalog).
- **Five Tips for Success**: positive practices to counterbalance the pitfalls.
  - When to use: throughout project planning and execution, not just at kickoff.
  - How: (1) Don't skimp on investment (hire real expertise, invest in the right tools, accept higher upfront cost for lower long-term cost); (2) Involve users early, show results often (agile releases, quick wins, training); (3) Add real value to migrated reports/dashboards (new KPIs/features, not just re-creation); (4) Have end users build a prototype (hands-on, concrete understanding of new capability); (5) Secure a project champion/sponsor (executive advocate who secures resources, aligns cross-functional priorities, manages change resistance) and a project manager targeting ~80% team efficiency (leaving ~20% for non-project overhead).

## Key Concepts
- **Data product owner**: the data-mesh-specific role responsible for a domain's data product lifecycle, quality, accessibility, and business alignment — the clearest organizational marker that principle #2 (data as a product) is being taken seriously.
- **87% statistic**: cited VentureBeat (2019) figure that 87% of data science projects never reach production — used to frame the chapter's urgency.
- **80% efficiency target**: the project-management heuristic that team members should spend ~80% of work hours on the project, with 20% reserved for vacation/sick time, other meetings, and learning — falling below this risks missed deadlines/cost overruns.
- **Project champion vs. project manager**: distinct roles — the champion operates at the strategic/executive level (advocacy, resourcing, cross-functional alignment, change management), while the project manager operates "in the trenches" (day-to-day coordination, dependency management, staffing).

## Mental Models
- "Skyscraper" analogy: even a flawless blueprint (architecture) and top-notch materials (technology) fail if builders/engineers/managers aren't aligned — people and process are the foundation, not an afterthought.
- "Driving a Lamborghini in first gear": end users who don't know their reporting tool's full feature set waste most of the solution's value — training and prototyping fix this.
- Treat each listed pitfall as a pair: (failure mode) + (specific, concrete prevention) — this chapter is explicitly structured as a diagnostic checklist, not abstract advice.

## Anti-patterns
- **Letting executives believe BI is "easy"**: leads to unrealistic timelines; counter by educating executives directly (e.g., having them sit through a coding session).
- **Using the wrong technology out of ignorance ("when you have a hammer, everything is a nail")**: e.g., using SQL Server as a data lake; counter by forming research committees per product category (ETL, analytical, reporting tools) before deciding.
- **Gathering too many business requirements**: year-long requirements phases produce unread, outdated documents; counter by spending only a few weeks on initial requirements, then gathering more in parallel with development.
- **Gathering too few business requirements**: developers guess at user needs and build the wrong thing; counter with an iterative development method incorporating ongoing feedback.
- **Presenting unvalidated reports**: an incorrect number in a first report destroys user trust, which is very hard to rebuild; counter with rigorous QA comparing new-solution totals against known-valid production reports, ideally with end-user involvement.
- **Hiring an inexperienced consulting company**: consultants may show relevant experts during sales but substitute less experienced staff once the project starts; counter by explicitly asking whether the people demonstrated will be the people doing the work.
- **Hiring a consultancy that offshores without disclosure**: time zone/language barriers and lower quality result; counter by directly asking about offshore use.
- **Passing full project ownership to consultants**: creates a "black box" with no visibility; counter by keeping your own project managers closely engaged, not relying on status reports alone.
- **Neglecting knowledge transfer from consultants back to the organization**: leaves the company unable to maintain/extend the solution without re-hiring consultants; counter by having internal staff work alongside consultants and scheduling explicit training/review sessions near project end.
- **Slashing the budget mid-project**: cutting staff, hardware, or end-user training near the finish line undermines the whole investment; counter by reducing scope (e.g., fewer initial data sources) rather than cutting critical resources.
- **Starting with a fixed end date and working backward**: compresses discovery/exploration, causes rushed/error-prone work, and can't accommodate emergent requirements; counter by keeping deadlines flexible as new information emerges.
- **Structuring the warehouse around source-system shape instead of business needs**: operational systems aren't designed for BI/analysis and don't align with business KPIs, causing instability as operational systems evolve; counter by designing the MDW/lakehouse around business needs, not mirroring source structure.
- **Shipping a solution with slow performance**: a slower experience than the legacy system destroys user confidence just as badly as wrong data; counter by proactively measuring and tuning performance against a baseline rather than waiting for complaints.
- **Overdesigning or underdesigning the architecture**: underdesign risks scalability/performance/security gaps; overdesign causes complexity, inflexibility, analysis paralysis, and cost bloat; counter by deliberately balancing robustness against simplicity/adaptability.
- **Poor communication between IT and business domains**: differing languages/priorities cause misaligned objectives, inefficient resourcing, and low morale; counter with roles like business analyst acting as translators, regular meetings, and documented requirements.

## Reference Tables
| Team (Data Mesh) | Core Responsibility | Key Unique Role |
|---|---|---|
| Domain team | Build domain's analytical solution, implement mesh principles locally | Data product owner |
| Self-service infrastructure platform team | Build/maintain shared tooling (principle #3) | Data mesh architect, platform architect |
| Federated governance platform team | Define/coordinate global governance (principle #4) | Data governance lead, domain representative |

| Pitfall Category | Representative Failure | Core Fix |
|---|---|---|
| Expectations | Executives think BI is "easy" | Educate executives directly |
| Technology choice | Wrong tool for the job | Dedicated research committees per tool category |
| Requirements | Too many or too few gathered | Time-box initial gathering; iterate afterward |
| Trust | Unvalidated reports, slow performance | Rigorous QA; proactive performance tuning |
| Vendor management | Inexperienced/offshore/unsupervised consultants | Vet experience, ask direct questions, retain internal PM oversight |
| Knowledge | No knowledge transfer from consultants | Embed staff with consultants; schedule explicit training |
| Budget/schedule | Mid-project cuts; fixed end-date-backward planning | Protect critical resources; keep schedules flexible |
| Design | Mirrors source data; over/underdesigned | Design around business needs; balance robustness and simplicity |
| Communication | IT/business misalignment | Business analyst as translator; regular transparent meetings |

## Worked Example
A developer recalls a company that spent over a year gathering business requirements for a data solution, producing hundreds of pages that no one fully read because the gathering phase had already run so far past schedule that teams began development after skimming only a few sections — directly illustrating the "too many requirements" pitfall and motivating the time-boxed, parallel-gathering countermeasure.

## Key Takeaways
1. Staff roles deliberately: MDW/fabric/lakehouse need one core role set; data mesh requires three parallel team structures (domain, infrastructure platform, governance platform) with new roles like data product owner.
2. Treat the pitfall catalog as a pre-project checklist — nearly every failure mode here has a known, specific, actionable fix.
3. Protect critical resources (staffing, hardware, training) from mid-project budget cuts; cut scope instead.
4. Keep timelines flexible rather than working backward from a fixed end date — rigid deadlines compress discovery and degrade quality.
5. Invest in user involvement, prototyping, and visible quick wins throughout the project, not just at delivery, to build and sustain trust.
6. Secure both a project champion (executive-level advocacy/resourcing) and a day-to-day project manager (targeting ~80% team efficiency) — these are distinct, complementary roles.

## Connects To
- **Ch 13, 14**: Data mesh's expanded role structure directly operationalizes the four principles and addresses the "people" concerns raised in Ch 14 (hiring mini-IT teams per domain, generalist vs. specialist trade-offs).
- **Ch 3**: ADS discovery/requirements practices connect to this chapter's requirements-gathering pitfalls.
- **Ch 16**: The book explicitly hands off from "people and process" to "technology" as the next and final major topic.
- **External concept**: VentureBeat's 2019 "87% of data science projects never make it into production" statistic.
