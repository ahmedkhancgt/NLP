# Chapter 3: The Challenges of Scaling Data Infrastructure

## Core Idea
Data development has a fundamentally different workflow than software development (discovery/hypothesis-driven vs. requirements-driven), and because software's change-management tooling (PRs, code review) has no data equivalent, scaling an organization past Dunbar's Number without a programmatic communication mechanism like data contracts inevitably breaks data quality.

## Frameworks Introduced
- **Data Discovery Questions**: does the data exist / where is it / who owns it / can the owner explain it / how do I use it / do I have permission.
  - When to use: at the start of any data development effort, before building a query or model.
  - How: answer each sequentially; a "no" to existence or ownership forces a branch decision (ask producer to generate it, or report back that no trustworthy answer is available).
- **Change Management Design Surface Requirements**: prevention-first, communicative, contextual, at-the-right-time, including-the-right-people.
  - When to use: designing any data change-management/contract tooling.
  - How: integrate with producer CI/CD (prevention), connect producers directly to consumers (communicative), surface expected-vs-actual schema/semantics/PII/usage (contextual), alert right before deployment not before/after (timing), include both technical and nontechnical stakeholders via a background-agnostic interface.
- **Concept Drift Typology** (Jie Lu et al.): sudden, gradual, incremental, recurring drift — ML's model for how underlying data distributions change over time, applied to data quality.
  - When to use: diagnosing why "correct" data suddenly looks wrong, or classifying a change for communication purposes.
  - How: match the observed change pattern (e.g., abrupt regulatory cutoff vs. slow demographic shift) to the appropriate type to set expectations with stakeholders.
- **Dunbar's Number + Conway's Law applied to data orgs**: ~150 is the relationship limit before communication structures must change; a system mirrors the communication structure of the org that built it.
  - When to use: deciding whether a data-quality problem is actually an org-structure/communication problem.
  - How: when potential relationships (which grow exponentially with headcount) exceed what individuals can track, replace ad hoc communication with a programmatic mechanism (data contracts) that limits who needs to be notified.

## Key Concepts
- **Pre-development (data) vs. requirements-gathering (software)**: data devs start from a business question/hypothesis and must discover whether answerable data exists; software devs start from a requirements doc and architecture spec.
- **Data discovery**: the process of searching for data to answer a question, covering existence, location, ownership, meaning, usage, and access.
- **Floss refactoring vs. root canal refactoring** (Murphy-Hill & Black): small opportunistic cleanups vs. large, expensive, overdue refactors.
- **Data drift / concept drift**: unforeseeable changes in the underlying data distribution over time (borrowed from ML).
- **Entropic behavior of data**: data decays at different rates (like radioactive half-life) — e.g., Social Security numbers decay slowly, telemetry decays fast — and the decay rate (not value) determines appropriate change management.
- **Dunbar's Number**: ~150, the cognitive limit on stable relationships a person can maintain; potential relationships grow combinatorially with headcount.
- **Conway's Law**: a system's design mirrors the communication structure of the organization that built it.
- **Alert fatigue**: too many non-actionable monitors causing teams to ignore all alerts by default.

## Mental Models
- Think of data development as "flux and discovery" and software development as "structured and rigid" — tools built for one will misfit the other.
- Use the data discovery checklist before any query/model effort, the same way software uses a requirements doc.
- Think of data as having a decay rate / half-life; choose monitoring cadence and change-management rigor based on where an asset sits on that entropy spectrum, not on its perceived importance.
- Use Dunbar's Number as a diagnostic: if coordination is breaking down, the fix is structural/communication tooling, not more headcount or more alerts.
- Think of data contracts as "limiting the graph" — they reduce an individual's exposure to the full combinatorial web of org relationships down to just the relevant producer/consumer link.

## Anti-patterns
- **Treating data change management like code review**: there's no PR-equivalent for data, so producers have no mechanism to see who depends on their data before they change it.
- **Alerting too early or too late**: notifying producers long before a change (ignored, backlogged) or after it ships (producer has moved on) both fail — the right moment is immediately pre-deployment.
- **Making all data visible / monitoring everything**: most data is in prototyping/discovery phase (not yet trustworthy); full visibility causes alert fatigue and renders monitors useless.
- **Scaling technical systems without scaling communication structure**: adding headcount or infrastructure without addressing Dunbar's Number / Conway's Law effects just adds complexity and breaks further.
- **"Hands off my stuff" territorialism and pure goodwill-based prioritization**: asking teams to volunteer data quality priority against their own roadmap doesn't work; data teams must be structurally empowered, not just polite.

## Code Examples
*(omitted — this chapter has no code/config examples; it is organizational/process-focused, using diagrams, surveys, and narrative case studies)*

## Reference Tables
| Concept drift type | Data quality example |
|---|---|
| Sudden drift | GDPR-driven removal/blocking of EU user data |
| Gradual drift | Netflix moving from DVD sales to streaming |
| Incremental drift | Roblox's user base aging from children to teens |
| Recurring concepts | Recessions reducing discretionary consumer spending |

| Design surface requirement | What it means |
|---|---|
| Prevention first | Catch breaking changes via CI/CD integration, not after the fact |
| Communicative | Directly connect data producers and consumers |
| Contextual | Surface expected vs. actual schema, semantics, PII, downstream usage |
| At the right time | Alert immediately before deployment, not earlier or later |
| Including the right people | Reach both technical and nontechnical stakeholders via an abstraction layer |

## Worked Example
**Alan's large-scale refactor (2016–2021, health insurance)**: Alan assumed a strict one-to-one company-to-contract relationship, embedded throughout code, docs, and culture. By 2019 enough customers broke the assumption that Alan patched around it with extra entities (floss refactoring / tech debt); by 2021 the volume forced a root-canal refactor. Following the seven-step survey framework: (1) determined change was needed once enough customers broke the assumption; (2) chose to redesign the data model from company-contract to company-contract-population and contract-population-contract (enabling one-to-many); (3) implemented it with a dedicated team, since one legacy property was referenced over 500 times; (4) generated/migrated tests; (5) validated by running old and new implementations in parallel with observability monitors; (6) recertified for compliance (health data regulation, SOC 2, data-leakage risk under the new one-to-many model); (7) updated documentation and retrained nonengineering roles who had internalized the "one active contract" assumption for five years. Lesson: large-scale refactors are business decisions, not just technical ones, and must include cultural/documentation change alongside code.

**Atlassian Confluence Cloud team**: first addressed Dunbar's Number by deliberately shrinking team size (<5 direct reports, colocated teammates) before scaling back up; then struggled with Conway's Law under a "squads and tribes" model whose structure didn't match the system being built, eventually converging duplicative teams into single ownership when signals showed misalignment.

## Key Takeaways
1. Data development and software development have incompatible pre-development workflows — tools and processes built for one will fail the other.
2. There is no "pull request" for data; building that missing change-management layer (prevention-first, communicative, contextual, timely, inclusive) is the design target for data contracts.
3. Large-scale refactors and database migrations are business decisions requiring ROI justification, not purely technical cleanup — and they require updating culture/documentation, not just code.
4. Data drift (sudden/gradual/incremental/recurring) should be classified and mapped to business impact, not treated generically.
5. Past Dunbar's Number (~150 people), ad hoc communication about data changes breaks down; organizations need programmatic mechanisms, not more goodwill or headcount.
6. Conway's Law means your data org structure will be mirrored in your data architecture — misaligned team structure produces misaligned systems.
7. Data contracts work by narrowing the "graph" of people who must be notified about a change down to the actually-relevant owners and consumers.

## Connects To
- **Ch 1**: Extends the startup-size/communication argument (small teams avoid GIGO) into a formal framework (Dunbar's Number).
- **Ch 4**: Will formally define data contracts as the mechanism that operationalizes the "design surface" requirements described here.
- **Concept drift / ML**: Explicitly borrows terminology and framework from machine learning's data drift literature.
- **Conway's Law / software architecture**: Direct application of a classic software-engineering principle to data team structure.
