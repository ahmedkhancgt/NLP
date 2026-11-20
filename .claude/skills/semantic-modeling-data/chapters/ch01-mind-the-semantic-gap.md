# Chapter 1: Mind the Semantic Gap

## Core Idea
Data-related projects often fail not from lack of data or technology but from the "semantic gap" — the disconnect between how data is modeled on the supply side and how it is understood/used on the exploitation side. Closing this gap requires deliberate semantic modeling, which is distinct from (and complementary to) machine learning.

## Frameworks Introduced
- **The Semantic Gap**: the situation where data models produced by the supply side are misunderstood/misused by the exploitation side, or where exploitation-side requirements are misunderstood by the supply side.
  - When to use: diagnosing why a data/AI initiative underperforms despite adequate data and infrastructure.
  - How: trace breakdowns to either (a) model consumers misinterpreting what the model's elements mean, or (b) model builders failing to capture what consumers actually need.
- **Pitfalls vs. Dilemmas** (the book's organizing distinction):
  - Pitfall: a clearly wrong decision/omission in modeling that has undesired consequences — avoidable, has a "right answer."
  - Dilemma: a choice between options each with real trade-offs and no universally correct answer — must be treated as a decision-making problem (enumerate options, evaluate against feasibility/cost-benefit/strategic criteria for your context).
  - When to use: classify any modeling problem you hit as one or the other before trying to "solve" it — dilemmas don't have a single correct fix.

## Key Concepts
- **Semantics**: the study of meaning — the relationship between signifiers (words, symbols) and the things they denote.
- **Semantic data modeling**: developing descriptions/representations of data so meaning is explicit, accurate, and commonly understood by humans and machines; spans metadata schemas, controlled vocabularies, taxonomies, ontologies, knowledge graphs, E-R models, property graphs.
- **Symbolic vs. subsymbolic representation**: semantic models use symbolic, human-interpretable representations of discrete facts and identities; ML models use latent/subsymbolic representations good at statistical regularities but without inherent human interpretation.
- **Supply side vs. exploitation side**: supply side collects/represents/stores data; exploitation side builds analytics/AI applications on top of it. Many practitioners wear both hats, but the two sides can talk past each other.
- **Vagueness treated as objective**: presenting subjective/context-dependent knowledge (e.g., "essential" skills) as if it were an objective, universal fact.
- **Complementarity of ML and semantic modeling**: ML can help build semantic models faster; semantic models can inform and constrain ML systems — not competing approaches.

## Mental Models
- **Gold mining metaphor**: data is the "gold mine," but you can have abundant data, real gold in it, and mature extraction technology, and still fail to extract value — because the semantic gap breaks the chain between supply and exploitation.
- **Two hats**: most data professionals operate on both the supply and exploitation side at different times; the semantic gap can originate from either side, so diagnosis must consider both.
- **Modeler balancing act**: software engineers/data engineers tend to under-specify meaning; ontologists/linguists/domain experts tend to over-specify it and argue over distinctions nobody downstream cares about. Good semantic modeling finds the expressiveness level that serves actual users without runaway cost.

## Anti-patterns
- **Presenting subjective knowledge as objective**: e.g., ESCO's `essential_for` relation between occupations and skills asserts specific skills are "essential" as if this were a universal fact, when essentiality is vague and context-dependent (country, industry, employer). Downstream consumers (e.g., career-advice software) then present disputable claims as indisputable facts.
- **Not informing model users about vagueness**: failing to flag a relation/attribute as vague leaves consumers unaware they're building on shaky, debatable semantics.
- **Conflating "concept" and "class"**: some OWL/Protégé material equates "concept" with "class" (a set of individuals), while SKOS correctly treats a concept as a unit of thought that need not be a set-denoting class. Following the OWL-tutorial framing causes real modeling errors later (covered in Ch. 2+).
- **Blaming only the modeler**: the author notes that the field itself (terminology inconsistency, dismissing pitfalls, flawed tooling/literature) often sets modelers up to fail — pitfalls aren't purely individual incompetence.

## Reference Tables
*Table 1-1 (reconstructed, abbreviated) — ESCO essential skills by occupation, used to illustrate the vagueness pitfall:*

| Occupation | Example "essential" skills listed |
|---|---|
| Data scientist | data mining, data models, information categorization/extraction, OLAP, query languages, SPARQL, statistics, visual presentation |
| Knowledge engineer | business intelligence, business process modeling, database dev tools, NLP, AI principles, SPARQL, systems theory, web programming |
| Data entry supervisor | LDAP, LINQ, MDX, N1QL, SPARQL, XQuery, company policies, database, documentation types, information confidentiality, query languages |

The table's point is not the content but that readers will likely disagree with several entries — demonstrating that "essential" is vague, not objective.

## Worked Example
ESCO (European Skills, Competences, Qualifications and Occupations classification), released 2017 after six years of development, aims to let software (a) semantically analyze labor-market data consistently across languages, (b) match job seekers to jobs, and (c) derive labor-market analytics for policymakers. It groups synonymous terms under single occupation/skill concepts (useful), and links occupations to skills via `essential_for`/`optional_for` relations. The pitfall: treating "essential" as crisp and universal. A downstream tool built on this (e.g., career-advice software telling a user "to become a knowledge engineer you must learn web programming") inherits an unstated, debatable premise as fact. The dilemma version: ESCO's developers could (a) flag the relation as "vague" (cheap, but doesn't reduce disagreement) or (b) build context-specific versions of the relation per country/industry/user group (reduces disagreement, but far more costly to build and maintain) — no universally correct choice.

## Key Takeaways
1. Projects fail even with good data and mature tech when the semantic gap between supply-side modeling and exploitation-side use is unaddressed.
2. Semantic models are symbolic and explicit; ML models are subsymbolic and implicit — treat them as complementary, not substitutes.
3. Distinguish pitfalls (avoidable mistakes with a right answer) from dilemmas (genuine trade-offs requiring contextual decision-making) before trying to fix a modeling problem.
4. Presenting vague/subjective relations as objective facts is a recurring, consequential pitfalls across real-world models (ESCO is one example among many).
5. Good semantic modeling is a balancing act between under-specification (engineers) and over-specification (ontologists/linguists) — aim for the expressiveness level your users and applications actually need.
6. "Concept" is not always equivalent to "class" — terminological sloppiness in the field itself causes downstream modeling errors.

## Connects To
- **Ch 2**: the "concept vs. class" confusion here is resolved in detail when Classes, Individuals, and Terms are formally defined.
- **Ch 3**: vagueness (used here as the ESCO example) is formally treated as a semantic/linguistic phenomenon.
- **Ch 4**: "semantic accuracy" and related quality dimensions give criteria for judging whether a model like ESCO is well-specified.
- **Ch 6-11**: the "bad X" chapters catalog the detailed pitfalls this chapter introduces at a high level.
- **Ch 12-14**: the dilemmas framework introduced here (options, no single right answer) is applied to representation, expressiveness, and governance dilemmas.
