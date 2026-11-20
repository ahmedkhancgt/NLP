# Chapter 4: Semantic Model Quality

## Core Idea
Semantic model quality is multidimensional (accuracy, completeness, consistency, conciseness, timeliness, relevancy, understandability, trustworthiness, plus availability/versatility/performance), and each dimension requires its own measurement approach, causes of failure, and remediation — no single metric captures "good."

## Frameworks Introduced
- **Application-centered vs. application-neutral quality evaluation**: two fundamentally different measurement strategies.
  - When to use: application-centered when you want to know if a specific application's effectiveness improves with the model (e.g., search, QA); application-neutral when you want quality that transfers across any application using the same domain/data.
  - How: application-centered compares an application's before/after performance (but can't isolate whether the model or its usage is at fault, and improving for one app may worsen another); application-neutral measures fit against the domain/data the model should describe (consistent across apps, used for all dimensions except relevancy in this chapter).
- **Nine (+3) Quality Dimensions**: Semantic Accuracy, Completeness (schema + population), Consistency, Conciseness, Timeliness, Relevancy, Understandability, Trustworthiness, plus Availability/Versatility/Performance (treated lightly, out of scope for the rest of the book).
  - When to use: as a structured checklist whenever auditing or comparing semantic models.
  - How: evaluate each dimension independently using its own metrics (below) rather than assuming one dimension implies another (e.g., consistency ≠ accuracy).
- **Gold standard / Partial gold standard / Silver standard**: three tiers of reference data for measuring completeness.
  - When to use: full gold standards are rarely available (especially for population completeness); use partial gold standards (a verified subset) or silver standards (not fully accurate but "good enough" reference) instead.
  - How: compare model content against the standard to reveal (not conclusively prove) incompleteness.

## Key Concepts
- **Semantic accuracy**: degree to which a model's assertions are accepted as true. Measured by sampling statements and having human judges (experts, users, or crowds) rate true/false, ideally with inter-agreement metrics, especially for vague statements. Automatable via outlier detection or logical-consistency-violation detection.
- **Schema completeness vs. population completeness**: schema completeness = are all necessary classes/relations/attributes/axioms defined; population completeness = are all the actual instances/assertions/values populated. A model can have complete schema but sparse population, or vice versa.
- **Consistency**: freedom from logical/semantic contradictions (e.g., two different "natural mothers" for one person, or an entity instantiating two disjoint classes). Caused by missing/unenforced constraints, framework limitations (e.g., Neo4j lacks native cardinality constraints), or by constraint-checking being computationally intractable (undecidable/NP-hard in some OWL2 profiles).
- **Conciseness**: absence of redundant elements — either semantically-equivalent duplicate representations (e.g., DBpedia's `dbo:child` vs `dbp:children`) or no-longer-needed legacy/temporary elements.
- **Timeliness**: degree to which the model reflects the current state of the world (e.g., recognizing Yugoslavia's dissolution, or Kazakhstan's capital rename). Depends on domain volatility and maintainer responsiveness.
- **Relevancy**: the one dimension that IS application-dependent — whether the model's structure/content is actually useful for a specific task, independent of how accurate/complete it otherwise is.
- **Understandability (comprehensibility)**: ease with which human consumers correctly interpret and use model elements without confusion. Chronically underrated by modelers who over-focus on computational properties; low understandability degrades accuracy, relevancy, and trust downstream (Ch. 6 covers the fix).
- **Trustworthiness**: subjective user confidence in model quality — correlated with but distinct from accuracy/completeness. Driven by reputation/adoption (e.g., schema.org's broad industry backing), availability of formal evaluations/experience reports, and provenance (centralized expert curation vs. open community editing vs. automatic extraction from unstructured sources).
- **Availability, Versatility, Performance**: availability = model/part of it is obtainable; versatility = different access forms (SPARQL endpoint, downloadable RDF, API); performance = efficiency/scalability of querying/reasoning — strongly dependent on the underlying framework and tech stack (e.g., some OWL variants have nonscalable reasoning).
- **Inferred inaccuracy multiplication**: one wrong base assertion (e.g., an incorrect subclass relation) can balloon into thousands of incorrect inferred statements once a reasoner propagates it across all instances.

## Mental Models
- **Quality is multidimensional, not a single score**: a model can be simultaneously consistent-but-inaccurate, or accurate-but-irrelevant, or complete-but-untrustworthy — treat each dimension as an independent axis to audit.
- **Consistency ≠ Accuracy**: a model that violates no logical constraints can still be substantively wrong about the world; conversely a model with real factual errors may remain logically consistent (unless the contradicting statements are non-vague, in which case both can be false but not both true).
- **Trust as a social/psychological construct, not just a formula**: reputation, endorsement, provenance, and honest (non-hyped) reporting of quality metrics shape trust independently of raw accuracy numbers — a model claiming 60% accuracy and delivering it can be more trustworthy than one claiming 90% and delivering 75%.

## Anti-patterns
- **Treating vague statements as straightforwardly "true/false" when measuring accuracy**: vague assertions can be true for one group and false for another without either being wrong — measure with inter-agreement, not a single verdict.
- **Assuming full gold standards exist for completeness measurement**: they rarely do, especially for population completeness — relying on their absence as an excuse not to measure at all is itself a pitfall; use partial/silver standards instead.
- **Letting inferred inaccuracy multiply unchecked**: not limiting inference or not fixing a root-cause inaccurate assertion lets a single mistake propagate into thousands of wrong derived facts.
- **Conflating consistency with correctness**: declaring a model "good" because it passes consistency checks, without separately verifying accuracy.
- **Uncoordinated multi-party modeling without governance**: different modelers independently choosing different representations for the same construct (e.g., multiple n-ary relation patterns) creates redundancy/inconcision.
- **Leaving in "temporary" hacks or legacy elements**: shortcuts taken under deadline pressure (e.g., adding unresolved duplicate synonym-entities to be merged "later") or obsolete domain elements that are never cleaned up, degrading conciseness and raising maintenance/inconsistency risk.
- **Building/choosing a model without considering the target task's requirements**: leads to low relevancy and, per Ch. 10, can actively harm an application even when the model looks accurate/complete on paper.
- **Overclaiming or hyping a model's quality**: e.g., a vendor's press-release claim of "a 1 billion word HR ontology" — misrepresentation erodes trust once scrutinized, more than modest but honest claims.

## Code Examples
```text
# Illustrative reasoning-based inconsistency check (not literal book syntax)
Constraint: capitalOf(x, y) => type(x, City) AND type(y, Country)
Assertion:  capitalOf(EiffelTower, France)   # City-typed check fails -> flagged inconsistent

# Illustrative cardinality constraint for consistency
Constraint: hasNaturalMother(x, y) has max-cardinality 1 on y
Assertion1: hasNaturalMother(John, Jane)
Assertion2: hasNaturalMother(John, Kim)      # violates max-cardinality -> inconsistency
```
- **What it demonstrates**: how explicit axioms (type restrictions on relation arguments, cardinality limits) let a reasoner automatically flag accuracy/consistency violations — but only if the model is well-axiomatized enough to support this ("adequately axiomatized and not contain too many errors already").

## Reference Tables
*Quality dimension summary (reconstructed):*

| Dimension | Definition | Typical measurement |
|---|---|---|
| Semantic accuracy | % of assertions accepted as true | human judges (experts/users/crowd) + inter-agreement; outlier detection; logical consistency checks |
| Completeness (schema) | are needed classes/relations/attributes/axioms present | gold/partial/silver standard comparison |
| Completeness (population) | are needed instances/assertions/values present | gold/partial/silver standard; cardinality/average-value heuristics |
| Consistency | free of logical/semantic contradiction | constraint violation checks; reasoner-based consistency checking |
| Conciseness | absence of redundant/duplicate/legacy elements | query-equivalence analysis; similarity metrics; corpus/gold-standard comparison |
| Timeliness | reflects current state of world | accuracy/completeness vs. contemporary knowledge; update frequency + domain volatility |
| Relevancy | useful for the specific task/application (application-dependent) | task-fit analysis; missing-critical-info checks; accessibility checks |
| Understandability | ease of correct human interpretation | direct user assessment of docs; observed usage-error analysis |
| Trustworthiness | subjective confidence in quality | reputation/adoption; formal evaluations/reports; provenance transparency |
| Availability/Versatility/Performance | obtainability / access forms / efficiency-scalability | out of this book's main scope |

*Causes of inaccuracy*: IE-method error rates, source data errors (e.g., ~2.8% of Wikipedia statements wrong per cited study), misunderstood element semantics, lack of domain expertise, vagueness mistaken for falsity.

*Causes of incompleteness*: domain size/complexity, IE inaccuracy, lack of appropriate source data, vagueness (disagreement overhead), domain volatility/semantic change.

*Causes of inconciseness*: uncoordinated multi-party modeling, multi-application optimization conflicts, unremoved temporary hacks, unremoved legacy elements.

## Worked Example
DBpedia's incorrect typing of the former country Yugoslavia as a "Musical Artist" (vs. its correct statement that Belgrade is Serbia's capital) is used to illustrate accuracy measurement directly: with only these two assertions, the model would be 50% accurate. The chapter then layers on a timeliness angle with the same entity: a model that still treats Yugoslavia as an active single country, unaware of the successor states (Serbia, Croatia, etc.), is not timely — showing how one entity can expose two different quality dimensions simultaneously. Contrast with DBpedia Live's same-day propagation of Kazakhstan's capital rename (Astana → Nur-Sultan) as a positive timeliness example.

## Key Takeaways
1. There is no single "quality score" for a semantic model — evaluate accuracy, completeness, consistency, conciseness, timeliness, relevancy, understandability, and trustworthiness independently.
2. Relevancy is the one dimension that is inherently application-dependent; the rest are application-neutral.
3. Vagueness undermines straightforward accuracy/completeness measurement — use inter-agreement metrics and treat disagreement as expected, not necessarily an error.
4. Reasoning-based inference can multiply a single inaccurate assertion into many; contain this by limiting inference scope or fixing root causes.
5. Consistency and accuracy are orthogonal — don't assume one implies the other.
6. Completeness measurement almost always relies on partial gold standards or silver standards rather than true gold standards.
7. Understandability is systematically underweighted by modelers relative to its downstream impact on accuracy, relevancy, and trust.
8. Trustworthiness is partly social/psychological (reputation, provenance, honest reporting) and not reducible to a technical accuracy number.

## Connects To
- **Ch 1**: ESCO's `essential_for` vagueness example reused here to show accuracy-measurement complications.
- **Ch 3**: vagueness, as defined there, directly undermines accuracy and completeness measurement here.
- **Ch 5**: model development process determines which quality dimensions are achievable and how.
- **Ch 6 (Bad Descriptions)**: expands on the understandability dimension's root causes.
- **Ch 7 (Bad Semantics)**: expands on the misunderstood-elements/inconsistency root causes of inaccuracy.
- **Ch 8 (Bad Model Specification/Knowledge Acquisition)**: expands on domain-expert limitations and source-data reliability issues feeding inaccuracy/incompleteness/bias.
- **Ch 9 (Bad Quality Management)**: the operational practices for managing these dimensions over the model's lifecycle.
- **Ch 10 (Bad Application)**: expands on relevancy/harm when a model is mismatched to its task.
