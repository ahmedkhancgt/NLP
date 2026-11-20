---
name: semantic-modeling-data
description: "Knowledge base from \"Semantic Modeling for Data\" by Panos Alexopoulos. Use when building or evaluating semantic models, ontologies, or knowledge graphs — the semantic gap, modeling elements (classes, relations, attributes), linguistic phenomena, model quality, model development process, or diagnosing/avoiding bad descriptions, bad semantics, bad model specification, bad quality management, bad application, and bad strategy/organization in semantic modeling."
---

# Semantic Modeling for Data

Knowledge base distilled from Panos Alexopoulos's *Semantic Modeling for Data* (O'Reilly), organized around the book's core distinction: **pitfalls** (clearly wrong decisions with a correct fix) vs. **dilemmas** (genuine trade-offs with no universal answer, requiring context-specific decision procedures).

## Core Frameworks

### The Semantic Gap
The central thesis: data-related initiatives fail not from lack of data/technology but from the gap between how data is modeled on the supply side and how it is understood/used on the exploitation side. Closing it requires deliberate semantic modeling as a discipline distinct from (and complementary to) machine learning: semantic models are symbolic, explicit, and built for deductive reasoning; ML models are subsymbolic, implicit, and built for inductive generalization.

### Pitfall vs. Dilemma
- **Pitfall**: a clearly wrong decision/omission with undesired consequences — has a right answer. (Covered by Ch. 6-11.)
- **Dilemma**: a choice among options each with real pros/cons, no universal correct answer — treat as a decision-making problem: enumerate options, evaluate against feasibility/cost-benefit/strategic criteria for your context. (Covered by Ch. 12-14.)

### The Five General Modeling Elements (Ch. 2)
Entity, Relation, Class, Attribute, Term — plus standardized relation types: lexicalization/synonymy, instantiation, meaning inclusion (subsumption), part-whole (6 subtypes), semantic relatedness, mapping/interlinking. The recurring fallacy to guard against: **"concept = class."** A concept is a class only if it has (or can have) genuine instances — always run the instance test before modeling something as a class.

### Semantic/Linguistic Phenomena (Ch. 3) — never conflate these
- **Ambiguity**: multiple discrete readings of a term.
- **Vagueness**: borderline cases, no sharp boundary (degree-vagueness along a dimension, or combinatory vagueness from unsettled necessary/sufficient conditions). Sorites Paradox is the canonical illustration.
- **Uncertainty**: truth unknown from incomplete knowledge.
- **Inexactness**: precise but wide-ranged — not actually a problem.
- **OntoClean** (Rigidity, Identity, Unity, Dependence): validates whether a subsumption relation is semantically sound.
- **Open-World vs. Closed-World Assumption**: RDF(S)/OWL use OWA by default — `domain`/`range` behave as inference rules, not constraints (use SHACL for real validation).
- **Semantic change/drift**: a term's meaning shifts over time across label, intension, and extension.

### Quality Dimensions (Ch. 4) — none implies another
Semantic Accuracy, Completeness (schema + population), Consistency, Conciseness, Timeliness, Relevancy (the one application-dependent dimension), Understandability, Trustworthiness, plus Availability/Versatility/Performance. Key insight: **consistency ≠ accuracy**; a model can be logically consistent yet substantively wrong, or accurate yet technically inconsistent (if vague).

### Development Life Cycle (Ch. 5) — six iterative activities
Setting the Stage (what/why/how/who/who-cares) → Deciding What to Build (competency questions, feasibility/importance) → Building It (knowledge acquisition mechanisms) → Ensuring It's Good (quality metrics/measurement) → Making It Useful (application integration) → Making It Last (evolution/change management). Treat as continuous, never one-off.

## The "Bad X" Pitfall Catalog (Ch. 6-11)

| # | Chapter | Core failure mode |
|---|---|---|
| 6 | Bad Descriptions | Bad names, missing/bad definitions, ignored vagueness, undocumented biases — the human-interpretability side of the semantic gap. |
| 7 | Bad Semantics | Misusing formal elements correctly defined by the language: bad synonymy, bad mapping (`owl:sameAs` misuse, referential opacity), bad subclasses (instantiation-as-subclassing, parts-as-subclasses, rigid-under-nonrigid, disjoint multi-parenting), bad axioms (wrong transitivity assumptions, mistaking inference rules for constraints). |
| 8 | Bad Model Specification & Knowledge Acquisition | Building the wrong thing (expressed ≠ desired features), wrong knowledge sources (bad data, fallible experts — Dunning-Kruger, availability bias), wrong acquisition tools (Maslow's Hammer, non-transferable benchmarks). |
| 9 | Bad Quality Management | Not treating quality as trade-offs, not linking metrics to risk/benefit, using misleading/incomparable/arbitrary-threshold metrics, confusing metrics with signals, mismeasuring vague-assertion accuracy, conflating model quality with extraction-tool quality. |
| 10 | Bad Application | Assuming a domain-matched model is automatically beneficial for a specific task (entity resolution, semantic relatedness) — "knowledge can hurt you." |
| 11 | Bad Strategy & Organization | No strategy, or a strategy incompatible with context; buying into myths (must use RDF, public models are reusable as-is); underestimating cost/complexity; wrong team composition/attitudes; no governance → semantic divergence. |

## The Dilemmas Catalog (Ch. 12-14)

| # | Chapter | Key dilemmas |
|---|---|---|
| 12 | Representation | Class-or-Individual? Subclass-or-not? Attribute-or-Relation? To-Fuzzify-or-not? |
| 13 | Expressiveness & Content | What lexicalizations? How granular? How general? How negative (negative knowledge)? How many truths (contextualization)? How interlinked? |
| 14 | Evolution & Governance | Remember-or-forget removed statements? Run-or-pace releases? React-or-prevent? Semantic drift measurement. Centralized-or-decentralized governance? |

## Chapter Index

| Ch | Title | File |
|---|---|---|
| 1 | Mind the Semantic Gap | chapters/ch01-mind-the-semantic-gap.md |
| 2 | Semantic Modeling Elements | chapters/ch02-semantic-modeling-elements.md |
| 3 | Semantic and Linguistic Phenomena | chapters/ch03-semantic-and-linguistic-phenomena.md |
| 4 | Semantic Model Quality | chapters/ch04-semantic-model-quality.md |
| 5 | Semantic Model Development | chapters/ch05-semantic-model-development.md |
| 6 | Bad Descriptions | chapters/ch06-bad-descriptions.md |
| 7 | Bad Semantics | chapters/ch07-bad-semantics.md |
| 8 | Bad Model Specification and Knowledge Acquisition | chapters/ch08-bad-model-specification-and-knowledge-acquisition.md |
| 9 | Bad Quality Management | chapters/ch09-bad-quality-management.md |
| 10 | Bad Application | chapters/ch10-bad-application.md |
| 11 | Bad Strategy and Organization | chapters/ch11-bad-strategy-and-organization.md |
| 12 | Representation Dilemmas | chapters/ch12-representation-dilemmas.md |
| 13 | Expressiveness and Content Dilemmas | chapters/ch13-expressiveness-and-content-dilemmas.md |
| 14 | Evolution and Governance Dilemmas | chapters/ch14-evolution-and-governance-dilemmas.md |
| 15 | Looking Ahead | chapters/ch15-looking-ahead.md |

## Topic Index

- **Abduction/Induction/Deduction** → Ch 2
- **Ambiguity** → Ch 3, Ch 6, Ch 10, Ch 13
- **Attribute vs. Relation dilemma** → Ch 2, Ch 12
- **Bias** → Ch 4, Ch 9, Ch 11, Ch 12, Ch 15
- **Class vs. Individual dilemma** → Ch 2, Ch 12
- **Competency questions** → Ch 8
- **Completeness** → Ch 4, Ch 9
- **Consensus (semantics as consensus-based)** → Ch 1, Ch 15
- **Consistency** → Ch 4
- **Definitions (textual)** → Ch 6
- **Entity resolution / disambiguation** → Ch 10
- **Fuzzification / fuzzy logic** → Ch 12
- **Generality of entities** → Ch 13
- **Governance** → Ch 11, Ch 14
- **Granularity** → Ch 13
- **Instantiation vs. subsumption** → Ch 2, Ch 7, Ch 12
- **Interlinking (owl:sameAs, skos:exactMatch)** → Ch 2, Ch 7, Ch 13
- **Knowledge acquisition** → Ch 5, Ch 8
- **Knowledge graph (definitions, myths)** → Ch 1, Ch 5, Ch 11
- **Lexicalization / synonymy** → Ch 2, Ch 7, Ch 13
- **Machine learning vs. symbolic representation** → Ch 1, Ch 5, Ch 15
- **Metrics vs. signals** → Ch 9
- **Naming (element names)** → Ch 6
- **Negative knowledge** → Ch 13, Ch 14
- **OntoClean (Rigidity/Identity/Unity/Dependence)** → Ch 3, Ch 7, Ch 12
- **Open/Closed World Assumption** → Ch 3, Ch 7, Ch 15
- **Part-whole relations** → Ch 2, Ch 7
- **Pitfalls vs. Dilemmas** → Ch 1
- **Property graphs vs. RDF/OWL** → Ch 2, Ch 15
- **Quality dimensions** → Ch 4, Ch 9
- **Relevancy (application-dependence)** → Ch 4, Ch 10
- **Semantic drift / semantic change** → Ch 3, Ch 14
- **Semantic gap** → Ch 1, Ch 15
- **Semantic model mining** → Ch 5, Ch 8
- **SHACL** → Ch 7
- **Strategy (model strategy)** → Ch 11
- **Subclass relations (pitfalls)** → Ch 7, Ch 12
- **Team composition / attitudes** → Ch 11
- **Transitivity / symmetry / inversion** → Ch 3, Ch 7
- **Trustworthiness** → Ch 4, Ch 9
- **Uncertainty** → Ch 3
- **Upper ontologies** → Ch 5, Ch 13, Ch 15
- **Vagueness** → Ch 3, Ch 6, Ch 7, Ch 9, Ch 12, Ch 13

## Supporting Files

- **[glossary.md](glossary.md)** — alphabetical term definitions with chapter references.
- **[patterns.md](patterns.md)** — all named decision procedures/techniques with when-to-use, how, and trade-offs.
- **[cheatsheet.md](cheatsheet.md)** — quick-reference decision rules, diagnosis tables, and red flags.

## Scope & Limits

This skill covers the conceptual/methodological content of *Semantic Modeling for Data* — pitfalls and dilemmas in building, evaluating, and governing semantic models (ontologies, knowledge graphs, taxonomies, controlled vocabularies). It is **not** a tutorial on any specific modeling language syntax (RDF/OWL/SKOS/SPARQL), and does not cover machine learning algorithm internals beyond how they interact with semantic modeling (mining methods, embeddings-as-relatedness, bias propagation). Techniques described (fuzzification, drift measurement, entity-resolution diagnostics) are presented at the level of detail the book provides — illustrative/methodological rather than production-ready implementations. When applying this material, always adapt the book's decision procedures to the user's actual context rather than citing a single "correct" choice, consistent with the book's own central thesis that most hard modeling questions are dilemmas, not pitfalls, with no universal answer.
