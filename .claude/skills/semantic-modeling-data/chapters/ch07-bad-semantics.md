# Chapter 7: Bad Semantics

## Core Idea
Even when a modeling language defines its elements with precise formal semantics, modelers routinely misuse those elements — treating hyponyms as synonyms, instances as classes, part-whole as subclassing, nontransitive relations as transitive, and inference rules as constraints — producing machine-facing semantic errors that propagate through reasoning.

## Frameworks Introduced
- **Identity-criteria compatibility check for multiple superclasses**: before giving a class more than one superclass, verify the superclasses share compatible identity criteria.
  - When to use: whenever a class appears to need multiple parents (e.g., Customer under Person and Organization).
  - How: if the superclasses are disjoint / give incompatible identity criteria, split the subclass into one class per identity (e.g., `PrivateCustomer` under Person, `CorporateCustomer` under Organization) rather than keeping one class with two incompatible parents.
- **"Is a" vs. "is a kind of" test for instantiation-vs-subclass confusion**: natural-language test to detect disguised instantiation relations.
  - When to use: whenever a hierarchical "is a" link between two elements is ambiguous about whether it's instantiation or subsumption.
  - How: try rephrasing as "X is a kind of Y" — if it sounds wrong (e.g., "Parthenon is a kind of Monument," "a Human is a kind of Species"), the relation is instantiation, not subclassing; also check whether the "subclass" actually has any legitimate instances of its own, and whether the two classes share OntoClean identity criteria.
- **Rigidity-compatibility check for subclassing**: a rigid class cannot be a subclass of a nonrigid class.
  - When to use: whenever a candidate subclass relation involves role-like categories (Customer, Teacher, Contestant, InfectiveAgent).
  - How: check the rigidity of both classes (Ch. 3's OntoClean framework); if the subclass is rigid (essential to instances, e.g., Person) and the superclass is nonrigid (e.g., Customer), the relation is invalid — the rigid class would wrongly inherit nonrigidity.
- **Mixed-relation-type transitivity check**: before asserting a hierarchy is transitive, verify the relations combined are of the same semantic type and that type is actually transitive.
  - When to use: whenever chaining two or more hierarchical links (broader/narrower, part-whole, instantiation).
  - How: identify each link's relation type (instantiation vs. meaning-inclusion vs. one of the six part-whole subtypes); transitivity only holds, if at all, within a single consistent relation type, and even then not universally (e.g., Member-Collection part-whole is never transitive).

## Key Concepts
- **Bad synonymy**: labeling terms as synonyms/lexicalizations of the same entity when they aren't truly interchangeable (e.g., ESCO listing "Interest Analyst" as a synonym of "Labor Economist"; Babelnet listing "Manchester United" as a synonym of "Arsenal FC"). Practical consequence: search/booking systems built on these models return or act on the wrong entity.
- **False friends**: cross-language terms that look/sound alike but differ in meaning (English "embarrassed" vs. Spanish "embarazada" = pregnant; English "sensible" vs. French/Spanish "sensible" = sensitive) — a specific, systematic source of bad synonymy in multilingual models, especially from string-similarity-based automatic methods.
- **owl:sameAs vs. owl:equivalentClass vs. skos:exactMatch**: `owl:sameAs` asserts full identity (merges all statements about both entities, requires OWL Full for classes, with poor reasoning performance); `owl:equivalentClass` asserts only extensional equivalence (same instance set, not full identity); `skos:exactMatch` is similar in spirit to `owl:sameAs` but explicitly does NOT require or imply statement merging. Conflating these causes "contamination" of a model with the other model's unwanted implied statements.
- **Referential opacity**: when two entities refer to the "same thing" but the attributes/relations attached to them in their respective models are not mutually acceptable (e.g., linking OpenCyc's India concept to WordNet's synset brings along WordNet-specific assertions like "is an instance of NounSynset" that are meaningless in OpenCyc).
- **Instantiation-as-subclassing**: individual entities (Calgary, Parthenon) incorrectly modeled as subclasses of their type (City, Landmark) rather than as instances — caused by the ambiguity of the English "is a" construction, the unavailability of second-order classes in languages like OWL-DL, conflated dual senses of a term (e.g., "democracy" as governance-system vs. as a type of state), conflating narrower/broader taxonomic hierarchies with formal subclass hierarchies, and misleading tutorials that equate "concept" with "class."
- **Parts-as-subclasses**: representing part-whole relationships (Engine part-of Car) as subclass relationships (Engine subclassOf Car) — caused by confusing "subset of a set" with "part of a set's members," and by the illusion that class-based languages can only express hierarchy via subclassing.
- **Rigid classes as subclasses of nonrigid classes**: e.g., making Person a subclass of Customer, or Bacterium a subclass of InfectiveAgent — wrongly implies all persons are essentially customers, or all bacteria are essentially infective, when these are role/circumstance-dependent (nonrigid) categories a rigid class should not inherit from.
- **Common superclasses with incompatible identity criteria**: e.g., Customer as a subclass of both Person and Organization, implying every customer is simultaneously a person and an organization — invalid because Person and Organization are disjoint classes with incompatible identity criteria.
- **Transitivity misapplied to hierarchies**: chaining instantiation + meaning-inclusion, or different part-whole subtypes, across a "hierarchy" produces nonsensical conclusions (Einstein is a Physicist, Physicist is a Profession ⇏ Einstein is a Profession; Amsterdam part-of Netherlands, Brazil part-of UN ⇏ Amsterdam part-of UN).
- **SKOS's broaderTransitive trap**: SKOS deliberately avoids making `skos:broader`/`skos:narrower` transitive by default, instead offering `skos:broaderTransitive`/`skos:narrowerTransitive` as transitive super-relations — but because ANY two entities related by a relation are also related by its super-relation, asserting broader twice still lets a reasoner derive a (possibly nonsensical) broaderTransitive link across the whole chain. The relation's name suggests guaranteed transitivity, but it should only be used when the underlying broader/narrower chain is actually transitive.
- **Vague relations destroy transitivity**: isNearTo(Italy, Greece) and isNearTo(Greece, Turkey) does not imply isNearTo(Italy, Turkey); vagueness in applicability criteria means different pairs can satisfy the relation along different, non-composable dimensions. SKOS explicitly warns that `skos:related` is not transitive.
- **Vague classes and complementation (owl:complementOf)**: for a vague class (e.g., HappyPerson), its logical complement (entities NOT in the class) is not the same as its common-sense "opposite" (e.g., UnhappyPerson) — morphological antonyms ("unhappy"), genuine antonyms ("sad"), and explicit negations ("not happy") behave differently and are not interchangeable, especially for vague gradable predicates.
- **Mistaking inference rules for constraints (domain/range in RDF(S)/OWL)**: declaring a relation's domain/range (e.g., `wasBornIn`: domain=Person, range=Location) does NOT validate data the way an E-R cardinality constraint would — under the Open World Assumption, a reasoner instead infers that any entity used as subject/object of that relation IS a member of the domain/range class, even if nonsensical (e.g., asserting "Amsterdam wasBornIn Ajax" causes the reasoner to infer Amsterdam is a Person and Ajax is a Location).
- **SHACL as the fix for OWL/RDF's constraint gap**: the W3C Shapes Constraint Language (2017) was introduced specifically because RDF(S)/OWL are built for open-world inference, not closed-world data validation — SHACL lets you define real validating constraints (e.g., cardinality limits) separately from OWL's inference-rule semantics.

## Mental Models
- **"Is a" is dangerously overloaded**: the same English surface pattern expresses both instantiation ("John is a Football Player") and subsumption ("A Football Player is an Athlete") — never trust the pattern alone; always verify with the "is a kind of" test and identity-criteria check.
- **Rigidity flows downhill, not uphill**: a subclass inherits ALL the properties (including nonrigidity) of its superclass, so a genuinely rigid class can never safely sit beneath a nonrigid one.
- **Open-world inference rules ≠ closed-world constraints**: in RDF(S)/OWL, "defining what's allowed" (domain/range) is not the same speech-act as "validating what's present" — the reasoner will happily invent new facts to keep your nonsensical assertion consistent, rather than rejecting it.
- **Vagueness breaks compositionality**: anywhere vagueness appears (transitivity, negation/complementation, synonymy thresholds), formal operations that work cleanly on crisp predicates (transitive chaining, logical complement) silently produce wrong results — treat every vague relation as a candidate for broken composition.

## Anti-patterns
- **Treating near-synonyms or merely-related terms as true synonyms**: driven by (a) avoiding the overhead of creating a full new entity, (b) conflating synonymy with search-expansion similarity (synonym rings quietly absorbing "merely similar" terms like Java/C++), (c) false friends in multilingual contexts, (d) ignoring context-dependence of near-synonyms (big/large vs. vacation/holiday), (e) missing subtle domain-specific meaning differences (violin vs. fiddle for classical musicians; CEO vs. COO for laypeople).
- **Mapping with `owl:sameAs` as a reflex "good practice" rather than a deliberate decision**: adding interlinking to external models (DBpedia, etc.) purely because Linked Data culture encourages it, without a concrete business case justifying the maintenance/risk cost.
- **Blindly importing all statements from an interlinked external model**: mapping to ESCO (or any external KB) does not obligate accepting every one of its statements as valid for your model — selectively import only what's semantically compatible.
- **Reaching for `skos:closeMatch` to "soften" strict mapping semantics**: relaxing `owl:sameAs`/`skos:exactMatch` into vaguer relations like `skos:closeMatch` typically just relocates and multiplies disagreement/misuse rather than resolving it.
- **Instantiation disguised as subclassing**: individual real-world entities (Calgary, Parthenon) or singular occupation-holders (Factory Worker, Secretary in SNOMED) wrongly modeled as classes/subclasses of their type class, driven by "is a" ambiguity, second-order-class restrictions in languages like OWL-DL, dual-sense terms, taxonomy/subclass-hierarchy conflation, and misleading published tutorials equating "concept" with "class."
- **Parts modeled as subclasses**: e.g., Engine as `subclassOf` Car instead of `partOf` Car — caused by the subset/part-of-members confusion and the illusion that class languages can only express hierarchy via subclassing.
- **Rigid-under-nonrigid subclassing**: Person under Customer, Bacterium under InfectiveAgent — makes essential membership out of what is actually circumstantial/role-based.
- **Disjoint multi-parenting**: giving a class two superclasses with incompatible identity criteria (Customer under both Person and Organization) implies impossible dual identity.
- **Assuming all hierarchies are transitive by default**: chaining instantiation with meaning-inclusion, or chaining incompatible part-whole subtypes (Component-Integral + Member-Collection, as in "the heart is part of the musician, the musician is part of the orchestra ⇒ the heart is part of the orchestra" — false), produces nonsensical inferred facts.
- **Trusting a relation's name to guarantee a formal property**: e.g., assuming `skos:broaderTransitive` assertions are always safely transitive just because of the name — must verify the underlying broader/narrower chain is genuinely transitive first.
- **Chaining vague relations as if they composed transitively**: isNearTo, isSimilarTo, and other vague relatedness relations do not compose reliably across pairs.
- **Conflating a vague class's logical complement with its intuitive "opposite"**: NotHappyPerson (logical negation) ≠ UnhappyPerson (morphological antonym) ≠ SadPerson (genuine antonym) — these differ in extension precisely because "happy" is vague.
- **Mistaking RDF(S)/OWL domain/range declarations for data-validation constraints**: under OWA, nonsensical relation usage gets "explained away" via inferred (and wrong) class memberships rather than flagged as an error — use SHACL (or an equivalent validation layer) when actual constraint enforcement is needed.

## Code Examples
```turtle
# Illustrative: the domain/range "constraint" trap in RDF(S)/OWL
:wasBornIn rdfs:domain :Person ;
           rdfs:range  :Location .

# Nonsensical assertion:
:Amsterdam :wasBornIn :Ajax .

# Under OWA, a reasoner does NOT reject this. Instead it infers:
:Amsterdam rdf:type :Person .     # wrong but logically "consistent"
:Ajax      rdf:type :Location .   # wrong but logically "consistent"
```
```turtle
# Illustrative: SKOS's broaderTransitive trap
:Amsterdam      skos:broader :TheNetherlands .
:TheNetherlands skos:broader :UnitedNations .

# A reasoner infers (super-relation of skos:broader):
:Amsterdam      skos:broaderTransitive :TheNetherlands .
:TheNetherlands skos:broaderTransitive :UnitedNations .

# Because skos:broaderTransitive IS transitive, the reasoner further infers:
:Amsterdam skos:broaderTransitive :UnitedNations .   # nonsensical
```
- **What it demonstrates**: how open-world inference rules (domain/range) silently fabricate wrong facts instead of rejecting bad input, and how a relation deliberately engineered to carry transitive semantics (`broaderTransitive`) can still propagate a chain into an absurd conclusion if the underlying chain mixes incompatible real-world hierarchy types.

## Reference Tables
*Rigid vs. nonrigid class pairs (reconstructed from Table 7-3):*

| Rigid | Nonrigid |
|---|---|
| Person | Teacher |
| Organization | Customer |
| Fruit | Food |
| Country | Contestant |
| Engine | CarPart |

*Bad synonym examples (reconstructed from Table 7-1):*

| Source | Entity | Problematic "synonym" |
|---|---|---|
| ESCO | Economist | "Interest Analyst" (too narrow a sub-specialty to be a true synonym) |
| Babelnet | Arsenal FC | "Manchester United" (clearly a different entity) |
| WordNet | Chief Executive Officer | "Chief Operating Officer" (related but distinct role) |

*False friends (reconstructed from Table 7-2):* Ambulance/Ambulanz(DE: emergency room), Rat/Rat(DE: advice), Stadium/Stadion(DE: stage), Excited/Excité(FR: aroused), Pain/Pain(FR: bread), Library/Librairie(FR: bookshop), Travel/Travail(FR: work), Compromise/Compromiso(ES: commitment), Preoccupied/Preocupado(ES: worried), Lecture/Lectura(ES: reading).

*Mapping relation semantics summary*: `owl:sameAs` (full identity, merges statements, needs OWL Full for classes), `owl:equivalentClass` (extensional equivalence only), `skos:exactMatch` (high-confidence interchangeability, no statement merging implied), `skos:closeMatch`/`broadMatch`/`narrowMatch`/`relatedMatch` (progressively weaker, more vague — risk of misuse increases).

## Worked Example
The chapter's running "disjoint multi-parenting" fix, carried over from Ch. 6's Customer/Person/Organization example: instead of `Customer subclassOf Person` and `Customer subclassOf Organization` (which implies every customer is both a person and an organization simultaneously — impossible since Person and Organization have incompatible identity criteria), split into `PrivateCustomer subclassOf Customer`, `PrivateCustomer subclassOf Person`, `CorporateCustomer subclassOf Customer`, `CorporateCustomer subclassOf Organization`. This simultaneously fixes the naming problem (Ch. 6) and the identity-incompatibility problem (Ch. 7) with a single refactor — illustrating how bad descriptions and bad semantics are often two faces of the same root modeling error.

## Key Takeaways
1. Understand the exact formal behavior of every relation/axiom type in your chosen modeling language (domain/range, subclassOf, sameAs, etc.) — don't assume intuitive English meaning transfers directly.
2. Semantic inference is not the same as logical inference in a vacuum: garbage input to a logically valid rule produces garbage (but logically "consistent") output.
3. Treat synonymy as a high-threshold relation requiring multi-judge validation and explicit documentation of your similarity criteria — don't let search-expansion needs quietly relax your synonymy definition.
4. Before interlinking to an external model, evaluate whether the business case justifies it, scrutinize the external model's semantics, and selectively import only compatible statements — don't interlink reflexively "because Linked Data."
5. Before asserting a subclass relation, run it through the "is a kind of" test, check for shared identity criteria, and check rigidity compatibility.
6. Don't assume hierarchies (broader/narrower, part-whole) are transitive by default — verify relation-type consistency and check for known nontransitive subtypes (e.g., Member-Collection part-whole).
7. Vagueness breaks transitivity and complicates logical complementation — handle vague relations and vague classes with extra caution, never composing them as if they were crisp.
8. RDF(S)/OWL domain/range axioms are inference rules under the Open-World Assumption, not data-validation constraints — use SHACL (or similar) when you actually need validation.

## Connects To
- **Ch 2**: directly revisits and "breaks" the subsumption, part-whole, instantiation, and mapping relations defined there.
- **Ch 3**: applies rigidity/identity (OntoClean) and the OWA framework introduced there as diagnostic tools for bad subclassing and bad axioms.
- **Ch 6**: the Customer/Person/Organization naming example is directly extended here into a formal semantic-error diagnosis.
- **Ch 4**: bad semantics is a primary driver of the Semantic Accuracy and Consistency quality dimensions failing.
- **Ch 8 (Bad Model Specification and Knowledge Acquisition)**: shifts focus from "using the framework wrong" (this chapter) to "getting the wrong requirements/content in the first place."
- **Ch 12-13 (Dilemmas)**: revisits the interlinking cost-benefit dilemma and vagueness-handling trade-offs raised here.
