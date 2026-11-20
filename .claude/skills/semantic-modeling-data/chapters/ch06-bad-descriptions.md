# Chapter 6: Bad Descriptions

## Core Idea
The first major category of semantic modeling pitfall is failing at human-interpretability: bad element names, missing/bad definitions, ignored vagueness, and undocumented biases/assumptions silently break the human side of the semantic gap even when machine-side formalization is technically correct.

## Frameworks Introduced
- **The "subclass trap" diagnostic**: when deciding a subsumption hierarchy (e.g., Customer/Person/Organization), verbalize each candidate hierarchy as a natural-language universal statement and check it for absurdity.
  - When to use: before committing to any class hierarchy involving role-like or context-dependent categories (Customer, Employee, Student).
  - How: "all X are Y" (subclass direction) and "all X are both Y and Z" (multiple-inheritance direction) — if either sounds false, the classes are misnamed; the fix is usually to rename the subclass to make its role-specific nature explicit (e.g., `PrivateCustomer`/`CorporateCustomer` instead of bare `Person`/`Organization`).
- **Four-step vagueness documentation protocol**: (1) identify vague elements (manually or via automated detection), (2) verify they are genuinely vague rather than just badly-worded but crisp, (3) specify the vagueness type/dimensions/context as precisely as possible, (4) explicitly document this in the element's description.
  - When to use: whenever a model contains predicates admitting borderline cases (per Ch. 3's vagueness framework).
  - How: distinguish degree-vagueness (state its dimension(s), e.g., "isNearTo: degree-vagueness along distance") from combinatory vagueness (state why no sharp combination rule exists), and note subjectivity/context-dependence explicitly.
- **Four ways to obtain definitions**: hire domain experts (costly, non-scalable, but authoritative); reuse existing dictionaries/glossaries (cheap but check copyright, quality, and domain-specificity mismatch); extract definitions automatically from text via lexico-syntactic patterns (works best on structured texts like medical/math texts); synthesize definitions from the model's own relations (verbalize subsumption/synonymy/instantiation facts into prose).
  - When to use: choosing a definition-sourcing strategy based on domain specialization, scale, and budget.

## Key Concepts
- **Bad naming**: a name is bad when it fails to help (or actively misleads) human interpretation — via ambiguity (multiple meanings), inaccuracy (too generic/specific/off-target), or unnecessary vagueness.
- **Why ambiguous names happen**: (1) modeler doesn't know of other interpretations, (2) modeler assumes other interpretations are irrelevant to the domain, (3) modeler assumes context will disambiguate for the user — all three are risky assumptions that don't hold for all users/contexts.
- **Why inaccurate names happen**: natural human communication tolerates context-resolved imprecision (e.g., answering "what car do you drive" with the model name, not the specific vehicle) — but a semantic model lacks that conversational safety net, so names must be more literal/specific than everyday speech (e.g., `CarModel` not `Car` for a class of car models).
- **Definitions needed when background knowledge can't be assumed**: specialized terms (e.g., "Agonal"), acronyms (e.g., "AWOL"), and general terms with domain-specific specialized meanings (e.g., "Ability to pay" in finance) all require explicit textual definitions, since even a better/more specific name isn't sufficient.
- **Why definitions get omitted**: the "everybody knows this" assumption (expert blind spot), difficulty/cost at scale, optionality in many modeling frameworks (definitions treated as non-core), and the "doesn't contribute to inference" attitude among logicians (valid only if the model is purely for automated reasoning AND all human stakeholders can use it unaided).
- **Circular definitions**: defining a term using only etymologically/morphologically related terms (e.g., defining "Poetry" via "poetic expression") — leaves the reader with no independent grounding.
- **"Nonclass way" class definitions**: defining a class without making clear what kind of entities are meant to instantiate it (e.g., defining Democracy purely as a political-system concept without stating its instances are meant to be countries) — a specific manifestation of the "concept = class" fallacy from Ch. 1/2.
- **Vagueness as a feature, not (only) a bug**: attempting to eliminate all vagueness by forcing crisp definitions doesn't guarantee users will apply the same crisp boundary, and can actively reduce usefulness (e.g., a strict €20 price cutoff for "moderate" excludes a €19.90 restaurant that's intuitively just as moderate). The right response is often to make vagueness explicit and manageable (via fuzzification/contextualization, covered in Ch. 12-13), not to pretend it away.
- **Vagueness detection via ML classifier**: the author's own study trained a supervised classifier (on WordNet vague/nonvague adjective senses, bag-of-words features) achieving 84% accuracy on held-out senses and 82% on CiTO relations — demonstrating vagueness detection is automatable but imperfect, and that vagueness is distinct from subjectivity (a subjectivity classifier correctly flagged only 7% of genuinely vague CiTO relations as such).
- **Undocumented biases and assumptions**: design choices made under real-world constraints (e.g., resolving ambiguity by picking the majority-sense meaning for a term) are acceptable only if made consciously AND communicated to all stakeholders — the harm isn't the choice itself but its invisibility to future maintainers.

## Mental Models
- **Names are tested in isolation**: judge an element's name on its own, stripped of surrounding context/relations/definitions, and ask what interpretations a reasonable human would independently assign — context-dependent clarity is an illusion that breaks the moment someone encounters the element out of context.
- **The friend-at-the-bar test for definitions**: whether a listener needs a better *name* (they can't disambiguate which sense is meant) or a better *definition* (they know which sense is meant but lack the domain knowledge to understand it) are different failures requiring different fixes.
- **Vagueness documentation ≠ vagueness elimination**: the goal of describing vagueness is to let users use the model safely and predictably, not to force a false precision that users will simply reinterpret anyway.

## Anti-patterns
- **Subclass-of-role-as-subclass-of-kind (the Customer/Person/Organization trap)**: modeling `Customer` as a superclass of `Person` and `Organization` (implying all persons/orgs are customers) or as their joint subclass (implying a customer is simultaneously a person AND an organization) — both are logically false. Fix: rename to role-specific classes (`PrivateCustomer`, `CorporateCustomer`).
- **Ambiguous relation names with unclear direction**: e.g., SKOS's `skos:broader` gives no cue about which argument is the broader concept — requires a side-note to clarify ("read as 'has broader concept'"). Fix: name relations as verb phrases that encode direction, e.g., `hasBroader`.
- **Overly generic attribute/relation names relative to their actual definition**: e.g., DBpedia's `Agent.cost` actually means construction cost only, but the bare name `cost` implies any kind of cost. Fix: rename to `hasConstructionCost` or similarly specific.
- **Relation names that drop their parent/supertype's semantics**: e.g., Schema.org's `sportsEvent`/`sportsTeam` as subproperties of `location`/`participant` lose that semantic link in their surface name. Fix: `isLocatedAt`, `hasParticipatingSportsTeam`.
- **Mislabeling a relation's true meaning in organizational language ("naming in people's heads")**: a search-expansion relation internally called "synonym" when it actually captures broader semantic relatedness (e.g., Java/C# expansion) — misleads everyone who works with the model, not just outside users. Fix requires both renaming in the model AND actively correcting the informal terminology people use.
- **Partial/incomplete definitions**: e.g., IPTC's "Fishing Industry" defined merely as "raising or gathering of fish," dropping the "industry" sense entirely.
- **Domain-mismatched definitions baked into the name's scope**: e.g., IPTC's "statistic" defined narrowly as weather-related numerical facts, when the term itself implies no such restriction.
- **Circular definitions**: defining a term purely via its own morphological relatives (Poetry → "poetic expression") without external grounding.
- **Defining a class without implying its intended instance type ("nonclass way")**: e.g., defining Democracy as an abstract political-system concept without stating the instances should be countries.
- **Treating vagueness as a defect to be silently special-cased**: e.g., classifying borderline restaurant cuisines as "both exotic and nonexotic" doesn't actually resolve who decides the borderline — it just relocates the arbitrary cutoff.
- **Confusing vagueness with subjectivity and treating them interchangeably**: the author's empirical comparison showed vague terms are NOT reliably flagged by subjectivity-detection tools — the two phenomena need separate detection and separate documentation.
- **Not documenting constraints, context-dependence, provenance, quality, applicability, and design biases**: e.g., silently resolving the lexicalization "tester" to mean only `Software Tester` (because it was the majority sense in the data) without recording that decision anywhere — risks future maintainers unknowingly violating the implicit rule (e.g., adding "tester" as a label for `Car Tester` later, reintroducing the very ambiguity the original decision avoided).
- **The "everybody knows this" domain-expert blind spot**: skipping definitions for terms the modeling team finds obvious, while actual model consumers (generalist annotators, cross-domain users) do not share that background and must resort to ad hoc web searches to understand elements.
- **Training ML models on vague/coarsened features without flagging it**: using binary proxies like `includesFamousActors` or categorical buckets like `old`/`new` instead of precise values bakes the training data's particular (and uncommunicated) interpretation of the vague term into the model, which may not transfer to other data.

## Code Examples
```text
# Naming fix pattern (illustrative, from the chapter's Customer example)
BAD:   Customer <- subclassOf <- Person
       Customer <- subclassOf <- Organization
       # implies: all persons AND organizations are customers (false)

BAD:   Customer -> subclassOf -> Person
       Customer -> subclassOf -> Organization
       # implies: every customer is simultaneously a person and an org (impossible)

GOOD:  PrivateCustomer    subclassOf  Customer
       PrivateCustomer    subclassOf  Person
       CorporateCustomer  subclassOf  Customer
       CorporateCustomer  subclassOf  Organization
```
- **What it demonstrates**: how renaming subclasses to reflect their actual role-specific scope (rather than reusing the broader kind's name) resolves a subsumption hierarchy that is otherwise either over- or under-general.

## Reference Tables
*Bad vs. better names (reconstructed from Table 6-1):*

| Bad modeling | Natural language it implies | Better modeling |
|---|---|---|
| Class `Democracy`, instance `United States of America` | "USA is a democracy" (fine, but class scope unclear) | Class `DemocraticCountry`, instance `United States of America` |
| Entity `Deep Learning`, relation `scoredHighIn` to `Jane` | "Jane scored high in deep learning" | Entity `Deep Learning Course`, relation `scoredHighInCourse` |
| Entity `Barcelona`, relation `visited` to `Santiago Bernabeu` | "Barcelona visited Santiago Bernabeu" (ambiguous: city or club?) | Entity `Barcelona FC`, relation `visited` to `Santiago Bernabeu stadium` |

*Good vs. bad definitions (reconstructed from Table 6-2, IPTC taxonomy):*

| Good | Bad |
|---|---|
| Dance: "the expression of emotion or message through movement" | Cinema: "cinema as art and entertainment" (circular) |
| Newspapers: "daily or weekly publications that present the day to day history of the world, as well as features, comics, etc." | Homicide: "killing of one person by another" (arguably fine, but contrast with poor ones) |
| Farms: detailed, instance-rich definition | Alternative energy: "alternative energy business" (circular) |
| Forecast: "prediction of the course of the weather..." | Jewelry: "accessories to clothing" (too generic) / Traffic: "traffic reports and/or warnings" (circular) / Pollution: vague-but-workable / statistic: wrongly weather-specific |

*Vague element documentation examples (reconstructed from Table 6-3):*

| Element | Vagueness nature |
|---|---|
| `isNearTo` | Degree-vagueness along distance |
| `isFunctionalPartOf` | Degree-vagueness along part's functional contribution |
| `isCompetitorOf` | Degree-vagueness along business-area and target-market overlap |
| `belongsToCategory` | Combinatory vagueness (no sharp membership conditions) |
| `isExpertAt` | Degree-vagueness along level of knowledge |

*Undocumented information categories*: constraints/restrictions, context-dependence, provenance, quality metrics, applicability scope, design decisions/biases.

## Worked Example
A real project to build an electricity-market ontology surfaced the practical cost of ignoring vagueness: domain experts could not agree on example instances of `Critical System Process` or `Strategic Market Participant` — not merely because of imprecise measurement, but because the underlying predicates ("critical," "strategic") are genuinely vague (combinatory and degree-vagueness respectively) with no consensual applicability criteria. A parallel tender-call evaluation process modeled decision conditions like "budget is high" and "company's experience is adequate" — both degree-vague — showing that even an explicit business rule (e.g., "a competitor is one with clients in the same industries and similar services") doesn't resolve the vagueness, since the threshold ("how many similar clients/services is enough?") remains undefined. The lesson: vagueness in business process knowledge spans structure, domain, and organizational-context dimensions, and different people executing the "same" process will diverge in outcome purely because of unaddressed vagueness, not because of data or rule deficiencies.

## Key Takeaways
1. Don't overestimate users' ability to correctly interpret ambiguous or inaccurate element names from context alone — be maximally specific in the formal name even if informal labels can stay colloquial.
2. Textual definitions are not optional "nice-to-haves" for specialized, acronym-heavy, or cross-domain terms — omitting them pushes real cost onto annotators/users who must research meanings externally.
3. Check every definition for circularity, partial coverage, and domain-mismatch before accepting it.
4. Vagueness should be detected, characterized (type, dimension, context), and explicitly documented — not silently ignored and not falsely eliminated by an arbitrary crisp cutoff that doesn't match how users actually think.
5. Vagueness and subjectivity are empirically distinct phenomena and need separate detection/handling — don't conflate them.
6. Every real-world modeling compromise (ambiguity resolution, scope restriction, optimization target) must be documented and communicated — undocumented bias is the actual risk, not the compromise itself.
7. When defining a class, explicitly indicate what kind of entity its instances are meant to be, or you risk the "concept = class" confusion from Ch. 1/2.

## Connects To
- **Ch 1 / Ch 2**: extends the "concept = class" fallacy into concrete "defining classes in a nonclass way" pitfalls.
- **Ch 3**: directly operationalizes the ambiguity/vagueness distinctions defined there into naming and definition remediation techniques.
- **Ch 4**: bad descriptions are a root cause of the Understandability quality dimension's degradation.
- **Ch 7 (Bad Semantics)**: shifts focus from human-facing description problems to machine-facing formal/logical problems.
- **Ch 12-13 (Representation / Expressiveness Dilemmas)**: revisits fuzzification and contextualization as structured techniques for handling vagueness beyond mere documentation.
