# Patterns

## Class-or-Individual Decision Procedure
**When to use**: An entity could legitimately be modeled as either a class or an individual (e.g., Eagle, Data Scientist), but your framework forces a choice.
**How**: Ask (1) does it have convincing, non-artificial instances? (2) do you care to describe those instances with relations/attributes? (3) are there facts about the entity itself you can't express if it's a class? If (1) is no, never make it a class. If (1)/(2) yes but (3) no, model as individual. If all three yes, use a workaround: custom relation instead of instantiation; dual differently-named entities (one class, one individual); or OWL2 punning.
**Trade-offs**: Workarounds trade inference power (losing automatic subclass-based inference) for expressivity (being able to state facts about both roles), or trade synchronization ease for full dual expressivity.

## To-Subclass-or-Not-to-Subclass
**When to use**: Deciding whether to specialize a class via a subclass vs. a relation/attribute to a separate class (e.g., restaurant cuisine).
**How**: Ask whether candidate subclasses carry genuinely distinguishing attributes/relations/axioms, whether they're common natural-discourse qualifiers in the domain, and how rigid they are (OntoClean).
**Trade-offs**: Subclassing is simpler and enables automatic subsumption inference, but makes it hard to talk about the specializing dimension directly or attach it to other entities; low-rigidity subclasses add disproportionate maintenance overhead.

## Attribute-or-Relation Decision
**When to use**: A characteristic could be modeled as a literal attribute or a relation to a semantically-described entity (e.g., film's original language).
**How**: Model as a relation only if you actually care to describe the semantics of the characteristic's possible values; otherwise use a plain attribute. Never reify purely numeric/date values as entities just to use them as relation objects.
**Trade-offs**: Relations add expressivity and linkability at the cost of extra modeling/maintenance overhead.

## Fuzzification (Six-Step Process)
**When to use**: You want to make a model's vagueness machine-processable rather than merely documented, and have validated it's worth the investment.
**How**: (1) detect/analyze vague elements, (2) decide fuzzification approach (single-dimension membership function, multivariate function, t-norm/t-conorm combination, or direct per-statement degree), (3) harvest degrees (ask directly, ask indirectly via application context, or mine from data), (4) assess fuzzy model quality (correct elements fuzzified? degrees accurate/consistent? provenance documented?), (5) represent degrees in your chosen framework, (6) apply the fuzzy model in consuming applications.
**Trade-offs**: Reduces (not eliminates) disagreement at real acquisition/maintenance cost; most mainstream languages lack native fuzzy support; increases (not decreases) documentation burden.

## Vagueness Documentation Protocol
**When to use**: A model contains elements admitting borderline cases.
**How**: (1) identify vague elements, (2) verify genuine vagueness (not just vaguely-worded but crisp), (3) specify vagueness type/dimension/applicability context as precisely as possible, (4) document this explicitly in the element's description.
**Trade-offs**: Cheap relative to fuzzification but doesn't reduce disagreement, only makes expectations clear.

## Lexicalization Cost/Benefit Assessment
**When to use**: Deciding whether to add a candidate term as a lexicalization of a model element.
**How**: For browsing/reference models, add freely. For automated tagging/detection, A/B test (build with/without the term, measure net correctness of tagging differences) or use fast heuristics: check for shortened forms, acronyms, cross-language translation shifts, multi-part-of-speech ambiguity, word count, and known ambiguity in external lexicons (WordNet/DBpedia).
**Trade-offs**: More lexicalizations improve recall/clarity but raise ambiguity risk and precision cost; rejected lexicalizations should be retained in reserve, not discarded.

## Granularity Decision Procedure
**When to use**: Deciding whether two closely related terms/meanings should be distinct model elements or merged (e.g., violin/fiddle).
**How**: Ask whether the difference is substantial/pragmatic vs. benign/theoretical, what the consequences of (not) making the distinction are, and whether you can actually support it (data, framework, downstream application capability).
**Trade-offs**: Finer granularity preserves meaning but increases disambiguation and semantic-matching difficulty; expect uneven granularity across a model as long as it's documented.

## Entity Resolution Ambiguity Diagnosis
**When to use**: An entity resolution system has low precision/recall when using a semantic model as disambiguation evidence.
**How**: Manually annotate a text sample with target AND nontarget entities; measure five ambiguity types (lexical, target-to-target, target-to-nontarget, nontarget-to-target, global) as specific ratios; map high scores on each type to a targeted fix (improve WSD, filter off-domain texts, prune noisy evidence model entities/relations, enrich model, or lower evidence threshold).
**Trade-offs**: More knowledge in the evidence model isn't automatically better — pruning noisy, low-prevalence elements often raises precision more than enrichment would.

## "Semantics-First" Relatedness Elicitation
**When to use**: An application needs a semantic-relatedness/similarity capability.
**How**: Elicit a concrete, decision-anchored relatedness definition from actual users (not a generic "should be related"), then escalate: (1) check if the model already has a matching relatedness relation, (2) check other existing models, (3) try standard measures (e.g., shortest-path), (4) define custom weighted rules on existing relations, (5) mine relatedness from scratch via external data.
**Trade-offs**: No single relatedness measure transfers across meaningfully different application contexts (e.g., bookstore vs. library) — always validate per application.

## Interlinking Risk/Benefit Framework
**When to use**: Considering `owl:sameAs`/`skos:exactMatch`-style mapping to an external model.
**How**: (1) scrutinize the external model's quality/compatibility (default low trust); (2) demand to know its management/evolution strategy; (3) assess interlinking cost/effort; (4) assess concrete benefit (standard-adoption value vs. simple content merging). If you proceed, selectively curate imported content rather than accepting it wholesale.
**Trade-offs**: Interlinking can reduce, not just increase, a model's value if the external model's quality/governance/evolution pattern doesn't fit.

## Semantic Model Specification (Three-Phase Process)
**When to use**: Starting (or iterating on) any semantic model development project.
**How**: (1) investigate context (stakeholder applications, existing models, pain points, remediation history); (2) specify features via core entity types (right generality level) and competency questions; (3) assess feasibility (framework capability + resource availability/cost) and importance (business/strategic) separately, then prioritize using both.
**Trade-offs**: Requirements must be actively discovered, not passively gathered — expect iteration and conflict resolution.

## Quality Trade-off Management
**When to use**: Specifying or managing a semantic model's development priorities.
**How**: Identify active trade-off pairs (accuracy-vs-completeness, conciseness-vs-completeness, conciseness-vs-understandability, relevancy-to-context-A-vs-B), make a deliberate strategic choice, document it.
**Trade-offs**: Pursuing 100% on every dimension simultaneously is not realistic; resources spent on one dimension are resources not spent on another.

## Semantic Drift Measurement (Domain-Adapted)
**When to use**: Prioritizing which parts of a model need more frequent maintenance.
**How**: Adapt the generic label/intension/extension drift model to your domain (determine which aspects actually carry meaning-relevant change and weight them accordingly); choose appropriate similarity metrics (set similarity for lexicalization/intension sets; rank-correlation like Kendall's Tau for fuzzy-ranked relations); build a parameterized measurement framework (concept types, time scope, relations, context/provenance) since different parameters yield different interpretations.
**Trade-offs**: No universal drift formula — must be adapted per domain; unparameterized drift reports are uninterpretable.

## Governance Centralization Diagnostic
**When to use**: Designing or reforming a semantic model's governance system.
**How**: Assess ten factors (development automation, modularity, quality-monitoring automation, domain breadth/vagueness, quality priorities, downstream change impact, shared goals/practices, model complexity, conflict-resolution effectiveness, organizational culture) to decide centralized vs. decentralized governance — potentially different degrees for different parts/tasks of the same model.
**Trade-offs**: Neither centralized nor decentralized is inherently better; both have valid supporting arguments depending on context.
