# Chapter 12: Representation Dilemmas

## Core Idea
Many pieces of semantic information can be legitimately represented in more than one way within the same modeling language (class vs. individual, subclass vs. relation/attribute, attribute vs. relation, crisp vs. fuzzy) — each option has real trade-offs, and the right choice depends on context-specific questions rather than a universal rule.

## Frameworks Introduced
- **Class-or-Individual Decision Procedure**: three diagnostic questions to resolve whether an entity should be modeled as a class or individual when the framework forces a choice.
  - When to use: whenever an entity (e.g., Data Scientist, Eagle) could legitimately be either, per Ch. 2's "instance test."
  - How: ask (1) does the entity have (convincing, non-artificial) instances? (2) do you care to describe those instances with relations/attributes in your model? (3) are there facts about the entity itself you can't easily express if it's a class? — No to (1): never model as class. Yes/No/anything to (1)/(2): model as individual (simpler). Yes to all three: use a workaround (custom relation instead of instantiation; dual-entity with disambiguating names; or OWL2 punning).
- **To-Subclass-or-Not-to-Subclass Decision Procedure**: extends the class/individual questions with two more, for deciding whether to specialize a class via subclassing vs. via a relation/attribute to a separate class.
  - When to use: e.g., deciding whether cuisine-of-restaurant should be `AsianRestaurant subclassOf Restaurant` vs. `Restaurant hasCuisine AsianCuisine`.
  - How: additionally ask (4) do the candidate subclasses carry extra attributes/relations/restrictions not shared by siblings or the parent? (5) are the subclasses commonly used distinguishing qualifiers in the domain's natural discourse? (6) how rigid are the candidate subclasses (per Ch. 3's OntoClean)? — low rigidity subclasses (highly volatile category membership) should make you reconsider subclassing in favor of an attribute/relation.
- **Attribute-or-Relation Decision Procedure**: resolves whether to model a characteristic as a literal-valued attribute or as a relation to a semantically-described entity.
  - When to use: whenever a characteristic (e.g., a film's original language, a film's color type) could go either way.
  - How: the deciding factor is simply how much you care to formally describe the semantics of that characteristic's possible values — if you want the value itself to have its own attributes/relations (e.g., Language with iso6391Code), model as a relation to an entity; if you're content with a bare literal, model as an attribute. Purely numeric/date-valued characteristics are almost never worth modeling as relations to reified number/date entities (impractical for infinite value spaces).
- **Six-Step Fuzzification Process**: detect/analyze vague elements → decide how to fuzzify each → harvest fuzzy degrees → assess fuzzy model quality → represent fuzzy degrees → apply the fuzzy model in downstream applications.
  - When to use: when deciding whether and how to make a model's vagueness machine-processable rather than merely documented.
  - How: see Key Concepts and Reference Tables below for each step's specific techniques.
- **Six "When to Fuzzify" Gating Questions**: a pre-investment checklist before committing to fuzzification.
  - When to use: before starting any fuzzification effort.
  - How: (1) which elements are genuinely/unavoidably vague (not just vaguely described)? (2) how severe/impactful are the resulting disagreements? (3) is the disagreement really caused by vagueness, not uncertainty/ambiguity? (4) would fuzzy degrees actually reduce disagreement? (5) can the consuming application actually exploit truth degrees to improve effectiveness? (6) can you acquire/maintain fuzzy degrees at a cost below the benefit they bring? — answer 4 and 5 via A/B testing (compare inter-agreement scores and downstream task effectiveness between fuzzy and crisp model versions).

## Key Concepts
- **Representation dilemma (defined)**: a situation with no single clearly correct or wrong modeling choice — all options are legitimate, each with its own pros/cons, requiring context-specific analysis rather than a universal rule (contrasted with a pitfall, which has a right answer).
- **OWL-DL's class/individual expressivity asymmetry**: you cannot define a class as an instance of another class, nor relate classes directly to other entities except via predefined axioms — this technical restriction is why many entities that are legitimately "both" a class and an individual (Eagle, Data Scientist) force an either/or choice.
- **Three workarounds for "both class and individual" entities**: (1) model as individual + custom binary relation instead of instantiation (keeps flexibility, loses standard subclass-based inference); (2) model as two differently-named entities, one class one individual (e.g., `Data Scientist (Profession)` as individual, `Data Scientist (Professional)` as class — gains full expressivity for both, loses automatic synchronization between the two); (3) OWL2 punning (same identifier, reasoner infers class-or-individual role from local context) — Michael Uschold's critique: punning is "syntactic trickery," not a real solution to the underlying metaclass problem, since it just builds an invisible wall between two interpretations that can confuse modelers/users.
- **Subclass vs. relation-based specialization trade-off**: subclassing is simpler/more intuitive and enables automatic subsumption inference (e.g., ChineseRestaurant inferred as AsianRestaurant), but makes it hard to talk directly about the specializing dimension itself (e.g., can't easily relate "Asian cuisine" to "recipes" or state that one cuisine influences another) or attach characteristics to the dimension values themselves.
- **When subclassing earns its keep**: when the candidate subclasses carry genuinely distinguishing attributes/relations/axioms not shared across siblings (mammals vs. reptiles re: body heat regulation) — vs. when siblings have no real distinguishing characteristics beyond the differentiator itself (left lung vs. right lung; red car vs. blue car), where subclassing adds no value.
- **Domain-discourse commonality as a subclassing signal**: if a qualifier is commonly used as a natural distinguishing term in how people talk about the domain (e.g., "comedy films," "Tarantino movies"), it's often worth promoting to a subclass even without deep internal elaboration; less-commonly-foregrounded qualifiers (film duration, dubbing languages) are better left as attributes/relations.
- **Rigidity as a subclassing caution signal**: low-rigidity subclasses (whose membership is especially volatile, e.g., `RestaurantWith50Employees`, more volatile than the already-nonrigid `AsianRestaurant`) should prompt reconsidering whether the maintenance overhead of subclassing is worth it versus a simple attribute/relation.
- **Subclass naming brevity vs. ambiguity trade-off**: shortening subclass names (e.g., `TarantinoFilm` instead of `QuentinTarantinoDirectedFilm`) is acceptable when disambiguation/inference is safely assumable by most users, but modelers must consciously bound how far they let this assumption stretch.
- **Fuzzification (concept fuzzification, Zadeh)**: assigning a real-valued truth degree in [0,1] to a vague statement — 1 = completely true, 0 = completely false, intermediate values = "partly true to a quantifiable extent." Explicitly NOT a probability: probability quantifies the likelihood of a well-defined event occurring; a fuzzy degree quantifies the extent to which a statement with undefined/blurry truth conditions can be perceived as true — different underlying mathematical frameworks (probability theory vs. fuzzy logic).
- **Fuzzy membership functions**: map a vague predicate's dimension value (e.g., age) to a [0,1] truth degree; common shapes include trapezoidal, triangular, left-shoulder, right-shoulder, and linear functions (Figure 12-4). Multi-dimensional quantitative vagueness can use a multivariate function, or separate single-dimension functions combined via fuzzy logic operators (t-norms for conjunction, t-conorms for disjunction).
- **T-norms / t-conorms**: functions combining two truth degrees while preserving classical AND/OR behavior at 0/1. Popular t-norms: minimum/Gödel (`min(a,b)`), product (`a·b`), Łukasiewicz (`max(0, a+b-1)`). Popular t-conorms: maximum (`max(a,b)`), probabilistic sum (`a+b-a·b`), bounded sum (`min(a+b,1)`). No universal rule picks the "right" t-norm/t-conorm for a domain — must be chosen empirically.
- **Direct-degree assignment**: for elements too multidimensional or qualitatively vague to fit a membership function, assign truth degrees directly per statement (e.g., "Google is CompetitorCompany to degree 0.3") rather than deriving them from a formula — trades harvesting scalability for applicability when no clean dimension exists.
- **Truth-degree harvesting methods**: ask people directly (structured/controlled but slow, doesn't scale, and requires careful question framing since vague predicates are non-complementary — "not tall" ≠ "short," per Ch. 7); ask people indirectly through application-embedded feedback (more natural/scalable but less predictable/controllable, requires careful system design); mine from data (comments/reviews/discussions — fully automatable and scalable, but noisy/polarized source data risks inaccurate extracted degrees).
- **Fuzzy model quality dimensions**: did you fuzzify the correct (genuinely vague, not just vaguely-worded-but-crisp) elements? Are the truth degrees accurate (best judged by whether they feel natural/intuitive to users — measured via fuzzy distance metrics rather than a single "golden value")? Are truth degrees consistent (treated as a distance, not a binary pass/fail, between potentially conflicting degree assignments)? Is the provenance of the truth degrees documented (how/by whom/from what data were they derived)?
- **Fuzzification bias**: like any modeling decision, assigning truth degrees can encode and propagate harmful bias (e.g., an age-based "old person" membership function used by an insurance company to set premiums) — fuzzification does not exempt a model from bias scrutiny (Ch. 9).
- **Fuzzy model representation gap**: most mainstream modeling languages (RDF/OWL, E-R) lack native support for truth degrees; academic fuzzy extensions exist (fuzzy OWL variants, fuzzy E-R) but are not industry-strength. Simple cases (a single relation's truth degree) can be hand-rolled (e.g., reifying the relation as an entity with a "truth degree" attribute when relation attributes aren't natively supported); complex cases (full membership functions, fuzzy rules/axioms with reasoning) may require adopting an academic framework.
- **Fuzzification documentation is additive, not a substitute**: fuzzifying a vague model does NOT reduce the need to document vagueness types/dimensions/contexts/provenance from Ch. 6 — it increases that need, since you now also must document the sources/methods used to derive the fuzzy degrees themselves.

## Mental Models
- **Dilemmas have no universally correct answer — only context-appropriate ones**: unlike pitfalls, representation dilemmas require you to run your specific context through a set of diagnostic questions and accept that different contexts will yield different (equally legitimate) answers.
- **"Don't make your model harder than it needs to be"**: if you don't actually need to represent facts about an entity's instances, don't force class-hood on it just because other models did, or because it theoretically could have instances.
- **Fuzzy truth degrees smooth disagreement, they don't eliminate vagueness**: the goal of fuzzification is not to make a vague predicate crisp, but to give disagreement-prone judgments a shared, quantifiable middle ground that reduces (not resolves) friction.
- **Fuzzification is an investment decision, not a default**: given its acquisition/maintenance cost and representation-framework immaturity, fuzzification should only be pursued when the sizing (severity of disagreement) and benefit (A/B-tested improvement) clearly justify the cost — "if you cannot develop a scalable and cost-effective way to acquire fuzzy degrees, any benefit may not be worth it."

## Anti-patterns
- **Forcing class-hood on an entity with no convincing instances, just because another model did**: repeats the Ch.1/Ch.2 "concept = class" fallacy; other models' choices are not binding and may not even mean the same thing in your context.
- **Over-complicating a model with unnecessary class machinery for instances you'll never populate**: when you don't care about describing an entity's instances, adding class structure for them just adds maintenance overhead with no payoff.
- **Treating OWL2 punning as a free pass to avoid choosing**: punning hides, rather than resolves, the semantic distinction between an entity's class-role and individual-role — can genuinely confuse collaborators/users if not carefully documented.
- **Subclassing categories with no real distinguishing content**: creating subclasses (e.g., for restaurant cuisines, or left/right lung) when siblings share identical characteristics beyond the differentiator itself, wasting modeling effort for no inferential gain.
- **Subclassing highly volatile (low-rigidity) categories without reconsidering maintenance cost**: e.g., a class like `RestaurantWith50Employees` whose membership churns frequently — probably better modeled as an attribute/relation than a subclass.
- **Modeling a characteristic as a bare attribute when its values genuinely need their own semantics**: e.g., leaving `filmColourType` as an unstructured string when downstream use cases actually need to reason about color-type relationships.
- **Reifying numeric/date values as entities just to use them as relation objects**: technically possible (DBpedia does this in some cases) but "rather strange and highly impractical," especially for characteristics with effectively infinite value ranges.
- **Confusing fuzzy truth degrees with probabilities**: applying probability-theory intuitions/operations to fuzzy degrees (or vice versa) is a category error — they quantify fundamentally different things (likelihood of a well-defined event vs. degree of truth of an ill-defined predicate) and require different mathematical machinery (probability theory vs. fuzzy logic/t-norms).
- **Fuzzifying elements that aren't actually vague**: wasting fuzzification effort on elements that are merely vaguely-worded but have a genuinely crisp intended meaning (per Ch. 6's detection guidance) — always re-verify genuineness before investing in fuzzification.
- **Asking vague-predicate questions in a way that assumes complementarity**: e.g., asking "Is John short?" to derive a degree for "tall" — ignores that vague predicates are non-complementary (not-tall ≠ short), producing misleading harvested degrees.
- **Assuming fuzzy degrees are automatically "better" without A/B validation**: skipping the step of empirically comparing a fuzzy vs. crisp model version's actual downstream task effectiveness (inter-agreement scores, application accuracy) before committing to the fuzzification investment.
- **Treating fuzzification as a substitute for documenting vagueness**: assuming that once something has a numeric truth degree, you no longer need to explain its vagueness type/dimensions/context/provenance — the opposite is true.
- **Ignoring fuzzification bias**: deploying membership functions (e.g., age-to-"old" functions) in consequential downstream decisions (insurance premiums) without scrutinizing their disparate impact.

## Code Examples
```text
# Illustrative fuzzy membership function example (age dimension; from Figure 12-3 narrative)
YoungPerson(age)      : membership degree decreasing from 1 (very young) to 0 (older)
MiddleAgedPerson(age) : membership degree peaking in a middle age range
OldPerson(age)        : membership degree increasing toward older ages

# Example evaluation: Maria, age 30
YoungPerson(30)      = 0.5
MiddleAgedPerson(30) = 0.5
OldPerson(30)        = 0.0

# Illustrative t-norm combination for a two-dimension vague class
SimilarRevenueCompany(X)        = f_revenue(X)        # membership function on revenue
SimilarEmployeeCountCompany(X)  = f_employees(X)       # membership function on employee count
CompetitorCompany(X) = T(SimilarRevenueCompany(X), SimilarEmployeeCountCompany(X))
   where T = min, product, or Łukasiewicz t-norm (chosen empirically)
```
- **What it demonstrates**: how quantitative vagueness in one dimension (age) maps directly through a membership function to truth degrees for several related vague classes simultaneously, and how multi-dimensional vagueness (revenue + employee count defining "competitor") can be decomposed into per-dimension membership functions combined via a chosen t-norm, rather than requiring one complex multivariate function.

## Reference Tables
*Class-or-individual dual-naming workaround examples (reconstructed)*:

| Entity | As individual | As class |
|---|---|---|
| Data Scientist | Data Scientist (Profession) | Data Scientist (Professional) |
| Pneumonia | Pneumonia (Disease) | Pneumonia (Incident) |
| Eagle | Eagle (Species) | Eagle (Animal) |
| Samsung A8 | Samsung A8 (Mobile Phone Model) | Samsung A8 (Mobile Phone Device) |

*Six fuzzification steps*: (1) detect/analyze vague elements, (2) decide fuzzification approach per element, (3) harvest fuzzy degrees, (4) assess fuzzy model quality, (5) represent fuzzy degrees, (6) apply the fuzzy model in consuming applications.

*Popular t-norms / t-conorms (reconstructed)*:

| T-norm (AND) | Formula | Dual t-conorm (OR) | Formula |
|---|---|---|---|
| Minimum (Gödel) | min(a,b) | Maximum | max(a,b) |
| Product | a·b | Probabilistic sum | a+b−a·b |
| Łukasiewicz | max(0, a+b−1) | Bounded sum | min(a+b, 1) |

*Three truth-degree harvesting methods*: ask directly (controlled, slow, doesn't scale), ask indirectly via application context (natural, more scalable, less predictable), mine from data (fully automatable, noisy/potentially inaccurate).

## Worked Example
Two fuzzification case studies from the author's own projects demonstrate measurable benefit: (1) the Greek electricity-market ontology (reprised from Ch. 10) had three inherently vague relations (`isImportantPartOfProcess`, `isImportantPartOfAction`, `isInterestedInTheProcess`) that domain experts couldn't crisply agree on; fuzzifying them via harvested expert-disagreement degrees (e.g., "Issuing of Distribution Orders is important part of Distribution Process to degree 0.5") and A/B testing against a 25-query gold set improved search precision and recall by ~7% over the non-fuzzy version. (2) Knowledge Tagger (KT), an entity-resolution/thematic-scope-resolution system, was tested with fuzzy relations like `wasAnImportantActorInFilm` (vs. a flat `hasPlayedInFilm`) on 100 IMDb film reviews covering 20 similar, hard-to-distinguish films — fuzzifying actor/director/character importance improved scope-resolution accuracy by 10%; a parallel test on 100 military-conflict texts using a DBpedia-derived fuzzy ontology (relations like `wasAnImportantPartOfConflict`, `playedMajorRoleInConflict`) improved accuracy by 13%. Both cases followed the same methodology: identify genuine vagueness, harvest degrees from domain experts, A/B test fuzzy vs. crisp versions on a held-out gold set, and only adopt fuzzification because it demonstrably improved the downstream task.

## Key Takeaways
1. Representation dilemmas (class-vs-individual, subclass-vs-relation, attribute-vs-relation) have no universally correct answer — resolve each using the specific diagnostic questions for your context, not a generic rule.
2. Don't model an entity as a class unless it has genuine instances AND you care to describe them AND doing so doesn't block facts you need about the entity itself — otherwise, prefer the simpler individual representation.
3. Subclassing earns its complexity cost only when subclasses carry real distinguishing content, are commonly used qualifiers in the domain's discourse, and aren't excessively volatile (low rigidity).
4. Fuzzy truth degrees are categorically distinct from probabilities — never conflate the two mathematical frameworks.
5. Fuzzification is a costly, multi-step investment (detect, decide, harvest, assess, represent, apply) that should only be pursued after confirming genuine, impactful, fuzzification-amenable vagueness and validating real downstream benefit via A/B testing.
6. Fuzzification increases, never decreases, the need to document a model's vagueness characteristics and the provenance of its truth degrees.
7. Watch for bias in fuzzification just as in any other modeling decision — truth degrees can encode and amplify disparate impact when used in consequential decisions.

## Connects To
- **Ch 2**: directly extends the class/individual and attribute/relation definitions and the "instance test" into concrete decision procedures.
- **Ch 3**: reuses OntoClean rigidity and the quantitative/qualitative vagueness typology as diagnostic inputs here.
- **Ch 6 / Ch 7**: builds directly on vagueness detection/documentation (Ch. 6) and subsumption/naming pitfalls (Ch. 7) as prerequisites for sound representation choices.
- **Ch 8-10**: "applying the fuzzy model" explicitly reuses the specification, quality-management, and application-fit frameworks from these chapters.
- **Ch 13 (Expressiveness and Content Dilemmas)**: the next chapter continues the dilemmas theme, shifting from "how to represent" to "what to include."
