# Chapter 3: Semantic and Linguistic Phenomena

## Core Idea
Human language exhibits recurring phenomena — ambiguity, uncertainty, vagueness, rigidity/identity/unity/dependence, symmetry/inversion/transitivity, open/closed world assumptions, and semantic change — that underlie most semantic modeling pitfalls; understanding them precisely is a prerequisite for avoiding those pitfalls.

## Frameworks Introduced
- **Ambiguity taxonomy**: phonological, syntactic, anaphoric, term-level (lexical), and sentence-level semantic ambiguity.
  - When to use: diagnosing why a statement or term resists a single clear model representation.
  - How: identify which layer the ambiguity lives at (sound, structure, reference, single term, or whole sentence) before attempting disambiguation.
- **Vagueness vs. inexactness vs. ambiguity vs. uncertainty (the four-way distinction)**: these are frequently confused but are logically distinct.
  - When to use: before handling any "fuzzy"-seeming data element, classify it correctly, since each needs a different modeling treatment.
  - How: vagueness = borderline cases / no sharp boundary (e.g., "tall"); inexactness = precise but wide range (e.g., "170-180cm"); ambiguity = multiple discrete readings (e.g., "bank"); uncertainty = truth unknown due to lack of knowledge (e.g., "probably raining").
- **Degree-vagueness vs. combinatory vagueness**: two kinds of vagueness.
  - When to use: deciding how to represent a vague predicate.
  - How: degree-vagueness = lack of precise boundary along one (or more) continuous dimension (e.g., "bald" along hair quantity); combinatory vagueness = a cluster of partially-relevant necessary/sufficient conditions with no sharp combination rule (e.g., "Religion").
- **OntoClean's four notions — Rigidity, Identity, Unity, Dependence**: a methodology for validating subsumption correctness.
  - When to use: before asserting a subclass/superclass relation, to check it is semantically sound.
  - How: check whether the class is rigid/anti-rigid/semi-rigid (is membership essential to instances?), has an identity criterion (can you tell two instances apart?), has a unity criterion (are instances well-defined wholes?), and whether it is dependent on another class (must some other class's instance exist for this one to exist?).
- **Closed-World vs. Open-World Assumption (CWA/OWA)**: governs how "unknown" is treated.
  - When to use: before choosing a modeling language/framework and before writing any axiom meant to act as a constraint.
  - How: CWA — unknown implies false (use when you fully control what enters the model); OWA — unknown implies "can't conclude" (use when completeness can't be guaranteed). RDF(S)/OWL default to OWA, which changes how axioms behave (constraints can act as inference rules instead).

## Key Concepts
- **Ambiguity**: a piece of information can be interpreted in more than one plausible way (e.g., "Tripoli" — Libya, Lebanon, or Greece).
- **Uncertainty**: a statement's truth cannot be determined from incomplete knowledge; explicit (hedged with "probably," "might") or implicit (doubt arising from an untrustworthy source or imperfect extraction method).
- **Vagueness / Sorites Paradox**: illustrated by the "heap of wheat" paradox — a predicate like "heap" has no sharp cutoff point, so valid-seeming step-by-step reasoning leads to an absurd conclusion. Vagueness = admits borderline cases / lacks sharp boundaries, and is context-dependent (e.g., "tall" relative to population vs. relative to NBA players).
- **Rigidity**: a class is rigid if membership is essential to instances in all worlds/times (e.g., Human); anti-rigid if not essential (e.g., Student, Food); semi-rigid if essential to some instances but not others (e.g., HardThing — essential to hammers, not sponges).
- **Identity**: whether instances of a class can be told apart from one another via some criterion (Person has identity via DNA/etc.; RedThing does not — distinct from a mere membership criterion).
- **Unity**: whether instances of a class count as "whole" entities, and under what criteria (Ocean has unity; AmountOfWater does not; LegalAgent has non-unity since people and organizations have different unity criteria).
- **Dependence**: class C1 depends on class C2 if every instance of C1 requires an instance of C2 to exist (Food depends on something/someone eating it).
- **Symmetry, Inversion, Transitivity**: properties of relations — symmetric (cousin-of), transitive (located-in), inverse (brother-of / sister-of). Critical for correct reasoning; relations that look like they have these properties sometimes don't (detailed in Ch. 7).
- **Closed-World Assumption (CWA)**: unknown statements are inferred false (e.g., no seat assignment implies not checked in).
- **Open-World Assumption (OWA)**: unknown statements remain undetermined (e.g., no allergy listed doesn't mean the patient has none). RDF(S)/OWL use OWA by default.
- **Semantic change / semantic drift**: a term's meaning shifts over time (e.g., "awful" from "inspiring awe" to "very bad"). Measured along three aspects: labels (terms), intension (defining characteristics), extension (set of instances). Distinct from "concept drift" in ML (statistical property shift of a target variable).

## Mental Models
- **The Sorites Paradox as the canonical vagueness demonstration**: a chain of individually plausible "if X doesn't make a heap, X+1 doesn't either" steps leads to a false conclusion — showing vagueness is a property of the predicate's boundary, not a flaw in reasoning.
- **Rigidity/Identity/Unity/Dependence as a subsumption "lint checker"**: running a proposed subclass relation through these four questions surfaces hidden category errors before they corrupt a model's reasoning.
- **Three faces of "fuzzy"**: whenever something feels imprecise, explicitly ask whether it's vague (no sharp boundary), inexact (precise range), ambiguous (multiple discrete senses), or uncertain (unknown truth) — conflating them leads to the wrong fix.
- **CWA/OWA as a default-behavior lens**: the same axiom ("every employee has a manager") behaves as an enforced constraint under CWA but as a soft inference trigger under OWA — know which world your framework assumes.

## Anti-patterns
- **Treating vague predicates as crisp** (reprise of ESCO's `essential_for` from Ch. 1): harms model quality/usability when borderline cases are inevitable but the model presents a hard boundary.
- **Confusing vagueness with ambiguity/uncertainty/inexactness**: applying the wrong remediation (e.g., trying to "disambiguate" something that is actually vague, which has no discrete alternate readings to choose between).
- **Mistaking a membership criterion for an identity criterion**: assuming that knowing whether X belongs to a class also lets you distinguish X from other members — these are different problems.
- **Assuming symmetry/transitivity/inverse properties hold without verification**: relations that look symmetric/transitive in common cases can fail in general (fully explored in Ch. 7).
- **Ignoring the CWA/OWA mismatch between framework and intended semantics**: using RDF(S)/OWL (OWA) while mentally modeling constraints as CWA-style hard rules, leading to axioms silently behaving as inference rules instead of validation checks.
- **Using a model whose entities have drifted in meaning without realizing it**: analyzing data from one era/community with a semantic model whose terms mean something different now (e.g., applying a contemporary ontology to 15th-century historical texts), or embedding disputed/offensive semantic shifts (e.g., outdated clinical terminology) without awareness.

## Code Examples
```text
# Sorites Paradox, formalized as a chain argument (illustrative, not literal book syntax)
P1: grains(1) -> NOT heap
P2: forall n: (grains(n) -> NOT heap) => (grains(n+1) -> NOT heap)
...
Pn: grains(999999) -> NOT heap  =>  grains(1000000) -> NOT heap
Conclusion: grains(1,000,000) -> NOT heap   # false, yet validly derived
```
- **What it demonstrates**: how a vague predicate (`heap`) with no sharp boundary makes an otherwise valid chain of inference produce an absurd result — the formal signature of vagueness as opposed to ambiguity or uncertainty.

## Reference Tables
*Ambiguity types (reconstructed):*

| Type | Level | Example |
|---|---|---|
| Phonological | sound | "ice cream" / "I scream" |
| Syntactic | sentence structure | "John ate the cookies on the couch" |
| Anaphoric | reference | "...she gave her a good lunch" — who is "she"? |
| Term-level (lexical) | single term | "Kashmir" — song or region |
| Sentence-level semantic | whole sentence | "John and Jane are married" — to each other, or separately? |

*Vague vs. crisp adjective examples (from WordNet, reconstructed):* Abnormal, Impenitent, Notorious, Yellowish, Outermost = vague; Compound, Biweekly, Unarmed, Aroused, Unfeathered = crisp (per the chapter's framing, several of these pairs illustrate the vague/crisp contrast rather than all being crisp — treat the table as illustrative of the spectrum).

*OntoClean dimensions summary:*

| Notion | Question it answers | Example values |
|---|---|---|
| Rigidity | Is membership essential to the instance, always? | Rigid (Human), anti-rigid (Student), semi-rigid (HardThing) |
| Identity | Can instances be distinguished from one another? | Person (yes), RedThing (no) |
| Unity | Are instances well-defined wholes? | Unity (Ocean), non-unity (LegalAgent), anti-unity (AmountOfWater) |
| Dependence | Must another class's instance exist for this instance to exist? | Food depends on an eater |

*Semantic change types (Bloomfield/Blank, reconstructed):* Specialization (narrower), Generalization (broader), Metaphor (similarity), Metonymy (contiguity), Synecdoche (part-whole), Hyperbole (weaker→stronger), Meiosis (stronger→weaker), Auto-antonymy (opposite), Folk-etymology (name similarity), Antiphrasis (contrastive irony).

| Word | Old meaning | Modern meaning |
|---|---|---|
| Awful | inspiring awe | exceptionally bad |
| Demagogue | popular leader | politician who panders to emotion/prejudice |
| Meat | solid food | flesh of animals |
| Sly | skillful, wise | sneaky, deceitful |
| Egregious | remarkably good | remarkably bad/flagrant |
| Matrix | female breeding animal | pattern of lines and spaces |

## Worked Example
Modeling "John and Jane are married": this single sentence intersects two phenomena at once. (1) Anaphoric/identity ambiguity — which "John" and "Jane" in the domain? Resolve by using unique identifiers. (2) Sentence-level semantic ambiguity — married to each other, or each married separately? If the former, model as a relation `isMarriedTo(John, Jane)`; if the latter, model as an attribute `maritalStatus=married` on each entity independently. The correct representation depends entirely on resolving the ambiguity first — picking a representation before resolving meaning risks baking in a wrong interpretation.

## Key Takeaways
1. Vagueness, inexactness, ambiguity, and uncertainty are four distinct phenomena requiring different modeling treatments — don't conflate them.
2. Every element in a model you create should be described unambiguously, with the exact intended meaning (identity, relation vs. attribute framing) resolved before encoding.
3. If a model is uncertain, its provenance and confidence level must be communicated to users (globally, per-section, or per-statement).
4. Know whether your modeling framework (and the model you're using/building) assumes OWA or CWA — this changes whether axioms act as hard constraints or soft inference triggers.
5. Validate subsumption relations against Rigidity, Identity, Unity, and Dependence (OntoClean) to catch semantically unsound class hierarchies.
6. Symmetry, transitivity, and inversion must be verified, not assumed, for each relation you define.
7. Track semantic change/drift for model elements whose meaning evolves quickly, and build mechanisms (Ch. 14) to keep the model current; be especially alert to disputed or offensive semantic shifts.

## Connects To
- **Ch 1**: directly follows up on ESCO's vague `essential_for` relation as the motivating case.
- **Ch 2**: applies these phenomena to the elements (entities, relations, classes) defined there.
- **Ch 6 (Bad Descriptions)**: expands on ambiguity-free description techniques.
- **Ch 7 (Bad Semantics)**: expands on broken subsumption, part-whole, symmetry/transitivity errors, and OWA-driven inference pitfalls.
- **Ch 10 (Bad Application)**: covers using semantic models for disambiguation tasks.
- **Ch 13**: revisits vagueness as an asset-vs-liability trade-off (expressiveness dilemmas).
- **Ch 14 (Evolution and Governance Dilemmas)**: covers mechanisms for tracking and managing semantic change over time.
