# Glossary

**Abstract entity** — an entity that does not exist in a particular time/place but as an idea/category/concept (e.g., Biology, Justice) (Ch 2).

**Abduction** — reasoning that infers premises from a known conclusion, working opposite to deduction; not guaranteed true (Ch 2).

**Ambiguity** — a piece of information that can be interpreted in more than one plausible way; types include phonological, syntactic, anaphoric, term-level (lexical), and sentence-level (Ch 3).

**Anti-rigid class** — a class not essential to any of its instances (e.g., Student, Food) (Ch 3).

**Application-centered quality evaluation** — measuring a model's quality by its effect on a specific application's effectiveness (Ch 4).

**Application-neutral quality evaluation** — measuring model quality against the domain/data it describes, independent of a specific application (Ch 4).

**Attribute** — a literal-valued characteristic (number, string, date) of an entity or relation, used instead of a relation to another entity (Ch 2).

**Class** — an abstract entity representing a kind of thing, serving as a semantic type for instances; a concept is a class only if it has (or can have) instances (Ch 2).

**Class subsumption (subclassing)** — meaning-inclusion applied to classes; if A is a subclass of B, all instances of A are also instances of B (Ch 2).

**Closed-World Assumption (CWA)** — unknown statements are inferred false (Ch 3).

**Combinatory (qualitative) vagueness** — vagueness from a cluster of partially-relevant conditions with no sharp rule for sufficiency/necessity (e.g., Religion) (Ch 3).

**Competency questions** — natural-language questions a model should be able to answer, used to elicit concrete requirements (Ch 8).

**Completeness** — degree to which a model contains all elements it should (schema completeness: classes/relations defined; population completeness: instances/assertions populated) (Ch 4).

**Conciseness** — degree to which a model lacks redundant/legacy/duplicate elements (Ch 4).

**Concept drift (ML sense)** — statistical shift in a target variable's properties over time; distinct from semantic drift (Ch 3).

**Consistency** — freedom from logical/semantic contradictions in a model (Ch 4).

**Deduction** — reasoning from true premises and valid rules to a certain conclusion; the primary reasoning type semantic models support (Ch 2).

**Degree-vagueness (quantitative)** — vagueness from lack of a precise boundary along a continuous dimension (e.g., tall along height) (Ch 3).

**Dependence (OntoClean)** — class C1 depends on C2 if every instance of C1 requires an instance of C2 to exist (e.g., Food depends on an eater) (Ch 3).

**Dilemma (semantic modeling)** — a situation with multiple legitimate courses of action, each with real trade-offs and no single correct answer (Ch 1).

**Distant supervision** — using an existing semantic model to auto-generate labeled training data for supervised mining (Ch 5).

**Entity** — anything that may exist concretely or abstractly; the core element of a semantic model (Ch 2).

**Entity resolution** — detecting entity mentions in text and mapping them to a semantic model's entities (disambiguation) (Ch 10).

**False friends** — cross-language terms that look/sound alike but differ in meaning (e.g., English "sensible" vs. French/Spanish "sensible" = sensitive) (Ch 7).

**Fuzzification** — assigning a real-valued truth degree (0-1) to a vague statement, distinct from probability (Ch 12).

**Global ambiguity** — confusion between target entities and entities from entirely unrelated domains (Ch 10).

**Gold standard / partial gold standard / silver standard** — reference data tiers for measuring completeness, from fully verified (rare) to partial or imperfect-but-useful (Ch 4).

**Hearst patterns** — lexico-syntactic patterns (e.g., "Y such as X") for extracting is-a relations from text (Ch 5).

**Identity (OntoClean)** — whether instances of a class can be distinguished from each other via some criterion (Ch 3).

**Induction** — reasoning that infers general rules from premise-conclusion pairs; what supervised ML does (Ch 2).

**Inheritance Richness (IR)** — average subclasses per class; a representation-dependent (and thus unreliable) structural quality metric (Ch 9).

**Instantiation** — the relation linking an entity to the class(es) it is an instance of (`rdf:type`) (Ch 2).

**Knowledge acquisition** — mechanisms/processes for generating model elements from sources (experts, data, users) (Ch 5).

**Lexicalization** — linking a model element to one or more natural-language terms (synonyms/variants) that express it (Ch 2).

**Mapping/interlinking relations** — relations linking elements across different models (e.g., `owl:sameAs`, `skos:exactMatch`) (Ch 2).

**Meaning inclusion** — one element's meaning is included in another's (hyponymy/hypernymy) (Ch 2).

**Mention-level vs. global-level IE** — mention-level IE returns extracted elements plus their source location; global-level IE returns only distinct elements (Ch 5).

**OntoClean** — a methodology (Rigidity, Identity, Unity, Dependence) for validating semantic correctness of class subsumption relations (Ch 3).

**Open-World Assumption (OWA)** — unknown statements remain undetermined, not inferred false; the default for RDF(S)/OWL (Ch 3).

**Ostensive definition** — a definition given by pointing to examples (Ch 2).

**Part-whole relation** — meronymy/holonymy; has at least six subtypes (component-integral, member-collection, portion-mass, stuff-object, feature-activity, place-area) (Ch 2).

**Pitfall (semantic modeling)** — a clearly wrong decision/omission in modeling with undesired consequences; has a right answer, unlike a dilemma (Ch 1).

**Punning (OWL2)** — using the same entity as both a class and an individual, with the reasoner inferring the role from context; "syntactic trickery," per Michael Uschold (Ch 12).

**Referential opacity** — when two linked entities refer to the "same thing" but their attached attributes/relations aren't mutually acceptable between the two models (Ch 7).

**Relevancy** — the one quality dimension that is inherently application-dependent: whether a model's content is actually useful for a specific task (Ch 4).

**Rigidity (OntoClean)** — whether class membership is essential to an instance across all worlds/times (rigid, e.g. Human; anti-rigid, e.g. Student; semi-rigid, e.g. HardThing) (Ch 3).

**Semantic accuracy** — degree to which a model's assertions are accepted as true (Ch 4).

**Semantic change / semantic drift** — a term's meaning shifting over time, measured via label, intension, and extension (Ch 3).

**Semantic gap** — the disconnect between how data is modeled on the supply side and understood/used on the exploitation side (Ch 1).

**Semantic model** — any data representation whose goal is to make meaning explicit and commonly understood among humans and machines (Ch 1).

**Semantic model mining** — acquiring/incorporating model elements from data with limited human effort via IE/NLP/ML methods (Ch 5).

**Semantic relatedness** — a vague relation indicating two elements' meanings are somehow related, without specifying how (Ch 2).

**SHACL** — Shapes Constraint Language; W3C standard (2017) for validating RDF data with real constraints, addressing OWL/RDF(S)'s inference-rule-not-constraint gap (Ch 7).

**Signal (quality)** — a measurement indicating elevated probability of a quality problem, as opposed to a direct measurement (metric) of the quality dimension itself (Ch 9).

**Sorites Paradox** — the "heap of wheat" argument illustrating vagueness: a chain of individually valid steps leads to an absurd conclusion (Ch 3).

**Symmetry, Inversion, Transitivity** — formal properties of relations that must be verified, not assumed (Ch 3).

**Synset** — a WordNet unit of meaning grouping words sharing a sense (Ch 2).

**t-norm / t-conorm** — functions combining two fuzzy truth degrees preserving classical AND/OR at 0/1 (e.g., minimum/Gödel t-norm, product t-norm) (Ch 12).

**Term** — a string (word/phrase) used to lexically express a model element; has no meaning on its own (Ch 2).

**Truth degree** — a real number in [0,1] representing the fuzzy truth of a vague statement; not a probability (Ch 12).

**Unity (OntoClean)** — whether instances of a class count as "whole" entities and under what criteria (Ch 3).

**Upper (foundational) ontology** — a highly abstract, domain-neutral ontology (e.g., DOLCE, BFO) meant to serve as shared top-level categories for interoperability (Ch 5).

**Vagueness** — a phenomenon where a predicate admits borderline cases / lacks sharp boundaries; distinct from ambiguity, uncertainty, and inexactness (Ch 3).
