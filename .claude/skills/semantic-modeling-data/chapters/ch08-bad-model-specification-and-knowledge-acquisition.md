# Chapter 8: Bad Model Specification and Knowledge Acquisition

## Core Idea
Even when a model is formally correct, it can still fail if its requirements were wrong (specification pitfalls) or if the data/people/tools used to populate it were inappropriate (knowledge acquisition pitfalls) — both require deliberate, iterative investigation rather than passive assumption.

## Frameworks Introduced
- **Three-phase specification process**: (1) investigate the model's context, (2) specify features via competency questions, (3) assess feasibility and importance.
  - When to use: at the start of (and iteratively throughout) any semantic model development project.
  - How: Phase 1 — identify stakeholder applications, existing models in use, pain points, and prior remediation history via four context questions. Phase 2 — identify "core" entity types at the right generality level, then derive required elements via competency questions (natural-language questions the model must answer). Phase 3 — separately assess feasibility (constrained by modeling-language capabilities and knowledge-acquisition resource availability/cost) and importance (a business/strategic judgment only stakeholders can make), then prioritize using both.
- **Competency questions**: natural-language questions the model should be able to answer, used to elicit requirements from non-expert stakeholders.
  - When to use: translating vague stakeholder requests ("I want entities from the legal domain") into concrete modeling requirements.
  - How: collect questions like "What is the average salary of a data scientist in the US?" and reverse-engineer the implied schema (e.g., salary must be country-specific); probe each question for hidden vagueness, scope, and granularity assumptions.
- **Tool/Source Scrutiny Questionnaire**: six questions to vet any knowledge-acquisition tool before adopting it (what exactly does it extract; does it deliver what it promises; does it avoid bias; what's its real effectiveness range and under what conditions; what's the required "recipe" to use it well; how do you troubleshoot/optimize and operationalize it).
  - When to use: before adopting any third-party extraction tool, framework, or data source for model population.
  - How: never trust a single reported precision/recall number — test on your own data/conditions, since effectiveness swings widely across datasets (same tools scoring from 31% to 83% F1 across different benchmark corpora in the chapter's examples).

## Key Concepts
- **"Building the wrong thing"**: a model can be internally accurate and complete yet still fail because it was specified against the wrong requirements — covering features not needed, omitting features that were needed, including features harmful to downstream use, or including features nobody can yet use.
- **Expressed vs. desired features gap**: what stakeholders literally say ("we want synonyms") often doesn't match what they actually need ("semantically related terms for search expansion") — requires active probing, not literal requirement-taking.
- **Core entity types at the right generality**: the central nouns of the model's one/two-sentence self-description (e.g., ESCO = "skills, competences, qualifications, occupations") — picking too generic (e.g., "Person" instead of "Athlete") risks irrelevant requirements; picking too specific risks repetitive, redundant requirements.
- **Premature modeling**: naturally occurring exploratory technical thinking during requirements elicitation (e.g., wondering whether "salary" should be an attribute or entity) is fine in moderation but risks losing focus on requirement-gathering and alienating non-technical stakeholders if it dominates the conversation.
- **Feasibility vs. importance as independent axes**: feasibility depends on (a) the modeling language/framework's capabilities and (b) knowledge-acquisition resource availability/cost/quality; importance depends on business/strategic factors only stakeholders can supply. Requirements must be assessed on both axes, not conflated.
- **"Tyranny of low-hanging fruits"**: easy-but-unimportant requirements tend to get prioritized over hard-but-important ones purely because they're easy — acceptable only if postponing the hard ones doesn't create accumulating technical debt in model maintenance.
- **Four ways data serves as a knowledge source**: reusable as-is (existing semantic models/databases), as raw input for extraction, as training data for supervised mining, as a source for distant supervision.
- **Two ways people serve as a knowledge source**: direct providers/curators/validators of model elements, or annotators of training data for supervised mining.
- **Three ways data can be "wrong" as a source**: factually/semantically inaccurate; accurate but mismatched to what's needed (e.g., CV data reflects skill supply, not market demand); accurate and relevant but too unstructured/ambiguous to extract efficiently.
- **Systematic vs. anecdotal data errors**: the diagnostic habit of checking whether an apparent data error (e.g., DBpedia classifying "Machine Learning" as a "Disease") is an isolated mistake or part of a broader systematic failure pattern (e.g., DBpedia's classes generally being unsuited for typing abstract entities) — only systematic problems should drive a source-selection decision.
- **Expert fallibility**: domain experts are not automatically a reliable knowledge source — due to (a) narrow true expertise mismatched to the specific sub-question asked, (b) failure to recognize the limits of their own expertise (Dunning-Kruger-type overconfidence, or over-rigid worldview in highly competent people), and (c) susceptibility to cognitive/behavioral biases (availability bias, groupthink, confirmation bias, etc. — see Reference Table).
- **Tool capability misunderstanding (Maslow's Hammer)**: over-relying on a familiar tool (e.g., "just use Word2Vec" for synonym extraction) without scrutinizing whether it actually solves the specific semantic task (Word2Vec conflates synonymy with antonymy/hyponymy/other similarity, and gives one vector per term regardless of polysemy).
- **Benchmark non-transferability**: a tool's reported precision/recall on its creators' test set is not a reliable predictor of its performance on your data/conditions — effectiveness is highly sensitive to task/domain/dataset shift (demonstrated by named-entity-resolution systems scoring from ~31% to ~83% across different published benchmarks).
- **Failing humans-in-the-loop**: asking ambiguous/inaccurate elicitation questions, withholding guidelines in the presence of vagueness, pushing for forced consensus (treating disagreement as noise rather than signal), under-sampling viewpoints, failing to prioritize annotation work, and not personally doing the annotation task before assigning it to others.

## Mental Models
- **The lamp-post fallacy (Mulla Nasreddin parable)**: looking for your (data/knowledge) "ring" where the light is good (easy data is available) rather than where you actually lost it (where the model's real requirements point) — a direct metaphor for choosing data sources by convenience rather than fitness.
- **Specify-then-source, never source-then-specify**: pick knowledge acquisition data/people/tools only after precisely specifying the target semantics — diving into convenient existing data before defining what you're looking for guarantees mismatch.
- **Disagreement is signal, not noise**: in the presence of genuine vagueness or ambiguity, multiple human judges disagreeing is valuable information about the predicate's nature, not a data-quality defect to be suppressed via forced consensus.
- **Credentials are not a free pass**: neither a data source's prestigious origin nor a human expert's professional credentials exempt them from systematic scrutiny — test both directly against your specific task.

## Anti-patterns
- **Specifying features that don't match true requirements**: implementing an overly strict synonymy relation when the application actually needed a looser relatedness relation optimized for recall (author's own Textkernel story).
- **Omitting crucial constraints from requirements**: building a profession hierarchy deeper than the downstream parsing system could handle, because the depth limit was never elicited.
- **Over-specifying harmful features**: adding so much lexicalization that the downstream disambiguation system's precision dropped more than its recall improved.
- **Building unused features**: populating a skills-demand relation that no product could yet consume — wasted effort, no value delivered.
- **Specifying the model in isolation from its intended applications**: treating "domains and data to cover" as the whole specification task while leaving application-integration requirements as an afterthought, only to have them retroactively invalidate earlier design decisions.
- **Ignoring conflicting requirements across stakeholders**: taking single-sided, biased requirements from different teams without reconciling them against each other up front.
- **Dismissing legacy models without investigating their design rationale**: discarding an existing (seemingly flawed) semantic model from scratch without first understanding that some "flaws" were actually deliberate decisions responding to still-applicable application constraints.
- **Projecting past-project pain points onto a new project**: assuming the challenges faced in one semantic modeling project are equally relevant in a new, different context, leading to redundant or badly prioritized features.
- **Trusting vendor/tool hype about ease of model-building**: "it's easy to build a model with our tool" is often true but misleading — you can build *a* model easily, not necessarily *the* model you actually need.
- **Blindly trusting a data source based on its creator's reputation or documentation alone**: ESCO's documentation covers methodology extensively but provides no concrete accuracy/coverage figures — reputation and documentation depth are not substitutes for direct empirical scrutiny.
- **Treating any "relevant" data as usable data just because it's abundant or was expensive to obtain** (the lamp-post trap in both directions: easy data gets used because it's there; expensive data gets used because "we already paid for it").
- **Deploying domain experts without checking the specificity of their expertise**: assuming any "history expert" is equally qualified across all sub-periods/sub-regions of history.
- **Letting expert overconfidence and biases silently populate the model**: e.g., ESCO's "essential skill" assignments (RDF Query Language for data scientists, web programming for knowledge engineers) traceable to availability bias — experts recalling salient but statistically rare associations as if they were typical.
- **Over-relying on a single familiar tool without verifying it matches the target semantics ("Maslow's Hammer")**: defaulting to Word2Vec for synonym extraction despite it conflating multiple relation types and failing to disambiguate polysemous terms.
- **Accepting a tool's self-reported benchmark scores at face value**: a reported 94% precision / 96% recall (Shalaby et al.'s profession/skill/company/school classifier) dropped to 60-75% precision in the author's own evaluation due to a hidden methodological flaw in the original test set (it excluded the "Unknown" category entirely).
- **Assuming systematic tool errors documented nowhere won't affect you**: FRED's undocumented systematic biases (inconsistent individual-vs-class modeling for near-identical sentences, near-universal "is-a" misclassification as instantiation rather than subsumption, artificial entity creation, unnecessary structural complexity) were discoverable only through the author's own qualitative evaluation, not the tool's own papers/docs.
- **Asking knowledge-acquisition questions that bake in ambiguity**: asking "Is A a B?" instead of "Are all A's instances also instances of B?" when trying to determine subclass relationships conflates instantiation and subsumption answers.
- **Suppressing disagreement among annotators/experts via forced consensus mechanisms**: masks genuine vagueness as if it were resolvable noise.
- **Skipping pilot/self-testing of an annotation process before deploying it to others**: designing elicitation tasks without personally attempting them first, missing usability flaws that only surface in practice.

## Code Examples
```turtle
# Illustrative SKOS-based aspect-evaluation-polarity model (from the chapter's ABSA case study)
@prefix : <http://vocab.isoco.net/absa/> .
@prefix ex: <http://example.org/> .

ex:DecentFood a :AspectEvaluation ;
    :evaluationExpression "decent" ;
    :polarity "neutral" .

ex:TastyFood a :AspectEvaluation ;
    :evaluationExpression "tasty" ;
    :polarity "positive" .

ex:Food a :AspectEntity ;
    :hasAspectEvaluation ex:DecentFood ;
    :hasAspectEvaluation ex:TastyFood .
```
- **What it demonstrates**: a model specified directly from competency questions ("what entities can have sentiment expressed about them," "what evaluation expressions go with an aspect," "what's the typical polarity") translated into a concrete SKOS-based schema (`AspectEntity` as subclass of `skos:Concept` to leverage broader/narrower taxonomy support; `AspectEvaluation` carrying an expression and polarity); shows how a well-specified competency question directly determines schema shape (e.g., the need for aspect taxonomies with inheritance of evaluations from parent to child aspects).

## Reference Tables
*Expressed vs. desired features (reconstructed from Table 8-1):*

| What people say | What they most likely mean |
|---|---|
| "We want entities from the legal domain" | "...but only for the US legal system" |
| "We want the model used for inference" | "...for inductive inference" (not deductive) |
| "We want to use the model for semantic search" | "...in the particular search system we just bought" |
| "Our biggest problem is ambiguity of business terms" | "Our biggest problem is vagueness of business terms" |

*Cognitive/behavioral biases affecting human knowledge acquisition (reconstructed from Table 8-4):*

| Bias | Description |
|---|---|
| Groupthink | minimizing conflict, reaching consensus without critically evaluating alternatives |
| Self-contradiction | giving different answers to the same question asked twice (common with vague/context-dependent info) |
| Desirability bias | assigning higher probability to desired outcomes |
| Confirmation bias | searching/interpreting/remembering info that confirms preconceptions |
| Distinction bias | viewing two options as more different when evaluated jointly vs. separately |
| Framing effect | different conclusions from the same info depending on presentation |
| Pygmalion effect | letting others' expectations influence judgment |
| Naïve realism | believing one sees the world objectively; disagreement implies the other is biased/uninformed |
| Shared information bias | over-discussing info all group members already know; under-discussing unique info |
| False causality bias | citing sequential events as evidence of causation |
| Action bias | acting despite high ambiguity/vagueness, especially under time pressure |
| Ambiguity bias | favoring options with more knowable outcomes |
| Anchoring bias | being swayed by already-known or first-shown information |
| Availability bias | over-weighting easily-recalled instances when estimating frequency (root cause of ESCO's skewed "essential skills") |

*Tool benchmark non-transferability examples*: AIDA named entity resolution: 83% (AIDA-YAGO2) vs. 62% (Reuters-21578). DBpedia Spotlight: 81% (wikilinks) vs. 56% (NYT paragraphs) vs. 34% F1 (AIDA/CoNLL-TestB). AGDISTIS: 76% (AQUAINT) vs. 60% (AIDA/CoNLL-TestB) vs. 31% (IITB).

## Worked Example
The Aspect-Based Sentiment Analysis (ABSA) ontology case study walks through the full specification-to-population pipeline: competency questions (what can be an aspect; what evaluation expressions apply; what polarity does an expression carry per aspect) directly shaped the SKOS-based schema (`AspectEntity` as `skos:Concept` subclass, enabling broader/narrower aspect taxonomies with evaluation-inheritance rules). Population used a semiautomatic pipeline: subjectivity filtering (OpinionFinder) → named entity resolution for known aspects → pattern-based relation extraction (dependency-grammar based) → tf-idf-based confidence ranking → sentiment-based polarity scoring (averaged per-sentence polarity from the Stanford Sentiment tool) → manual validation of newly discovered aspects and generality-expression promotion (e.g., promoting "tasty" from "chicken" up to the more generic "Food" aspect). Evaluation on 2,000 restaurant-review sentences achieved up to 80% precision for known aspects, 72% for unknown ones (with lower recall, limited by the corpus's pattern complexity), and 80% accuracy on polarity detection against human judges — illustrating realistic, imperfect but usable outcomes from a well-specified, carefully executed acquisition pipeline.

## Key Takeaways
1. Specification failures ("building the wrong thing") can produce a technically high-quality model that is still useless — always validate against the applications that will actually consume it.
2. Treat requirements gathering as active investigation (context, competency questions, feasibility/importance) rather than passive collection of stakeholder statements at face value.
3. Pick knowledge sources and tools only after specifying the target semantics precisely — never let convenient data availability drive specification (the lamp-post fallacy).
4. Scrutinize both data sources and human experts for systematic (not merely anecdotal) errors and documented biases before trusting them.
5. Never accept a tool's self-reported benchmark performance as transferable to your own data/conditions — test independently.
6. Treat annotator/expert disagreement as a signal about genuine vagueness/ambiguity, not as noise to be eliminated via forced consensus.
7. Separate feasibility assessment (technical/resource-driven) from importance assessment (business/strategic) and use both to prioritize, resisting the "low-hanging fruit" trap when it risks accumulating technical debt.

## Connects To
- **Ch 1**: directly revisits ESCO's "essential skills" vagueness example, now explained via the availability bias mechanism.
- **Ch 3 / Ch 6**: vagueness-driven disagreement among domain experts (e.g., "Critical System Process") connects directly to documentation/description practices.
- **Ch 5**: extends the "Deciding What to Build" and mining-method sections into concrete specification and sourcing pitfalls.
- **Ch 7**: tool-induced semantic errors (FRED's instance/class confusion) are instances of the Ch. 7 subsumption pitfalls, now viewed from the tool-selection angle.
- **Ch 9 (Bad Quality Management)**: builds directly on the feasibility/importance prioritization framework to discuss quality trade-off management.
- **Ch 11 (Bad Strategy and Organization)**: expands on team/expert assembly pitfalls touched on here.
