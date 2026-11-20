# Chapter 10: Bad Application

## Core Idea
Having a semantic model that formally covers the right domain is not enough — applying it to a specific application (entity resolution, semantic relatedness) can fail or actively backfire because the model's extra elements introduce noise, or because a generic relation (like "related to") doesn't match what the specific application context actually needs.

## Frameworks Introduced
- **Five Ambiguity Types for Entity Resolution Diagnosis**: lexical, target-to-target, target-to-nontarget, nontarget-to-target, and global ambiguity.
  - When to use: diagnosing low precision/recall in any entity resolution (named entity disambiguation) system that uses a semantic model as contextual evidence.
  - How: manually annotate a representative text sample with target AND nontarget entities, run the system unassisted, then measure each ambiguity type as the percentage of terms misclassified in that specific way (see Reference Tables for exact formulas); map high scores on each type to a specific remediation action (Table 10-2 pattern).
- **Semantic Model Evidential Adequacy measurement**: two-part framework — model richness (how populated are the relevant relations/attributes) and text prevalence (how often does that evidence actually appear in the target texts).
  - When to use: when an entity resolution system has low effectiveness even after addressing ambiguity.
  - How: compute richness metrics (% of target entities with no related entities at all; average related-entities per target entity; average related-entities via a specific relation) purely from the model; compute prevalence metrics (% of target entities with at least one evidential entity actually appearing in the annotated texts; average evidential entities found per target entity; same per specific relation) by cross-referencing the model against the annotated text sample.
- **"Semantics-first" requirement elicitation for relatedness** (reprise of Ch. 8's approach): make implicit "should be related" assumptions explicit and context-specific before choosing or building a relatedness measure.
  - When to use: whenever an application needs a semantic-relatedness/similarity capability.
  - How: elicit concrete, falsifiable relatedness definitions from actual users (e.g., "A is related to B when a user interested in buying books about A is also interested in buying books about B"), then work down a five-step escalation ladder: (1) check if the model already has a matching relatedness relation, (2) check other existing models for a reusable one, (3) test standard relatedness measures (e.g., shortest-path distance) on your model, (4) define custom metrics/rules/weights on your model's existing relations, (5) mine the target relatedness from scratch using external data if nothing else works.

## Key Concepts
- **"Knowledge can hurt you"**: the core thesis of this chapter — a semantic model's extra, domain-relevant-but-contextually-irrelevant elements (e.g., DBpedia's non-actor Person entities when disambiguating actors in film reviews, or a former player's still-present team relation when disambiguating a current match report) actively reduce precision rather than merely being harmless extra information.
- **Entity resolution's four inputs**: the target texts, the target entity set, an entity thesaurus (unique IDs + natural-language surface forms), and a contextual-evidence knowledge resource (often the semantic model itself) used for disambiguation via the "terms with similar meaning appear in similar contexts" hypothesis (Miller and Charles).
- **Precision and recall in entity resolution**: precision = correctly resolved terms ÷ all detected terms; recall = correctly resolved terms ÷ all actual entity mentions in the texts. Low precision stems from high ambiguity or inadequate contextual evidence; low recall stems from thesaurus incompleteness (missing entities/surface forms) or an evidence threshold too strict to be met.
- **Lexical ambiguity**: target entities confused with ordinary lexical terms (e.g., company "Factual" vs. the adjective "factual").
- **Target-to-target entity ambiguity**: two or more target entities (both within the thesaurus) confused with each other (e.g., Tripoli, Greece vs. Tripoli, Libya, when both are target locations).
- **Target-to-nontarget entity ambiguity**: a target entity confused with a nontarget entity present in the evidence model (e.g., soccer team Barcelona confused with the city of Barcelona, when the model also contains locations).
- **Nontarget-to-target entity ambiguity**: a nontarget entity (from the evidence model) wrongly resolved as a target entity.
- **Global ambiguity**: target entities confused with entities from domains entirely outside the thesaurus/evidence model (e.g., company "Apple" vs. the fruit) — signals the input texts themselves may often not even be about the target domain.
- **Model richness vs. text prevalence**: richness asks "does the model even have enough relational data to serve as evidence," while prevalence asks "does that evidential data actually show up in the specific texts we're disambiguating" — a rich model with low text prevalence (for this corpus) is just as useless as a poor model.
- **Model pruning as a precision-boosting action**: when target-to-nontarget ambiguity is high and nontarget-to-target is low, remove nonessential/noisy entities and underused relations from the evidence model (starting with the lowest-prevalence relations) — directly operationalizes the "knowledge can hurt you" thesis.
- **Threshold adjustment as a recall-boosting action**: when model richness/prevalence is adequate but recall remains low due to a too-strict minimum-evidence threshold, lowering that threshold increases recall with limited precision cost, provided ambiguity scores are otherwise low.
- **Vague relatedness requires context-specific definitions, not generic ones**: the same two entities (e.g., Prolog and Python) are "related" or "not related" depending entirely on the application's actual decision context (general domain similarity vs. hiring-fit similarity) — a one-size-fits-all relatedness measure will systematically mismatch some application's real needs.
- **The bookstore vs. library relatedness contrast**: the same history ontology used for query expansion in two superficially similar systems (an online bookstore vs. a university library) requires fundamentally different relatedness definitions, because the underlying user intent differs (willingness to pay for a tangential match vs. willingness to accept any free, tangentially-useful information).

## Mental Models
- **A model is a liability as well as an asset for any given application**: don't assume "more knowledge in the model = better"; extra elements not relevant to a specific downstream task are pure noise that reduces precision.
- **Relatedness feels obvious in the abstract, breaks down in context**: people readily agree two things are "related" when asked with no stakes attached, but disagree sharply once a concrete decision (hire this person? buy this book?) is on the line — always elicit relatedness definitions anchored to the actual downstream decision, never in the abstract.
- **Diagnosis before remediation**: measure the specific ambiguity type and evidential-adequacy metric driving a low precision/recall score before choosing a fix — different root causes (lexical ambiguity vs. global ambiguity vs. low model prevalence) call for entirely different interventions (WSD improvements vs. domain classifiers vs. model pruning vs. threshold tuning).
- **Diagnoses and fixes are inexact — always re-measure**: both the diagnosis-to-action mapping and the action's actual effect can be wrong even when metrics support them; every corrective action should be validated by re-measuring precision/recall on a fresh test set (to avoid bias from reusing the same tuning set).

## Anti-patterns
- **Assuming domain-matched models are automatically beneficial**: using a general-purpose model like DBpedia wholesale as disambiguation evidence for a narrow task (e.g., resolving actor mentions in film reviews) when the model's many non-actor Person entities actively create confusable candidates that wouldn't realistically occur in the target texts.
- **Retaining low-value relations in the evidence model "just in case"**: e.g., keeping a soccer player's relation to their former team when disambiguating contemporary match reports, when that relation essentially never corresponds to what the text is actually describing.
- **Using a single generic relatedness relation/measure across all applications**: reusing a model's built-in "related to" relation, or a generic shortest-path measure, without checking whether it matches the specific decision context of the application (hiring similarity vs. general domain similarity vs. linguistic similarity, as in the Prolog/Python and German-language examples).
- **Treating "should be related" as self-evidently specified**: leaving relatedness requirements as an unexamined, generic phrase instead of eliciting a concrete, falsifiable, context-anchored definition from real users.
- **Expecting one relatedness solution to transfer across superficially similar applications**: assuming the bookstore's query-expansion relatedness measure will also work for the library's query-expansion use case, when the two systems' underlying user goals (willingness to pay vs. free access to tangential info) differ fundamentally.
- **Not re-measuring after a corrective action, or reusing the same tuning set for validation**: risks either falsely concluding an action helped, or silently overfitting to the diagnostic sample rather than genuinely improving general performance.
- **Treating a mediocre out-of-the-box precision/recall score as the system's ceiling rather than investigating root causes**: the Knowledge Tagger soccer case started at precision 60%/recall 55% (comparable to AIDA+YAGO's 62%/58%) and jumped to 82%/80% purely through evidence-model pruning guided by text-prevalence measurement — illustrating that low baseline scores are often fixable via diagnosis rather than inherent tool limitations.
- **Ignoring global ambiguity when most input texts are off-domain**: applying entity resolution to a general news corpus for a narrow target domain (startups) without first filtering for domain relevance, as in the companies case where 40% of ambiguity came from the global-ambiguity category.

## Code Examples
```text
# Illustrative: ambiguity-type measurement formulas (reconstructed, not literal code)
lexical_ambiguity        = count(terms: lexical_term AND no_manual_target_annotation AND system_mapped_to_target) / total_detected_terms
target_to_target_ambig   = count(terms: manually_annotated_target AND system_also_mapped_other_targets) / total_detected_terms
target_to_nontarget_ambig= count(terms: manually_annotated_target AND system_also_mapped_nontargets) / total_detected_terms
nontarget_to_target_ambig= count(terms: manually_annotated_nontarget AND system_mapped_to_targets) / total_detected_terms
global_ambiguity         = count(terms: real_entity AND no_manual_annotation_at_all AND system_mapped_to_target) / total_detected_terms
```
- **What it demonstrates**: how each of the five ambiguity types is operationalized as a precise, separately-measurable ratio over a manually annotated diagnostic sample — enabling root-cause-specific remediation instead of a single undifferentiated "ambiguity problem."

## Reference Tables
*Diagnosis-to-action map (reconstructed from Table 10-2):*

| Observed metric pattern | Diagnosis | Recommended action |
|---|---|---|
| High lexical ambiguity | System's word-sense disambiguation (WSD) is inadequate | Improve the linguistic analysis component (heuristics to full WSD) |
| High global ambiguity | Many input texts aren't actually in the target domain | Apply a domain/topic classifier to filter texts before resolution |
| High target-to-nontarget, low nontarget-to-target ambiguity | Evidence model contains noisy nontarget entities hampering disambiguation | Prune the evidence model (remove nonessential entities/relations, starting from lowest-prevalence) |
| Low semantic model richness | Model inadequate as evidence regardless of text | Enrich the model, prioritizing the most text-prevalent relations |
| High richness but low text prevalence | Model's evidence doesn't actually occur in these texts | Change/expand model with entities more likely to appear in texts, or lower the minimum evidence threshold (if other ambiguity types are low) |

*Soccer case ambiguity metrics (reconstructed)*: Lexical 1%, Target-to-target 30%, Target-to-nontarget 56%, Nontarget-to-target 4%, Global 2% → fix: prune to keep only current-team/co-player/manager relations (85-95% text prevalence) → precision 60%→82%, recall 55%→80%.

*Companies case ambiguity metrics (reconstructed)*: Lexical 10%, Target-to-target 4%, Target-to-nontarget 4%, Nontarget-to-target 3%, Global 40% → fix: domain classifier (90% accuracy) raised precision to 72%; capitalization heuristic raised precision to 78%/recall 57%; lowering evidence threshold (given low other ambiguity types) raised recall to 62%.

*Five-step relatedness escalation ladder*: (1) reuse model's own relatedness relation if compatible, (2) reuse another existing model's relatedness relation, (3) try standard relatedness measures (e.g., shortest-path), (4) define custom metrics/weighted rules on existing relations, (5) mine relatedness from scratch via external data (Ch. 5/8 methods).

## Worked Example
The Hellenic Transmission System Operator (HTSO) electricity-market semantic search case: after eliciting that users wanted query expansion ordered by "how much it helps them know what they need to do to participate in the market" (processes first, then actions/rights/obligations), the team implemented a custom relatedness measure assigning manual weights from -1 (exclude from relatedness) to +1 (treat as identical) to each ontology relation — e.g., `isImportantPartOfProcess=1`, `isImportantPartOfAction=1`, `isInterestedInProcess=1`, `performsAction=0.8`, `isPerformedByParticipant=-0.5`, `hasObligation=0.5`, `hasRight=0.5`. Tested against a 25-query gold set with ground-truth ideal responses, the custom weighted measure improved precision and recall by about 8% over a standard distance-based relatedness measure — demonstrating that even simple, hand-tuned relation weighting (not full external mining) can meaningfully outperform generic relatedness measures once the application's real relatedness semantics are made explicit.

## Key Takeaways
1. A model's domain fit is not sufficient for a given application — extra elements that are accurate and relevant to the domain can still actively harm a specific downstream task (entity resolution, relatedness calculation).
2. Diagnose entity-resolution precision/recall problems by decomposing ambiguity into its five specific types (lexical, target-to-target, target-to-nontarget, nontarget-to-target, global) rather than treating "ambiguity" as one undifferentiated issue.
3. Separately measure model richness (how much evidential data exists) and text prevalence (how often that evidence actually appears in your specific corpus) — a model can be rich but useless for your texts, or vice versa.
4. Prune evidence models of noisy, low-prevalence, or off-target elements when target-to-nontarget ambiguity is high — more knowledge is not automatically better.
5. Semantic relatedness is maximally context-dependent — elicit a concrete, decision-anchored definition from actual users before selecting or building any relatedness measure, and expect that no single measure transfers across meaningfully different application contexts (e.g., bookstore vs. library).
6. Always re-measure precision/recall on a fresh test set after any corrective action, since both the diagnosis and the remediation can fail even when metrics seemed to support them.

## Connects To
- **Ch 3**: directly builds on the ambiguity and semantic-relatedness/vagueness definitions established there.
- **Ch 4**: operationalizes the Relevancy quality dimension (the one explicitly application-dependent dimension) in concrete diagnostic detail.
- **Ch 5**: entity resolution and relation-extraction techniques referenced here were introduced as mining methods there.
- **Ch 8**: the "semantics-first" requirement-elicitation approach for relatedness directly reuses the specification methodology from that chapter.
- **Ch 9**: "knowledge can hurt you" extends the Relevancy-to-context-A-vs-B quality trade-off discussed there into concrete application-level diagnostics.
- **Ch 13 (Expressiveness and Content Dilemmas)**: revisits the cost/benefit calculus of enriching vs. pruning a model's content for specific applications.
