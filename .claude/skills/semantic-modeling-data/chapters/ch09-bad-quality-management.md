# Chapter 9: Bad Quality Management

## Core Idea
Beyond specification and modeling errors, quality itself can be mismanaged: by ignoring inherent trade-offs between quality dimensions, by failing to link quality metrics to real risks/benefits, and by using metrics that are structurally misleading, incomparable across models, arbitrarily thresholded, or that are really just risk signals mistaken for direct quality measurements.

## Frameworks Introduced
- **Quality Trade-off Management**: explicit recognition and strategic management of tensions between quality dimensions (Ch. 4), rather than assuming all dimensions can be simultaneously maximized.
  - When to use: whenever specifying or managing a semantic model's development priorities.
  - How: identify the specific trade-off pairs active in your context (accuracy-vs-completeness, conciseness-vs-completeness, conciseness-vs-understandability, relevancy-to-context-A-vs-B), make a deliberate strategic choice about which to favor, and document the decision.
- **Risk/Benefit-Linked Quality Measurement**: three-step discipline for making quality metrics actionable.
  - When to use: before investing resources into improving any quality dimension.
  - How: (1) know during specification which quality dimensions actually drive success/failure for your model's use cases, and to what extent; (2) pick metrics/measurement processes tailored to your specific model and context rather than generic off-the-shelf metrics; (3) explicitly connect metric values to a risk/benefit estimate (e.g., "this relation needs ≥80% accuracy for deployment in the French pharma industry").
- **Metric vs. Signal distinction**: a metric directly measures a quality dimension's value; a signal only indicates elevated probability of good/bad quality.
  - When to use: before adopting any quality metric from the literature (e.g., outlier count, provenance-based trust scores).
  - How: ask whether the proposed measurement actually quantifies the dimension itself, or merely correlates with/predicts it — use signals for triage/error-detection, never for quality reporting.

## Key Concepts
- **Accuracy vs. Completeness trade-off**: the two dimensions are mathematically independent (a model can be 100% accurate/low-complete, or low-accurate/100%-complete simultaneously, as the EuropeanUnionMemberState example shows) — this independence creates a tempting but risky shortcut of loosening accuracy controls to accelerate completeness.
- **Conciseness vs. Completeness trade-off**: thoroughly deduplicating/merging new terms into existing entities (for conciseness) takes time and tooling that a rush toward completeness will often skip, leading to duplicate entities; also, "redundancy" itself is subjective/context-dependent (e.g., a product owner excluding low-frequency entities as "unnecessary burden" vs. an outsider calling the same omission an incompleteness).
- **Conciseness vs. Understandability trade-off**: aggressively minimizing entity names/lexicalizations/definitions (e.g., banning definitions for non-class entities, or banning names longer than three words) for the sake of conciseness directly degrades human comprehension — e.g., forbidding the full expansion of acronyms like ADHD or DARE.
- **Relevancy-to-context-A vs. Relevancy-to-context-B trade-off**: when a model serves multiple application scenarios with conflicting needs (rich lexicalization vs. low ambiguity; accuracy-priority vs. completeness-priority client; single-truth vs. diversity-of-perspective regional deployment; complex-axiom-tolerant vs. performance-constrained application) — optimizing for one scenario can actively harm another.
- **Multiple-models-as-escape-valve**: maintaining separate model variants tuned for different quality priorities avoids some trade-off decisions but multiplies maintenance burden — justified mainly when maintenance of extra variants can be outsourced to the party that needs them (e.g., a client assuming responsibility for their custom knowledge-graph version).
- **Risk/benefit-blind quality reporting**: reporting raw metric values (e.g., "accuracy is 85%") without connecting them to actual downstream risk/benefit makes the numbers "meaningless" for prioritization — risk severity varies drastically by application (95% biomedical ontology accuracy is fine for a trivia system like Watson's Jeopardy! play, but dangerous for a medical diagnosis system).
- **Harm-aware quality management**: when a model affects real people, undetected bias is a severe quality management failure, not a minor defect — illustrated by Amazon's 2018 recruiting-engine gender discrimination case (trained on male-skewed résumé data), which led to the system's withdrawal and the team's disbandment.
- **Metrics with misleading interpretations**: a metric's suggested meaning can be formally wrong — e.g., OntoQA's Inheritance Richness (average subclasses per class) is claimed to indicate domain-coverage detail vs. generality, but two semantically identical models (genres as subclasses vs. genres as individual entities related via `hasGenre`) produce wildly different IR scores despite covering the same knowledge equally.
- **Metrics with little comparative value**: metrics like Average Population (instances ÷ classes) or Class Richness (classes-with-instances ÷ total classes) implicitly assume classes matter more than individuals, so two models with identical content but different class/individual representation choices get very different, misleading scores — undermining cross-model comparison.
- **Metrics with arbitrary value thresholds**: OQuaRE's 1-5 quality-score bands mapped onto metric value ranges (e.g., "Tangledness > X parents per class = not acceptable") are presented as objective but are actually arbitrary cutoffs lacking contextual justification — usable only if YOU define and get stakeholder agreement on your own thresholds, never adopted wholesale from generic literature.
- **Metrics that are actually signals**: e.g., "number of statistical outliers" (suggested as an accuracy metric) only indicates elevated probability of error, not actual error rate, since not all outliers are wrong; similarly, provenance-based "trustworthiness scores" (expert-manual=1.0 down to automatic-unstructured=0) encode assumptions (experts > community; manual > automatic; structured > unstructured sources) that don't universally hold.
- **Measuring accuracy of vague assertions in a crisp way**: four specific measurement mistakes — (1) requesting truth judgments with no stated context (judges invent their own, misaligned context), (2) using only one judge per assertion (hides disagreement), (3) forcing consensus among multiple judges on genuinely controversial (not merely confused) assertions (induces consensus bias, masking real disagreement), (4) reporting an aggregate accuracy score without reporting the underlying disagreement level (hides how risky/uncertain that number actually is).
- **Equating model quality with information-extraction quality**: a model's quality depends not just on the extraction method's reported precision/recall but also on (a) the quality of the underlying knowledge source it was derived from (e.g., DBpedia's reported "11.93% inaccurate statements" actually measured extraction-fidelity to Wikipedia infoboxes, not real-world truth — so true inaccuracy could be substantially higher), and (b) the mismatch between extraction-level metrics (per-mention precision/recall over a corpus) and model-level metrics (per-entity accuracy/completeness over a domain) — e.g., a 70%-precision NER system's effective accuracy for a model entity depends on how many corroborating occurrences you require before adding that entity, and its recall is bounded by the corpus's actual domain coverage, not the full domain.

## Mental Models
- **Quality is a portfolio allocation, not a single optimization target**: resources spent maximizing one dimension (completeness) are resources not spent on another (accuracy, conciseness) — treat quality management as an explicit trade-off/prioritization exercise, not a universal "make it all better" mandate.
- **A number without a stake is not information**: any quality metric value is only useful to the extent it's tied to a concrete downstream risk or benefit — a metric reported in isolation can't drive any real decision.
- **Signals detect, metrics report**: use signal-type measurements (outlier counts, provenance heuristics) to find candidate problems to investigate, never to state "the model is X% good."
- **Structural metrics can be representation-invariant-blind**: any metric computed purely from schema structure (subclass counts, instance/class ratios) risks conflating "how the same knowledge happens to be represented" with "how much/good the knowledge actually is" — always check whether an equivalent alternative representation of the same content would produce a wildly different score.

## Anti-patterns
- **Pursuing 100% on every quality dimension simultaneously**: ignores the real, often mathematically guaranteed trade-offs (e.g., accuracy vs. completeness are independent, not jointly maximizable without cost).
- **Loosening accuracy controls to accelerate completeness without a strategic decision**: a "tempting but risky option" — letting unchecked knowledge enter the model to hit coverage targets, without weighing whether completeness genuinely outweighs accuracy for the use case.
- **Rushing completeness at the expense of deduplication/conciseness**: adding thousands of new terms as distinct entities without running synonym-detection/grouping steps, because that detection pipeline is slow or unavailable.
- **Banning helpful human-facing content (definitions, longer names/acronym expansions) purely for conciseness**: as in the author's own project where a client forbade definitions for non-class entities and capped names at three words, leaving acronyms like ADHD/DARE unexplained.
- **Optimizing a shared model for one application's needs while silently degrading another's**: rich lexicalization (helps text-analytics application) vs. minimal ambiguity (needed for a different, precision-sensitive application) is a recurring conflict that surfaces only when all usage contexts are considered together.
- **Reporting quality metrics without any risk/benefit linkage**: spending significant resources to raise accuracy by 10% only to discover the business impact is zero — "meaningless numbers" syndrome.
- **Allowing a biased model/system into production without harm-aware scrutiny**: directly referencing the Amazon recruiting-engine case as a cautionary example of catastrophic quality-management failure when a model affects real people.
- **Trusting OntoQA-style structural richness metrics (Inheritance Richness, Average Population, Class Richness) as if they measure actual domain knowledge quality**: these metrics assume classes are inherently more meaningful than individuals, so two models with identical real-world content can score wildly differently purely due to a representational choice (class-based genres vs. individual-based genres) — unsuitable for cross-model comparison or as standalone quality indicators.
- **Adopting OQuaRE-style arbitrary 1–5 quality-score thresholds wholesale from the literature**: using someone else's "more than 8 parents per class = not acceptable" cutoff without re-deriving thresholds specific to your own model and getting stakeholder buy-in.
- **Treating outlier counts as an accuracy metric rather than an error-detection signal**: not all statistical outliers are wrong; using outlier count as "the" accuracy score overstates confidence in what is really just a triage heuristic.
- **Treating provenance-based "trustworthiness" formulas as ground truth**: assuming expert-curated-manual models are always more trustworthy than community-curated or automatically-extracted ones, when in practice expert bias, community-consensus quality, and source-structuredness all vary case by case.
- **Measuring vague-assertion accuracy without context, with a single judge, or by forcibly resolving genuine controversy into consensus**: each of these four specific mistakes (no context, single judge, forced consensus on real disagreement, reporting accuracy without disagreement level) produces an accuracy score that misrepresents how risky/uncertain the underlying assertions actually are.
- **Conflating extraction-method benchmark scores with the resulting model's quality**: assuming a model built with a 70%-precision/60%-recall NER system will itself have 70% accuracy/60% completeness — ignores (a) the underlying source's own reliability (DBpedia's accuracy study actually measured fidelity-to-Wikipedia, not real-world truth) and (b) the mismatch between per-mention extraction metrics and per-entity/per-domain model metrics.

## Code Examples
```text
# Illustrative: same knowledge, two representations, wildly different "Inheritance Richness" (IR)

# Representation 1: genres as subclasses
Film
 ├─ subclassOf ComedyFilm
 ├─ subclassOf ActionFilm
 └─ ... (100 genre subclasses total)
# IR (avg subclasses per class) is high (driven up by Film's 100 subclasses)

# Representation 2: genres as individuals
Film        (class, instantiated by films)
FilmGenre   (class, instantiated by genre entities)
<film> hasGenre <genre>
<genreA> narrowerThan <genreB>
# IR = 0 (no subclass relations used at all)

# Same real-world content (10,000 films, 100 genres, same genre assignments)
# -> IR scores differ from "non-zero / high" to "zero", despite equal domain coverage
```
- **What it demonstrates**: how a purely structural metric (Inheritance Richness) is representation-dependent rather than content-dependent, making it an unreliable (and potentially misleading) proxy for a model's actual domain-coverage detail or generality — reinforcing the chapter's core warning against trusting structural quality metrics without checking representation-invariance.

## Reference Tables
*Quality trade-off pairs (reconstructed):*

| Trade-off | Core tension |
|---|---|
| Accuracy vs. Completeness | faster coverage growth vs. correctness of what's added |
| Conciseness vs. Completeness | deduplication effort/time vs. speed of adding new content |
| Conciseness vs. Understandability | minimalism (short names, few definitions) vs. human comprehensibility |
| Relevancy to Context A vs. Context B | optimizing for one application's needs can actively harm another's |

*OQuaRE-style metrics (reconstructed, illustrative only — exact thresholds are context-specific and should not be reused as-is):* Class Richness (CROnto: mean instances/class), Attribute Richness (RROnto: attributes ÷ [relations+attributes]), Weighted Method Count (WMCOnto: mean attributes+relations/class), Number of Children (NOCOnto: mean direct subclasses), Tangledness (TMOnto: mean parents/class) — each mapped to a 1 (not acceptable) to 5 (exceeds requirements) score band in the source framework, but the chapter stresses these specific bands are arbitrary and must be re-derived per context.

*Provenance-based trustworthiness scoring (reconstructed, flagged as flawed)*: expert+manual curation = 1.0; open-community manual curation = 0.75; expert-curated but automatically inserted from structured sources = 0.5; automatically curated+inserted from structured sources = 0.25; automatically curated+inserted from unstructured sources = 0. Flagged because each of its underlying assumptions (experts > community, manual > automatic, structured > unstructured) can fail in specific cases.

*Four mistakes in measuring vague-assertion accuracy*: (1) no context given to judges, (2) single judge only, (3) forced consensus on genuinely controversial items, (4) reporting accuracy without reporting disagreement level.

## Worked Example
The EuropeanUnionMemberState class example cleanly demonstrates accuracy/completeness independence: a model listing all 27 real EU members plus 100 non-members has 100% completeness but ~21% accuracy; the same model listing only 2 correct members has 100% accuracy but ~7% completeness. Neither configuration is "better" in the abstract — which one a team should prefer depends entirely on the deployment context's risk tolerance, which is exactly the kind of decision this chapter argues must be made explicitly and strategically rather than left to drift (e.g., by simply relaxing validation checks whenever coverage targets loom).

## Key Takeaways
1. Quality dimensions genuinely trade off against each other (accuracy/completeness, conciseness/completeness, conciseness/understandability, relevancy across contexts) — manage them as a deliberate strategic allocation, not an afterthought.
2. Never report or act on a quality metric value without linking it to a concrete risk or benefit for a specific use case — otherwise the number is not actionable.
3. Treat model-affecting-real-people bias detection as a first-class quality management responsibility, not an edge case — the Amazon recruiting case shows the stakes.
4. Scrutinize any quality metric for representation-dependence before trusting its interpretation or using it to compare models — structural metrics like Inheritance Richness, Average Population, and Class Richness can mislead when the same content is representable in more than one way.
5. Never adopt generic value-to-score thresholds (e.g., OQuaRE's bands) wholesale — define and validate thresholds specific to your own model and context with stakeholder agreement.
6. Distinguish quality signals (elevated probability of a problem, e.g., outlier counts, provenance heuristics) from quality metrics (direct measurement) — use signals for triage, not for reporting.
7. When measuring accuracy of vague assertions, provide context to judges, use multiple judges, avoid forcing consensus on genuinely controversial items, and always report disagreement alongside the accuracy score.
8. Don't assume your model's quality equals your extraction tooling's reported benchmark performance — source reliability and metric-level mismatches (per-mention vs. per-entity/per-domain) both break that assumption.

## Connects To
- **Ch 4**: directly extends the quality dimensions defined there into trade-off management and metric-selection practice.
- **Ch 3 / Ch 6**: vagueness-aware accuracy measurement here builds on the vagueness framework and documentation guidance from those chapters.
- **Ch 5**: the "Ensuring It's Good" development activity is operationalized in detail here.
- **Ch 8**: extends the knowledge-acquisition quality discussion (extraction-tool scrutiny, source reliability) into the model-quality-vs-extraction-quality distinction.
- **Ch 10 (Bad Application)**: shifts focus from managing a model's own quality to how it's actually used (and misused) by downstream applications.
- **Ch 11 (Bad Strategy and Organization)**: the "multiple models" escape valve and strategic trade-off decisions connect to organizational/strategic planning covered there.
