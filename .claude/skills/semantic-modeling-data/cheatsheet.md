# Cheatsheet

## Pitfall vs. Dilemma — which am I facing?
- Has a single correct answer, violating it has clear bad consequences → **Pitfall** (fix it).
- Multiple legitimate options, each with real trade-offs, no universal right answer → **Dilemma** (run the relevant decision procedure for your context).

## Four "fuzzy-feeling" phenomena — don't conflate them
| Phenomenon | Signature | Fix |
|---|---|---|
| Ambiguity | multiple discrete readings (e.g., "bank") | disambiguate / rename more specifically |
| Vagueness | borderline cases, no sharp boundary (e.g., "tall") | document type/dimension/context; consider fuzzification |
| Uncertainty | truth unknown from lack of knowledge | state confidence/provenance |
| Inexactness | precise but wide range (e.g., "170-180cm") | not a problem — it has clear limits |

## Is it a class? Quick test
Can you find convincing, non-artificial **instances**? No → don't model as a class, no matter what other models do. Yes, but you don't care to describe them AND don't need facts about the entity itself → model as individual (simpler). Need both class-level facts and instance-level facts → use a workaround (custom relation, dual-entity, or punning) — see patterns.md.

## Subsumption sanity checks (run before committing to `subclassOf`)
1. **"Is a kind of" test**: does "X is a kind of Y" sound right? If not, it's instantiation, not subsumption.
2. **Rigidity check**: is the subclass rigid (Person) but the superclass nonrigid (Customer)? → invalid, rigid can't sit under nonrigid.
3. **Identity check**: do the (candidate) superclasses share compatible identity criteria? Disjoint superclasses (Person + Organization) → split into separate subclasses instead.
4. **Transitivity check**: are you chaining the SAME relation type? Mixing instantiation + meaning-inclusion, or different part-whole subtypes, breaks transitivity.

## OWA vs. CWA — which does your framework assume?
- RDF(S)/OWL → **Open-World Assumption** by default. `rdfs:domain`/`rdfs:range` behave as **inference rules**, not constraints — a reasoner will infer new (possibly wrong) class memberships rather than reject bad data. Use **SHACL** if you need real validation.
- E-R models, most databases → **Closed-World Assumption**. Unknown = false.

## Quality dimensions — metric or signal?
A **metric** directly measures a dimension. A **signal** only indicates elevated probability of a problem (e.g., outlier count, provenance-based "trust" scores). Never report a signal as if it were a metric.

## Structural metrics are representation-dependent — sanity-check before trusting
Before trusting Inheritance Richness / Average Population / Class Richness (or any purely structural metric): could the SAME real-world content be represented a different way (e.g., genres as subclasses vs. genres as individuals) and get a wildly different score? If yes, the metric isn't measuring content quality — don't use it for cross-model comparison.

## Vague-assertion accuracy measurement — 4 things to always do
1. Give judges **context** (never ask in the abstract).
2. Use **multiple judges**, not one.
3. **Don't force consensus** on genuinely controversial items — report disagreement, use it to improve the model instead.
4. Report **disagreement level alongside** the accuracy score, never accuracy alone.

## Fuzzify? Contextualize? Gate it first
Before investing in fuzzification or truth-contextualization, confirm: (1) the vagueness/context-dependence is genuine (not just badly worded), (2) disagreement is severe and hard to resolve by consensus, (3) you can acquire/identify degrees or contexts cost-effectively, (4) the consuming application can actually use them, (5) A/B testing shows real improvement, (6) benefit > acquisition/maintenance cost. They are **complementary**, not alternatives.

## Entity resolution precision/recall — diagnosis table
| Symptom | Likely cause | Fix |
|---|---|---|
| High lexical ambiguity | weak word-sense disambiguation | improve linguistic analysis / WSD |
| High global ambiguity | texts mostly off-domain | add a domain/topic classifier filter |
| High target-to-nontarget, low nontarget-to-target | noisy nontarget entities in evidence model | **prune** the model (start with lowest-prevalence relations) |
| Low model richness | evidence model too sparse | enrich, prioritizing most text-prevalent relations |
| High richness, low text prevalence | evidence doesn't occur in these texts | change/expand model, or lower evidence threshold (if other ambiguity is low) |

## Knowledge source red flags
- **Data**: has it been scrutinized for *systematic* errors (not just anecdotal ones)? Does it match your target semantics, or just your target domain?
- **Experts**: narrow true-expertise mismatch, overconfidence (Dunning-Kruger), availability bias — scrutinize with test sessions, don't give a free pass on credentials alone.
- **Tools**: "just use Word2Vec" ≈ Maslow's Hammer. Benchmark scores do NOT transfer across datasets (same NER tool: 83%→62%, or 81%→34% F1, across different corpora). Always test on YOUR data.

## Interlinking — default stance: skeptical, not enthusiastic
Linked Data's "four rules" are best practices, not obligations. Before `owl:sameAs`/`skos:exactMatch`: scrutinize quality, demand to know the change pattern, cost it out, and ask what concrete benefit you get. If you proceed: link entities, but don't auto-import the external model's attached lexicalizations/attributes/relations — curate selectively.

## Governance: lean centralized when... lean decentralized when...
| Lean centralized | Lean decentralized |
|---|---|
| low dev automation | high dev automation |
| low modularity (parts interdependent) | high modularity |
| low quality-monitoring automation | high quality-monitoring automation |
| narrow domain, low vagueness | broad domain, high vagueness, need many perspectives |
| accuracy/conciseness priority | completeness priority |
| high downstream change impact | low downstream change impact |
| weak conflict-resolution mechanisms | strong, fast conflict-resolution mechanisms |

## Red-flag words that should trigger a pause
- "essential" / "important" / "strategic" / "critical" applied to a category → likely **vague**, check for borderline cases before treating as crisp.
- "synonym" used for anything merely *similar* or *related* → likely **bad synonymy** (Ch 7) — rename the relation (e.g., `hasAttractor`, not `rdfs:label`).
- "concept = class" stated as a rule → **fallacy** (Ch 1/2) — check for actual instances first.
- A relation named without direction (e.g., `broader`) → rename as a verb phrase with explicit direction (`hasBroader`).

## Bias checkpoints
Bias can enter at: specification, (semi-)automatic construction algorithms/sources, quality-dimension prioritization, or evolution strategy. Symbolic/crisp models are **not** inherently less biased than ML models — abstract/vague category boundaries (e.g., age → "Old Person") can encode harmful bias just as easily.
