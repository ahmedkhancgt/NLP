# Chapter 11: Bad Strategy and Organization

## Core Idea
Technical modeling skill is insufficient — semantic modeling initiatives fail without a concrete strategy (goals, high-level approach, decision mechanisms) tailored to real organizational context, and without a team with the right complementary skills and moderated attitudes, backed by real governance.

## Frameworks Introduced
- **Three-Part Semantic Model Strategy**: Goals (why/what/who/value/cost-of-not-building) → High-level Approach (guiding principles/philosophy, not a detailed plan) → Decision-making Mechanisms (criteria/processes for ongoing choices, not just upfront ones).
  - When to use: before or during any semantic modeling initiative, as a living (not one-time) strategic document.
  - How: answer concrete goal questions (never buzzword-laden ones like "make search semantic"); define guiding principles (e.g., open standards vs. proprietary, expert-sourced vs. mined, release cadence, centralized vs. open-contribution); establish ongoing decision criteria so future choices (what domain to add next, what quality aspect to improve) aren't made ad hoc or left undecided.
- **Context-Driven Strategic Decision Rules**: a strategy must be derived from your own organization's strengths/weaknesses/constraints, not copied from other organizations' public models.
  - When to use: whenever tempted to imitate another organization's semantic modeling approach (representation language, automation level, source reuse).
  - How: ask whether your model's requirements (structure, content, quality, usage) genuinely match the model you're imitating; if the answers aren't clearly "yes," don't base your approach on it (illustrated by the Neo4j-vs-RDF decision, which was driven by existing infrastructure investment, not Semantic Web community preference).
- **Governance Question Checklist**: five questions any semantic model governance framework must answer.
  - When to use: before a model is maintained by more than one person/team, especially across organizational boundaries.
  - How: define (1) what kinds of changes are allowed and under what circumstances, (2) who is allowed to make them, (3) what process changes must follow, (4) who owns/is accountable for the model, (5) how conflicts/disagreements are resolved and how rules are enforced.

## Key Concepts
- **Strategic goals vs. approach vs. means**: a goal answers "why/for whom/what value," not "how" (e.g., "develop a knowledge graph using deep learning" is a methodological choice, not a goal) and not a vague buzzword claim (e.g., "make search semantic" conveys nothing actionable; "increase search precision for subdomain X" does).
- **High-level approach as philosophy, not detailed plan**: a small set of guiding principles and priorities (illustrated by ESCO's three strategic choices — open standards (RDF(S)/SKOS), exclusively expert-sourced content, infrequent releases — contrasted with Diffbot's proprietary query language, web-crawled population, and near-continuous (4-5 day) release cadence).
- **Decision-making mechanisms**: the ongoing criteria/process for resolving future choices (what to add, what to improve) throughout the model's life — without them, teams either freeze in indecision or make low-value ad hoc choices.
- **The Product Owner gap**: a semantic model/knowledge graph without a dedicated product owner to set roadmap/priorities results in wasted development work (features built with no immediate client value) and missed critical features (never prioritized because nobody owned the backlog) — illustrated by the author's own Textkernel experience.
- **Strategic myths and half-truths**: five specific misconceptions the author has encountered: (1) semantic models must use Semantic Web (RDF/OWL) standards — false (Cyc, LinkedIn KG use other representations); (2) full automatic population with little human intervention works because others did it — only valid if your model's requirements genuinely match theirs; (3) public semantic models/standards are easily reusable — false, per Ch. 8's inaccuracy/mismatch findings; (4) semantic interoperability is just a matter of shared representation language — false, requires actual agreement/consensus on meaning, not just syntax; (5) uncertainty/vagueness are "noise" to exclude — a dangerous, dismissive stance (quoted from an actual Semantic Web conference speaker) that denies a core phenomenon covered throughout the book; and a sixth caution: academic/industrial papers' described methods/tools rarely reproduce their promised results directly in a new context without significant adaptation effort.
- **Cost/complexity drivers of semantic model development**: degree of automation (manual triples cost $2-$6 each vs. $0.0083-$0.14 automatically, per the "How Much Is A Triple?" cost study, with manually-created statements more likely correct); domain diversity/nuance; element complexity (axioms/rules harder than simple entities/relations); entity abstractness (abstract entities harder to rigorously define — Ch. 2/8); vagueness intensity; desired applicability/reusability scope (more contexts = more dev/maintenance overhead); source data structuredness (structured infoboxes/profiles vs. unstructured text/CVs); infrastructure/tooling maturity in the organization.
- **Team skill composition**: no single specialist type suffices — pure formal ontologists are too slow/costly for real-world deadlines; pure logicians produce axioms that break on noisy real data; pure data scientists can hit high accuracy on the wrong semantics entirely. The core required skill across all team members is conceptual/semantic thinking (understanding ambiguity/vagueness/other phenomena, correctly using modeling-language elements, deciphering inconsistent stakeholder terminology, and empathizing with how users might misinterpret the model) — supplemented, as needed, by IE/NLP/ML expertise, data engineering, UX/UI design, domain expertise, and applied-semantics expertise (e.g., librarians/taxonomists who understand user warrant and social context).
- **Harmful team attitudes (five specific ones)**: being pedantic (obsessing over valid-but-insignificant semantic distinctions regardless of practical relevance); being a "semantics nihilist" (the opposite extreme — ignoring obvious, important semantic distinctions); believing everything is boolean (denying the real-world prevalence of vagueness/uncertainty); being a data/expert/crowd "fanatic" (blind trust in one knowledge-source type at the exclusion of scrutinizing all three, per Ch. 8); and "Maslow's Hammer" thinking (reusing a familiar tool/relation without checking it actually fits the problem, per Ch. 10).
- **Semantic divergence**: when multiple teams independently and uncoordinatedly expand the same part of a shared model, their results can diverge on fundamental choices (e.g., granularity of synonymy) in ways that require costly after-the-fact reconciliation — illustrated by the author's own Textkernel case of two client-driven skill-concept expansions that took 2-3 months to merge due to incompatible synonymy granularity.
- **Governance as risk/cost reduction, not bureaucratic overhead**: establishing clear rules/roles/decision-rights/accountabilities is explicitly framed as necessary to minimize risk, reduce cost, and ensure the model's continued existence/evolution — a "laissez-faire" default invites semantic anarchy.

## Mental Models
- **The general's strategy analogy**: a battle strategy that ignores terrain/weather/opponent, or troops that can't execute it, both guarantee defeat — directly mapped onto semantic modeling: strategy divorced from organizational context, or a team that can't execute the strategy, both guarantee project failure.
- **Strategy is continuous, not a one-time document**: since both the model and its organizational context change over time, strategy must be continuously formed, applied, monitored, and revised — never "set and forget."
- **Skills and attitudes are separate axes**: having the right skills on a team is necessary but insufficient; the team members must also moderate counterproductive attitudes (pedantry, nihilism, boolean-thinking, source fanaticism, hammer-seeking-nails) to actually collaborate effectively.
- **Know your own context before imitating anyone else's**: public models and famous organizations' practices are data points, not templates — a strategic choice (RDF vs. property graph, manual vs. automatic, open vs. closed) is only valid relative to your own constraints, existing infrastructure, and requirements.

## Anti-patterns
- **Drafting a semantic modeling plan with no strategic/organizational content, focusing solely on technical/procedural issues**: produces plans that look sound but fail against the real-world nontechnical complexity of execution (the author's own repeated mistake, by his own account).
- **Adopting a strategy that is technically sound but incompatible with your actual organizational context**: strategy that ignores constraints, existing infrastructure, or stakeholder realities fails regardless of its technical merit.
- **Stating strategic goals in buzzwords or generic terms**: e.g., "make our search engine semantic" — conveys no concrete, actionable information about who benefits and how.
- **Confusing a development approach/methodology with a strategic goal**: e.g., "our goal is to build a knowledge graph using deep learning" states a means, not a goal (who needs it, why, what value) — the deep learning choice may be a legitimate strategic decision, but it isn't itself the goal.
- **Leaving a semantic model/knowledge graph without a dedicated product owner**: results in wasted, unprioritized development effort and missed critical features, as happened with Textkernel's knowledge graph before a product owner was assigned.
- **Believing you must use Semantic Web (RDF/OWL) standards to build "a real" semantic model**: ignores the many successful non-RDF models (Cyc, LinkedIn Knowledge Graph).
- **Assuming full automatic population works for you because it worked for DBpedia/Diffbot/LinkedIn**: without first verifying your target entities/relations actually exist findably in your available data, and that your quality bar matches theirs.
- **Assuming public models/standards can be reused wholesale without scrutiny**: contradicts the Ch. 8 finding that public models often contain inaccurate or mismatched information.
- **Assuming shared representation language (e.g., everyone using RDF) automatically yields semantic interoperability**: syntax agreement is not the same as agreement/consensus on meaning.
- **Dismissing uncertainty/vagueness as "noise to exclude"**: an explicitly cited real-world example (a Semantic Web conference speaker's stated position) of a strategic stance that denies a fundamental, unavoidable linguistic phenomenon — "consume at your own risk."
- **Assuming a published paper's method/tool will reproduce its promised results directly in your context**: underestimates the real implementation and methodological adaptation effort required.
- **Underestimating semantic model development cost/complexity**: failing to account for automation level, domain diversity, element complexity, entity abstractness, vagueness, desired reusability scope, source structuredness, and organizational tooling maturity when budgeting time/resources.
- **Copying another organization's technology/approach choice without validating it against your own context**: e.g., switching to RDF purely because the Semantic Web community insists it's "correct," while ignoring years of working, context-fit investment already made in an alternative (Neo4j) that meets the organization's actual needs.
- **Building a semantic modeling team of only one specialist archetype**: all-ontologists (too slow/costly), all-logicians (brittle on real noisy data), or all-data-scientists (fast but semantically off-target) each fail in a characteristic way.
- **Being pedantic**: insisting on including every theoretically valid semantic distinction (e.g., between near-synonyms) regardless of whether it's used or needed in practice.
- **Being a "semantics nihilist"**: the opposite failure — not noticing or caring about important, obvious semantic distinctions, and not pushing for clarity/specificity (directly causing the Ch. 6/7 pitfalls).
- **Believing everything is boolean**: denying the pervasiveness of vagueness/uncertainty in the real world, producing models that are technically crisp but practically useless.
- **Being a data/expert/crowd fanatic**: blind, exclusive trust in one knowledge-source type, ignoring that experts, crowds, and data can each independently be wrong (Ch. 8).
- **"Seeing everything as a nail" (Maslow's Hammer, reprised from Ch. 8/10)**: reusing a familiar tool/relation (e.g., an existing relatedness relation) without checking its actual fit for the application's real requirement.
- **Letting independent teams expand the same shared model region without coordination ("semantic divergence")**: produces incompatible granularity/synonymy choices that cost months of after-the-fact reconciliation, as in the Textkernel skill-concept case.
- **Adopting a laissez-faire governance stance ("semantic anarchy")**: no defined rules for what changes are allowed, who can make them, how disagreements are resolved, or how rules are enforced — invites wasted resources and model quality erosion as more parties get involved.

## Code Examples
Not applicable — this chapter is strategic/organizational, with no formal modeling syntax. The one structured artifact is the governance checklist (see Reference Tables).

## Reference Tables
*Sample strategic "what" and "how" questions (reconstructed from Table 11-1):*

| Questions about the model and its usage | Questions about the model's development |
|---|---|
| What knowledge domains should the model cover? | Use open standards or develop your own representation? |
| What parts will users pay for vs. get free? | Will content come primarily from domain experts or be mined from data? |
| What is the minimum acceptable accuracy level? | How often will the model be updated/released? |
| Single model for all customers, or customized per customer? | Will maintenance/evolution stay centrally controlled, or open to public contribution? |

*ESCO vs. Diffbot strategic contrast (reconstructed)*: ESCO = open standards (RDF(S)/SKOS), exclusively expert-sourced, infrequent releases. Diffbot = proprietary Diffbot Query Language, web-crawled/mined population, releases every 4-5 days.

*Governance's five core questions*: (1) what changes are allowed and under what circumstances, (2) who is allowed to make them, (3) what process must changes follow, (4) who owns/is accountable for the model, (5) how are conflicts resolved and rules enforced (e.g., Wikipedia's detailed consensus-building process; FIBO's permanent-ban policy for sustained community-standard violations).

*Semantic model development cost/complexity drivers*: automation degree, domain diversity/nuance, element complexity (axioms/rules > entities/relations), entity abstractness, vagueness intensity, desired applicability/reusability scope, source data structuredness, organizational tooling/process maturity.

*Triple cost estimate (reconstructed, approximate, from "How Much Is A Triple?")*: manual creation ≈ $2-$6 per triple (higher accuracy); automatic creation ≈ $0.0083-$0.14 per triple (lower accuracy) — illustrating the fundamental automation-vs-accuracy cost trade-off underlying Ch. 9's accuracy/completeness trade-off.

## Worked Example
The Textkernel "semantic divergence" story is the chapter's clearest cautionary worked example: in early 2019, two separate Textkernel teams independently began expanding the same region of the company's knowledge graph (skill concepts) for two different clients, without coordinating. When the teams eventually tried to merge their work, they discovered their underlying granularity assumptions about synonymy were incompatible — one client wanted strict, fine-grained distinctions the other client's data would never accept, while the other client's looser synonym groupings were unacceptable to the first. Reconciling the two expansions into one coherent merged model took an additional 2-3 months of negotiation and rework — work the author notes could have been avoided entirely had the two teams simply communicated their differing requirements before starting independent, uncoordinated work. This directly motivates the chapter's governance checklist: had a governance process defined who could make granularity-level decisions and required cross-team coordination before any changes to that model region, the costly divergence and rework would likely have been prevented.

## Key Takeaways
1. A semantic model strategy must explicitly cover goals (concrete, not buzzword-laden), high-level approach (guiding principles, not a detailed plan), and ongoing decision-making mechanisms — omitting any of the three invites failure.
2. Don't imitate another organization's semantic modeling approach (standards, automation level, source strategy) without first verifying your own requirements and context genuinely match theirs.
3. Semantic interoperability requires actual meaning consensus, not merely shared representation syntax.
4. Development cost/complexity scales with automation level, domain diversity, element complexity, entity abstractness, vagueness, desired reusability, source structuredness, and organizational tooling maturity — budget accordingly.
5. Build teams with complementary skills (semantic/conceptual thinking as the non-negotiable baseline, plus IE/NLP/ML, data engineering, UX, domain expertise, and applied-semantics expertise as needed) and actively moderate counterproductive attitudes (pedantry, nihilism, boolean-thinking, source fanaticism, tool-fixation).
6. Establish governance (what changes are allowed, who can make them, how, who's accountable, how conflicts/violations are resolved) before multiple parties begin independently modifying a shared model, to prevent costly semantic divergence.
7. Treat strategy as a continuous, revisable process, not a one-time plan — both the model and its organizational context change over time.

## Connects To
- **Ch 1**: directly echoes the "finding the right balance of meaning explicitness" theme as an organizational/team-composition challenge, not just an individual modeling skill.
- **Ch 5**: extends the "Setting the Stage" development activity (what/why/how/who/who cares) into a full strategic framework.
- **Ch 8**: reuses and extends the knowledge-source scrutiny (experts/data/crowd fallibility) into the "fanatic attitude" anti-pattern and the myths about public model reuse.
- **Ch 9**: the triple-cost accuracy/automation trade-off connects directly to the accuracy-vs-completeness quality trade-off discussed there.
- **Ch 10**: "seeing everything as a nail" directly reuses the Maslow's Hammer concept and the relatedness-relation reuse pitfall from that chapter.
- **Ch 12-14 (Dilemmas)**: this chapter closes Part II ("Pitfalls") and transitions directly into Part III, where many of the strategic choices raised here (open standards vs. custom, expert-sourced vs. mined, centralized vs. open governance) are reframed as structured dilemmas.
