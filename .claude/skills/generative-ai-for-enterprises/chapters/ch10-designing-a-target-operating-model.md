# Chapter 10: Designing a Target Operating Model

## Core Idea
A holistic operating model (distinct from the operating system of Chapter 5) aligns technology, people, process, and culture through seven interlocking layers — actor, requirements, consumption, platform, execution, landing zone, and applicability — unified by a continuous feedback loop, so that Generative AI investment translates into strategic business outcomes rather than disconnected pilots.

## Frameworks Introduced
- **The Seven-Layer Operating Model**: (1) Actor layer — human stakeholders and end-user devices participating in the AI journey; (2) Requirements layer — scalability, reliability, maintainability, ethics, governance, IP rights, culture; (3) Consumption layer — co-creation/co-design/co-deployment of business requirements and AI solutions, starting from PoC/MVP to enterprise-wide scale, defining ROI/TCO; (4) Platform layer — the Generative AI studio spanning the AI lifecycle (tune/validate/evaluate/test/deploy); (5) Execution layer — practical application domains (app dev, ITOps, digital transformation, software engineering, industry-specific use cases) plus technical governance; (6) Landing zone layer — hosting environments aligned to strategic goals/partnerships; (7) Applicability layer — AI-native/augmented workflows integrated into business/technical processes, measured by percentage automated/augmented.
  - When to use: As the master design framework when building or auditing an enterprise's end-to-end GenAI operating model.
  - How: Work through layers in order (actors and requirements first, applicability last), recognizing each layer constrains and informs the next; close the loop with continuous feedback.
- **Four Core Principles of Enterprise-Class Holistic Infusion**: Hybrid cloud-based ecosystem (consistency across on-prem/cloud); data training and ingestion (strengthening data architecture, modeling, governance, quality, integration); digital operationalization (aligning execution strategy to business value via digital capabilities); API economy (integration across silos as the "mantra for winning").
  - When to use: As guiding principles underpinning all seven layers, especially when scaling from pilot to enterprise-wide infusion.
  - How: Audit each principle's maturity before scaling AI initiatives beyond a single business unit.
- **Feedback Loop as Continuous Improvement Mechanism**: Systematically collect performance/outcome data across all seven layers (technical metrics, user satisfaction, business impact), then iteratively adjust algorithms, training data, and interfaces; also serves ethical/governance oversight by surfacing bias or drift for review.
  - When to use: Continuously, as a structural feature of the operating model rather than a periodic retrospective.
  - How: Embed feedback collection into actor and applicability layers; require organizational culture that values and acts on feedback, not just collects it.

## Key Concepts
- **Holistic operating model**: Organization-wide operational framework comprising seven layers plus feedback loop, governance, and cultural shift.
- **Seven layers**: Actor, requirements, consumption, platform, execution, landing zone, applicability.
- **Actor layer**: Human stakeholders (C-suite, developers, DevOps, etc.) and end-user devices participating in the AI journey.
- **Consumption layer**: Where business requirements and AI solutions are co-created/co-deployed/co-tuned, from PoC to enterprise scale.
- **API economy**: Integration across organizational silos as a precondition for real-time AI-driven outcomes.
- **Feedback loop**: The mechanism for continuous refinement of algorithms, training data, and UX based on multi-source performance data.

## Mental Models
- Think of the seven layers as a vertical stack from "who" (actors) through "what's needed" (requirements) to "where it runs" (landing zone) to "how it's applied" (applicability) — each layer must be solid before the next can bear weight.
- Use "operating model is the organizational blueprint; operating system (Ch 5) is the technical plumbing" to keep the two concepts distinct but complementary.
- Treat the feedback loop as a thermostat, not a survey: it must actively trigger retraining/UX/governance adjustments, not just report temperature.

## Anti-patterns
- **Skipping the requirements layer's governance/ethics components to move faster**: Creates compliance and bias blind spots that surface later at higher cost.
- **Starting consumption layer efforts enterprise-wide instead of with a focused PoC/MVP**: Skips the validation and ROI/TCO-definition step that justifies wider rollout.
- **Treating the platform layer as merely "pick a model"**: The platform layer must support the full lifecycle (tune, validate, evaluate, test, deploy), not just initial model selection.
- **Building feedback collection without organizational commitment to act on it**: A feedback loop that doesn't drive retraining, UX changes, or governance review is performative, not functional.
- **Confusing this operating model with the Chapter 5 operating system**: Using the terms interchangeably leads to incomplete planning — organizational alignment and technical infrastructure are both necessary but distinct.

## Worked Example
An enterprise rolling out GenAI starts at the consumption layer with a specific use case — say, automating first-line customer support responses — built as a proof of concept with a small team. Co-created with business and technical stakeholders (consumption layer), it draws on the platform layer's GenAI studio to tune and evaluate candidate models, while the requirements layer mandates ethical guardrails, scalability targets, and clearly assigned governance roles. The actor layer identifies which human roles (support agents, their supervisors) and devices (desktop support tools) interact with the system, feeding usage data back into the feedback loop. Once the PoC proves ROI, the execution layer extends the capability into broader ITOps and software-engineering workflows, hosted in the chosen landing zone (e.g., a hybrid cloud environment chosen per Chapter 4's deployment patterns), and finally the applicability layer measures success by what percentage of support workflows are now AI-augmented versus fully manual — feeding that measurement back into the loop to refine further rollout.

## Key Takeaways
1. Treat the operating model as seven distinct, ordered layers (actor, requirements, consumption, platform, execution, landing zone, applicability) rather than a single "strategy document."
2. Start new AI capabilities at the consumption layer with a focused PoC/MVP before scaling enterprise-wide, defining ROI/TCO explicitly at this stage.
3. Bake governance, ethics, and clearly assigned roles into the requirements layer from the start — it is the foundation all other layers build on.
4. The platform layer must support the full AI lifecycle (tune, validate, evaluate, test, deploy), not just model hosting.
5. Success in the applicability layer is measured by percentage of workflows automated/augmented — define this metric explicitly.
6. A functioning feedback loop must actively drive retraining, UX refinement, and governance review — not just collect data passively.
7. Keep "operating model" (this chapter, organizational) distinct from "operating system" (Chapter 5, technical) in planning conversations.

## Connects To
- **Ch 5**: Explicitly distinguished from, and complementary to, the technical operating system covered earlier.
- **Ch 3**: The platform layer explicitly references the model-selection criteria established earlier.
- **Ch 8**: The platform layer corresponds directly to the orchestration-platform architecture detailed previously.
- **Ch 9**: The requirements layer embeds the six ethical dimensions and governance structures established in the previous chapter.
- **Ch 11**: The consumption layer's ROI/TCO definition sets up the dedicated cost-optimization discussion that follows.
