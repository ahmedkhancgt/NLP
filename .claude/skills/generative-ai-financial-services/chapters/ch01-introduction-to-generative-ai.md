# Chapter 1: Introduction to Generative AI

## Core Idea
Generative AI is a branch of AI/deep learning built on foundation models (FMs) — pre-trained once on vast unlabeled data and then lightly customized per use-case, in contrast to predictive AI, which requires a new model trained from scratch on labeled data for every use-case.

## Frameworks Introduced
- **AI Taxonomy**: AI ⊃ Machine Learning ⊃ Deep Learning ⊃ Generative AI. Generative AI leverages neural network architectures to generate new content (text, image, audio, video) rather than only predicting labels.
  - When to use: framing where generative AI sits relative to classical ML/AI for stakeholders.
- **Generative AI Lifecycle** (contrasted with predictive ML lifecycle of Data prep → Train → Deploy → Monitor → Automate with MLOps):
  - When to use: planning any generative AI project in an enterprise/financial setting.
  - How: 1) Browse FM catalogs and shortlist <10 close-fit models; 2) Evaluate shortlisted models against the use-case; 3) Customize the chosen model (prompt engineering, RAG, fine-tuning — detailed in Ch3); 4) Perform inference directly via API, or deploy to an endpoint if self-hosting; 5) Enable monitoring for drift and feed corrective action back into model selection/customization.

## Key Concepts
- **Foundation model (FM)**: A model pre-trained on vast, curated data that can perform many tasks with little or no task-specific retraining.
- **Predictive AI**: Supervised/unsupervised ML that predicts outcomes based on labeled historical data; requires a new model per use-case.
- **Self-supervised learning** (contextualized further in Ch2): training approach FMs use, distinct from supervised/unsupervised predictive ML.
- **MLOps**: Operational discipline for automating predictive model lifecycles.
- **FMOps**: Operational discipline for automating generative AI/foundation model lifecycles (detailed in Ch14).
- **Prompt engineering, RAG, fine-tuning**: The three customization approaches for FMs, introduced here and detailed in Ch3.

## Mental Models
- Think of predictive AI as "one model per problem" and generative AI as "one model, many problems" — the shift from bespoke model-building to prompting/customizing a shared FM.
- Use the "train once vs. train per use-case" lens to explain to stakeholders why generative AI lowers the technical barrier to entry compared to predictive AI, even though it raises infrastructure cost (expensive hardware to host FMs).

## Anti-patterns
- None introduced yet in this chapter (anti-patterns are the explicit focus of Ch6); this chapter is purely foundational/definitional.

## Code Examples
No code examples in this chapter — it is conceptual grounding for the book.

## Reference Tables
**Table 1.1: Predictive AI vs. Generative AI**

| Dimension | Predictive AI | Generative AI |
|---|---|---|
| Output | Predicts behavior/pattern/result from past training data only | Generates completely new content without use-case-specific training |
| Skill required | Complex data science to train and produce predictions | Natural language skill (a good prompt) |
| Industry reach | Limited — new model per task is costly | Broad — low technical barrier via natural language interface and versatile FMs |
| Ops discipline | MLOps | FMOps (Ch14) |
| Hosting hardware | Less powerful/expensive hardware sufficient | Needs powerful, expensive hardware (trend toward smaller hostable FMs) |

## Worked Example
Bank savings-plan sales prediction (predictive AI): train a supervised model on past customer interactions + financial health + circumstances to predict whether a *new* customer will buy a savings plan — illustrates that predictive AI needs contextual labeled data and a dedicated model per use-case (fraud detection would need its own separate model). Contrast with generative AI: a single large FM trained once on broad unlabeled data (e.g., web text) can be given natural-language instructions to handle many different tasks without retraining.

## Key Takeaways
1. Generative AI's power comes from not needing to train from scratch per use-case — natural-language instructions unlock FM capabilities directly.
2. Financial organizations should expect FM customization (not full training) as the normal path: prompt engineering → RAG → fine-tuning, in increasing order of effort.
3. Generative AI needs its own operational discipline (FMOps), distinct from MLOps, because the lifecycle steps (model selection/evaluation/customization/monitoring) differ fundamentally from predictive ML.
4. Hardware/cost trade-offs are real: generative AI capability currently demands more expensive hosting than most predictive models, though this is improving.

## Connects To
- **Ch2**: Explains what foundation models actually are (architectures, training lifecycle, attributes) underlying the lifecycle introduced here.
- **Ch3**: Expands the "customization" step of the generative AI lifecycle into concrete implementation approaches (prompt engineering, RAG, fine-tuning).
- **Ch14**: Builds out the FMOps practices only named here.
