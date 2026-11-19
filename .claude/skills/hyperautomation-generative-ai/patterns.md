# Patterns

## RPA-to-Hyperautomation Maturity Progression
**When to use**: When an organization has existing RPA (task-level, rule-based automation) and needs to decide whether/how to evolve toward full Hyperautomation.
**How**: Verify the gating condition first — does the remaining manual work genuinely require intelligence (judgment, unstructured-data handling, prediction)? If no, stay with RPA (cheaper, simpler). If yes, layer AI/ML/NLP/decision management on top of the existing RPA foundation rather than replacing it; follow the Four-Stage Journey (Research → Decision making → Action → Optimization) and the Five-Milestone plan (Identify problem → Automate fundamentals via RPA → Identify AI/ML needs → Validate results → Deliver polished output).
**Trade-offs**: Hyperautomation has higher implementation difficulty and longer time-to-market than RPA alone, but higher long-term ROI (per Table 4.3); skipping the gating check wastes cost on unnecessary AI/ML complexity.

## Five-Ingredient Solution Readiness Check
**When to use**: Before committing resources to any new Hyperautomation build.
**How**: Confirm all five are present — (1) a clearly defined problem statement/use case, (2) a genuine manual/semi-automated process needing intelligence, (3) a dedicated, skill-matched team, (4) adequate infrastructure, (5) the right technology mix (process/task mining, RPA, AI, ML, OCR, NLP). Treat any missing ingredient as a stop condition, not a risk to accept.
**Trade-offs**: Thorough upfront ingredient-checking slows project kickoff but prevents the two most common failure modes cited in the book: poor process selection and infrastructure/team gaps discovered mid-project.

## Human-in-the-Loop Document Processing
**When to use**: For processes triggered by incoming documents/emails containing semi-structured or unstructured data (invoices, claims, onboarding documents).
**How**: RPA/script collects the incoming document → ML model extracts machine-readable data → ML/rules validate the extraction (e.g., VAT compliance, fraud check) → route low-confidence-score outputs to a human reviewer for correction → feed corrections back for continuous model retraining.
**Trade-offs**: Adds latency for low-confidence cases but prevents silently accepting low-accuracy extractions (a named failure mode for Automation Anywhere's basic document preprocessing); requires an organizational process for validator review, not just technology.

## RPA Scaling via Center of Excellence (CoE)
**When to use**: When expanding RPA from a successful pilot to enterprise-wide deployment.
**How**: Assess automation potential across candidate processes → prioritize by impact/feasibility → standardize/optimize the process *before* automating it → select the right RPA tool/platform for scale → establish a CoE for governance and standards → automate gradually (pilot → expand) → continuously track performance and upskill employees.
**Trade-offs**: CoE governance adds coordination overhead but prevents the "struggle to scale past 10 bots" pattern the book cites (per Forrester, 52% of RPA clients struggle to scale); choosing centralized vs. federated CoE must match organizational culture or it becomes its own bottleneck.

## Three-Layer Architecture Design (Automation / Orchestration / Optimization)
**When to use**: When architecting any enterprise-scale Hyperautomation platform rather than point-solution bots.
**How**: Build the Automation layer first (RPA/AI/ML executing discrete tasks) → add the Orchestration layer to coordinate multiple automated processes across systems/departments into end-to-end workflows → close the loop with the Optimization layer (analytics/ML-driven continuous improvement of the automated processes).
**Trade-offs**: Skipping Orchestration leaves automation "islands" that don't combine into end-to-end value; skipping Optimization locks in whatever inefficiencies existed at automation time (echoing the "automation magnifies inefficiency" risk from Ch1).

## Fully-Digitalized vs. Document-Triggered Process Routing
**When to use**: When deciding the entry point/technology mix for a new automation candidate process.
**How**: Classify the process — if it's triggered by already-structured internal/client data (e.g., Accounts Payable, AML screening), route directly to RPA+ML validation. If triggered by incoming documents/emails with unstructured content, add an extraction step (OCR/NLP) before the validation stage.
**Trade-offs**: Misclassifying a document-heavy process as "fully digitalized" and skipping the extraction step produces poor-quality structured data downstream, undermining every subsequent automation stage.

## MLOps/ModelOps Governance Layering
**When to use**: Whenever ML models (not just rule-based RPA) are part of a Hyperautomation deployment, especially at enterprise scale with multiple models in production.
**How**: Use MLOps (Levels 0/1/2, matched to how often the model/data changes) to manage the model-building-to-deployment pipeline; layer ModelOps on top for enterprise governance (data validation, model versioning/scoring, evaluation for bias, CI/CD, drift/performance monitoring).
**Trade-offs**: Full Level-2 MLOps + ModelOps is resource-intensive and only justified for models needing frequent retraining or operating in regulated/high-stakes domains; for rarely-changing models, Level 0 MLOps is sufficient and cheaper.

## Human-Bot Complementary Workforce Design
**When to use**: When planning the workforce impact of any Hyperautomation or Generative-AI-augmented automation initiative.
**How**: Assign bots to repetitive, rule-based, structured-data tasks; reserve humans for unstructured-data interpretation, empathy-dependent customer interactions, and judgment calls. Plan around the empirically observed ~1.8:1–2:1 bot:human effectiveness ratio rather than assuming 3:1 (24/7 availability ≠ full-task superiority).
**Trade-offs**: Over-automating human-judgment-dependent tasks degrades customer experience and risks compliance/ethical failures; under-automating structured repetitive tasks wastes the clearest ROI opportunity.

## Generative AI + Hyperautomation Content Pipeline
**When to use**: For any use case needing new content/insight creation (marketing copy, product designs, financial reports, knowledge articles) combined with automated distribution/integration.
**How**: Generative AI model creates/analyzes the content (text, image, summary, design variant) → Hyperautomation automates its distribution, A/B testing, scheduling, or integration into downstream systems → feedback/performance data flows back to refine the Generative AI model (adaptive learning loop).
**Trade-offs**: Requires validating generated content for bias/accuracy before automated distribution at scale (a single bad generative output can be mass-distributed by the automation layer); needs the Six-Collaboration-Domains checklist (Ch11) to ensure the right domain-specific design (content gen, design/prototyping, data analysis, workflow optimization, process automation, adaptive learning) is applied.

## Contact Center Generative AI + Hyperautomation Pattern
**When to use**: For customer-service/contact-center operations with high call volume and repetitive post-contact administrative work.
**How**: Use Generative AI to auto-generate knowledge articles from chat transcripts (human-reviewed before publishing), group conversations by intent to surface automation opportunities, power virtual-agent bot-building from existing FAQs/manuals, expand conversational AI coverage for previously out-of-scope queries, and generate real-time trend summaries from multi-source consumer input; use Hyperautomation to integrate, schedule, and deploy these outputs, and to alert supervisors to agent-level issues.
**Trade-offs**: Expands self-service coverage and reduces agent repetitive workload, but requires continued human review/oversight for published knowledge content and for complex/emotionally-sensitive interactions that bots cannot adequately handle.
