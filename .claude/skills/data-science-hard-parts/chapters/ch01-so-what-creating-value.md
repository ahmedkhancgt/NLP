# Chapter 1: So What? Creating Value with Data Science

## Core Idea
Data by itself creates no value; value comes from the quality of decisions made with it. Data scientists create incremental value by improving an organization's decision-making capabilities, following a what / so what / now what progression.

## Frameworks Introduced
- **What / So What / Now What**: a three-stage approach to every DS project.
  - **What**: understand the business — metrics, levers, economic/psychological drivers.
  - **So What**: find the actionable implication — what levers exist, how will the output be used, is it even actionable.
  - **Now What**: be a go-getter — drive the decision through the org end to end (approvals, stakeholders, launch).
  - When to use: at project kickoff (backwards-design from "how will this be used") and again at delivery.
- **Value of X on Metric M framework**: `Impact of X = M(A(X)) − M(A(baseline))`.
  - Key insight: M is a function of actions (A), not of the model/analysis (X) directly — a model that doesn't change an action has zero impact.
  - How: identify the action layer between your output and the metric; never claim credit for M(X) alone.

## Key Concepts
- **Value creation ladder**: describe (BI/dashboards) → predict (ML) → decide/automate (optimization) — ascending levels of value.
- **Decentralized vs. centralized DS structure**: embedded teams build business expertise; centralized "consultant" teams struggle to.
- **Incremental value**: value attributable only to the change under consideration, not to existing baseline operations.
- **False positive cost**: cost of treating (e.g., discounting) a user who was never going to churn anyway.
- **False negative cost**: opportunity cost of revenue lost from churners the model/baseline failed to flag.

## Mental Models
- Data → Decisions → Value: value is never a direct function of data or models, only of the decisions/actions they inform.
- Own `M(A(X))`, not just `X`: end-to-end ownership of the whole chain from model to metric movement, not just the modeling step.

## Anti-patterns
- **Reading your charts at the meeting**: presenting visualizations without articulating the so-what/actionable implication.
- **Stopping at "churn decreased"**: failing to translate impact into a dollar figure, which weakens the case for more resources.
- **Functional specialization**: treating the job as ending at the technical deliverable and leaving business translation to stakeholders.

## Code Examples
```text
Impact of X = M(A(X)) − M(A(baseline))
Cost of Churn(A,X) = B × Churn(A,X) × R
ΔCost of Churn(A,baseline,X) = B × ΔChurn(A;X,baseline) × R
```
- **What it demonstrates**: how to convert a model's incremental effect on a rate metric (churn) into a dollar value using base size (B) and revenue per user (R).

## Reference Tables
(None — chapter uses worked formulas rather than tables.)

## Worked Example
Churn prediction model: 100 users, $7/month ARPU (R), baseline (segmentation) monthly churn 10% → cost of churn = $70/month. A new ML-targeted retention model reduces this to a cost of churn equivalent to $90/month in savings (i.e., baseline saved $70, model-driven approach saves $90) → incremental value = $90 − $70 = $20/month. A richer version also nets out false positive cost (users given a discount who would not have churned: `5 × 0.1 × P` for 5 non-churners given a 10% discount) and false negative cost (lost revenue from undetected churners).

## Key Takeaways
1. Companies exist to create value; a DS team that creates none is a cost center, not an asset.
2. Value comes from better decisions, not from data or models in isolation — always trace the action.
3. Start every project by solving backwards: what action will use this output, and is it actionable?
4. Write the "so what" down early; use it as a North Star during the technical work.
5. Translate impact into a monetary figure whenever possible to secure resources and buy-in.
6. Account for false positives and false negatives when quantifying incremental value, not just the headline metric change.
7. Soft skills (stakeholder management, end-to-end ownership) are as important as technical depth for driving real impact.

## Connects To
- **Ch 2**: Metrics design — "so what" presumes you already have actionable, well-designed metrics.
- **Ch 7**: Narratives — the what/so-what/now-what structure reappears as the narrative framework for delivery.
- **External concept**: marginal/incremental analysis from economics, used throughout the book's business-case chapters.
