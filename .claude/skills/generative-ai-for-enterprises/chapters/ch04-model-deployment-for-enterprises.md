# Chapter 4: Model Deployment for Enterprises

## Core Idea
LLM deployment is a choice among five distinct patterns (cloud, open-on-premises, own-on-premises, open-cloud-IaaS, own-cloud-IaaS), each with different trade-offs in cost, control, and compliance; this choice must then be paired with an application-release strategy (Blue/Green, Canary, A/B Testing) to safely ship AI-powered features.

## Frameworks Introduced
- **Five Model Deployment Patterns**: (1) Cloud service LLM pattern — security/SLOs/HA/automation all managed by the cloud vendor, enterprise data not owned by vendor; (2) Open LLM on-premises — self/third-party managed, open model with possible active-user limits; (3) Own LLM on-premises — fully enterprise-owned model and data, fully self-managed; (4) Open LLM cloud IaaS — model-as-a-service or real-time endpoint on cloud infrastructure; (5) Own LLM cloud IaaS — enterprise-owned model and data running on vendor-managed cloud IaaS.
  - When to use: Immediately after model selection, to decide hosting architecture.
  - How: Rank regulatory posture (on-prem for highly regulated industries) against agility needs (cloud for speed); hybrid combinations are common.
- **Positioning Matrix (Regulation vs. Cost/Agility)**: Less-regulated, speed-focused industries → cloud deployment for elasticity and shorter time-to-value. Highly regulated industries (healthcare, finance) → on-premises for data control despite higher CapEx. Hybrid → sensitive data on-premises, less-critical workloads in the cloud.
  - When to use: Translating the five deployment patterns into an actual organizational decision.
  - How: Plot the organization's regulatory intensity against its cost-sensitivity/agility needs; cloud is generally most optimal, own-on-premises is generally most expensive.
- **AI Application Deployment Strategies**: Blue/Green (two full parallel environments, swap traffic after validation — safe and predictable rollback, costlier); Canary (no duplicate infrastructure, a user subset gets the new version first for early bug detection — cheaper than Blue/Green); A/B Testing (measures usability/experience impact on business metrics; can be combined with Blue/Green or Canary as the deployment mechanism).
  - When to use: When releasing new versions or features of an LLM-powered application, not the underlying model itself.
  - How: Choose Canary for cost-sensitive early-feedback cycles, Blue/Green for safe predictable cutover, A/B for measuring business-metric impact — these are complementary, not mutually exclusive.

## Key Concepts
- **Deployment patterns**: The distinct architectural options (cloud/on-prem/hybrid, own/open model) for hosting an LLM.
- **Deployment strategies**: Release-management techniques (Blue/Green, Canary, A/B) for shipping LLM-powered application changes safely.
- **Landing zone**: The pre-configured secure/compliant environment hosting the deployment (detailed further in Ch 5).
- **High availability (HA) and disaster recovery (DR)**: Redundancy/failover mechanisms ensuring continuous model availability.
- **Security by design**: Embedding encryption (at rest/in transit) and access controls at every architecture layer rather than retrofitting them.

## Mental Models
- Think of the five deployment patterns as a 2x2-plus-one grid: who owns the model (open vs. own) crossed with where it runs (on-premises vs. cloud IaaS), plus a fifth fully-managed "cloud service" option that abstracts both.
- Use "regulation pulls toward on-premises, agility pulls toward cloud" as the simplest heuristic for positioning — then layer hybrid as the practical middle path.
- Treat Canary vs. Blue/Green as a cost-vs-safety dial: Canary is cheaper/faster feedback, Blue/Green is safer/more predictable rollback.

## Anti-patterns
- **Choosing a deployment pattern based on vendor preference rather than regulatory/latency needs**: Leads to compliance gaps or poor data-proximity performance.
- **Deploying application updates directly to 100% of production without Canary/Blue-Green**: Increases blast radius of bugs in LLM-powered features.
- **Conflating deployment pattern decisions with the Chapter 5 operating-system/landing-zone requirements**: The two must be considered together, not sequentially in isolation.
- **Assuming cloud is always cheaper**: Hybrid/on-premises can be more cost-effective at very high, predictable utilization despite higher upfront CapEx.

## Worked Example
A healthcare enterprise evaluating deployment: on less-regulated marketing workloads, it adopts the cloud service LLM pattern for elasticity and low time-to-market. For patient-data-touching clinical-decision-support workloads, strict compliance pushes it toward the "own LLM on-premises" pattern despite higher CapEx and GPU procurement burden. The overall architecture becomes hybrid: sensitive processing stays on-premises while less-critical operations (e.g., marketing content generation) run in the cloud. When shipping a new conversational feature to the clinical-support app, the team uses Canary release to a small clinician subset first (cheap early bug detection) before considering Blue/Green for the full cutover, and uses A/B testing to measure whether the new UI improves clinician task completion time.

## Key Takeaways
1. There are five distinct deployment patterns, differentiated by model ownership (open/own) and hosting location (cloud service/on-prem/cloud IaaS) — match the pattern to regulatory and cost needs, not vendor default.
2. Highly regulated industries typically favor on-premises despite higher CapEx; less-regulated, agility-focused industries favor cloud.
3. Hybrid deployment (sensitive data on-premises, less-critical workloads in cloud) is a legitimate and common middle path.
4. Application deployment strategy (Blue/Green, Canary, A/B) is a separate decision layered on top of the model deployment pattern — don't conflate the two.
5. Canary is generally cheaper and faster for early feedback; Blue/Green is safer for predictable, reversible full cutovers; A/B testing measures business-metric impact and can combine with either.
6. Deployment pattern choice must be made jointly with the operating-system/landing-zone requirements covered in the next chapter.

## Connects To
- **Ch 3**: Directly follows from the model selection decision — deployment pattern is the next strategic choice after choosing the model.
- **Ch 5**: Deployment patterns here must align with the operating-system demarcation and co-location requirements detailed next.
- **Ch 11**: CapEx-vs-OpEx cost implications of each deployment pattern connect to the dedicated cost-optimization chapter.
- **Ch 13**: The cloud service LLM pattern here is the foundation for the Model-as-a-Service model explored in depth later.
