# Chapter 3: The Architecture Design Session

## Core Idea
An Architecture Design Session (ADS) is a structured, whiteboard-driven discovery workshop with business and technical stakeholders that produces a vendor-neutral architecture blueprint and action plan — and doing discovery thoroughly, before proposing any solution, is the single most important discipline of running one.

## Frameworks Introduced
- **The ADS (Architecture Design Session)**: a structured discussion with stakeholders to define a high-level data solution design.
  - When to use: at the start of any data architecture project, internal or with a customer, before product/technology selection.
  - How: produce two deliverables — an architecture blueprint and a high-level plan of action (POCs, demos, follow-ups). Not a tech workshop, training, demo, or detailed requirements session.
- **ADS Questionnaire**: a vendor-neutral discovery question set (~20+ questions) covering cloud adoption, data variety/volume/velocity, usage patterns (dashboards vs. ad hoc), SLAs, ML needs, HA/DR, MDM, security/sovereignty, budget, timeline, skill levels, and self-service BI needs.
  - When to use: during the discovery phase of every ADS, before any whiteboarding of architecture or product talk.
  - How: ask each question without naming products; let answers narrow the architecture options before introducing specific technologies.
- **"Thinking big, starting small"**: a framing for scoping delivery.
  - When to use: whenever a full architecture would ingest dozens of sources.
  - How: design the complete high-level architecture, but select 2–3 high-value, low-effort data sources as the first quick win.

## Key Concepts
- **Discovery**: the first 1–2 hours of an ADS spent asking questions (not proposing solutions) about pain points, current/future architecture, decisions already made, and use cases.
- **Whiteboarding**: the live, collaborative diagramming of the architecture plus goals, pain points, and "parking lot" items — explicitly preferred over presentation slides.
- **Parking lot**: off-topic items raised during the ADS that are written down for follow-up rather than discussed in the moment.
- **SME (Subject Matter Expert)**: a specialist brought in for topics outside the facilitator's expertise, rather than the facilitator trying to learn the topic on the fly.
- **GA (Generally Available)**: the stage after private preview (invited customers only) and public preview (anyone); using pre-GA products carries risks (bugs, limited support, pricing uncertainty).
- **Recovery Time Objective / Recovery Point Objective (RTO/RPO)**: disaster-recovery requirements that can force major architecture additions.

## Mental Models
- "Stay away from the solution" during discovery — don't let early product mentions short-circuit the questioning phase (the author's own early-career mistake: jumping to a Microsoft product demo before confirming the customer even needed a data warehouse).
- The medical-instrument-van analogy: ask diagnostic questions first, then present only the few relevant "instruments" (technologies), rather than showing everything up front.
- Read the room: treat stakeholder engagement (body language, phone-checking) as a live signal to switch topics or take a break.

## Anti-patterns
- **Jumping to product demos before discovery is complete**: risks educating the customer on something they don't need and wastes the ADS's value.
- **Running an ADS as a slide presentation**: slides turn an interactive discovery session into a one-way lecture; whiteboarding keeps it collaborative.
- **Letting a "science project" (technology-first, no business case) go unchallenged**: an ADS should surface whether a project has a real business case or is just a tech experiment, and limit resources for the latter.
- **Arguing with a difficult stakeholder**: escalating disagreement derails the session; defer contentious points to an offline follow-up instead.
- **Skipping breaks or letting account-team members over-talk**: both drain stakeholder engagement and derail focus.

## Reference Tables
| ADS Phase | Duration (in-person) | Focus |
|---|---|---|
| Introductions | Short | Roles, learning goals, prioritize goals |
| Discovery | 1–2 hours | Pain points, current/future architecture, use cases, business details |
| Whiteboarding | Remainder of day | Architecture diagram, goals, pain points, parking lot |
| Follow-up | Last 30 min | Action items, surveys |

| Format | Recommended Duration |
|---|---|
| In-person (full group, unfamiliar tech) | 7–8 hours, 1 day |
| In-person (familiar tech) | 6–7 hours |
| Remote | Two 4-hour sessions, back-to-back days or within a week |

## Worked Example
An ADS for a modern data warehouse project: facilitator preps via an account-team pre-call (budget, timeline, decision maker, customer pain points), builds a draft agenda, runs morning discovery (business/goals) and afternoon technical/architecture whiteboarding (so C-level and technical attendees can each focus on their relevant half-day), ends with a prioritized pain-point checklist on the whiteboard, and follows up with a summary document, exported whiteboard image, action items, parking-lot details, and a survey.

## Key Takeaways
1. Always complete discovery (vendor-neutral questions) before discussing any architecture or product.
2. Use the ADS questionnaire systematically — each question maps directly to an architecture decision (e.g., nonrelational data → need a data lake; low end-user skill → no-code/low-code products; data sovereignty → multiple data lakes).
3. Prefer whiteboarding over slides to keep the session collaborative and discovery-driven.
4. Build in explicit time for pain points, goals, and parking-lot follow-ups, and close every item before the ADS ends or schedule a follow-up.
5. Facilitation skills (humility, humor, reading the room, defusing conflict, stamina) matter as much as technical content.
6. Debrief internally and send a structured follow-up email (summary, architecture export, action items, parking lot, survey) after every ADS.

## Connects To
- **Ch 2**: The six architecture types are the output space the ADS questionnaire narrows down to.
- **Ch 4–14**: Specific questionnaire answers (nonrelational data, self-service needs, data sovereignty, etc.) map directly onto when to choose RDW, data lake, MDW, fabric, lakehouse, or data mesh.
- **External concept**: Agile discovery/requirements-gathering workshops in general software design.
