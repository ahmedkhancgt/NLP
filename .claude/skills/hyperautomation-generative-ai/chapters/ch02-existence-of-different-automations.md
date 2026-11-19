# Chapter 2: Existence of Different Automations

## Core Idea
Automation exists on a spectrum — from fixed/hard automation through programmable and flexible automation to full Robotic Process Automation (RPA) — and is executed by distinct agents (robots, cobots, bots) whose coexistence with humans, not replacement of them, produces the best outcomes.

## Frameworks Introduced
- **Three types of automation (Fixed / Programmable / Flexible)**: A classification of industrial/business automation by flexibility vs. volume trade-off.
  - When to use: To diagnose which automation category a candidate process already falls into before choosing a tool.
  - How: Fixed (Hard) automation = high volume, no change, high upfront cost, low per-unit cost. Programmable (Soft-leaning) automation = low/medium volume, batch production, program-controlled sequence changes. Flexible (Soft) automation = medium volume, no time lost reprogramming, higher per-unit cost, best for variable product mixes.
- **Global/Integrated Automation umbrella**: Groups CAM, RPA, Cognitive Intelligence, and Conversational Automation as the techniques that unify independently automated systems.
  - When to use: When multiple separately automated systems need to be streamlined into one coordinated system.
  - How: Use CAM for manufacturing process integration with CAD; use RPA for back-office/API-level task automation; layer Cognitive Intelligence for unstructured-data precision; add Conversational Automation (NLP-driven) for human-like customer interaction.
- **Human-Robot Interaction (HRI) coexistence model**: Research-backed finding (Univ. of Duisburg-Essen/Trier/Göttingen study, Prof. Matthias Klumpp) that joint human-robot teams outperform fully robotic teams in logistics simulations.
  - When to use: When deciding whether to fully automate a task or keep a human-machine team.
  - How: Design workflows that allow robot-to-human and human-to-robot handoffs, not just robot-to-robot or human-to-human; measure efficiency and accident/setback rates, not just automation %.

## Key Concepts
- **Fixed/Hard automation**: Automation with a configured, unchanging sequence of steps; high upfront cost, high volume, low flexibility (e.g., Flow Production).
- **Programmable automation**: Equipment whose operation sequence is set by a program/code and can be changed for batch production (e.g., cloth printing machines).
- **Flexible/Soft automation**: Produces varied product combinations with no lost time reprogramming; medium volume, higher per-unit cost.
- **Cobot (collaborative robot)**: A robot designed to learn multiple tasks and work alongside humans; compact, easy to install/program, flexible, cost-reducing.
- **Robot vs. Bot**: A robot is a machine performing physical tasks (categorized by appearance); a bot is a software program performing automated digital tasks (categorized by function), with no physical body.
- **RPA (Robotic Process Automation)**: A hallmark of Intelligent Process Automation (IPA) describing logic-driven execution via pre-programmed rules on structured data; the six standard functions are internal system interface, internet data extraction, external system interface, data check/collection, data manipulation, and commercial application.
- **Cognitive automation**: Simulates the human learning process to extract patterns from any data type (images, video, audio, text) for decision-making, independent of data nature.

## Mental Models
- Use the fixed→programmable→flexible spectrum as a dial: as production volume rises and variability falls, move left (fixed); as variability rises and volume falls, move right (flexible).
- Think of RPA as "doing" and Cognitive Automation as "thinking + doing in a loop" — RPA alone cannot reason about button-pushing or data entry, it just executes.
- Distinguish robots (bodies, physical tasks) from bots (software, digital tasks) when scoping a project — this avoids conflating hardware and software automation requirements.

## Anti-patterns
- **Treating RPA as a silver bullet for all repetitive work**: RPA only fits work that is repetitive, independently/cognitively stable, well-defined, rule-based, and logic/data-driven — poor process selection is the primary cause of RPA project failure.
- **Over-scaling without governance**: Per a Forrester report, 52% of RPA clients struggle to scale their program, and few RPA efforts grow past the first ten bots — scaling requires organizational readiness, not just more bots.
- **Ignoring organizational culture impact**: Treating RPA purely as a technical rollout while ignoring workforce fear of job loss and the need for upskilling undermines adoption.

## Reference Tables
*(No standalone comparison table reproduced in this chapter; the fixed/programmable/flexible characteristics above function as the chapter's core reference matrix.)*

## Worked Example
The **enterprise sales-procedure scenario**: a company changes its sales policy to require a 50% upfront payment before confirming reservations, but this rule is not yet encoded in the IT structure (Figure 2.3/2.4). Without RPA, the company must hire and train staff to manually validate billing/payment information and edit sell orders whenever a 50% promotion occurs — a recurring technical/organizational debt. With RPA, a quick software change (encoding the new rule in a bot) replaces the slower, costlier path of retraining hundreds of employees every time a process changes — illustrating why "changing code" beats "retraining people" as business rules evolve.

## Key Takeaways
1. Classify any automatable process into fixed, programmable, or flexible automation first — the right tooling and ROI math follow from that classification.
2. RPA's six core functions (system interfaces, internet extraction, data checks, manipulation, commercial application) define its practical scope; it has no "thinking" capability — that's cognitive automation's job.
3. Human-robot collaborative teams can outperform fully robotic teams in real workflows (per the Göttingen-led logistics study) — full automation is not always the most efficient design.
4. Robots and bots are categorically different: robots = physical bodies/appearance-defined; bots = software/function-defined.
5. RPA adoption benefits (boosted productivity, accuracy, security, scalability, customer experience) must be weighed against real scaling and cultural-adoption challenges.

## Connects To
- **Ch1**: Provides the automation taxonomy (fixed/programmable/flexible, RPA) that Chapter 1's automation-vs-Hyperautomation comparison assumes as background.
- **Ch3**: Picks up RPA specifically and goes deep into the concrete tools (UiPath, Automation Anywhere, Blue Prism) that implement the RPA concept defined here.
