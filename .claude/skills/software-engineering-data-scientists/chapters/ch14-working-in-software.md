# Chapter 14: Working in Software

## Core Idea
Working effectively in a software/tech organization means understanding standard development processes (SDLC, waterfall vs. Agile), the varied technical roles you'll collaborate with, and the value of engaging with the wider developer community through open source and events.

## Frameworks Introduced
- **Software Development Lifecycle (SDLC)**: plan → design → build → test → deploy → maintain.
  - When to use: as a mental map of what phases any software feature/product passes through, regardless of methodology.
  - How: plan/analyze requirements (what's needed) → design architecture → build (write code) → test (QA, user inputs) → deploy (release to users) → maintain (fix bugs) — organizations vary in how rigidly or fluidly these steps are sequenced.
- **Waterfall vs. Agile**: two contrasting ways to organize the SDLC's steps over time.
  - When to use: waterfall suits projects with fixed, unchanging requirements and high failure cost (safety-critical, one-shot systems like deep-space probes); Agile suits most modern software work where requirements evolve and frequent deployment/feedback is valuable.
  - How: waterfall = each step fully completes before the next starts, no revisiting; Agile = all steps can happen concurrently/iteratively in short cycles, with frequent re-planning based on feedback.
- **Scrum / Kanban (Agile methodologies)**: concrete implementations of Agile.
  - When to use: structuring a team's day-to-day workflow.
  - How: Scrum = fixed-length sprints (commonly 2 weeks) with planning/review meetings, daily stand-ups, and a goal of a working increment each sprint; Kanban = a visual board tracking tasks through planned→in-progress→done, pulled from a backlog, more flexible than sprints (combining both = "scrumban").
- **Adapting Agile for data science**: take the useful structural parts of Agile, skip the parts that assume predictable, deployable-every-sprint output.
  - When to use: structuring a data science team's work without forcing a poor-fit methodology.
  - How: keep frequent planning meetings (re-prioritize based on early findings), a product-owner-like stakeholder voice, and regular demos/retrospectives; relax or drop the expectation that every sprint ends with a deployable increment, since exploratory research timelines are inherently unpredictable.

## Key Concepts
- **Front-end / back-end / full-stack engineer**: front-end = user-facing UI (HTML/CSS/JavaScript); back-end = server-side data processing/storage (Python, PHP, Ruby, Java, SQL); full-stack = both.
- **DevOps engineer**: proactively builds tools/processes to make development, testing, deployment, and maintenance more efficient.
- **Site Reliability Engineer (SRE)**: similar focus to DevOps but more reactive — "firefighting" production issues as they happen, then implementing preventive measures.
- **ML/AI engineer**: designs/implements ML systems, from training models to integrating multiple models into a product and monitoring performance — often overlapping with or encompassed by the data scientist role.
- **QA (Quality Assurance) / Test engineer**: finds bugs (including minor/edge-case ones affecting a small subset of users), maintains test plans/documentation, checks accessibility compliance, and helps gate releases.
- **Data engineer**: builds/maintains data pipelines turning raw data into usable aggregated tables, maintains data-capture/storage standards, ensures compute/storage efficiency, enforces privacy/access controls, sets up data quality checks, and builds KPI dashboards.
- **Data analyst**: derives insights from data (internal or market research) via SQL/Python/R, defines/tracks KPIs, builds visualizations for stakeholders; variants include product analysts (embedded with product teams) and business analysts (focused on operations/finance/competitive analysis).
- **Product manager**: sets product strategy/roadmap balancing customer and business needs, communicates it to engineers/designers, and manages execution — holds responsibility for product success without direct authority over the team, requiring strong cross-functional communication. A "product owner" (sometimes a separate role) handles day-to-day Agile execution.
- **UX researcher**: investigates *why* users behave as they do (vs. a data analyst's *what*) via interviews, usability sessions, and surveys; may split into qualitative (interviews/usability) and quantitative (large-scale surveys/statistics) specializations.
- **Designer (UI vs. UX)**: UI designer focuses on look (wireframes/mockups, visual style guides, individual elements) and works with front-end engineers who implement it; UX designer focuses on feel/usability (user research, navigation wireframes) and works with UX research/product management. Designers don't implement their designs in code.
- **Open source contribution**: a way to build skills and community involvement — via tagged "good first issue" entry points, project sprints at conferences, fixing something you personally hit while using a package (checking existing issues first), or publishing your own well-documented code publicly.
- **Python Software Foundation (PSF)**: the nonprofit (founded 2001) behind core Python's distribution, PyCon US, sponsorships/donations, and grants for sprints/user groups/development; open membership.
- **PyCon / regional Python conferences**: PyCon US (since 2003, 2,000–3,000 attendees) plus regional/topic conferences (PyCon Japan, LatAm, Namibia, India; SciPy for scientific computing) as community/learning hubs.
- **PyLadies**: a global network of groups supporting and mentoring women in the Python community, addressing underrepresentation; held its first online conference in 2023.

## Mental Models
- **SDLC as the shared skeleton, methodology as the muscle**: every organization roughly passes through plan/design/build/test/deploy/maintain — what differs is *how* those steps are sequenced and paced (waterfall's strict sequence vs. Agile's iterative overlap).
- **Waterfall for high-stakes irreversibility, Agile for evolving uncertainty**: the right methodology depends on whether requirements are fixed and failure is catastrophic (NASA updating Voyager 1's software 12 billion miles away — no second chances) versus whether frequent course-correction based on feedback is more valuable than upfront certainty (most commercial software today).
- **Data science as a partial, not perfect, fit for Agile**: data science's inherent unpredictability (you don't know if the 1st or the 15th experiment will work) breaks Scrum's "always ship something working each sprint" assumption — the useful parts (frequent planning, stakeholder involvement, demos, retrospectives) can be kept without forcing the ill-fitting parts.
- **Roles differentiated by what question they answer**: data engineers answer "is the data available and correct?"; data analysts answer "what happened?"; UX researchers answer "why did it happen, and what do users want?"; product managers answer "what should we build and why?" — a useful lens for quickly placing a new colleague's role.
- **Community participation as a career accelerator, not just altruism**: open source contribution and conference speaking build both technical skills (even public speaking "easier to get into than you think") and visibility/connections — benefits flow back into your own work via exposure to ideas outside your day job.

## Anti-patterns
- **Applying Scrum rigidly to data science without adaptation**: forcing "a deployable increment every sprint" onto exploratory research work misrepresents the nature of the work and sets unrealistic expectations.
- **Jumping straight to a large conference talk as your first public-speaking experience**: skips the lower-pressure skill-building available at local meetups/lightning talks.
- **Starting open-source work on an issue without checking if someone's already on it**: wastes effort and can create conflict; check current issues first, and for small projects, ask the maintainer how best to help.
- **Publishing personal code publicly without documentation**: undermines the value of sharing it — good documentation (Ch 9) is a prerequisite for a useful public contribution.
- **Treating waterfall as universally obsolete**: it remains appropriate for genuinely fixed-requirement, failure-intolerant contexts (safety-critical systems, certain government/defense/aerospace projects) — the right choice depends on context, not fashion.

## Code Examples
No code examples in this chapter — it covers team processes, industry roles, and community engagement rather than programming techniques.

## Reference Tables
| SDLC step | What happens |
|---|---|
| Plan/analyze | Define needed features/improvements |
| Design | Draft overall code architecture |
| Build | Write the code |
| Test | QA and user-input validation |
| Deploy | Release to end users |
| Maintain | Fix bugs post-release |

| Methodology | Structure | Best fit |
|---|---|---|
| Waterfall | Strict sequential phases, no revisiting | Fixed requirements, high failure cost, one-shot systems |
| Scrum (Agile) | Fixed sprints (~2 weeks), daily stand-ups, sprint goal | Iterative product development with evolving requirements |
| Kanban (Agile) | Visual board, pull from backlog, continuous flow | Flexible, ongoing task management |

| Role | Core question / focus |
|---|---|
| Software Engineer | Build and maintain the product's code |
| QA/Test Engineer | Does it work correctly for all users? |
| Data Engineer | Is the data pipeline correct, efficient, and accessible? |
| Data Analyst | What do the numbers say happened? |
| Product Manager | What should we build, and why? |
| UX Researcher | Why do users behave this way, what do they need? |
| UI/UX Designer | How should it look and feel? |

## Worked Example
A data science team adopts a modified Scrum: they keep two-week sprints with planning meetings at the start (to re-prioritize based on the latest experimental results) and review/retrospective meetings at the end (to extract lessons and standardize recurring project patterns), and they involve a stakeholder akin to a "product owner" who represents what the eventual users of the analysis/model actually need. However, instead of committing to "a working, deployable feature every sprint" (which doesn't fit exploratory research where you can't know in advance whether experiment 1 or experiment 15 will pan out), they replace that commitment with frequent demos of in-progress findings — keeping the team accountable and aligned with stakeholder needs without pretending research timelines are as predictable as feature development.

## Key Takeaways
1. The SDLC (plan, design, build, test, deploy, maintain) is the common skeleton of software development, organized either sequentially (waterfall) or iteratively (Agile/Scrum/Kanban).
2. Waterfall suits fixed-requirement, high-stakes, failure-intolerant projects; Agile suits most modern work needing flexibility and frequent feedback.
3. Agile practices (frequent planning, stakeholder involvement, demos, retrospectives) can benefit data science teams, but should be adapted rather than applied rigidly, given the inherent unpredictability of exploratory research.
4. Technical roles you'll likely encounter include software engineers (front-end/back-end/full-stack, plus specializations like DevOps/SRE/ML engineer), QA engineers, data engineers, data analysts, product managers, UX researchers, and UI/UX designers — each with distinct focuses and contributions to the SDLC.
5. Engaging with the wider community — contributing to open source (via tagged beginner issues, project sprints, or your own fixes/publications) and speaking at events (starting small, with clear structure and audience takeaways) — builds skills, visibility, and connections beyond your day job.
6. The Python Software Foundation, PyCon and regional conferences, and groups like PyLadies offer structured ways to connect with and learn from the broader Python/data science community.

## Connects To
- **Ch 7**: QA/test engineer roles and automated testing both serve the "does it work" question from different angles (team role vs. code practice).
- **Ch 9**: Documentation quality is a prerequisite for a meaningful open-source contribution.
- **Ch 10 / Ch 12**: Version control, code review, and CI/CD practices described earlier are the concrete day-to-day mechanics underlying the SDLC and Agile sprint cycles described here.
- **Ch 15**: Next Steps — continues the theme of professional growth beyond writing code.
- **External**: the Manifesto for Agile Software Development (2001); Eugene Yan's blog post on Agile for data science; Amazon's example SDLC; pandas/scikit-learn contributor guides; Nina Zakharenko's articles on speaking at events; the Python Software Foundation and PyLadies.
