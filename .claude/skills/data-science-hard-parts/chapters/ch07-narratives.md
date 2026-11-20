# Chapter 7: Narratives

## Core Idea
A data science narrative is a connected sequence of events built to drive action; it must be clear, credible, memorable, and actionable, and is best built iteratively from the start of a project rather than fitted onto results afterward.

## Frameworks Introduced
- **Four Properties of Successful DS Narratives**: Clear and to the point, Credible, Memorable, Actionable.
  - When to use: as a checklist before any delivery (deck, memo, TL;DR).
  - How: see Key Concepts below for sub-tips per property.
- **What / So What / Now What narrative structure** (vs. the classic setup–struggle–resolution arc): What = business problem context and sizing; So What = actionable results and Aha! moments; Now What = concrete next steps and who's needed.
  - When to use: structuring any delivery-stage presentation or document; preferred over the narrative arc because DS doesn't benefit from artificial suspense.
- **Iterative Narrative-Building Process** (vs. "do the work, then fit a narrative"): start with a business question → draft initial hypotheses/stories → test/refine with data iteratively → arrive at delivery-stage narrative. The key messages at the end are "direct descendants" of the initial hypotheses, purified through iteration.
- **TL;DR Construction**: write the full narrative/deck first, then compress into a TL;DR structured the same way (What/So What/Now What); every TL;DR line must have a backing slide.

## Key Concepts
- **Aha! moment**: an unexpected or surprisingly-quantified/actionable result; the mechanism for memorability in DS (preferred type: expected direction but surprising magnitude/actionability, not pure surprise).
- **Data credibility**: results must check out against known order-of-magnitude business facts; requires data-quality checks throughout development.
- **Technical credibility**: earned by using the right tool for the problem and surviving peer review (internal seminar practice).
- **Business credibility**: being as knowledgeable about the business as your stakeholder; faulty assumptions about the product/customer destroy credibility fast.
- **Elevator pitch**: a 10–20 second summary test; if you can't compress your narrative this far, the story has a structural problem.
- **Appendix discipline**: technical jargon and equations belong in an appendix, never the main narrative; keep the appendix itself curated, not a dump.

## Mental Models
- Narrative-first workflow (Figure 7-1): Business Question → Hypotheses/Stories → Iterative Testing → Delivery Narrative — storytelling happens at the start, middle, AND end, not just at delivery.
- "The reader is king": always write/design for the audience's background and attention span, not your own technical pride.
- Write-then-cut: draft fully, then aggressively delete anything not reinforcing the key message; iterate until "dropped so much the message isn't clear anymore," then back off one step.

## Anti-patterns
- **Fit-a-narrative-after-the-fact**: doing all technical work first, then bolting on a story — produces a "collage of findings" lacking a throughline.
- **Including every technical detail to prove rigor**: overloading slides/appendices to showcase effort, which distracts from and dilutes the key message.
- **Using unexpected-but-expected-direction findings as if routine**: showing a result everyone already expects (e.g., "lower price boosts sales") without quantification fails to create an Aha! moment.
- **Channel-dependent clarity**: assuming narrative only needs to work live; it must be self-explanatory for asynchronous readers (e.g., C-level reading later).
- **TED-style suspense in business settings**: injecting struggle/suspense into DS narratives generally backfires; actionable simplicity beats dramatic arcs.

## Code Examples
(No code in this chapter — narrative/communication-focused.)

## Reference Tables
| Property | Core Tip |
|---|---|
| Clear | Identify audience; jargon to appendix; short sentences (<10 words); self-explanatory regardless of channel |
| Credible | Check data vs. business reality; present technical work to peers; match stakeholder business knowledge |
| Memorable | Aha! moments — prefer expected-direction + surprising quantification/actionability over pure novelty |
| Actionable | Confirm actionable insight exists before starting the narrative at all |

## Worked Example
Pricing narrative: "Users are relatively price insensitive above $5.30, but below that, each additional $1.00 discount boosts sales by 1,000 units." This beats the generic "lower prices increase sales" because it quantifies a threshold and a specific lever-to-outcome rate, creating both credibility and a direct call to action for the pricing/marketing team (the Now What). The chapter also walks through 3 TL;DR iterations for itself: version 0 (cluttered, tiny font, unreadable), version 1 (simplified sentences), version 2 (restructured cleanly around What/So What/Now What) — illustrating the write-then-cut process in action.

## Key Takeaways
1. A narrative must drive action — if there's no actionable insight, go back to the drawing board before building the story.
2. Prefer iterative narrative-building (hypotheses from day one) over bolting a story onto finished technical work.
3. Structure delivery as What / So What / Now What rather than the classical narrative arc.
4. Build credibility across three dimensions: data, technical, and business — losing any one undermines the whole narrative.
5. Aha! moments are best created by quantifying or making actionable an already-expected directional finding, not by shock value.
6. Write the full story first, then compress to a TL;DR; every TL;DR point needs a backing slide.
7. Use the elevator-pitch test before and after you think you're "done" to validate narrative coherence.
8. Always quantify claims ("decreases NPS by 3pp" vs. "decreases NPS"), but don't overstate precision beyond what your estimates support.

## Connects To
- **Ch 1**: Directly extends the what/so-what/now-what framework introduced there into delivery format.
- **Ch 8**: Datavis principles (labeling, simplicity, avoiding clutter) are the visual complement to narrative clarity.
- **Ch 13**: Storytelling in ML reuses these same credibility/actionability principles specifically for model explanations.
