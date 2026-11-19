# Chapter 2: History Of Generative Models

## Core Idea
Generative AI's evolution runs from rule-based/pattern-recognition systems (1950s-60s) through probabilistic models (1980s-90s: Hidden Markov Models, Bayesian Networks) to deep-learning generative architectures (2010s: GANs, VAEs) that can produce photorealistic images and natural text — each era adding more capacity for complex, diverse output.

## Frameworks Introduced
- **Three-era history of generative AI**: rule-based era → probabilistic era → deep-learning era.
  - When to use: framing how far a given generative technique has come, or scoping expectations for a new technique's maturity.
  - How: rule-based systems only did pattern-matching/decision rules; probabilistic models (HMMs, Bayesian Networks) added the ability to reason over uncertainty and generate more diverse outputs; deep learning (GANs, VAEs) unlocked highly realistic, complex generation.
- **Generative AI evaluation framework**: human evaluation + quantitative metrics (perplexity, inception score) + perceptual/user-experience metrics.
  - When to use: judging the quality of any generative model's output, not just text.
  - How: combine an objective score (perplexity/inception score) with subjective human judgment, since "creativity" and "quality" resist pure quantification.

## Key Concepts
- **Generative AI**: AI that creates new content/data from inputs, typically via deep learning and neural networks, as opposed to purely classifying or retrieving existing content.
- **Generative Adversarial Networks (GANs)**: deep learning models (2010s) that generate highly realistic outputs such as photorealistic images.
- **Variational Autoencoders (VAEs)**: deep learning models enabling generation of complex, diverse outputs alongside GANs.
- **Hidden Markov Models (HMMs)**: 1980s-90s probabilistic models for sequential decision-making, an early step toward more complex generative behavior.
- **Bayesian Networks**: probabilistic graphical models used in the same era for complex decision-making.
- **Perplexity / Inception score**: quantitative metrics used to evaluate generative model output quality.

## Mental Models
- Use the **capability-over-time staircase**: each generative AI era didn't replace the last so much as add a new rung of capability (rules → probability → deep learning) — useful for explaining to stakeholders why "AI" now feels qualitatively different from AI a decade ago.
- Think of **evaluation as triangulation**, not a single score — no single metric (perplexity, inception score, human rating) alone certifies a generative model's quality.

## Key Takeaways
1. Generative AI is a subfield of AI defined by creating new content/data, not just classifying existing data.
2. The deep-learning breakthrough (GANs/VAEs in the 2010s) is the inflection point that enabled today's realistic generative outputs — this is the direct ancestor of the GPT lineage covered in Ch 9.
3. Evaluating generative output quality is inherently harder than evaluating classification accuracy and requires blending human and automated metrics.
4. The book anticipates broad industry impact (entertainment, fashion, healthcare, finance) from generative AI well beyond text — a theme the later industry-specific chapters (Ch 3, 19-22, 26) each instantiate.

## Connects To
- **Ch 9**: picks up the generative-model lineage specifically for the GPT/transformer branch, in much greater technical depth.
- **Ch 3, 19-22**: apply the "generative AI across industries" forecast made here to specific sectors.
