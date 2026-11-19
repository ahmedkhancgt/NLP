# Chapter 1: Exploring the World of Generative AI

## Core Idea
Generative AI is a subfield of deep learning where models learn the probability distribution of training data well enough to generate new, never-before-seen content (text, image, audio, video, code) rather than merely classify or predict a label.

## Frameworks Introduced
- **AI → ML → Deep Learning → Generative AI hierarchy**: generative AI is nested inside deep learning, which is nested inside machine learning, which is a subfield of AI.
  - When to use: to explain where a technique sits relative to classic ML before diving into GANs/VAEs/transformers.
  - How: AI (systems that reason/act autonomously) → ML (supervised/unsupervised/semi-supervised learning from data) → Deep learning (multi-layered ANNs) → Generative AI (uses labeled + unlabeled data to create novel outputs).
- **Discriminative vs. Generative models**: discriminative models learn P(y|x) to label/classify data; generative models learn the joint distribution P(x,y) to create new data instances resembling training data.
  - When to use: deciding whether a use case is "generative AI" at all.
  - How: ask what y (the output) is — a number/discrete class/probability = not generative; natural language, image, audio, or video = generative AI.
- **GAN (Generative Adversarial Network)**: a generator and discriminator trained adversarially — generator tries to fool the discriminator, discriminator tries to catch fakes, both improve iteratively.
  - When to use: image/video/voice synthesis tasks.
  - How: generator produces candidate data; discriminator classifies real vs. generated; backpropagate both networks competitively.
- **VAE (Variational Autoencoder)**: compresses input into a lower-dimensional latent representation, then generates new samples from that learned latent space.
  - When to use: when you need a compact learned representation to reconstruct/generate variations of data.
- **Foundation model paradigm**: one model trained unsupervised on massive unstructured data can be transferred to drive many different downstream tasks/applications (vs. training a separate model per task).

## Key Concepts
- **Foundation model**: a model pre-trained on massive unsupervised/unstructured data that transfers across many downstream tasks (term coined by Stanford researchers).
- **Supervised learning**: trains on labeled data to predict a tag/class/number for new inputs.
- **Unsupervised learning**: finds structure/groupings in unlabeled data.
- **Semi-supervised learning**: trains on a small labeled set plus a large unlabeled set — the gateway concept into generative AI.
- **Hallucination**: a model-generated output that is grammatically incorrect, nonsensical, or factually wrong, often from insufficient/noisy training data, missing context, or missing constraints.
- **Transformer (encoder-decoder)**: began the 2018 NLP revolution; encoder encodes an input sequence, decoder decodes it for the target task.
- **Multi-modal language model (MMLM)**: processes/generates across modalities (text, image, audio) rather than a single modality.
- **Text-to-X model types**: text-to-text, text-to-image, text-to-video/3D, text-to-task (autonomous agents), text-to-code — each with distinct application tables.

## Mental Models
- Think of a discriminative model as an art critic (spotting fakes/labeling) and a generative model as a forger (creating new convincing works) — this is the GAN intuition generalized to all generative AI.
- Use "y = f(x)" as a litmus test: if y is a number, discrete class, or probability, it's conventional ML; if y is language/image/audio/video, it's generative AI.
- Think of traditional programming as hard-coded rules ("Cat: legs=4, ears=2...") versus generative AI as pattern-learned behavior (show examples, let the model infer the rule).

## Anti-patterns
- **Treating any ML output as "generative AI"**: predicting a number, class or probability (house prices, spam/not-spam, rain probability) is not generative AI even if a neural network is used — the output type is what matters.
- **Ignoring hallucination risk**: deploying transformer output directly into production decisions without accounting for insufficient data, noisy data, missing context, or missing constraints as hallucination causes.

## Code Examples
```text
Cat:
type: animal
legs: 4
ears: 2
fur: yes
likes: catnip, yarn
```
- **What it demonstrates**: the traditional-programming approach of hard-coding rules to classify a "cat," contrasted against generative AI's pattern-learned approach where a model is simply shown examples and asked to generate/classify.

## Reference Tables
| Model type | Applications |
|---|---|
| Text-to-text | Generation, Classification, Summarization, Translation, (Re)Search, Extraction, Clustering, Content editing/re-writing |
| Text-to-image | Image generation, Image editing |
| Text-to-video / Text-to-3D | Video generation, Video editing, Game assets |
| Text-to-task | Software agents, Virtual assistants, Automation |
| Text-to-code | Code generation, Bug fixing, Source code documentation, Code summarization |

## Worked Example
The book walks through classifying a cat image two ways: (1) traditional programming hard-codes attributes (4 legs, 2 ears, fur, likes catnip/yarn) as explicit rules; (2) a neural network is instead shown many labeled cat/dog images and learns to predict "cat" — illustrating supervised learning. It then extends this to the generative case: instead of asking a model to classify an existing image as a cat (discriminative), you ask the model to generate new images of cats it has never seen (generative) — the model has learned the joint distribution of what "catness" looks like, not just a decision boundary.

## Key Takeaways
1. Before calling something "generative AI," check what the output y actually is — only novel language/image/audio/video counts.
2. Foundation models are valuable because one unsupervised-pretrained model can be reused across many tasks, rather than training per-task models from scratch.
3. GANs (adversarial generator/discriminator) and VAEs (latent-space compression) are the two classic generative architectures underlying image/content generation.
4. Hallucinations are a structural risk of transformer-based generation, not a rare bug — expect and plan for them in any application.
5. Transformers (2018+) and their encoder-decoder structure underlie the current generative AI wave, including GPT-family models.

## Connects To
- **Ch2**: Applies this chapter's taxonomy (GANs, LLMs, foundation models) to concrete real-world use cases in content, art, conversational agents, and VR.
- **Ch4**: Revisits foundation models and deep-dives into specific ones (GPT-3/4, Claude, Gemini, Llama 2) with integration details.
- **Ch5**: Builds on hallucination and vector-embedding concepts introduced here to discuss RAG and fine-tuning as mitigations.
