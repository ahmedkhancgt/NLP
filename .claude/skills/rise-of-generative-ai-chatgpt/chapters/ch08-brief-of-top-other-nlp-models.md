# Chapter 8: Brief of top other NLP models

## Core Idea
GPT is not the only serious large language model family — BERT, XLNet, RoBERTa, ALBERT, PaLM, and Megatron each made distinct architectural trade-offs (bidirectionality, parameter efficiency, scale, multimodality) that define the competitive landscape ChatGPT sits within.

## Frameworks Introduced
- **Model-comparison rubric (Model / Performance / Capabilities)**: the table format the chapter uses for every competing model.
  - When to use: evaluating or pitching any LLM choice for a project.
  - How: for each candidate model, document (1) its architecture and parameter count, (2) its benchmark performance (e.g., GLUE, SQuAD, SuperGLUE scores), (3) its practical capability fit (chatbots, sentiment analysis, Q&A, classification, code, translation).

## Key Concepts
- **BERT**: bidirectional transformer with random-masked-token training (~340M parameters, 24 layers); strong for sentiment analysis and chatbot customer experience; GLUE ~80.4%, SQuAD 1.1 accuracy 93.2%.
- **XLNet**: combines TransformerXL's autoregressive approach with BERT's bidirectionality to overcome both models' limitations; outperformed BERT on 20 of the NLP tasks tested.
- **RoBERTa**: BERT trained on ~10x more data, longer training, larger batches (8000), 50k+ subword vocabulary; outperforms BERT on nearly every benchmark.
- **ALBERT**: reduces BERT's parameter bloat via factorized embedding parameterization and cross-layer parameter sharing — 18x fewer parameters, 1.7x faster training, with minimal performance loss (F1 92.2 on SQuAD, GLUE 89.4).
- **PaLM**: 540B-parameter model trained via data parallelization across two Cloud TPU v4 pods (57.8% hardware utilization); outperformed on 28 of 29 major NLP tasks; notably sample-efficient for code tasks versus Codex.
- **Megatron (MT-NLG)**: 530B parameters, 105 layers, 20,480 hidden dimensions, 128 attention heads, trained on 339B tokens from 15 blended datasets; strong on LAMBADA, RACE-h, logical reasoning, and mathematical inference.

## Mental Models
- Use the **"bidirectional vs. autoregressive" axis** to quickly classify any new LLM: BERT-style models read context in both directions (strong for understanding/classification tasks); GPT-style models are autoregressive left-to-right (strong for open-ended generation). XLNet is notable precisely because it tries to get both.
- Treat **parameter-efficiency models (ALBERT) as a counter-trend** to the "bigger is better" scaling race — useful when cost/latency constraints matter more than raw benchmark score.

## Key Takeaways
1. No single large language model dominates every NLP task — model choice should match the task profile (understanding/classification vs. open-ended generation vs. multi-tasking).
2. RoBERTa and ALBERT demonstrate two opposite optimization strategies on the same BERT base: RoBERTa scales data/training, ALBERT shrinks parameters — both improve on BERT from different directions.
3. PaLM's multi-task breadth (28/29 benchmarks) and training efficiency (less code needed than Codex) make it a credible large-scale GPT-3 alternative.
4. Megatron's blended, quality-weighted dataset sampling strategy is a reusable pattern for training on heterogeneous data sources.

## Connects To
- **Ch 7**: supplies the NLP-evolution context (transfer learning, pre-training) that these specific models all build on.
- **Ch 9**: contrasts these competitor architectures with GPT's own specific evolution (GPT-1 through GPT-3.5/InstructGPT).
- **Ch 30**: revisits several of these same models (BLOOM, PaLM, OPT) as explicit "ChatGPT alternatives" from a market/competitive angle.
