# Chapter 3: Introduction to Foundational Models on Kubernetes

## Core Idea
You rarely need to train a foundational model — you consume open-source ones from Hugging Face via lightweight frameworks (Gradio, Streamlit), containerize them, and deploy them on Kubernetes using the same Deployment/Service patterns from Ch2, with GPUs providing roughly 10x inference speedup over CPU.

## Frameworks Introduced
- **Hugging Face model evaluation pattern**: inspect a model's Model Card (intended use, limitations, training data, biases), Files tab (size/update recency), Community tab (issues/discussions), and license tags before adoption.
  - When to use: before adopting any open-source model, even for a demo.
  - How: visit `huggingface.co/<model>`, review all four areas, check the Inference API widget for a quick smoke test.
- **Gradio `Interface` vs `Blocks` UI pattern**: `gr.Interface` gives a one-line simple input/output UI; `gr.Blocks` gives full control (rows/columns/custom components) for exposing multiple tunable parameters.
  - When to use: `Interface` for a quick single-input/output demo (e.g. text completion); `Blocks` when you need sliders, radio selects, galleries, or multi-parameter tuning (e.g. image generation settings).
  - How: define a generation function → wrap with `gr.Interface(fn=..., inputs=..., outputs=...)` or compose `with gr.Blocks(): ...` and wire `.click(fn=..., inputs=[...], outputs=[...])`.
- **Container-then-deploy pattern for model demos**: package model + framework (Gradio/Streamlit) in a Dockerfile → build with `minikube image build` → deploy via Deployment + Service YAML → `kubectl port-forward` to access.
  - When to use: every hands-on model demo in this book and a template for early-stage model serving.
  - How: `FROM python:3.9-slim` → install requirements → `COPY . .` → `EXPOSE <port>` → `CMD ["gradio"/"streamlit", "app.py"]`.

## Key Concepts
- **Inference**: using a trained model to generate output from a prompt — far less compute-intensive than training.
- **Safetensors**: a secure model weight storage format (vs. pickle, which can execute arbitrary code on load); prefer `use_safetensors=True`.
- **Scheduler (diffusion)**: controls the noise-adding/removal process during diffusion training and inference (e.g. `DPMSolverMultistepScheduler`).
- **VAE (Variational Autoencoder)**: encodes/decodes between pixel space and the compressed latent space used during diffusion denoising; different VAEs (MSE vs EMA) trade off smoothness vs. realism (e.g. eyes).
- **Prompt guidance scale**: how strongly the diffusion model weights the text prompt vs. image quality — higher values follow the prompt more closely but can reduce quality.
- **num_inference_steps**: number of denoising iterations; more steps = higher quality but slower generation.
- **torch_dtype precision**: float32 vs float16 trade memory for fidelity when loading pretrained pipelines.

## Mental Models
- Think of the **DiffusionPipeline as a mix-and-match framework**: model + scheduler + VAE are independently swappable components, not a monolith — experiment with each independently.
- Use **CPU vs GPU as a 10x rule of thumb for diffusion inference** observed directly in this chapter's walkthrough (~12 min CPU vs ~1 min GPU for the same image generation).
- Model card warnings (e.g. "not tuned for fact generation", "includes biases") are **risk disclosures, not boilerplate** — treat them as inputs to a go/no-go decision for your use case.

## Anti-patterns
- **Skipping the model card/license review**: some models (e.g. OpenJourney) explicitly restrict certain harmful use cases in their license — deploying without checking can create legal/compliance exposure.
- **Exposing all configuration knobs to end users by default**: the book found that simply reducing image size (400x600 → 200x304) caused unexpectedly poor quality — exposing every parameter to naive users risks confusing failure modes; test edge cases before exposing a setting.
- **Using the pickle format for model weights when safetensors is available**: pickle allows arbitrary code execution on deserialization — a real security exposure (expanded in Ch10).
- **Running production inference workloads on CPU-only infra "because it works in a demo"**: CPU inference is viable for learning but the chapter shows roughly a 10x slowdown vs GPU — budget GPU infra for real latency-sensitive production use.

## Code Examples
```python
import gradio as gr
from transformers import pipeline

# Initialize the GPT-2 model pipeline
gpt2_pipeline = pipeline('text-generation', model='gpt2')

def generate_text(prompt, max_length=100, num_return_sequences=1):
    """ Generate text using the GPT-2 model """
    results = gpt2_pipeline(prompt,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True)
    return results[0]['generated_text']

gpt2_iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter a prompt..."),
    outputs='text',
    title="GPT-2 Text Generator",
    description="Enter a prompt to generate text using GPT-2."
)
gpt2_iface.launch(server_name="0.0.0.0", server_port=7800)
```
- **What it demonstrates**: minimal end-to-end Hugging Face pipeline + Gradio `Interface` wrapper — the base pattern reused (with Blocks) for image/audio generation later in the chapter.

```python
# Initialize OpenJourney model pipeline (Creates a StableDiffusionPipeline)
openj_pipeline = DiffusionPipeline.from_pretrained(
    "prompthero/openjourney-v4",
    torch_dtype=torch.float32,
    use_safetensors=True
)
```
- **What it demonstrates**: secure model loading (`use_safetensors=True`) and explicit precision control (`torch_dtype`) — both are production-relevant security/performance knobs.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: GradioGPT2Chat
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ggpt2chat
  template:
    metadata:
      labels:
        app: ggpt2chat
    spec:
      containers:
      - name: ggpt2chat
        image: gpt2gradio:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 7800
```
- **What it demonstrates**: standard pattern for deploying a containerized model-serving demo as a K8s Deployment, paired with a matching Service for `kubectl port-forward` access.

## Reference Tables
| Foundational Model Training Scale (excerpt) | # Training Parameters | Training Tokens | Year |
|---|---|---|---|
| GPT-2 | 1.5 Billion | 10 Billion | 2019 |
| GPT-3 | 175 Billion | 300 Billion | 2020 |
| Claude | 52 Billion | 400 Billion | 2021 |
| PaLM | 540 Billion | 768 Billion | 2022 |
| LLaMa | 65 Billion | 1.4 Trillion | 2023 |
| Falcon 180B | 180 Billion | 3.5 Trillion | 2023 |
(Table 3.1)

## Worked Example
**OpenJourney (Stable Diffusion fine-tune) image generation on Kubernetes:** Build the `GradioOpenJourney` Dockerfile → configure `IMAGE_HEIGHT=400`, `IMAGE_WIDTH=600`, `IMAGE_VARIATIONS=3` → load `DiffusionPipeline.from_pretrained("prompthero/openjourney-v4", torch_dtype=torch.float32, use_safetensors=True)` → in `generate_image()`, set `prompt_guidance=9`, random seeds per variation, swap in `DPMSolverMultistepScheduler`, and select VAE (`sd-vae-ft-mse` default vs `sd-vae-ft-ema`) → build Gradio Blocks UI with prompt textbox, inference-steps slider, VAE radio select, and output Gallery → run with prompt "A black and white photograph of a bird. Hi-resolution. HD 8k image" → CPU (D4s) run takes ~12 minutes; same run on an NC4as_T4_v3 GPU VM (`docker run --runtime=nvidia --gpus all`) takes ~1 minute, confirming the 10x GPU speedup.

## Key Takeaways
1. Hugging Face's Model Card / Files / Community / license tabs are the standard pre-adoption checklist for any open-source model.
2. `use_safetensors=True` should be the default for loading any pretrained pipeline — pickle format is an active security risk (foreshadows Ch10).
3. Diffusion image quality depends on an interacting set of knobs (scheduler, VAE, guidance scale, inference steps, resolution) — tune and test each, don't assume defaults are optimal for your use case.
4. GPU inference delivers roughly an order-of-magnitude speedup over CPU for diffusion workloads — budget accordingly for production.
5. The Dockerfile → `minikube image build` → Deployment+Service → `port-forward` pipeline is the reusable deployment pattern for every model demo in this book.

## Connects To
- **Ch2**: reuses the minikube/GPU setup and Deployment+Service patterns established there.
- **Ch4**: continues model evaluation (benchmarks, prompt engineering) and explores shrinking techniques (quantization) hinted at by the CPU/GPU performance gap here.
- **Ch10**: the safetensors-vs-pickle security distinction raised here is revisited as a supply-chain security concern.
