# Chapter 8: Building and Deploying Generative Models

## Core Idea
Turning a trained GAN or VAE into a production system follows a consistent pipeline — serialize, containerize, push to a registry, create a managed-platform model/endpoint — and the same pipeline (with platform-specific CLI syntax) applies across AWS SageMaker, Azure Machine Learning, and Google Cloud AI Platform.

## Frameworks Introduced
- **Generative-model development lifecycle**: choose architecture (GAN/VAE/normalizing flow) → collect/prepare training data → train (often on cloud compute) → evaluate on held-out data → deploy to production.
  - When to use: as the top-level checklist for any new generative-model project, before diving into platform-specific deployment steps.
- **Universal deployment pipeline**: (1) serialize the model (SavedModel/ONNX/TorchScript), (2) assess scalability needs, (3) build an API (REST/GraphQL), (4) containerize (Docker), (5) choose cloud vs. edge deployment, (6) add monitoring/logging, (7) secure the endpoint.
  - When to use: deploying *any* trained generative model to a real-world serving environment, regardless of cloud provider.
- **AWS SageMaker GAN/VAE deployment via Docker + ECR**: train → write inference script → Dockerize → push image to Amazon ECR → create SageMaker model referencing the ECR image → create endpoint configuration → create endpoint → wait for in-service.
  - When to use: deploying a trained GAN or VAE to AWS when you need a managed, scalable inference endpoint.
- **Deployment-readiness considerations**: compute resources, model size, model latency, model accuracy, model fairness — each with its own measurement approach and mitigation techniques.
  - When to use: as a pre-launch checklist before promoting any generative model from training to production.

## Key Concepts
- **Model serialization**: converting a trained model into a storable/transportable format (TensorFlow SavedModel, ONNX, PyTorch TorchScript).
- **Containerization**: packaging a model and its dependencies into a Docker container for consistent deployment across environments.
- **Amazon ECR (Elastic Container Registry)**: AWS's managed Docker image registry used to store the inference container before SageMaker deployment.
- **SageMaker endpoint configuration**: the AWS resource that specifies the model, instance type, and instance count backing a live inference endpoint.
- **Model quantization / pruning / knowledge distillation**: techniques to shrink model size and reduce latency for deployment-constrained environments (mobile, edge).
- **FID / Inception Score (deployment context)**: used to evaluate generative output quality pre- and post-deployment, since classification accuracy doesn't apply.
- **Model fairness metrics**: demographic parity and equalized opportunity — statistical techniques to audit a deployed generative model for biased outputs.

## Mental Models
- Treat cloud deployment as "the same five boxes, different labels": every platform (AWS/Azure/GCP) needs a model artifact, a container/runtime, a registry, an endpoint config, and a live endpoint — only the CLI verbs differ.
- Use the five deployment-readiness dimensions (compute, size, latency, accuracy, fairness) as a pre-flight checklist; a model that passes accuracy but fails fairness, or passes latency but fails compute-cost sustainability, is not actually production-ready.
- When choosing among AWS SageMaker, Google AI Platform (Unified), and Azure Machine Learning, let the decision follow from what you value most: SageMaker for maturity/breadth, Google AI Platform for ease of use, Azure ML for deep cloud-service integration and responsible-AI tooling.

## Anti-patterns
- **Skipping the inference script/Dockerization step**: a trained model file alone is not deployable — SageMaker (and equivalents) require a defined inference process bundled with the model.
- **Deploying without model-fairness auditing**: the chapter explicitly calls out those with biased training data producing biased generative outputs; statistical fairness metrics must be checked before production rollout, not just accuracy.
- **Ignoring model size/latency trade-offs**: deploying a large, unoptimized GAN to a latency-sensitive or mobile context without pruning/quantization/distillation causes unacceptable response times.
- **Treating cloud deployment commands as universal boilerplate**: always replace placeholder values (`<your-region>`, `<your-account-id>`, ARNs) with environment-specific values — copy-pasting without substitution is the most common deployment failure.

## Code Examples

### AWS: push a Docker inference image to ECR and deploy a VAE on SageMaker
```bash
# Authenticate Docker to your ECR registry
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com

# Tag your Docker image
docker tag <your-image> <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/<your-repository>:<tag>

# Push the Docker image to ECR
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/<your-repository>:<tag>

# Create a SageMaker model
aws sagemaker create-model --model-name <your-model-name> --primary-container Image="<your-image>",ModelDataUrl="<your-model-artifacts-S3-url>"

# Create an endpoint configuration
aws sagemaker create-endpoint-config --endpoint-config-name <your-endpoint-config-name> --production-variants VariantName=<your-variant-name>,ModelName=<your-model-name>,InstanceType=<instance-type>,InitialInstanceCount=<instance-count>

# Create an endpoint
aws sagemaker create-endpoint --endpoint-name <your-endpoint-name> --endpoint-config-name <your-endpoint-config-name>

# Wait for the endpoint to be in-service
aws sagemaker wait endpoint-in-service --endpoint-name <your-endpoint-name>
```
- **What it demonstrates**: the full AWS path from a Docker-packaged VAE inference container through ECR to a live, production SageMaker endpoint.

### AWS: deploy a GAN to SageMaker via CLI (S3 + model + endpoint)
```bash
# Create an S3 bucket for model artifacts and code
aws s3 mb s3://<your-bucket-name>

# Upload model artifacts and inference code to S3
aws s3 cp model/ s3://<your-bucket-name>/model/ --recursive
aws s3 cp code/ s3://<your-bucket-name>/code/ --recursive

# Create a SageMaker model referencing a Docker image in ECR
aws sagemaker create-model \
  --model-name gan-model \
  --execution-role-arn <your-role-arn> \
  --primary-container ImageUri.dkr.ecr.<your-region>.amazonaws.com/<your-gan-image>:latest

# Create an endpoint configuration
aws sagemaker create-endpoint-config \
  --endpoint-config-name gan-endpoint-config \
  --production-variants VariantName=gan-variant,ModelName=gan-model,InstanceType=<your-instance-type>,InitialInstanceCount=1

# Create the endpoint
aws sagemaker create-endpoint \
  --endpoint-name gan-endpoint \
  --endpoint-config-name gan-endpoint-config

# Update the endpoint with the latest model/config
aws sagemaker update-endpoint --endpoint-name gan-endpoint --endpoint-config-name gan-endpoint-config
```
- **What it demonstrates**: packaging convention (`model/` + `code/` with `inference.py` and `requirements.txt`) and the S3-to-SageMaker-endpoint deployment sequence for a GAN.

### Azure: deploy a GAN/VAE via Azure ML CLI
```bash
# Log in to Azure
az login

# Create an Azure Machine Learning workspace
az ml workspace create -w WORKSPACE_NAME -g RESOURCE_GROUP

# Register the model with Azure Machine Learning
az ml model register -m MODEL_PATH -n MODEL_NAME --asset-path ASSET_PATH --model-framework MODEL_FRAMEWORK

# Deploy the registered model as a web service
az ml model deploy -n SERVICE_NAME --service-usage SCORING_ENDPOINT -w WORKSPACE_NAME --cpu CORES --memory MEMORY
```
- **What it demonstrates**: the Azure-equivalent four-step flow (login, workspace creation, model registration, web-service deployment) to the AWS SageMaker pipeline above.

### Google Cloud: deploy a GAN/VAE via gcloud CLI
```bash
# Authenticate and set project
gcloud auth login
gcloud config set project PROJECT_ID

# Create the model resource and deploy a version
gcloud ai-platform models create MODEL_NAME

gcloud ai-platform versions create VERSION_NAME \
  --model=MODEL_NAME \
  --origin=gs://YOUR_MODEL_BUCKET_PATH \
  --runtime-version=2.6 \
  --framework="TENSORFLOW" \
  --python-version=3.7
```
- **What it demonstrates**: the GCP AI Platform (Unified) equivalent flow — model resource creation followed by a versioned deployment pointing at a GCS model artifact path.

## Reference Tables
**Table 8.1 — Comparison between AWS SageMaker, Google AI Platform (Unified), and Microsoft Azure Machine Learning**

| Feature | AWS SageMaker | Google AI Platform (Unified) | Microsoft Azure Machine Learning |
|---|---|---|---|
| Ease of use | Medium | Easy | Easy |
| Scalability | High | High | High |
| Cost-effectiveness | Medium | Medium | Medium |
| Comprehensive capabilities | High | High | High |
| Pre-trained models | Wide range | Wide range | Wide range |
| Code samples and tutorials | Wide range | Wide range | Wide range |
| Integration with other cloud services | Deep AWS integration | Deep Google Cloud integration | Deep Azure integration |
| Open-source support | TensorFlow, PyTorch, scikit-learn | TensorFlow, PyTorch, scikit-learn | TensorFlow, PyTorch, scikit-learn |
| Responsible AI | Model fairness/explainability features | Model fairness/explainability features | Model fairness/explainability features |

**Deployment-readiness dimensions and mitigation techniques**

| Dimension | Risk if ignored | Mitigation |
|---|---|---|
| Compute resources | Training/inference cost spirals | Cloud elasticity, distributed training frameworks |
| Model size | Storage/bandwidth/compute overload | Smaller architectures (e.g., MobileNet-style), pruning, quantization, knowledge distillation |
| Model latency | Poor user experience in real-time apps | Smaller architecture, quantization/pruning, specialized hardware (GPU/TPU), model-serving frameworks |
| Model accuracy | Unreliable/misleading generated outputs | Larger/more diverse dataset, longer training, regularization |
| Model fairness | Biased or discriminatory outputs | Debiasing algorithms, fair training process, human-in-the-loop monitoring |

## Worked Example
**End-to-end VAE deployment to AWS SageMaker**: (1) train the VAE in TensorFlow or PyTorch and save the artifacts; (2) write an inference script that loads the model, preprocesses input, and runs inference; (3) Dockerize the inference script and its dependencies; (4) authenticate Docker to ECR, tag, and push the image (`aws ecr get-login-password ... | docker login ...`, `docker tag ...`, `docker push ...`); (5) create a SageMaker model pointing at the pushed image and the model's S3 artifact URL (`aws sagemaker create-model ...`); (6) create an endpoint configuration specifying instance type/count (`aws sagemaker create-endpoint-config ...`); (7) create the endpoint (`aws sagemaker create-endpoint ...`); (8) wait for it to report in-service (`aws sagemaker wait endpoint-in-service ...`). At this point the VAE is reachable as a live inference endpoint that can generate new samples on request — the same pattern (model → container → registry → endpoint) generalizes directly to GANs and to Azure/GCP equivalents.

## Key Takeaways
1. Every cloud deployment of a generative model reduces to the same five steps — serialize, containerize, push to registry, create model/endpoint-config, create endpoint — regardless of provider.
2. AWS SageMaker deployment specifically requires Docker + Amazon ECR for custom inference code; always authenticate, tag, and push in that exact order before creating the SageMaker model.
3. Platform choice should follow your priorities: SageMaker (maturity/breadth), Google AI Platform (ease of use), Azure ML (service integration + responsible-AI tooling) — see Table 8.1.
4. A model is not production-ready until it clears all five deployment-readiness dimensions (compute, size, latency, accuracy, fairness), not just accuracy.
5. Model compression techniques (quantization, pruning, knowledge distillation) are the standard toolkit for reconciling large GAN/VAE architectures with latency- or resource-constrained deployment targets.
6. Treat every CLI placeholder (`<your-region>`, `<your-account-id>`, `MODEL_NAME`) as a required substitution — the commands in this chapter are templates, not literal copy-paste scripts.

## Connects To
- **Ch3/Ch7**: supplies the GAN/VAE architectures and training considerations that this chapter's deployment pipeline packages and serves.
- **Ch6**: the generative-design/business-innovation use cases from Chapter 6 are the kinds of models this chapter's deployment pipeline would put into production.
