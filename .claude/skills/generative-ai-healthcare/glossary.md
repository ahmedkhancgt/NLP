# Glossary

**Activation function** — a non-linear function applied to a neuron's weighted input sum (sigmoid, ReLU, tanh), enabling networks to learn complex relationships. (Ch2)

**AI ethics committee** — a multidisciplinary internal oversight body (ethics experts, data scientists, legal/compliance, DEI specialists, end-user representatives) that reviews AI projects for risk, fairness, and regulatory compliance. (Ch7, Ch9)

**AI Risk Management Framework (NIST)** — US framework for implementing trustworthy AI via risk mapping, governance structure, performance monitoring, and bias assessment. (Ch4)

**Artificial neuron** — the basic computational unit of a neural network; receives weighted inputs, applies a bias/activation function, and produces an output. (Ch1, Ch2)

**Backpropagation** — the algorithm that propagates prediction error backward through a network's layers to adjust weights and reduce future error. (Ch1, Ch2)

**Backpropagation Through Time (BPTT)** — the variant of backpropagation used to train RNNs on sequential data. (Ch2)

**Bias audit** — systematic evaluation of an AI model's outputs across demographic subgroups to detect and correct disparate performance. (Ch4, Ch6, Ch7)

**Blockchain (for data integrity)** — a technique for providing immutable audit trails, granular access control, and patient-controlled data sharing without requiring direct data transfer between parties. (Ch6)

**Conditional GAN** — a GAN variant used to generate realistic synthetic EHRs that preserve temporal associations between health events. (Ch4)

**Convolutional Neural Network (CNN)** — a neural network architecture specialized for grid-like/image data via convolutional, pooling, and fully connected layers; used for medical imaging and segmentation. (Ch1, Ch2)

**Counterfactual explanation** — an explainability technique showing what change to input variables would alter an AI model's output. (Ch5, Ch7)

**Data augmentation** — expanding training data (e.g., flipping/rotating images, paraphrasing text) to improve model robustness without new real-world data. (Ch1, Ch6)

**Data masking** — making sensitive data invisible to humans while remaining computer-processable, via character shuffling, substitution, encryption, or nulling. (Ch6)

**De-identification** — removing or obscuring identifiers from a dataset to protect patient privacy; includes the HIPAA Safe Harbor method, expert determination, k-anonymity, l-diversity, and t-closeness. (Ch6)

**Differential privacy** — a framework adding calibrated statistical noise so aggregate data can be released without exposing individual contributions. (Ch6)

**Digital twin (patient)** — a virtual model of an individual patient used to simulate disease progression and personalized treatment response. (Ch4, Ch8)

**Explainable AI (XAI)** — AI design and tooling (e.g., LIME, SHAP, saliency maps, counterfactuals, model cards) aimed at making model decisions interpretable to clinicians, patients, and regulators. (Ch5, Ch7, Ch8, Ch9)

**Federated learning** — training a shared AI model across multiple institutions' distributed datasets without exchanging the underlying raw data; only model updates are shared. (Ch4, Ch6, Ch8, Ch9)

**Feedforward Neural Network (FNN / MLP)** — the simplest neural network architecture, with unidirectional information flow; suited to independent, static-input prediction tasks. (Ch1, Ch2)

**Fine-tuning** — adapting a pre-trained model's parameters on a smaller, task- or domain-specific dataset. (Ch1, Ch2, Ch6)

**Fraud, Waste, and Abuse (FWA)** — the payer-industry term for healthcare cost leakage from fraudulent, wasteful, or abusive claims activity. (Ch1, Ch3)

**Generative Adversarial Network (GAN)** — a generative architecture pairing a generator (creates synthetic data) and a discriminator (judges real vs. fake) trained adversarially. (Ch1, Ch2, Ch3, Ch4, Ch5)

**HITECH Act** — US law strengthening HIPAA requirements, notably breach notification. (Ch3)

**HITRUST CSF (Common Security Framework)** — a certifiable security/privacy framework harmonizing HIPAA, GDPR, NIST, and ISO requirements for healthcare organizations. (Ch7)

**HL7 FHIR (Fast Healthcare Interoperability Resources)** — the data-exchange standard recommended for integrating generative AI with payer/provider systems and reducing data silos. (Ch3, Ch6)

**Homomorphic encryption** — a cryptographic technique enabling computation directly on encrypted data without decrypting it first. (Ch6)

**Human-in-the-Loop (HITL)** — an AI system design that keeps a human clinician empowered to review, accept, or override AI-generated recommendations. (Ch7)

**k-anonymity / l-diversity / t-closeness** — a progression of statistical de-identification guarantees ensuring individual records are indistinguishable within a group and that sensitive attributes are sufficiently diverse/representative. (Ch6)

**Knowledge graph** — a structured representation connecting healthcare data points (patients, conditions, medications, procedures) as nodes with defined relationships, enriched with medical ontologies (SNOMED CT, RxNorm). (Ch6)

**Large Language Model (LLM)** — a neural network trained on vast text data to generate new text by learning word patterns and relationships. (Ch1)

**LIME (Local Interpretable Model-Agnostic Explanations)** — an XAI technique producing interpretable, model-agnostic explanations of individual predictions. (Ch7)

**Mode collapse** — a GAN failure mode in which the generator produces a limited variety of outputs. (Ch2)

**Model card** — a standardized documentation format (originated by Google) summarizing an AI model's capabilities, limitations, and training data. (Ch7)

**Pharmacogenomics** — the study of how genetic factors influence individual drug response, used by generative AI to predict personalized medication efficacy/risk. (Ch4)

**Population Health Management (PHM)** — a payer strategy to improve health outcomes and lower costs across a member population via integrated, multi-source data analysis. (Ch1, Ch3)

**Pre-trained model** — a model trained on a large general dataset (e.g., ImageNet, large text corpora) that is reused as a starting point via transfer learning/fine-tuning. (Ch1, Ch2)

**Protected Health Information (PHI)** — individually identifiable health information regulated under HIPAA. (Ch3, Ch6)

**Recurrent Neural Network (RNN)** — a neural network architecture with loop connections that retain memory of past inputs; suited to sequential/time-series data (ECG, vitals). (Ch1, Ch2)

**Regulatory sandbox** — a controlled, regulator-supervised environment (used in UK, Singapore, Australia) for testing new AI capabilities before mass deployment. (Ch7)

**Secure Multi-Party Computation (SMPC)** — a cryptographic protocol letting multiple parties jointly compute a function over their private inputs without revealing those inputs to each other. (Ch6)

**SHAP (Shapley Additive Explanations)** — an XAI technique attributing a model's prediction to the contribution of each input feature. (Ch7)

**Synthetic data** — artificially generated data (claims, patient records, images, molecules) that mimics the statistical properties of real data without mapping to real individuals. (Ch1, Ch3, Ch4, Ch5, Ch6, Ch8)

**Tokenization** — replacing sensitive data with a non-sensitive placeholder symbol that can be mapped back to the original only via a secure lookup. (Ch6)

**Transfer learning** — reusing knowledge from a model trained on one task to improve learning on a related task, reducing data/compute requirements. (Ch1, Ch2, Ch6)

**Up-coding** — a fraudulent billing practice of charging for a more expensive service than was actually provided; a key target of generative anomaly-detection models. (Ch3)

**Variational Autoencoder (VAE)** — a generative architecture with an encoder (compresses input to a latent space) and decoder (reconstructs/generates data from that latent space); valued for interpretable latent structure. (Ch1, Ch2, Ch3, Ch4)

**Virtual patient** — a GenAI-generated synthetic patient simulation (symptoms, history, physiological responses) used for medical training, research, and engagement without real-patient exposure. (Ch5)
