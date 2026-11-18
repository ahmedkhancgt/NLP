# Glossary

**Activation function** — introduces non-linearity into a neuron's output, enabling networks to model complex patterns (Ch1).
**Algorithm** — rules/instructions that let an AI program learn and decide (Ch1, Ch2, Ch4, Ch5).
**Anomaly detection** — identifying data points/patterns that deviate from a learned "normal" baseline (Ch2, Ch3).
**Artificial intelligence (AI)** — simulation of human intelligence (learning, reasoning, self-correction) by machines (all chapters).
**Attention mechanism** — lets a model selectively weight different parts of input when predicting (Ch4).
**Autoencoder** — neural network that encodes input into a compressed representation then decodes/reconstructs it (Ch3, Ch5).
**Autoregressive generation** — predicting each next token/element conditioned only on previous ones (GPT-style) (Ch4, Ch6).
**Backpropagation** — algorithm computing loss gradients w.r.t. network weights for training (Ch1, Ch2, Ch4).
**Beam search / top-k / top-p / temperature** — decoding parameters controlling generation coherence, diversity, repetition (Ch4, Ch6).
**BERT** — bidirectional Transformer pre-trained on unlabeled text, fine-tuned for understanding tasks (Ch4).
**Bias (in AI)** — systematic error from nonrepresentative training data; also a neuron's firing offset (Ch1, Ch7, Ch9).
**CLIP** — Contrastive Language-Image Pre-training; jointly embeds text and images for multimodal tasks (Ch6).
**CNN (Convolutional Neural Network)** — network using convolutional filters to detect spatial patterns in grid data (images) (Ch1, Ch2, Ch5).
**Content vs. style** — content = what is depicted; style = how it's depicted (basis of style transfer) (Ch2, Ch5).
**Data augmentation** — artificially expanding dataset diversity via transformations or synonym replacement to reduce overfitting (Ch2, Ch3, Ch5).
**Deep learning (DL)** — ML subset using layered neural networks for representation learning (Ch1, Ch2, Ch4, Ch6-8).
**Deep Reinforcement Learning (DRL)** — RL combined with deep neural networks for high-dimensional state spaces (Ch8).
**Deepfake** — realistic synthetic video/audio/image replacing a person's likeness, raising misinformation/privacy risk (Ch1, Ch2, Ch5, Ch7).
**Discriminator** — GAN network that classifies input as real or generator-produced fake (Ch2, Ch5, Ch7).
**Encoder / Decoder** — encoder compresses data to a latent representation; decoder reconstructs/generates from it (Ch3, Ch5).
**Ensemble learning** — combining multiple models' predictions to improve robustness/accuracy or balance speed vs. accuracy (Ch8, Ch10).
**Ethical AI** — ensuring AI systems are fair, transparent, accountable, and privacy-respecting (Ch7-9).
**Exploration vs. exploitation** — RL trade-off between trying new actions and using known-good ones (Ch8).
**Feedback loop** — system where outputs feed back as inputs; central to RL and adaptive systems (Ch1, Ch8).
**Fine-tuning** — continuing training of a pre-trained model to adapt it to a specific task/dataset (Ch2, Ch4, Ch6, Ch7, Ch10).
**GAN (Generative Adversarial Network)** — generator and discriminator trained adversarially to produce realistic synthetic data (Ch1, Ch2, Ch5, Ch7, Ch8).
**Generative AI** — AI that creates new content (text, images, music) resembling but distinct from training data (all chapters).
**Generator** — GAN network that transforms noise into synthetic data meant to fool the discriminator (Ch2, Ch5).
**GPT (Generative Pre-trained Transformer)** — autoregressive Transformer for open-ended text generation (Ch4, Ch6, Ch7).
**Gram matrix** — feature-correlation matrix used to represent image "style" independent of spatial content (Ch2, Ch5).
**Hidden layer** — layer between input/output where neurons extract features via weighted computation (Ch1, Ch4, Ch6).
**Hyperparameter tuning** — optimizing settings (learning rate, batch size, latent dim) not learned from data (Ch2, Ch5, Ch10).
**Inference** — using a trained model to make predictions on new, unseen data (Ch2, Ch4, Ch6, Ch10).
**Internet of Things (IoT)** — networked sensor/device systems generating data for AI-driven monitoring (Ch9, Ch10).
**KL divergence (Kullback-Leibler)** — measures how one probability distribution diverges from another; regularizes VAE latent space (Ch3).
**Latent space** — compressed, continuous representation space where VAEs/GANs encode and sample data features (Ch1, Ch3, Ch5).
**Loss function** — quantifies difference between predicted and actual values, guiding training updates (all technical chapters).
**LSTM (Long Short-Term Memory)** — RNN variant with forget/input/output gates for long-range sequence dependencies (Ch1, Ch4).
**Markov Decision Process (MDP)** — RL formalism: state, action, reward, transition probabilities (Ch1, Ch8).
**Mode collapse** — a GAN/VAE generator producing limited output diversity, failing to represent the full data distribution (Ch2, Ch3, Ch5, Ch8).
**Model drift** — deployed model accuracy degrading as real-world data distributions shift from training data (Ch10).
**Multimodal learning** — modeling relationships across multiple data types (text, image, audio) jointly (Ch6, Ch9).
**Neural network** — layered computational model of interconnected artificial neurons for learning patterns (all chapters).
**Overfitting** — model performs well on training data but fails to generalize to unseen data (Ch3, Ch10).
**Perceptron** — simplest neural network unit: weighted sum of inputs through an activation function (Ch1).
**Policy (RL)** — strategy mapping states to actions that an RL agent follows (Ch8).
**Policy gradient methods** — RL technique directly optimizing a stochastic action-selection policy (Ch8).
**Positional encoding** — injects sequence-order information into Transformer embeddings (self-attention is order-blind otherwise) (Ch4).
**Quantization** — reducing model weight precision (e.g., 32-bit to 8-bit) to speed inference (Ch10).
**Reinforcement Learning (RL)** — agent learns via trial-and-error reward/penalty feedback from an environment (Ch1, Ch8).
**Reparameterization trick** — `z = mean + exp(0.5*logvar) * epsilon`; makes VAE latent sampling differentiable (Ch3, Ch5).
**ResNet (Residual Network)** — deep network with skip connections addressing vanishing/exploding gradients (Ch1).
**RNN (Recurrent Neural Network)** — network maintaining hidden state across sequence timesteps (Ch1, Ch4).
**Self-attention** — mechanism where sequence elements attend to all others to capture dependencies (Ch1, Ch4).
**Style transfer** — applying one image's artistic style to another image's content via neural networks (Ch2, Ch5, Ch7).
**StyleGAN / StyleGAN2** — NVIDIA GAN variants enabling fine latent-space control over generated image attributes (Ch5, Ch7).
**Supervised / Unsupervised / Semi-supervised learning** — ML paradigms using labeled, unlabeled, or mixed data respectively (Ch1).
**Text generation** — NLP task producing coherent, contextually relevant text (Ch4, Ch6, Ch7).
**Tokenization** — splitting text into smaller units (tokens) for model processing (Ch4, Ch6).
**Transfer learning** — reusing knowledge from a pre-trained model to speed up learning on a related task (Ch4, Ch7, Ch10).
**Transformer** — self-attention-based architecture ("Attention Is All You Need") underlying BERT/GPT (Ch1, Ch4, Ch6).
**VAE (Variational Autoencoder)** — generative model combining autoencoders with probabilistic latent-space modeling (Ch1, Ch3, Ch5, Ch8).
**Variational inference** — learning an approximate probability distribution over a model's latent space (Ch3).
**Word embedding** — dense vector representation of words capturing semantic relationships (Ch4, Ch6).
**Zero-shot learning** — recognizing classes never seen during training, via transferred knowledge (Ch4, Ch6).
