# ai-news-classifier/ai_keywords.py

# --- Define Core Phrases for Higher Weighting ---
# These are unambiguous terms that should strongly indicate AI relevance
CORE_PHRASES_FOR_HIGH_WEIGHT = {
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "neural net",
    "computer vision", "natural language processing", "nlp",
    "natural language generation", "nlg", # Add NLG base term
    "generative ai", "large language model", "llm",
    "reinforcement learning", "supervised learning", "unsupervised learning",
    "gradient descent", # Crucial term often appearing alone
    "backpropagation",
    "transformer", # Core architecture term
    "transformer model",
    "machine translation", # Specific application
    "facial recognition",
    "speech recognition",
    "ai engine",
    "ai-powered robot", # Specific term combo
    "ai-powered robots" # Plural
}

# --- Keywords for PhraseMatcher (Case-Insensitive Exact Matches) ---

# Core Concepts & Foundational Terms
CORE_CONCEPTS = [
    "ai", "artificial intelligence", "machine learning", "ml",
    "deep learning", "neural network", "neural net",
    "data science", "cognitive computing", "expert system",
    "knowledge representation", "pattern recognition", "generative ai",
    "artificial general intelligence", "agi",
    "large language model", "large language models", "llm",
]

# Specific Techniques, Algorithms & Architectures
TECHNIQUES_CONCEPTS = [
    "algorithm", "machine learning algorithm", "machine learning algorithms",
    "transformer", "transformer model", "attention mechanism",
    "generative adversarial network", "gan", "cnn", "rnn", "lstm", "gru",
    "convolutional neural network", "recurrent neural network",
    "supervised learning", "unsupervised learning", "reinforcement learning",
    "self-supervised learning", "semi-supervised learning",
    "clustering", "classification",
    "dimensionality reduction",
    "feature extraction", "feature engineering",
    "backpropagation", "gradient descent", "stochastic gradient descent",
    "loss function", "cost function", "activation function",
    "hyperparameter", "hyperparameter tuning", "optimization",
    "optimization algorithm", "optimization algorithms",
    "overfitting", "underfitting", "regularization", "bias variance tradeoff",
    "decision boundary", "support vector machine", "svm",
    "random forest", "decision tree", "k-means", "dbscan",
    "principal component analysis", "pca",
    "transfer learning", "fine-tuning", "pre-training", "unsupervised pretraining",
    "prompt engineering", "retrieval-augmented generation", "rag",
    "vector database", "vector embedding", "word embedding", "embedding",
    "neurosymbolic", "fuzzy logic", "bayesian network", "markov chain",
    "federated learning", "explainable ai", "xai", "ai ethics", "ai safety",
    "model compression", "quantization",
    "quantum machine learning",
]

# AI Application Areas & Tasks
APPLICATION_AREAS = [
    "computer vision", "cv", "natural language processing", "nlp",
    "natural language understanding", "nlu", "natural language generation", "nlg",
    "text generation",
    "speech recognition", "speech synthesis", "text-to-speech", "tts",
    "object detection", "image recognition", "image segmentation", "semantic segmentation",
    "facial recognition", "video analysis",
    "sentiment analysis", "topic modeling", "machine translation",
    "recommendation system", "recommender system", "collaborative filtering",
    "predictive analytics", "predictive modeling", "forecasting", "time series analysis",
    "anomaly detection", "fraud detection",
    "robotics", "autonomous", "autonomous vehicle", "autonomous vehicles", "self-driving car",
    "autonomous drone", "swarm intelligence",
    "chatbot", "virtual assistant", "conversational ai",
    "drug discovery", "computational biology",
    "ai engine", "ai-powered robot", "ai-powered robots", # Added specific terms
]

# Common AI Data Terms
DATA_TERMS = [
    "big data", "dataset", "data set", "training data", "validation data", "test data",
    "labeled data", "unlabeled data", "data augmentation", "data preprocessing",
    "corpus", "corpora", "knowledge graph",
]

# Software, Libraries, Frameworks & Hardware
TOOLS_LIBRARIES_HARDWARE = [
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "opencv", "spacy", "nltk", "hugging face", "transformers", # library
    "jupyter", "cuda", "gpu", "tpu", "tensor processing unit",
    "onnx", "mlflow", "kubeflow",
]

# Specific Models & Platforms (Examples - add more as needed)
SPECIFIC_MODELS_PLATFORMS = [
    "gpt-3", "gpt-4", "chatgpt", "dall-e", "stable diffusion", "midjourney",
    "bert", "roberta", "xlnet", "t5", "bart",
    "llama", "claude", "gemini", "mistral", # Foundational models
    "alexnet", "resnet", "vgg", "yolo", "efficientnet", # CV models
    "alphago", "alphafold", "alphazero", # DeepMind specific
    "watson", # IBM
    "openai api", "google ai platform", "azure machine learning", "aws sagemaker",
]

# Companies & Research Labs (Prominent ones)
COMPANIES_RESEARCH = [
    "openai", "google", "deepmind", "meta ai", "facebook ai", "fair", # facebook ai research
    "microsoft", "amazon", "aws", "nvidia", "intel", "amd",
    "ibm", "apple", "tesla", "waymo", "cruise",
    "anthropic", "cohere", "ai21 labs", "mistral ai", "stability ai",
    "element ai", "baidu", "tencent", "alibaba", "huawei",
    "sensetime", "megvii",
    "boston dynamics",
    "allen institute for ai", "ai2", "mila", "vector institute",
]

# --- Keywords/Lists for Contextual Matcher Patterns ---

# Nouns often associated with AI actions/processes
AI_NOUNS_CONTEXT = [
    "model", "network", "agent", "system", "algorithm", "algo",
    "classifier", "regressor", "clusterer", "detector", "recognizer",
    "pipeline", "architecture", "framework", "platform",
    "data", "dataset", "parameters", "weights", "hyperparameters",
    "layer", "node", "neuron", "embedding", "vector", "feature",
    "prediction", "forecast", "output", "decision", "recommendation",
    "agent", "robot", "chatbot", "assistant", "network"
]

# Adjectives often describing AI entities or processes
AI_ADJECTIVES_CONTEXT = [
    "artificial", "intelligent", "machine", "deep", "neural",
    "learning",
    "supervised", "unsupervised", "reinforcement", "generative",
    "predictive", "descriptive", "prescriptive",
    "automated", "automatic", "autonomous",
    "cognitive", "semantic", "syntactic", "linguistic",
    "convolutional", "recurrent",
    "computational", "quantum",
    "ai-powered", "ai powered", # Added no-hyphen version
]

# Verbs that are strong indicators WHEN USED WITH AI_NOUNS_CONTEXT
CONTEXT_VERBS = [
    "train", "fine-tune", "deploy", "optimize", "tune", "calibrate",
    "predict", "forecast", "classify", "cluster", "segment", "detect",
    "recognize", "identify", "generate", "synthesize",
    "analyze", "process", "interpret", "understand", "learn",
    "embed", "vectorize", "compute", "calculate", "evaluate",
    "validate", "test", "benchmark", "simulate", "model", # verb 'to model'
    "infer", "reason", "decide", "recommend", "automate", "control",
    "adapt", "evolve", "build",
]

# Verbs that are *highly* specific to AI/ML and might be sufficient alone
HIGH_CONFIDENCE_VERBS = [
    "backpropagate", "regularize", "tokenize", "lemmatize", "vectorize",
]

# Combine phrase lists for PhraseMatcher - ensure no duplicates
# Include the core phrases in the main list as well for matching
ALL_PHRASE_KEYWORDS_LIST = list(CORE_PHRASES_FOR_HIGH_WEIGHT) + ( # Ensure core phrases are included
    CORE_CONCEPTS + TECHNIQUES_CONCEPTS + APPLICATION_AREAS + DATA_TERMS +
    TOOLS_LIBRARIES_HARDWARE + SPECIFIC_MODELS_PLATFORMS + COMPANIES_RESEARCH
)
# Convert to set to remove duplicates, then back to list
ALL_PHRASE_KEYWORDS = sorted(list(set(kw.lower() for kw in ALL_PHRASE_KEYWORDS_LIST if kw)))