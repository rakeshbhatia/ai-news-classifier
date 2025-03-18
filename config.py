# config.py
# Configuration file for AI World Press API

# RSS feeds configuration
RSS_FEEDS = {
    "Mainstream News": [
        {
            "name": "Bloomberg",
            "url": "https://feeds.bloomberg.com/technology/news.rss",
            "logo": "/images/logos/bloomberg.png",
            "category": "Mainstream News"
        },
        {
            "name": "New York Times",
            "url": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
            "logo": "/images/logos/nytimes.png",
            "category": "Mainstream News"
        },
        {
            "name": "Forbes",
            "url": "https://www.forbes.com/innovation/feed2",
            "logo": "/images/logos/forbes.png",
            "category": "Mainstream News"
        },
        {
            "name": "CNBC",
            "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910",
            "logo": "/images/logos/cnbc.png",
            "category": "Mainstream News"
        },
        {
            "name": "BBC",
            "url": "https://feeds.bbci.co.uk/news/technology/rss.xml?edition=uk",
            "logo": "/images/logos/bbc.png",
            "category": "Mainstream News"
        },
        {
            "name": "The Guardian",
            "url": "https://www.theguardian.com/technology/rss",
            "logo": "/images/logos/guardian.png",
            "category": "Mainstream News"
        }
    ],
    "Tech News": [
        {
            "name": "Wired",
            "url": "https://www.wired.com/feed/tag/ai/latest/rss",
            "logo": "/images/logos/wired.png",
            "category": "Tech News"
        },
        {
            "name": "Ars Technica",
            "url": "https://feeds.arstechnica.com/arstechnica/index",
            "logo": "/images/logos/ars-technica.png",
            "category": "Tech News"
        },
        {
            "name": "The Verge",
            "url": "https://www.theverge.com/tech/rss/index.xml",
            "logo": "/images/logos/the-verge.png",
            "category": "Tech News"
        },
        {
            "name": "ZDNet",
            "url": "https://www.zdnet.com/topic/artificial-intelligence/rss.xml",
            "logo": "/images/logos/zdnet.png",
            "category": "Tech News"
        },
        {
            "name": "TechCrunch",
            "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
            "logo": "/images/logos/techcrunch.png",
            "category": "Tech News"
        },
        {
            "name": "VentureBeat",
            "url": "https://venturebeat.com/category/ai/feed/",
            "logo": "/images/logos/venturebeat.png",
            "category": "Tech News"
        }
    ],
    "Alternative News": [
        {
            "name": "Zero Hedge",
            "url": "https://cms.zerohedge.com/fullrss2.xml",
            "logo": "/images/logos/zerohedge.png",
            "category": "Alternative News"
        },
        {
            "name": "The Intercept",
            "url": "https://theintercept.com/feed/?lang=en",
            "logo": "/images/logos/intercept.png",
            "category": "Alternative News"
        }
    ]
}

# AI-related keywords
AI_GENERAL_KEYWORDS = [
    # Core AI terms
    "artificial intelligence", "ai ", "machine intelligence", "computational intelligence",
    "ai technology", "ai system", "ai model", "ai tool", "ai application", "ai solution",
    
    # Machine learning
    "machine learning", "ml", "deep learning", "neural network", "supervised learning",
    "unsupervised learning", "reinforcement learning", "training data", "model training",
    "prediction model", "classification", "regression", "clustering", "ai training",
    
    # Language models
    "language model", "llm", "large language model", "nlp", "natural language processing",
    "chatbot", "chat bot", "gpt", "chatgpt", "text generation", "speech recognition",
    "text-to-speech", "language understanding", "prompt engineering", "language ai",
    "transformer model", "bert", "attention mechanism",
    
    # Computer vision
    "computer vision", "image recognition", "object detection", "facial recognition",
    "image processing", "visual ai", "image classification", "video analysis",
    "scene understanding", "image segmentation", "visual recognition", "image generation",
    
    # Robotics & Automation
    "robotics", "robot", "autonomous system", "automation", "autonomous vehicle",
    "self-driving", "drone", "robotic process", "industrial robot", "robot learning",
    "humanoid", "automated system",
    
    # AI companies & products
    "openai", "anthropic", "deepmind", "claude", "chatgpt", "gemini", "bard",
    "stable diffusion", "midjourney", "dall-e", "gpt-4", "gpt-3", "hugging face",
    "llama", "mistral", "meta ai", "microsoft ai", "google ai", "nvidia ai",
    
    # AI applications
    "ai assistant", "virtual assistant", "recommendation system", "decision support",
    "predictive analytics", "ai-powered", "generative ai", "foundation model",
    "ai algorithm", "ai agent", "multimodal ai", "synthetic data",
    
    # AI ethics & impact
    "ai ethics", "ai bias", "ai safety", "responsible ai", "ai regulation",
    "ai policy", "ai governance", "ai impact", "ai risk", "algorithmic bias",
    "ai transparency", "ethical ai", "ai privacy", "ai alignment"
]

# Policy & Regulation keywords
POLICY_REGULATION_KEYWORDS = [
    "policy", "regulation", "law", "compliance", "government", "legislation",
    "congress", "senate", "governance", "oversight", "legal", "regulatory",
    "regulator", "agency", "commission", "framework", "rule", "guideline",
    "standard", "restriction", "ban", "moratorium", "sanction", "fine", "penalty",
    "enforcement", "audit", "investigation", "hearing", "testimony", "subpoena",
    "lawsuit", "litigation", "court", "judge", "ruling", "verdict", "settlement",
    "consent decree", "executive order", "bill", "act", "statute", "constitution",
    "compliance", "non-compliance", "violation", "infringement", "oversight",
    "accountability", "transparency", "disclosure", "reporting", "certification"
]

# Research & Development keywords
RESEARCH_DEVELOPMENT_KEYWORDS = [
    "research", "study", "discovery", "breakthrough", "innovation", "development",
    "scientific", "experiment", "academic", "ai model", "neural network", "publication",
    "paper", "journal", "conference", "symposium", "workshop", "laboratory", "lab",
    "university", "institute", "fellowship", "grant", "funding", "peer review",
    "methodology", "hypothesis", "theory", "concept", "algorithm", "architecture",
    "prototype", "proof of concept", "poc", "mvp", "r&d", "advancement", "cutting edge",
    "state of the art", "sota", "benchmark", "baseline", "improvement", "enhancement",
    "optimization", "parameter", "hyperparameter", "fine-tuning", "training", "testing",
    "validation", "evaluation", "performance", "efficiency", "accuracy", "precision"
]

# Ethics & Social Impact keywords
ETHICS_SOCIAL_IMPACT_KEYWORDS = [
    "ethics", "privacy", "bias", "fairness", "transparency", "responsible", "safety",
    "risk", "human rights", "discrimination", "inequality", "society", "job displacement",
    "unemployment", "workforce", "labor", "worker", "automation", "social impact",
    "societal impact", "community", "public", "citizen", "civic", "civil rights",
    "civil liberties", "freedom", "justice", "equity", "equality", "diversity",
    "inclusion", "marginalized", "vulnerable", "underrepresented", "minority",
    "majority", "privilege", "disadvantage", "harm", "benefit", "wellbeing", "welfare",
    "health", "mental health", "psychological", "emotional", "addiction", "dependency",
    "trust", "mistrust", "skepticism", "criticism", "backlash", "protest", "activism"
]

# Defense & Security keywords
DEFENSE_SECURITY_KEYWORDS = [
    "military", "defense", "weapon", "warfare", "security", "cybersecurity",
    "intelligence", "espionage", "surveillance", "threat", "national security",
    "armed forces", "army", "navy", "air force", "marines", "Pentagon", "NATO",
    "alliance", "enemy", "adversary", "combat", "conflict", "war", "battle",
    "attack", "offensive", "defensive", "strategy", "tactical", "operation",
    "mission", "command", "control", "intelligence", "reconnaissance", "surveillance",
    "target", "missile", "drone", "UAV", "unmanned", "autonomous weapon",
    "cyber attack", "cyber defense", "cyber warfare", "hack", "breach", "vulnerability",
    "exploit", "malware", "ransomware", "threat actor", "APT", "state-sponsored",
    "terrorism", "terrorist", "counter-terrorism", "homeland security", "border security"
]

# Technology & Innovation keywords
TECHNOLOGY_INNOVATION_KEYWORDS = [
    "technology", "hardware", "software", "ai chip", "cloud", "robotics", "computing",
    "infrastructure", "automation", "iot", "5g", "future tech", "innovation",
    "disruption", "transformation", "digital", "platform", "app", "application",
    "system", "network", "internet", "web", "mobile", "wireless", "connectivity",
    "device", "gadget", "processor", "GPU", "TPU", "ASIC", "semiconductor", "chip",
    "memory", "storage", "server", "data center", "edge computing", "quantum computing",
    "blockchain", "distributed ledger", "API", "interface", "UI", "UX", "frontend",
    "backend", "full-stack", "programming", "coding", "developer", "engineer",
    "architecture", "microservice", "container", "virtual", "augmented reality", "AR",
    "virtual reality", "VR", "mixed reality", "XR", "metaverse", "digital twin"
]

# Healthcare & Biotech keywords
HEALTHCARE_BIOTECH_KEYWORDS = [
    "healthcare", "medicine", "biotech", "drug discovery", "hospital", "doctor",
    "genomics", "diagnostics", "pharmaceutical", "medical ai", "patient", "clinical",
    "treatment", "therapy", "drug", "vaccine", "disease", "illness", "condition",
    "symptom", "diagnosis", "prognosis", "radiology", "imaging", "x-ray", "MRI",
    "CT scan", "ultrasound", "pathology", "lab test", "biomarker", "DNA", "RNA",
    "gene", "genetic", "genome", "protein", "cell", "tissue", "organ", "biology",
    "medical device", "medical equipment", "telemedicine", "telehealth", "remote care",
    "digital health", "wearable", "implant", "prosthetic", "clinical trial",
    "FDA approval", "regulatory approval", "health record", "EHR", "EMR", "health data",
    "wellness", "preventive", "personalized medicine", "precision medicine"
]

# Entertainment & Media keywords
ENTERTAINMENT_MEDIA_KEYWORDS = [
    "entertainment", "movie", "music", "art", "creative", "media", "game", "hollywood",
    "film", "streaming", "ai-generated content", "television", "TV", "show", "series",
    "episode", "broadcast", "network", "channel", "studio", "production", "producer",
    "director", "actor", "actress", "star", "celebrity", "artist", "musician", "band",
    "record", "album", "song", "track", "concert", "performance", "tour", "festival",
    "award", "Grammy", "Oscar", "Emmy", "Golden Globe", "critic", "review", "rating",
    "audience", "viewer", "listener", "fan", "follower", "subscriber", "content",
    "content creator", "influencer", "social media", "platform", "YouTube", "TikTok",
    "Instagram", "Twitter", "Facebook", "Snapchat", "streaming service", "Netflix",
    "Disney+", "Hulu", "Amazon Prime", "HBO Max", "Apple TV+", "gaming", "video game",
    "console", "PC gaming", "mobile gaming", "esports", "virtual world", "publisher"
]

# Types of entities that can be key players
KEY_PLAYER_TYPES = ["ORG", "PERSON", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT"]

# Business & Finance keywords
BUSINESS_FINANCE_KEYWORDS = [
    "business", "company", "industry", "corporate", "enterprise", "startup",
    "stock", "investor", "market", "nasdaq", "investment", "shares", "trading",
    "revenue", "valuation", "venture capital", "vc", "fund", "funding", "series a",
    "series b", "ipo", "acquisition", "merger", "profit", "loss", "earnings",
    "quarterly", "fiscal", "ceo", "cfo", "executive", "board", "shareholder",
    "stakeholder", "portfolio", "asset", "liability", "equity", "debt", "loan",
    "credit", "finance", "financial", "commercial", "monetize", "monetization",
    "business model", "subscription", "customer", "client", "product", "service",
    "market share", "competitive", "industry", "sector", "economy", "economic",
    "recession", "inflation", "growth", "scale", "scaling", "unicorn", "exit strategy"
]