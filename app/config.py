# ai_news_classifier/app/config.py

import os
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """
    Application configuration settings.
    Reads settings from environment variables or uses defaults.
    """
    # --- MongoDB Settings ---
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "ai_news_db")
    MONGO_ARTICLES_COLLECTION: str = os.getenv("MONGO_ARTICLES_COLLECTION", "articles")

    # --- RSS Feed Configuration ---
    # Structure defined here, actual data might come from .env if complex,
    # but often kept here or loaded from another file/DB if very large.
    # Ensure this matches the structure expected by flatten_feed_config.
    rss_feeds: List[Dict[str, Any]] = [
        {
            "name": "Mainstream News",
            "sources": [
                {
                    "name": "Bloomberg",
                    "logo": "/images/logos/bloomberg.png",
                    "urls": ["https://feeds.bloomberg.com/technology/news.rss"]
                },
                {
                    "name": "New York Times",
                    "logo": "/images/logos/nytimes.png",
                    "urls": ["https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"]
                },
            ]
        },
        {
            "name": "Independent & Alternative News",
            "sources": [
                {
                    "name": "Zero Hedge",
                    "logo": "/images/logos/zerohedge.png",
                    "urls": ["https://cms.zerohedge.com/fullrss2.xml"]
                },
                {
                    "name": "Intercept",
                    "logo": "/images/logos/intercept.png",
                    "urls": ["https://theintercept.com/feed/?lang=en"]
                },
            ]
        },
        {
            "name": "Tech News",
            "sources": [
                {
                    "name": "Wired",
                    "logo": "/images/logos/wired.png",
                    "urls": ["https://www.wired.com/feed/tag/ai/latest/rss"]
                },
                {
                    "name": "Ars Technica",
                    "logo": "/images/logos/ars-technica.png",
                    "urls": ["https://feeds.arstechnica.com/arstechnica/index"]
                },
            ]
        }
    ]

    # --- Scheduler Settings ---
    RSS_FETCH_INTERVAL_MINUTES: int = int(os.getenv("RSS_FETCH_INTERVAL_MINUTES", 30))
    CLASSIFICATION_INTERVAL_MINUTES: int = int(os.getenv("CLASSIFICATION_INTERVAL_MINUTES", 60))

    # --- AI Classifier Settings ---
    CLASSIFICATION_BATCH_SIZE: int = int(os.getenv("CLASSIFICATION_BATCH_SIZE", 100))
    # Default threshold if not set in environment
    classifier_threshold: float = float(os.getenv("CLASSIFIER_THRESHOLD", 3.0))
    # spaCy model name - use environment variable or default
    spacy_model_name: str = os.getenv("SPACY_MODEL_NAME", "en_core_web_lg") # Example: Large model

    class Config:
        # If rss_feeds were complex JSON string in .env, you might need:
        # json_loads = json.loads
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra fields from environment


# Create a single instance of the settings to be imported elsewhere
settings = Settings()

# Example usage (optional, for testing):
if __name__ == "__main__":
    print("Loaded Settings:")
    print(f"MONGO_URI: {settings.MONGO_URI}")
    print(f"MONGO_DB_NAME: {settings.MONGO_DB_NAME}")
    print(f"SCHEDULER - RSS Interval (min): {settings.RSS_FETCH_INTERVAL_MINUTES}")
    print(f"SCHEDULER - Classification Interval (min): {settings.CLASSIFICATION_INTERVAL_MINUTES}")
    print(f"CLASSIFIER - Batch Size: {settings.CLASSIFICATION_BATCH_SIZE}")
    print(f"CLASSIFIER - Threshold: {settings.classifier_threshold}")
    print(f"CLASSIFIER - spaCy Model: {settings.spacy_model_name}")
    print(f"Number of Feed Categories defined: {len(settings.rss_feeds)}")



'''# ai_news_classifier/app/config.py

import os
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import Field

# Load environment variables from a .env file if it exists
# Useful for local development
load_dotenv()

class Settings(BaseSettings):
    """
    Application configuration settings.
    Reads settings from environment variables.
    """
    # MongoDB Settings
    MONGO_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/") # Default for local dev
    MONGO_DB_NAME: str = os.getenv("DB_NAME", "ai_world_press_test")
    MONGO_ARTICLES_COLLECTION: str = os.getenv("COLLECTION_NAME", "articles_test")

    # Optional: Add other settings as needed
    # e.g., API keys, logging levels etc.
    # LOG_LEVEL: str = "INFO"

    RSS_FETCH_INTERVAL_MINUTES: int = 30 # Fetch every 30 minutes
    CLASSIFICATION_INTERVAL_MINUTES: int = 60 # Classify every 60 minutes

    # RSS feed configuration (matches Next.js structure)
    rss_feeds: List[Dict[str, Any]] = [
        {
            "name": "Mainstream News",
            "sources": [
                {
                    "name": "Bloomberg",
                    "logo": "/images/logos/bloomberg.png",
                    "urls": ["https://feeds.bloomberg.com/technology/news.rss"]
                },
                {
                    "name": "New York Times",
                    "logo": "/images/logos/nytimes.png",
                    "urls": ["https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"]
                },
            ]
        },
        {
            "name": "Independent & Alternative News",
            "sources": [
                {
                    "name": "Zero Hedge",
                    "logo": "/images/logos/zerohedge.png",
                    "urls": ["https://cms.zerohedge.com/fullrss2.xml"]
                },
                {
                    "name": "Intercept",
                    "logo": "/images/logos/intercept.png",
                    "urls": ["https://theintercept.com/feed/?lang=en"]
                },
            ]
        },
        {
            "name": "Tech News",
            "sources": [
                {
                    "name": "Wired",
                    "logo": "/images/logos/wired.png",
                    "urls": ["https://www.wired.com/feed/tag/ai/latest/rss"]
                },
                {
                    "name": "Ars Technica",
                    "logo": "/images/logos/ars-technica.png",
                    "urls": ["https://feeds.arstechnica.com/arstechnica/index"]
                },
            ]
        }
    ]

    # Classifier Settings
    spacy_model_name: str = Field("en_core_web_md", validation_alias='SPACY_MODEL_NAME')
    classifier_threshold: float = Field(1.9, validation_alias='CLASSIFIER_THRESHOLD') # The tuned threshold

    # Optional: Logging level
    log_level: str = Field("INFO", validation_alias='LOG_LEVEL')

    class Config:
        # Optional: Specify a .env file explicitly if not using load_dotenv()
        # env_file = ".env"
        # env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra fields from environment

# Create a single instance of the settings to be imported elsewhere
settings = Settings()

# Example usage (optional, for testing):
if __name__ == "__main__":
    print("Loaded Settings:")
    print(f"MONGO_URI: {settings.MONGO_URI}")
    print(f"MONGO_DB_NAME: {settings.MONGO_DB_NAME}")
    print(f"MONGO_ARTICLES_COLLECTION: {settings.MONGO_ARTICLES_COLLECTION}")
    print(f"SpaCy Model: {settings.spacy_model_name}")
    print(f"Classifier Threshold: {settings.classifier_threshold}")
    print(f"MongoDB URI Loaded: {'Yes' if settings.MONGO_URI else 'No'}")
    # Simple test to check if MONGO_URI seems valid (basic check)
    if not settings.MONGO_URI.startswith("mongodb://") and not settings.MONGO_URI.startswith("mongodb+srv://"):
        print("Warning: MONGO_URI might not be in the correct format.")
    print("\nConfig loaded successfully.")'''