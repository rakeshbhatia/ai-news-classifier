# ai_news_classifier/app/config.py

import os
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application configuration settings."""

    # --- Database Settings ---
    # SUPABASE / POSTGRESQL connection string
    # Example: postgresql+asyncpg://user:password@host:port/dbname
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres.vkhbwommicblghygcots:0EwokbBi2K85639@aws-0-us-west-1.pooler.supabase.com:6543/postgres")

    CSV_OUTPUT_FILENAME: Optional[str] = os.getenv("CSV_OUTPUT_FILENAME", "rss_feed_data.csv")

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
                {
                    "name": "Forbes",
                    "logo": "/images/logos/forbes.png",
                    "urls": ["https://www.forbes.com/innovation/feed2"]
                },
                {
                    "name": "CNBC",
                    "logo": "/images/logos/cnbc.png",
                    "urls": ["https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910"]
                },
                {
                    "name": "Fox News",
                    "logo": "/images/logos/foxnews.png",
                    "urls": ["https://moxie.foxnews.com/google-publisher/tech.xml"]
                },
                {
                    "name": "BBC",
                    "logo": "/images/logos/bbc.png",
                    "urls": ["https://feeds.bbci.co.uk/news/technology/rss.xml?edition=uk"]
                }
            ]
        },
        {
            "name": "Independent & Alternative News",
            "sources": [
                {
                    "name": "ZeroHedge",
                    "logo": "/images/logos/zerohedge.png",
                    "urls": ["https://cms.zerohedge.com/fullrss2.xml"]
                },
                {
                    "name": "The Intercept",
                    "logo": "/images/logos/intercept.png",
                    "urls": ["https://theintercept.com/feed/?lang=en"]
                },
                {
                    "name": 'Mother Jones',
                    "logo": '/images/logos/mother_jones.png',
                    "urls": ['http://feeds.feedburner.com/motherjones/feed']
                }
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
                {
                    "name": 'The Verge',
                    "logo": '/images/logos/the-verge.png',
                    "urls": ['https://www.theverge.com/rss/index.xml']
                }
            ]
        }
    ]

    # --- Scheduler Settings ---
    RSS_FETCH_INTERVAL_MINUTES: int = int(os.getenv("RSS_FETCH_INTERVAL_MINUTES", 9999))
    CLASSIFICATION_INTERVAL_MINUTES: int = int(os.getenv("CLASSIFICATION_INTERVAL_MINUTES", 9999))

    # --- AI Classifier Settings ---
    CLASSIFICATION_BATCH_SIZE: int = int(os.getenv("CLASSIFICATION_BATCH_SIZE", 10000))
    classifier_threshold: float = float(os.getenv("CLASSIFIER_THRESHOLD", 3.0))
    spacy_model_name: str = os.getenv("SPACY_MODEL_NAME", "en_core_web_md")

    # --- SQLAlchemy Settings (Optional) ---
    # Set to True to echo SQL statements (useful for debugging)
    DB_ECHO_SQL: bool = bool(os.getenv("DB_ECHO_SQL", False))

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings()

# Example usage (optional, for testing):
if __name__ == "__main__":
    print("Loaded Settings:")
    print(f"DATABASE_URL: {settings.DATABASE_URL}") # Check partially masked URL
    print(f"SCHEDULER - RSS Interval (min): {settings.RSS_FETCH_INTERVAL_MINUTES}")
    print(f"CLASSIFIER - spaCy Model: {settings.spacy_model_name}")
    print(f"DB_ECHO_SQL: {settings.DB_ECHO_SQL}")