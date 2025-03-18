import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Union

import feedparser
import pymongo
import spacy
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from config import (AI_GENERAL_KEYWORDS, BUSINESS_FINANCE_KEYWORDS,
                   DEFENSE_SECURITY_KEYWORDS, ENTERTAINMENT_MEDIA_KEYWORDS,
                   ETHICS_SOCIAL_IMPACT_KEYWORDS, HEALTHCARE_BIOTECH_KEYWORDS,
                   KEY_PLAYER_TYPES, POLICY_REGULATION_KEYWORDS,
                   RESEARCH_DEVELOPMENT_KEYWORDS, RSS_FEEDS,
                   TECHNOLOGY_INNOVATION_KEYWORDS)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ai-world-press-api")

# Load environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "aiWorldPress")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "articles")
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "900"))  # 15 minutes in seconds

# Initialize FastAPI
app = FastAPI(
    title="AI World Press API",
    description="Backend API for AI World Press news aggregator",
    version="1.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB client
client = pymongo.MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Create indexes for faster queries
collection.create_index([("link", pymongo.ASCENDING)], unique=True)
collection.create_index([("pubDate", pymongo.DESCENDING)])
collection.create_index([("topicCategory", pymongo.ASCENDING)])
collection.create_index([("newsOutlet", pymongo.ASCENDING)])
collection.create_index([("isAIRelated", pymongo.ASCENDING)])

# Load spaCy model
@lru_cache(maxsize=1)
def load_spacy_model():
    logger.info("Loading spaCy model...")
    return spacy.load("en_core_web_md")

nlp = load_spacy_model()

# Define rate limiter
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_history = {}
        
    async def check(self, client_ip: str) -> bool:
        now = time.time()
        
        # Clean up old records
        self.request_history = {
            ip: timestamps for ip, timestamps in self.request_history.items() if (timestamps[-1] if timestamps else 0) > now - 60
        }
        
        # Check rate limit
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
            
        self.request_history[client_ip] = [
            t for t in self.request_history[client_ip] if t > now - 60
        ]
        
        if len(self.request_history[client_ip]) >= self.requests_per_minute:
            return False
            
        self.request_history[client_ip].append(now)
        return True

rate_limiter = RateLimiter()

# Cache for articles
articles_cache = {
    "last_updated": None,
    "data": []
}

# Pydantic models
class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    is_ai_related: bool
    category: Optional[str] = None
    key_player: Optional[str] = None
    confidence: float

class RSSArticle(BaseModel):
    title: str
    link: str
    author: str = "Unknown"
    pubDate: datetime
    description: str = ""
    content: str = ""
    topicCategory: Optional[str] = None
    newsOutlet: str
    newsOutletLogo: Optional[str] = None
    feedCategory: str
    imageUrl: Optional[str] = None
    isAIRelated: bool = False
    keyPlayer: Optional[str] = None
    classificationConfidence: float = 0.0

class RSSFetchResponse(BaseModel):
    message: str
    stats: dict

class Article(BaseModel):
    _id: Optional[str] = None
    title: str
    link: str
    author: str = "Unknown"
    pubDate: datetime
    description: str = ""
    content: str = ""
    topicCategory: Optional[str] = None
    newsOutlet: str
    newsOutletLogo: Optional[str] = None
    feedCategory: str
    imageUrl: Optional[str] = None
    isAIRelated: bool = False
    keyPlayer: Optional[str] = None
    classificationConfidence: float = 0.0
    
    class Config:
        schema_extra = {
            "example": {
                "title": "OpenAI Releases GPT-5",
                "link": "https://example.com/news/openai-gpt5",
                "author": "Jane Doe",
                "pubDate": "2025-03-15T12:00:00Z",
                "description": "OpenAI has released its latest AI model, GPT-5, with improved capabilities.",
                "content": "OpenAI has released its latest AI model, GPT-5, with improved capabilities...",
                "topicCategory": "Research & Development",
                "newsOutlet": "TechNews",
                "newsOutletLogo": "https://example.com/logos/technews.png",
                "feedCategory": "Tech News",
                "imageUrl": "https://example.com/images/gpt5.jpg",
                "isAIRelated": True,
                "keyPlayer": "OpenAI",
                "classificationConfidence": 0.95
            }
        }

# Middleware for rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    is_allowed = await rate_limiter.check(client_ip)
    
    if not is_allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
    response = await call_next(request)
    return response

# Classification helper functions
def clean_text(text: str) -> str:
    """Clean HTML and normalize text"""
    if not text:
        return ""
    
    # Simple HTML tag removal (for production, consider a proper HTML parser)
    import re
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Convert HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def is_ai_related(text: str) -> tuple:
    """
    Determine if text is AI-related using keyword matching and NLP.
    Returns (is_related, confidence)
    """
    if not text:
        return False, 0.0
    
    text = text.lower()
    
    # Direct keyword matching
    direct_matches = sum(1 for keyword in AI_GENERAL_KEYWORDS if keyword in text)
    
    # More weight to direct matches
    confidence = min(direct_matches / 2, 1.0) if direct_matches else 0
    
    # Lower threshold
    return confidence >= 0.2, confidence

def determine_category(text: str) -> tuple:
    """
    Classify text into one of the defined categories.
    Returns (category, confidence)
    """
    if not text:
        return None, 0.0
    
    text = text.lower()
    
    # Prepare category keywords for matching
    categories = {
        "Business & Finance": BUSINESS_FINANCE_KEYWORDS,
        "Policy & Regulation": POLICY_REGULATION_KEYWORDS,
        "Research & Development": RESEARCH_DEVELOPMENT_KEYWORDS,
        "Ethics & Social Impact": ETHICS_SOCIAL_IMPACT_KEYWORDS,
        "Defense & Security": DEFENSE_SECURITY_KEYWORDS,
        "Technology & Innovation": TECHNOLOGY_INNOVATION_KEYWORDS,
        "Healthcare & Biotech": HEALTHCARE_BIOTECH_KEYWORDS,
        "Entertainment & Media": ENTERTAINMENT_MEDIA_KEYWORDS
    }
    
    # Check keyword matches for each category
    category_scores = {}
    for category, keywords in categories.items():
        # Count how many keywords from this category appear in the text
        matches = 0
        for keyword in keywords:
            if keyword in text:
                matches += 1
                
        # Calculate score - more matches = higher score
        # Use a different formula that gives higher scores
        if matches > 0:
            # Score based on matches, with diminishing returns for many matches
            category_scores[category] = min(0.7 * (matches / 5), 1.0)
    
    # If no categories matched well, return None
    if not category_scores:
        return None, 0.0
    
    # Add bonus for entity types that align with categories
    doc = nlp(text[:10000])  # Limit text size for performance
    
    # Look for company/business entities for Business & Finance
    if any(ent.label_ in ["ORG"] and any(kw in ent.text.lower() for kw in ["company", "inc", "corp", "capital", "fund"]) for ent in doc.ents):
        category_scores["Business & Finance"] = category_scores.get("Business & Finance", 0) + 0.3
    
    # Look for government/regulatory entities for Policy & Regulation
    if any(ent.label_ in ["ORG", "GPE"] and any(kw in ent.text.lower() for kw in ["government", "commission", "authority", "agency", "regulator"]) for ent in doc.ents):
        category_scores["Policy & Regulation"] = category_scores.get("Policy & Regulation", 0) + 0.3
    
    # Look for research institutions for Research & Development
    if any(ent.label_ in ["ORG"] and any(kw in ent.text.lower() for kw in ["university", "institute", "lab", "research", "laboratory"]) for ent in doc.ents):
        category_scores["Research & Development"] = category_scores.get("Research & Development", 0) + 0.3
    
    # Get category with highest score
    top_category = max(category_scores.items(), key=lambda x: x[1])
    
    # Return category with confidence score
    return top_category[0], min(top_category[1], 1.0)

def extract_key_player(text: str) -> tuple:
    """
    Extract the key player (person, organization, etc.) from the text.
    Returns (key_player, confidence)
    """
    if not text:
        return None, 0.0
    
    doc = nlp(text[:10000])  # Limit text size for performance
    
    # Collect entities that might be key players
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in KEY_PLAYER_TYPES:
            # Basic relevance score based on entity length and position
            length_factor = min(len(ent.text.split()) / 5, 1)  # Prefer multi-word entities
            position_factor = 1 - (ent.start / len(doc)) if len(doc) > 0 else 0  # Prefer earlier entities
            
            # Higher score for entities in title (first ~15 tokens)
            title_bonus = 0.3 if ent.start < 15 else 0
            
            # Named organizations and people get higher relevance
            type_bonus = 0.4 if ent.label_ in ["ORG", "PERSON"] else 0.2
            
            score = (0.4 * length_factor) + (0.3 * position_factor) + title_bonus + type_bonus
            
            entities.append((ent.text, score))
    
    # Select the highest scoring entity
    if entities:
        sorted_entities = sorted(entities, key=lambda x: x[1], reverse=True)
        return sorted_entities[0][0], sorted_entities[0][1]
    
    return None, 0.0

async def parse_rss_feeds() -> List[RSSArticle]:
    """Fetch and parse RSS feeds concurrently"""
    async def fetch_feed(feed_info):
        try:
            feed_url = feed_info["url"]
            feed_data = feedparser.parse(feed_url)
            
            articles = []
            for entry in feed_data.entries:
                # Extract article data
                title = clean_text(entry.get("title", ""))
                link = entry.get("link", "")
                
                # Skip if no title or link
                if not title or not link:
                    continue
                
                # Get author
                author = "Unknown"
                if hasattr(entry, "author"):
                    author = entry.author
                elif hasattr(entry, "dc_creator"):
                    author = entry.dc_creator
                
                # Get publication date
                pub_date = datetime.now()
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    pub_date = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
                
                # Get content
                content = ""
                if hasattr(entry, "content") and entry.content:
                    content = entry.content[0].value
                elif hasattr(entry, "summary"):
                    content = entry.summary
                elif hasattr(entry, "description"):
                    content = entry.description
                
                content = clean_text(content)
                
                # Get description (shorter version of content)
                description = clean_text(entry.get("summary", "")) or content[:300]
                
                # Extract image URL
                image_url = None
                if hasattr(entry, "media_content") and entry.media_content:
                    for media in entry.media_content:
                        if "url" in media:
                            image_url = media["url"]
                            break
                
                # Create article object
                article = RSSArticle(
                    title=title,
                    link=link,
                    author=author,
                    pubDate=pub_date,
                    description=description,
                    content=content,
                    newsOutlet=feed_info["name"],
                    newsOutletLogo=feed_info.get("logo"),
                    feedCategory=feed_info["category"],
                    imageUrl=image_url,
                    isAIRelated=False  # Will be determined later
                )
                
                articles.append(article)
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching feed {feed_info['url']}: {str(e)}")
            return []
    
    tasks = []
    for category, feeds in RSS_FEEDS.items():
        for feed in feeds:
            tasks.append(fetch_feed(feed))
    
    results = await asyncio.gather(*tasks)
    return [article for sublist in results for article in sublist]

async def classify_and_store_articles(articles: List[RSSArticle]) -> Dict:
    """Classify articles and store AI-related ones in database"""
    ai_related_count = 0
    stored_count = 0
    
    for article in articles:
        # Combine title, description, and content for classification
        text = f"{article.title} {article.description} {article.content}"
        
        # Check if AI-related
        is_ai, ai_confidence = is_ai_related(text)
        
        if not is_ai:
            continue
        
        ai_related_count += 1
        article.isAIRelated = True
        article.classificationConfidence = ai_confidence
        
        # Determine category
        category, category_confidence = determine_category(text)
        article.topicCategory = category
        
        # Combined confidence score (weighted)
        article.classificationConfidence = (0.4 * ai_confidence) + (0.6 * category_confidence)
        
        # Extract key player
        key_player, player_confidence = extract_key_player(text)
        article.keyPlayer = key_player
        
        # Store in database
        try:
            # Convert to dict for MongoDB
            article_dict = article.dict()
            
            # Use upsert to avoid duplicates
            result = collection.update_one(
                {"link": article.link},
                {"$set": article_dict},
                upsert=True
            )
            
            if result.upserted_id or result.modified_count > 0:
                stored_count += 1
                
        except DuplicateKeyError:
            # Skip if already exists
            pass
        except Exception as e:
            logger.error(f"Error storing article {article.title}: {str(e)}")
    
    # Update cache if articles were stored
    if stored_count > 0:
        # Clear cache to force refresh on next request
        global articles_cache
        articles_cache = {
            "last_updated": None,
            "data": []
        }
    
    return {
        "total_fetched": len(articles),
        "ai_related": ai_related_count,
        "stored": stored_count
    }

@app.get("/")
async def root():
    return {"message": "AI World Press API is running"}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """Classify text for AI relevance and category"""
    text = request.text
    
    # Check if AI-related
    is_ai, ai_confidence = is_ai_related(text)
    
    if not is_ai:
        return ClassificationResponse(
            is_ai_related=False,
            confidence=ai_confidence
        )
    
    # Determine category
    category, category_confidence = determine_category(text)
    
    # Extract key player
    key_player, player_confidence = extract_key_player(text)
    
    # Combined confidence (weighted average)
    overall_confidence = (0.4 * ai_confidence) + (0.6 * category_confidence)
    
    return ClassificationResponse(
        is_ai_related=True,
        category=category,
        key_player=key_player,
        confidence=overall_confidence
    )

@app.post("/fetch-rss", response_model=RSSFetchResponse)
async def fetch_rss():
    """Fetch RSS feeds, classify articles, and store in database"""
    # Fetch and parse feeds
    articles = await parse_rss_feeds()
    
    # Classify and store articles
    stats = await classify_and_store_articles(articles)
    
    return RSSFetchResponse(
        message="RSS feeds processed successfully",
        stats=stats
    )

@app.get("/articles")
async def get_articles(
    feedCategory: Optional[str] = None,
    topicCategory: Optional[str] = None,
    newsOutlet: Optional[str] = None,
    days: Optional[int] = 7,
    limit: Optional[int] = 100
):
    """Get articles with optional filtering"""
    # Check cache first
    global articles_cache
    
    if (
        articles_cache["last_updated"] 
        and (datetime.now() - articles_cache["last_updated"]).total_seconds() < CACHE_EXPIRY
    ):
        # Filter cached results
        filtered_results = articles_cache["data"]
        
        if feedCategory:
            filtered_results = [a for a in filtered_results if a.get("feedCategory") == feedCategory]
        if topicCategory:
            filtered_results = [a for a in filtered_results if a.get("topicCategory") == topicCategory]
        if newsOutlet:
            filtered_results = [a for a in filtered_results if a.get("newsOutlet") == newsOutlet]
            
        # Limit results
        return filtered_results[:limit]
    
    # Build query
    query = {"isAIRelated": True}
    
    if feedCategory:
        query["feedCategory"] = feedCategory
    if topicCategory:
        query["topicCategory"] = topicCategory
    if newsOutlet:
        query["newsOutlet"] = newsOutlet
    if days:
        date_limit = datetime.now() - timedelta(days=days)
        query["pubDate"] = {"$gte": date_limit}
    
    # Fetch from database
    cursor = collection.find(query).sort("pubDate", -1).limit(limit)
    
    # Convert to list and format for response
    results = []
    for doc in await cursor.to_list(length=limit):
        # Convert ObjectId to string
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        results.append(doc)
    
    # Update cache
    articles_cache["last_updated"] = datetime.now()
    articles_cache["data"] = results
    
    return results

@app.delete("/clear-cache")
async def clear_cache():
    """Clear the articles cache"""
    global articles_cache
    articles_cache = {
        "last_updated": None,
        "data": []
    }
    return {"message": "Cache cleared successfully"}

# Add other endpoints as needed

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)