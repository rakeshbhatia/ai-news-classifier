# ai_news_classifier/app/models.py

from datetime import datetime
from typing import Optional, List, Any, Callable # Ensure Callable is imported

from pydantic import BaseModel, Field, HttpUrl, ConfigDict, field_validator, validator
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, UniqueConstraint, Index
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func # For server-side timestamp defaults
from sqlalchemy.dialects.postgresql import TIMESTAMP # More specific type

# --- SQLAlchemy Base Model ---
class Base(DeclarativeBase):
    pass

# --- SQLAlchemy ORM Model for the 'articles' table ---
class Article(Base):
    __tablename__ = "articles"

    id: int = Column(Integer, primary_key=True) # Standard integer PK
    title: str = Column(String(500), nullable=False, index=True)
    author: Optional[str] = Column(String(255), nullable=True, default='Unknown')
    pub_date: datetime = Column(TIMESTAMP(timezone=True), nullable=False, index=True) # Timezone-aware timestamp
    category: Optional[str] = Column(String(255), nullable=True)
    description: Optional[str] = Column(Text, nullable=True) # Use Text for longer content
    content: Optional[str] = Column(Text, nullable=True)
    link: str = Column(String(2048), nullable=False, unique=True) # Unique constraint on link
    topic_category: Optional[str] = Column(String(255), nullable=True)
    news_outlet: str = Column(String(255), nullable=False, index=True)
    news_outlet_logo: Optional[str] = Column(String(2048), nullable=True) # Store logo path/URL as string
    feed_category: str = Column(String(255), nullable=False, index=True)
    image_url: Optional[str] = Column(String(2048), nullable=True) # Store image URL as string

    is_ai_related: Optional[bool] = Column(Boolean, nullable=True, index=True, default=None) # Allow NULL for unclassified
    classification_confidence: Optional[float] = Column(Float, nullable=True, default=None)

    # Timestamps managed by SQLAlchemy/database
    # Use server_default for database-level timestamp generation
    created_at: datetime = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at: datetime = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Define constraints and indexes explicitly if needed beyond basic ones
    __table_args__ = (
        UniqueConstraint('link', name='uq_article_link'),
        Index('ix_article_feedCategory_pubDate', 'feed_category', 'pub_date'),
        Index('ix_article_isAIRelated_pubDate', 'is_ai_related', 'pub_date'),
     )

    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title[:30]}...', link='{self.link}')>"


# --- Pydantic Models for API Validation/Response---

class ArticleBase(BaseModel):
    # Fields common to creation and reading, without DB-generated ID/timestamps
    title: str
    author: Optional[str] = 'Unknown'
    pub_date: datetime = Field(..., alias="pubDate")
    category: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    link: HttpUrl # Keep HttpUrl for input validation
    topic_category: Optional[str] = Field(None, alias="topicCategory")
    news_outlet: str = Field(..., alias="newsOutlet")
    news_outlet_logo: Optional[str] = Field(None, alias="newsOutletLogo") # Store as string path
    feed_category: str = Field(..., alias="feedCategory")
    image_url: Optional[HttpUrl] = Field(None, alias="imageUrl") # Keep HttpUrl for input validation
    is_ai_related: Optional[bool] = Field(None, alias="isAIRelated")
    classification_confidence: Optional[float] = Field(None, alias="classificationConfidence", ge=0.0, le=1.0)

# Pydantic model for API responses (includes ID and timestamps)
class ArticleModel(ArticleBase):
    id: int # Integer ID from PostgreSQL
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    # Enable ORM mode for automatic mapping from SQLAlchemy objects
    model_config = ConfigDict(
        from_attributes=True, # Enable ORM mode (formerly orm_mode=True)
        populate_by_name=True, # Allow population by alias
        json_encoders={datetime: lambda dt: dt.isoformat()} # Ensure consistent datetime format
    )

# Pydantic Model for Paginated Response Structure (uses ArticleModel)
class PaginatedArticleResponse(BaseModel):
    total_count: int
    page: int
    page_size: int
    has_more: bool
    articles: List[ArticleModel] # Use the Pydantic model here

# Example additions to models.py's __main__ block
'''if __name__ == "__main__":
    print("--- Testing Pydantic Models ---")
    now = datetime.now(timezone.utc)
    sample_api_data = {
        "title": "API Test Article", "pubDate": now, "link": "http://apitest.com/article",
        "newsOutlet": "Test News", "feedCategory": "Tech News",
        # Add other required fields from ArticleBase...
        "author": "Test Author", "description": "API Desc", "topicCategory": "API Topic",
        "newsOutletLogo": "/logos/test.png", "imageUrl": "http://apitest.com/image.jpg",
        "isAIRelated": True, "classificationConfidence": 0.88
    }
    try:
        base_instance = ArticleBase(**sample_api_data)
        print("ArticleBase validation passed.")

        # Simulate data coming from DB for ArticleModel (add id/timestamps)
        orm_like_data = {
            **base_instance.model_dump(by_alias=False), # Use Python names
            "id": 123,
            "created_at": now - timedelta(days=1),
            "updated_at": now
        }
        # Use model_validate for ORM-like data
        api_model_instance = ArticleModel.model_validate(orm_like_data)
        print("ArticleModel validation (from dict) passed.")
        print(api_model_instance.model_dump_json(indent=2, by_alias=True)) # Test serialization

        # Test PaginatedArticleResponse
        paginated_resp = PaginatedArticleResponse(
            total_count=50, page=1, page_size=10, has_more=True,
            articles=[api_model_instance] # List of ArticleModel instances
        )
        print("\nPaginatedArticleResponse validation passed.")
        # print(paginated_resp.model_dump_json(indent=2, by_alias=True))

    except Exception as e:
        print(f"\nPydantic Model Test FAILED: {e}", exc_info=True)'''


# --- Test block ---
if __name__ == "__main__":
    # --- Add imports needed *specifically for testing* here ---
    from datetime import datetime, timezone, timedelta # Import timezone and timedelta
    import json # Import json for testing serialization output
    # --- End test-specific imports ---

    print("--- Testing Pydantic Models ---")
    now = datetime.now(timezone.utc) # This line should now work
    sample_api_data = {
        "title": "API Test Article", "pubDate": now, "link": "http://apitest.com/article",
        "newsOutlet": "Test News", "feedCategory": "Tech News",
        # Add other required fields from ArticleBase...
        "author": "Test Author", "description": "API Desc", "topicCategory": "API Topic",
        "newsOutletLogo": "/logos/test.png", "imageUrl": "http://apitest.com/image.jpg",
        "isAIRelated": True, "classificationConfidence": 0.88
    }
    try:
        base_instance = ArticleBase(**sample_api_data)
        print("ArticleBase validation passed.")

        # Simulate data coming from DB for ArticleModel (add id/timestamps)
        orm_like_data = {
            **base_instance.model_dump(by_alias=False), # Use Python names
            "id": 123,
            "created_at": now - timedelta(days=1), # timedelta is now imported
            "updated_at": now
        }
        # Use model_validate for ORM-like data
        api_model_instance = ArticleModel.model_validate(orm_like_data)
        print("ArticleModel validation (from dict) passed.")
        print(api_model_instance.model_dump_json(indent=2, by_alias=True)) # Test serialization

        # Test PaginatedArticleResponse
        paginated_resp = PaginatedArticleResponse(
            total_count=50, page=1, page_size=10, has_more=True,
            articles=[api_model_instance] # List of ArticleModel instances
        )
        print("\nPaginatedArticleResponse validation passed.")
        # print(paginated_resp.model_dump_json(indent=2, by_alias=True))

        print("\nPydantic Model Test PASSED") # Added explicit success message

    except Exception as e:
        import traceback # Import traceback for better error printing
        print(f"\nPydantic Model Test FAILED: {e}")
        traceback.print_exc() # Print full traceback for debugging