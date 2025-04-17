# ai_news_classifier/app/models.py

from pydantic import BaseModel, Field, HttpUrl, ConfigDict, field_validator, validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing import Optional, List, Any, Literal, Callable
from datetime import datetime
from bson import ObjectId

# Helper class to handle MongoDB ObjectId serialization/validation with Pydantic V2
class PyObjectId(ObjectId):
    """
    Custom Pydantic type for MongoDB ObjectId validation and serialization.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_object_id

    @classmethod
    def validate_object_id(cls, v: Any, handler) -> ObjectId:
        """Validate that the input is a valid ObjectId."""
        if isinstance(v, ObjectId):
            return v
        # Use the handler to potentially convert types before checking validity
        s = handler(v)
        if isinstance(s, str) and ObjectId.is_valid(s):
            return ObjectId(s)
        # Raise ValueError for invalid types or invalid ObjectId strings
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler # handler might not be needed here
    ) -> JsonSchemaValue:
        """
        Return JSON schema representation for ObjectId.
        Explicitly define it as a string type.
        """
        # Instead of returning handler(core_schema.str_schema()),
        # return the schema dictionary directly.
        return {'type': 'string',
                'description': 'MongoDB ObjectId represented as a 24-character hex string',
                'examples': ['60f1b7b3b3b3b3b3b3b3b3b3'] # Optional example
               }

    # --- Add __get_pydantic_core_schema__ for robust validation ---
    # This helps Pydantic understand the core validation logic better
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[Any], handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        """
        Define the core schema for validation.
        Ensures input is validated by validate_object_id.
        """
        # Define a validation schema that uses our custom validator
        # It expects input that can be validated into an ObjectId
        # and ensures the output type is ObjectId
        return core_schema.general_plain_validator_function(
            cls.validate_object_id, serialization=core_schema.to_string_ser_schema()
        )

# Pydantic Model for an Article
class ArticleModel(BaseModel):
    """
    Represents a single RSS feed article entry in the database.
    """
    # Use PyObjectId for the '_id' field, mapping it to 'id' in the model
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    # --- Fields from Article.js ---
    title: str = Field(..., description="Article title (indexed in DB)")
    author: Optional[str] = Field(default='Unknown', description="Article author, defaults to 'Unknown'")
    # Renamed published_date to pub_date for consistency with JS schema
    pub_date: datetime = Field(..., alias="pubDate", description="Article publication date (indexed in DB)")
    # Renamed summary to description for consistency with JS schema
    description: Optional[str] = Field(None, description="Article summary or description")
    content: Optional[str] = Field(None, description="Full article content, if available")
    link: HttpUrl = Field(..., description="Unique URL link to the article (unique index in DB)")
    topic_category: Optional[str] = Field(None, alias="topicCategory", description="Specific category if AI-related, null otherwise")
    news_outlet: str = Field(..., alias="newsOutlet", description="Name of the news source/outlet (indexed in DB)")
    news_outlet_logo: Optional[str] = Field(None, alias="newsOutletLogo", description="Path to news outlet's logo image file")

    # Using str for flexibility, consider Literal[...] or Enum if categories are strictly fixed
    # Example: feed_category: Literal['Mainstream News', 'Alternative News', 'Tech News'] = Field(...)
    feed_category: str = Field(..., alias="feedCategory", description="Category of the RSS feed source (e.g., 'Tech News') (indexed in DB)")

    image_url: Optional[HttpUrl] = Field(None, alias="imageUrl", description="URL of a relevant image for the article (optional)")

    # Changed from Optional[bool] to bool = False based on JS schema (required, default: false)
    is_ai_related: bool = Field(default=None, alias="isAIRelated", description="Boolean flag indicating AI relevance (indexed in DB)")

    # Added confidence score
    classification_confidence: float = Field(
        default=None,
        alias="classificationConfidence",
        ge=0.0, # Enforce >= 0
        le=1.0, # Enforce <= 1
        description="Confidence score (0-1) from the AI classification model"
    )

    # --- Timestamps (like Mongoose timestamps: true) ---
    # Note: 'updated_at' should ideally be updated by the database operation ($currentDate)
    # or application logic during updates, not just by Pydantic default_factory on load.
    created_at: datetime = Field(default_factory=datetime.utcnow, alias="createdAt")
    updated_at: datetime = Field(default_factory=datetime.utcnow, alias="updatedAt")

    # --- Fields from original Python model (if needed, else remove) ---
    # feed_source_url: str = Field(...) # Original field, maybe redundant if link is unique key? Check usage.
                                      # Keeping it commented out for now. If needed, add back.

    # Pydantic V2 configuration using model_config dictionary
    model_config = ConfigDict(
        populate_by_name=True,  # Allow populating by field name OR alias ('_id', 'pubDate', etc.)
        arbitrary_types_allowed=True, # Allow PyObjectId type
        json_encoders={ObjectId: str, datetime: lambda dt: dt.isoformat()}, # Serialize ObjectId to str, datetime to ISO str
        validate_assignment=True, # Re-validate fields on assignment
        extra='ignore' # Ignore extra fields from DB if any mismatch
    )

    # Optional: Validator to ensure updated_at is always >= created_at
    @validator('updated_at')
    def check_updated_at(cls, v, values):
        if 'created_at' in values and v < values['created_at']:
            raise ValueError('updated_at must be greater than or equal to created_at')
        return v


# Pydantic Model for the Paginated Response Structure (Remains the same)
class PaginatedArticleResponse(BaseModel):
    """
    Defines the response structure for fetching articles with pagination.
    """
    total_count: int = Field(..., description="Total number of articles matching the query")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of articles per page")
    has_more: bool = Field(..., description="Indicates if there are more pages available")
    articles: List[ArticleModel] = Field(..., description="List of articles for the current page")


# Example usage (optional, for testing - updated with new fields)
if __name__ == "__main__":
    print("--- Running Model Tests ---")
    # Test PyObjectId
    print("\nTesting PyObjectId...")
    oid_str = "60f1b7b3b3b3b3b3b3b3b3b3"
    if ObjectId.is_valid(oid_str):
        py_oid = PyObjectId(oid_str)
        print(f"String to PyObjectId: {py_oid} (type: {type(py_oid)})")
        assert isinstance(py_oid, ObjectId)
    else:
        print(f"'{oid_str}' is not a valid ObjectId string.")
    try:
        PyObjectId("invalid-oid")
    except ValueError as e:
        print(f"Caught expected error for invalid ObjectId string: {e}")

    # Test ArticleModel instantiation and JSON serialization
    print("\nTesting ArticleModel...")
    now = datetime.utcnow()
    sample_article_data_from_db = {
        "_id": ObjectId(),
        "title": "New AI Model Released",
        "author": "AI Research Lab",
        "pubDate": now,
        "description": "A cutting-edge AI model shows promise.",
        "content": "Full content here...",
        "link": "http://example.com/news/new-ai-model",
        "topicCategory": "Language Models",
        "newsOutlet": "Tech News Today",
        "newsOutletLogo": "http://example.com/logo.png",
        "feedCategory": "Tech News",
        "imageUrl": "http://example.com/image.jpg",
        "isAIRelated": True,
        "classificationConfidence": 0.95,
        "createdAt": now,
        "updatedAt": now
    }

    try:
        article = ArticleModel(**sample_article_data_from_db)
        print("ArticleModel instantiated successfully:")
        # Use model_dump instead of deprecated dict() in Pydantic V2
        print(article.model_dump())

        # Test JSON serialization (id should be string, dates ISO format)
        json_output = article.model_dump_json(indent=2, by_alias=True) # Use by_alias=True for JS keys
        print("\nJSON Output (with aliases):")
        print(json_output)
        import json
        json_data = json.loads(json_output)
        assert isinstance(json_data['_id'], str) # Check alias _id is string
        assert isinstance(json_data['pubDate'], str) # Check alias pubDate is string
        assert json_data['classificationConfidence'] == 0.95
        assert json_data['isAIRelated'] is True
        assert json_data['author'] == "AI Research Lab"
        print("\nJSON serialization successful (ObjectId as string, dates as ISO string, aliases used).")

        # Test default values
        minimal_data = {
            "title": "Minimal Article",
            "pubDate": now,
            "link": "http://example.com/minimal",
            "newsOutlet": "Min Feed",
            "feedCategory": "Mainstream News"
            # Other required fields filled by defaults or validation error if not provided
        }
        minimal_article = ArticleModel(**minimal_data)
        print("\nMinimal Article Defaults:")
        print(f"Author: {minimal_article.author}")
        print(f"Is AI Related: {minimal_article.is_ai_related}")
        print(f"Confidence: {minimal_article.classification_confidence}")
        assert minimal_article.author == 'Unknown'
        assert minimal_article.is_ai_related is False
        assert minimal_article.classification_confidence == 0.0
        print("Defaults applied correctly.")

    except Exception as e:
        print(f"Error during ArticleModel testing: {e}")
        import traceback
        traceback.print_exc()


    # Test PaginatedArticleResponse (remains structurally the same)
    print("\nTesting PaginatedArticleResponse...")
    # Need ArticleModel instances for the articles list
    article_instance = ArticleModel(**sample_article_data_from_db)
    paginated_data = {
        "total_count": 100,
        "page": 1,
        "page_size": 10,
        "has_more": True,
        "articles": [article_instance]
    }
    try:
        paginated_response = PaginatedArticleResponse(**paginated_data)
        print("PaginatedArticleResponse instantiated successfully.")
        # print(paginated_response.model_dump_json(indent=2, by_alias=True)) # Print if needed
        print("\nModels updated and tests passed.")
    except Exception as e:
        print(f"Error during PaginatedArticleResponse testing: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Model Tests Finished ---")




'''from beanie import Document, Indexed
from pydantic import Field, HttpUrl, validator, field_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import feedparser # For parsing time structs
import logging

logger = logging.getLogger(__name__)

def parse_datetime_flexible(value: Any) -> Optional[datetime]:
    """Utility to parse datetime from various feedparser formats."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    # Handle feedparser's time_struct tuple if present directly or via _parsed attribute
    time_tuple = None
    if isinstance(value, feedparser.FeedParserDict): # Check common attributes
        time_tuple = value.get('published_parsed', value.get('updated_parsed'))
    elif isinstance(value, tuple) and len(value) >= 6: # Direct time_struct
         time_tuple = value

    if time_tuple:
        try:
            # Ensure all components are integers before passing to datetime
            int_time_tuple = tuple(int(i) for i in time_tuple[:6])
            return datetime(*int_time_tuple)
        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"Could not parse time_tuple {time_tuple}: {e}")
            return None # Invalid time tuple components

    # Add more parsing logic here if needed for string formats etc.
    return None

class Article(Document):
    """
    Represents an article fetched from an RSS feed, including extended metadata.
    """
    # Core Identifiers
    link: Indexed(HttpUrl, unique=True) # Assumed stable URL, unique index
    guid: Optional[str] = Field(default=None) # Feed's unique ID (may differ from link), consider indexing if used for lookups

    # Core Content
    title: str
    summary: Optional[str] = None # Often a short description or excerpt
    content: Optional[str] = None # Potentially longer content, often HTML. Store first non-empty value.
    description: Optional[str] = None
    # Optional: Store all content entries if needed: content_entries: Optional[List[Dict[str, str]]] = None

    # Author(s)
    author: Optional[str] = None # Single author string if available
    authors: Optional[List[str]] = None # List of author names if multiple provided

    # Timestamps
    published_datetime: Optional[datetime] = Field(default=None)
    updated_datetime: Optional[datetime] = Field(default=None)

    # Feed/Source Info
    source_feed_url: HttpUrl
    source_feed_title: Optional[str] = Field(default=None) # Title of the feed itself

    # Categorization / Tags
    tags: Optional[List[str]] = None # List of tags/keywords

    # Classification Fields
    is_ai_related: Optional[bool] = Field(default=None) # Nullable boolean, None means unclassified
    classification_score: Optional[float] = Field(default=None)

    # Raw Data (Optional)
    raw_feed_entry: Optional[dict] = Field(default=None) # Store the raw entry for debugging/future use

    # --- Settings ---
    class Settings:
        name = "articles" # MongoDB collection name
        # Optional: Add more indexes if needed for querying
        indexes = [
            [("published_datetime", -1)], # Index for sorting by date (descending)
            [("is_ai_related", 1), ("published_datetime", -1)], # Compound index for filtering + sorting
            [("source_feed_url", 1)], # Index for querying by source
            # Consider indexing 'guid' if it's reliably unique and used for lookups
            # Consider text index if doing full-text search in Mongo:
            # [
            #    ("title", "text"),
            #    ("summary", "text"),
            #    ("content", "text"), # Be cautious with large content fields
            # ]
        ]

    # --- Validators ---
    # Use Pydantic v2 style validator
    @field_validator('published_datetime', 'updated_datetime', mode='before')
    @classmethod
    def parse_datetimes(cls, v):
        """Parse published and updated datetimes flexibly."""
        return parse_datetime_flexible(v)

    @field_validator('tags', mode='before')
    @classmethod
    def parse_tags(cls, v):
        """Extract tag terms from feedparser's tag list format."""
        if isinstance(v, list):
            tags_out = []
            for tag_dict in v:
                if isinstance(tag_dict, dict) and 'term' in tag_dict and isinstance(tag_dict['term'], str):
                    tags_out.append(tag_dict['term'])
            return tags_out if tags_out else None
        return None # Return None if input is not a list

    @field_validator('authors', mode='before')
    @classmethod
    def parse_authors(cls, v):
        """Extract author names from feedparser's authors list format."""
        if isinstance(v, list):
            authors_out = []
            for author_dict in v:
                 if isinstance(author_dict, dict) and 'name' in author_dict and isinstance(author_dict['name'], str):
                     authors_out.append(author_dict['name'])
            return authors_out if authors_out else None
        return None

    @field_validator('content', mode='before')
    @classmethod
    def parse_content(cls, v):
        """Extract first non-empty content value from feedparser's content list."""
        if isinstance(v, list):
            for content_entry in v:
                if isinstance(content_entry, dict) and 'value' in content_entry and content_entry['value']:
                    return content_entry['value'] # Return the first non-empty value found
        # Handle case where content might be a simple string (less common but possible)
        if isinstance(v, str) and v:
             return v
        return None # No suitable content found


    # Example of a simple representation
    def __repr__(self) -> str:
        return f"<Article title={self.title[:30]}... link={self.link}>"

    def __str__(self) -> str:
        return self.title'''