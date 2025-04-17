# ai_news_classifier/app/database.py

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure, OperationFailure
from typing import List, Optional, Dict, Any

from .config import settings
from .models import ArticleModel, PaginatedArticleResponse, PyObjectId # Import PyObjectId if needed for query construction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    """Handles MongoDB connection and operations."""
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None
    articles_collection: Optional[AsyncIOMotorCollection] = None

    async def connect_db(self):
        """Establishes an asynchronous connection to MongoDB."""
        if self.client is not None:
            logger.info("Database connection already established.")
            return

        logger.info(f"Attempting to connect to MongoDB at {settings.MONGO_URI}...")
        try:
            self.client = AsyncIOMotorClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
            # The ismaster command is cheap and does not require auth.
            await self.client.admin.command('ismaster')
            self.db = self.client[settings.MONGO_DB_NAME]
            self.articles_collection = self.db[settings.MONGO_ARTICLES_COLLECTION]
            logger.info(f"Successfully connected to MongoDB database: {settings.MONGO_DB_NAME}")
            logger.info(f"Using collection: {settings.MONGO_ARTICLES_COLLECTION}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            self.articles_collection = None
            raise  # Re-raise the exception to signal connection failure

    async def close_db(self):
        """Closes the MongoDB connection."""
        #if self.client:
        if self.client is not None:
            self.client.close()
            self.client = None
            self.db = None
            self.articles_collection = None
            logger.info("MongoDB connection closed.")

    async def get_articles_paginated(
        self,
        page: int = 1,
        page_size: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> PaginatedArticleResponse:
        """
        Fetches articles from the database with pagination and optional filtering.

        Args:
            page: The page number to retrieve (1-indexed).
            page_size: The number of articles per page.
            filter_criteria: A dictionary representing the MongoDB query filter (optional).

        Returns:
            A PaginatedArticleResponse object containing the articles and pagination info.

        Raises:
            ValueError: If the database is not connected.
            OperationFailure: If there's an issue querying MongoDB.
        """
        #if not self.articles_collection or not self.db:
        if self.articles_collection is None or self.db is None:
            raise ValueError("Database not connected. Call connect_db() first.")

        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 10 # Or a default value

        skip = (page - 1) * page_size
        query = filter_criteria or {} # Use provided filter or empty dict for all documents

        try:
            # Perform two queries: one for the total count and one for the data page
            total_count = await self.articles_collection.count_documents(query)

            cursor = self.articles_collection.find(query)\
                                            .sort("published_date", -1)\
                                            .skip(skip)\
                                            .limit(page_size)

            # Execute the query and retrieve the documents
            # Use .to_list(length=page_size) to fetch documents asynchronously
            articles_data = await cursor.to_list(length=page_size)

            # Convert raw BSON documents to Pydantic ArticleModel instances
            articles = [ArticleModel(**doc) for doc in articles_data]

            # Calculate if there are more pages
            has_more = (page * page_size) < total_count

            return PaginatedArticleResponse(
                total_count=total_count,
                page=page,
                page_size=page_size,
                has_more=has_more,
                articles=articles
            )
        except OperationFailure as e:
            logger.error(f"MongoDB query failed: {e.details}")
            raise # Re-raise to be handled by the caller (e.g., API endpoint)
        except Exception as e:
            logger.error(f"An unexpected error occurred during article retrieval: {e}")
            raise


# Create a single instance of the Database class
# Connection will be established via FastAPI startup event or dependency
db_handler = Database()

# --- Testing Block ---
async def run_db_tests():
    """Function to run async database tests."""
    print("--- Running Database Tests ---")
    # Ensure environment variables are set (e.g., via .env file loaded by config)
    # You MUST have a running MongoDB instance accessible via MONGO_URI
    if not settings.MONGO_URI or "localhost" in settings.MONGO_URI:
         print("\nWARNING: Testing requires MONGO_URI to be set in your environment")
         print("and point to an accessible MongoDB instance (local or Atlas).")
         print("Ensure MongoDB is running.")
         # return # Optionally skip tests if not configured

    try:
        # Connect
        print("\nConnecting to DB...")
        await db_handler.connect_db()
        assert db_handler.client is not None
        assert db_handler.db is not None
        assert db_handler.articles_collection is not None
        print("Connection successful.")

        # Optional: Add some test data if collection is empty
        # Note: This adds data on every test run if uncommented!
        # print("\nEnsuring some test data exists...")
        # if await db_handler.articles_collection.count_documents({}) < 5:
        #     print("Adding sample articles...")
        #     sample_articles_raw = [
        #         {
        #             "title": f"Test Article {i}", "link": f"http://test.com/{i}",
        #             "published_date": datetime.utcnow(), "summary": f"Summary {i}",
        #             "feed_source_url": "http://test.com/rss", "is_ai_related": i % 2 == 0,
        #             "processed_at": datetime.utcnow()
        #         } for i in range(15)
        #     ]
        #     await db_handler.articles_collection.insert_many(sample_articles_raw)
        #     print(f"Added {len(sample_articles_raw)} sample articles.")

        # Test fetching page 1
        print("\nFetching page 1 (size 5)...")
        page1_response = await db_handler.get_articles_paginated(page=1, page_size=5)
        print(f"Total Count: {page1_response.total_count}")
        print(f"Has More: {page1_response.has_more}")
        print(f"Articles on Page 1: {len(page1_response.articles)}")
        assert len(page1_response.articles) <= 5
        if page1_response.total_count > 0:
           assert len(page1_response.articles) > 0
           assert isinstance(page1_response.articles[0], ArticleModel)
           assert isinstance(page1_response.articles[0].id, ObjectId) # Internal check
        print("Page 1 fetch successful.")
        # print(page1_response.model_dump_json(indent=2)) # Uncomment to see full response

        # Test fetching page 2
        if page1_response.has_more:
            print("\nFetching page 2 (size 5)...")
            page2_response = await db_handler.get_articles_paginated(page=2, page_size=5)
            print(f"Articles on Page 2: {len(page2_response.articles)}")
            assert len(page2_response.articles) <= 5
            assert page2_response.page == 2
            print("Page 2 fetch successful.")
            # Verify items are different from page 1 (requires > 5 items total)
            if len(page1_response.articles) == 5 and len(page2_response.articles) > 0:
                assert page1_response.articles[0].id != page2_response.articles[0].id


        # Test filtering (example: fetch only AI-related articles)
        print("\nFetching AI-related articles (page 1, size 5)...")
        ai_filter = {"is_ai_related": True}
        ai_page1_response = await db_handler.get_articles_paginated(
            page=1, page_size=5, filter_criteria=ai_filter
        )
        print(f"Total AI Articles: {ai_page1_response.total_count}")
        print(f"AI Articles on Page 1: {len(ai_page1_response.articles)}")
        for article in ai_page1_response.articles:
            assert article.is_ai_related is True
        print("AI filter test successful.")


    except ConnectionFailure:
        print("\n*** DB TEST FAILED: Could not connect to MongoDB. Check MONGO_URI and ensure DB is running. ***")
    except Exception as e:
        print(f"\n*** DB TEST FAILED: An error occurred: {e} ***")
        logger.exception("Detailed traceback:") # Log full traceback
    finally:
        # Disconnect
        print("\nClosing DB connection...")
        await db_handler.close_db()
        print("Connection closed.")
        print("\n--- Database Tests Finished ---")


if __name__ == "__main__":
    import asyncio
    # Run the async test function
    asyncio.run(run_db_tests())




'''import motor.motor_asyncio
from beanie import init_beanie
import logging

from .config import settings
from .models import Article # Import your models here

logger = logging.getLogger(__name__)

async def init_db():
    """
    Initializes the MongoDB database connection and the Beanie ODM.

    This function sets up the asynchronous database client and links
    the Beanie Document models (like `Article`) to the database.
    It's typically called once during application startup.

    Note: In the Beanie ODM pattern, most database operations (CRUD) are
    performed directly on the model classes (e.g., `await Article.find_one(...)`,
    `await article.save()`) rather than through functions in this file.
    This file's primary role is the initial setup and connection.
    """
    logger.info("Attempting to connect to MongoDB...")
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.mongodb_connection_string
        )
        # Optional: Verify connection early. Can add latency to startup.
        # await client.admin.command('ping')
        # logger.info("MongoDB server ping successful.")

        db = client[settings.database_name]

        await init_beanie(
            database=db,
            document_models=[
                Article,
                # Add other Beanie Document models here if you create more
            ]
        )
        logger.info(f"Successfully connected to MongoDB database: '{settings.database_name}' and initialized Beanie.")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB or initialize Beanie: {e}")
        # Depending on your app's requirements, you might want to raise the exception
        # or handle it differently (e.g., allow app to start but log error)
        raise'''