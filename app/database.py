# ai_news_classifier/app/database.py

import uuid
import logging
from typing import AsyncGenerator, Optional, Dict, Any, List
from datetime import datetime, time, timezone # Import time for date_to adjustment

import psycopg2
from sqlalchemy import select, func, desc, or_
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.pool import NullPool

from pydantic import HttpUrl
from asyncpg import Connection

from .config import settings
from .models import Base, Article, PaginatedArticleResponse, ArticleModel # Import SQLAlchemy and Pydantic models

logger = logging.getLogger(__name__)


# --- SQLAlchemy Setup ---
try:
    connect_args = {
        "server_settings": {"statement_cache_size": "0"} # Use STRING "0"
    }
    async_engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DB_ECHO_SQL,
        pool_pre_ping=True,
        connect_args=connect_args
    )

    async_session_factory = async_sessionmaker(
        bind=async_engine,
        expire_on_commit=False,
        class_=AsyncSession
    )
    logger.info("SQLAlchemy async engine and session factory created successfully.")
except Exception as e:
     logger.critical(f"Failed to create SQLAlchemy engine or session factory: {e}", exc_info=True)
     async_engine = None
     async_session_factory = None


# --- Dependency for FastAPI Routes ---
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get a database session for API requests."""
    if async_session_factory is None:
        logger.error("Database session factory is not available.")
        # This indicates a critical startup failure
        raise RuntimeError("Database not configured correctly.")

    async with async_session_factory() as session:
        try:
            yield session
            # Note: commit/rollback typically handled within endpoint logic if needed
        except SQLAlchemyError as e:
            logger.error(f"Database session error: {e}", exc_info=True)
            # Depending on error, might await session.rollback() here,
            # but often better handled where the transaction occurs.
            raise # Re-raise to be caught by FastAPI error handlers
        # Session is automatically closed by the context manager


# --- Database Operation Functions ---

async def get_articles_paginated(
    session: AsyncSession,
    page: int = 1,
    page_size: int = 10,
    filter_criteria: Optional[Dict[str, Any]] = None
) -> PaginatedArticleResponse:
    offset = (page - 1) * page_size

    query = select(Article)
    count_query = select(func.count()).select_from(Article)

    # --- Apply Filtering ---
    if filter_criteria:
        # is_ai_related filter (base filter from endpoint)
        if "is_ai_related" in filter_criteria and filter_criteria["is_ai_related"] is not None:
            ai_filter_value = filter_criteria["is_ai_related"]
            query = query.where(Article.is_ai_related == ai_filter_value)
            count_query = count_query.where(Article.is_ai_related == ai_filter_value)

        # Search filter (title and description)
        if "search" in filter_criteria:
            search_term = filter_criteria["search"]
            # Case-insensitive search using ilike
            search_pattern = f"%{search_term}%"
            query = query.where(
                or_(
                    Article.title.ilike(search_pattern),
                    Article.description.ilike(search_pattern)
                    # Add Article.content.ilike(search_pattern) if content is stored and searchable
                )
            )
            count_query = count_query.where(
                or_(
                    Article.title.ilike(search_pattern),
                    Article.description.ilike(search_pattern)
                )
            )

        # Source (news_outlet) filter
        if "news_outlet" in filter_criteria:
            source_value = filter_criteria["news_outlet"]
            # Case-insensitive comparison for news_outlet
            query = query.where(func.lower(Article.news_outlet) == func.lower(source_value))
            count_query = count_query.where(func.lower(Article.news_outlet) == func.lower(source_value))

        # Category (feed_category) filter
        if "feed_category" in filter_criteria:
            category_value = filter_criteria["feed_category"]
            # Case-insensitive comparison for feed_category
            query = query.where(func.lower(Article.feed_category) == func.lower(category_value))
            count_query = count_query.where(func.lower(Article.feed_category) == func.lower(category_value))

        # Date range filters
        if "pub_date_from" in filter_criteria:
            date_from_value = filter_criteria["pub_date_from"]
            # Convert date to datetime at the start of the day
            datetime_from = datetime.combine(date_from_value, time.min, tzinfo=timezone.utc)
            query = query.where(Article.pub_date >= datetime_from)
            count_query = count_query.where(Article.pub_date >= datetime_from)

        if "pub_date_to" in filter_criteria:
            date_to_value = filter_criteria["pub_date_to"]
            # Convert date to datetime at the end of the day for inclusive range
            datetime_to = datetime.combine(date_to_value, time.max, tzinfo=timezone.utc)
            query = query.where(Article.pub_date <= datetime_to)
            count_query = count_query.where(Article.pub_date <= datetime_to)

    # Apply Ordering, Offset, Limit
    query = query.order_by(desc(Article.pub_date)).offset(offset).limit(page_size)

    try:
        total_count_result = await session.execute(count_query)
        total_count = total_count_result.scalar_one()

        results = await session.execute(query)
        articles_db = results.scalars().all()

        articles_api = [ArticleModel.model_validate(article) for article in articles_db]

        has_more = (page * page_size) < total_count

        return PaginatedArticleResponse(
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_more=has_more,
            articles=articles_api
        )
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving articles with filter {filter_criteria}: {e}", exc_info=True)
        raise


'''async def get_articles_paginated(
    session: AsyncSession,
    page: int = 1,
    page_size: int = 10,
    filter_criteria: Optional[Dict[str, Any]] = None # Will need SQL filtering later
) -> PaginatedArticleResponse:
    """
    Fetches articles from the database with pagination using SQLAlchemy.
    """
    offset = (page - 1) * page_size

    # --- Build the Query ---
    query = select(Article)
    count_query = select(func.count()).select_from(Article)

    # --- Apply Filtering (Placeholder - Adapt as needed) ---
    # Example: Filtering by is_ai_related
    # if filter_criteria and 'is_ai_related' in filter_criteria:
    #     ai_filter = filter_criteria['is_ai_related']
    #     if ai_filter is not None: # Handle None separately if needed
    #         query = query.where(Article.is_ai_related == ai_filter)
    #         count_query = count_query.where(Article.is_ai_related == ai_filter)
    #     else: # Handle query for None/NULL
    #         query = query.where(Article.is_ai_related.is_(None))
    #         count_query = count_query.where(Article.is_ai_related.is_(None))
    # Add more complex filtering logic here based on filter_criteria dict

    # Apply Ordering, Offset, Limit
    query = query.order_by(desc(Article.pub_date)).offset(offset).limit(page_size)

    try:
        # Execute count query
        total_count_result = await session.execute(count_query)
        total_count = total_count_result.scalar_one()

        # Execute main data query
        results = await session.execute(query)
        articles_db = results.scalars().all()

        # Convert SQLAlchemy Article objects to Pydantic ArticleModel objects
        # This leverages the from_attributes=True (orm_mode) in ArticleModel
        articles_api = [ArticleModel.model_validate(article) for article in articles_db]

        has_more = (page * page_size) < total_count

        return PaginatedArticleResponse(
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_more=has_more,
            articles=articles_api
        )
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving articles: {e}", exc_info=True)
        # Consider specific exceptions if needed (e.g., OperationalError)
        raise # Re-raise for FastAPI to handle'''


async def create_article_if_not_exists(session: AsyncSession, article_data: Dict[str, Any]) -> Optional[Article]:
    link = article_data.get("link")
    title = article_data.get('title', 'N/A') # Get title for logging

    if not link:
        logger.warning(f"!!! Attempted to create article '{title}' with no link. Returning None.")
        print(f"DEBUG: No link for article '{title}'") # Temporary Print
        return None

    # Convert potential Pydantic HttpUrl types to strings for database operations
    if isinstance(link, HttpUrl): link = str(link); article_data["link"] = link
    image_url = article_data.get("image_url")
    if isinstance(image_url, HttpUrl): article_data["image_url"] = str(image_url)

    try:
        # Check if article with this link string already exists
        stmt_exists = select(Article.id).where(Article.link == link).limit(1)
        result_exists = await session.execute(stmt_exists)
        exists = result_exists.scalar_one_or_none()

        if exists is not None:
            logger.debug(f"Article already exists (link): {link}, ID: {exists}")
            return None # OK if it exists

        logger.debug(f"Attempting to create article '{title}' (Link: {link})") # Log attempt

        # Prepare ORM data
        orm_data = {k: v for k, v in article_data.items() if hasattr(Article, k)}
        orm_data.setdefault("is_ai_related", None)
        orm_data.setdefault("classification_confidence", None)

        # --- Point of potential failure without specific IntegrityError ---
        new_article = Article(**orm_data)
        session.add(new_article)
        logger.debug(f"Attempting commit for article '{title}'")
        await session.commit() # <<<----- If this fails with non-IntegrityError, it goes to SQLAlchemyError or Exception below
        logger.debug(f"Commit successful for article '{title}'")
        await session.refresh(new_article)
        logger.info(f"--> Inserted new article: '{new_article.title}' (ID: {new_article.id})") # Make success obvious
        print(f"DEBUG: Inserted article ID {new_article.id}") # Temporary Print
        return new_article
        # --- End potential failure section ---

    except IntegrityError as e:
        await session.rollback()
        logger.debug(f"Article already exists (caught IntegrityError for link): {link}")
        return None
    except SQLAlchemyError as e:
        await session.rollback()
        # Make error log more prominent
        logger.error(f"!!! SQLAlchemyError creating article '{title}': {e}", exc_info=True)
        print(f"DEBUG: SQLAlchemyError for article '{title}': {e}") # Temporary Print
        return None # Indicate error
    except Exception as e:
         await session.rollback()
         # Make error log more prominent
         logger.error(f"!!! Exception creating article '{title}': {e}", exc_info=True)
         print(f"DEBUG: Exception for article '{title}': {e}") # Temporary Print
         return None


async def create_db_and_tables():
    """Development utility: Creates all tables defined in Base metadata."""
    if async_engine is None:
        logger.error("Cannot create tables, async_engine is not initialized.")
        return
    async with async_engine.begin() as conn:
        logger.info("Creating database tables...")
        try:
            # Use this line if you want it to ONLY create tables if they don't exist
            await conn.run_sync(Base.metadata.create_all, checkfirst=True)
            # Use this line if you want it to drop existing tables first (destructive!)
            #await conn.run_sync(Base.metadata.drop_all) # DANGEROUS in production
            #await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully.")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}", exc_info=True)


'''async def run_db_tests():
    print("--- Running Database Tests (SQLAlchemy) ---")
    # 1. Settings Check: Correctly checks DATABASE_URL. Good.
    if not settings.DATABASE_URL or "localhost" in settings.DATABASE_URL:
         print("\nWARNING: Testing requires DATABASE_URL set to your Supabase test instance.")
         # return # Or proceed cautiously

    # 2. Table Existence Assumption: Correctly comments out automatic table creation. Good.
    # Ensure tables exist (Run create_db_and_tables() manually once first if needed)
    # await create_db_and_tables() # Use with EXTREME caution - drops tables!

    # 3. Session Factory Check: Correctly checks if the factory was initialized. Good.
    session_factory_local = async_session_factory # Use the configured factory
    if not session_factory_local:
        print("Session factory not initialized. Aborting tests.")
        return

    added_article = None # Initialize variable used later. Good.
    try:
        # 4. Session Context Manager: Correctly uses async with for session management. Good.
        async with session_factory_local() as session:
            print("\nTesting article creation...")
            # 5. Test Data Generation: Uses datetime.now() for unique title/link. Good for preventing collisions across test runs.
            test_data = {
                "title": f"SQLAlchemy Test {datetime.now()}",
                "link": f"http://test.com/sql/{datetime.now().timestamp()}", # Using timestamp ensures uniqueness
                "pub_date": datetime.now(timezone.utc), # Needs timezone import
                "news_outlet": "DB Test",
                "feed_category": "Test",
                "description": "Testing..."
                # Note: Other non-nullable fields in Article ORM model (like feed_category)
                # are provided. Author will use default.
            }
            # 6. Call create_article_if_not_exists: Correctly calls the function.
            added_article = await create_article_if_not_exists(session, test_data)
            # 7. Assertions for Creation: Checks article is not None and has an ID. Good.
            assert added_article is not None
            assert added_article.id is not None
            print(f"Article created successfully (ID: {added_article.id}).")

            print("\nTesting duplicate creation prevention...")
            # 8. Call create_article_if_not_exists (Duplicate): Correctly calls again with same data.
            duplicate_article = await create_article_if_not_exists(session, test_data) # Same link
            # 9. Assertion for Duplicate: Checks that None is returned. Good.
            assert duplicate_article is None
            print("Duplicate prevention successful.")

            print("\nTesting pagination...")
            # 10. Call get_articles_paginated: Correctly calls the function.
            page1_response = await get_articles_paginated(session, page=1, page_size=5)
            print(f"Page 1: Total={page1_response.total_count}, Count={len(page1_response.articles)}, HasMore={page1_response.has_more}")
            # 11. Assertions for Pagination (Page 1): Checks page size limit, checks instance type. Good.
            assert len(page1_response.articles) <= 5
            if page1_response.articles:
                # Ensure ArticleModel is imported for this check
                assert isinstance(page1_response.articles[0], ArticleModel) # Check Pydantic conversion

            # 12. Conditional Page 2 Test: Correctly checks if more pages exist before requesting page 2. Good.
            if page1_response.total_count > 5:
                 page2_response = await get_articles_paginated(session, page=2, page_size=5)
                 print(f"Page 2: Total={page2_response.total_count}, Count={len(page2_response.articles)}, HasMore={page2_response.has_more}")
                 # 13. Assertions for Pagination (Page 2): Checks page size limit. Good.
                 assert len(page2_response.articles) <= 5
            print("Pagination test completed.")

    # 14. Exception Handling: Catches general exceptions, prints message, logs traceback. Good.
    except Exception as e:
        print(f"\n*** DB TEST FAILED: {e} ***")
        # Ensure logger is imported and configured if used here
        # logger.exception("Detailed traceback:") # Requires logger import/config
        import traceback # Or just use traceback directly
        traceback.print_exc()
    # 15. Finally Block (Engine Disposal): Ensures engine connections are closed. Good.
    finally:
        # Ensure async_engine is imported/available in this scope
        if async_engine: await async_engine.dispose() # Close connections
        print("\n--- Database Tests Finished ---")


async def test_database_write():
    """Simple test function to verify database writes are working."""
    if async_session_factory is None:
        logger.error("Cannot test database, session factory not initialized.")
        return False
        
    import uuid
    test_id = str(uuid.uuid4())
    test_data = {
        "title": f"Database Write Test {test_id}",
        "link": f"http://test.example.com/{test_id}",
        "pub_date": datetime.now(timezone.utc),  # Changed from pubDate to pub_date
        "news_outlet": "Test",                   # Changed from newsOutlet
        "feed_category": "Test",                 # Changed from feedCategory
        "description": "Testing database write capability"
    }
    
    try:
        async with async_session_factory() as session:
            article = Article(**test_data)
            session.add(article)
            await session.commit()
            logger.info(f"Successfully wrote test article with ID: {article.id}")
            return True
    except Exception as e:
        logger.error(f"Failed to write test article: {e}", exc_info=True)
        return False'''


# 16. Main Execution Block: Correctly uses asyncio.run, includes prerequisite message. Good.
'''if __name__ == "__main__":
    import asyncio
    # Ensure datetime and timezone are imported for test_data generation
    from datetime import datetime, timezone
    # Ensure ArticleModel is imported for the assertion check inside run_db_tests
    #from .models import ArticleModel
    # Ensure logger is configured if logger.exception is used, or import traceback
    import logging
    logging.basicConfig(level=logging.INFO) # Basic config for logger
    logger = logging.getLogger(__name__) # Define logger if used in except block

    # Ensure tables exist in your test DB first!
    print("Ensure the 'articles' table exists in your Supabase DB before running tests.")
    # print("You can run 'python -m app.database create' IF you add command line parsing.")
    asyncio.run(run_db_tests())'''

# Example for standalone table creation (run once during setup/dev)
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(create_db_and_tables())


# Simplified main function for database.py
async def db_test_main():
    """Main function for database module standalone execution."""
    logger.info("Starting database module test...")
    
    if async_session_factory is None:
        logger.critical("Cannot run tests: Database session factory not initialized.")
        return
        
    try:
        # Test the database connection with a simple write operation
        db_test_result = await test_database_connection()
        if db_test_result:
            logger.info("Database test succeeded!")
        else:
            logger.error("Database test failed.")
    except Exception as e:
        logger.exception(f"An error occurred during database testing: {e}")
    finally:
        logger.info("Database testing finished.")

if __name__ == "__main__":
    # Ensure environment variables are loaded if using .env for config
    from dotenv import load_dotenv
    import asyncio
    from datetime import datetime, timezone
    
    load_dotenv()
    
    # Run the test main function
    asyncio.run(db_test_main())