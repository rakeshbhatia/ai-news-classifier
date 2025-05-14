# ai_news_classifier/app/main.py

import logging
from contextlib import asynccontextmanager
from typing import Optional
from datetime import date, datetime, timezone

from fastapi import FastAPI, Request, Query, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
# REMOVE BackgroundTasks if manual triggers add jobs to scheduler directly
# from fastapi import BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession # Import AsyncSession for dependency typing
from sqlalchemy.exc import SQLAlchemyError # Import specific exception

# --- APScheduler Imports ---
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# --- Project Imports ---
from .config import settings
# Import SQLAlchemy specific components from database.py
from .database import async_engine, async_session_factory, get_db_session, get_articles_paginated
from .models import PaginatedArticleResponse, ArticleModel # API Models
# Import functions needed for jobs and startup
from .rss_processor import process_all_feeds, flatten_feed_config
from .ai_classifier import load_spacy_model_and_matchers, classify_pending_articles_in_db

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- APScheduler Setup ---
scheduler = AsyncIOScheduler(timezone="UTC")

# --- Job Functions (Wrappers using session factory) ---
async def run_rss_processing_job():
    """Wrapper for RSS processing job - manages its own session."""
    logger.info("Scheduler: Triggering RSS feed processing job.")
    if async_session_factory is None:
         logger.error("Scheduler: Cannot run RSS job, session factory not available.")
         return
    try:
        feed_list = flatten_feed_config(settings.rss_feeds)
        if feed_list:
            # Pass the factory, not a session
            await process_all_feeds(
                session_factory=async_session_factory,
                feed_list=feed_list,
                csv_filename=settings.CSV_OUTPUT_FILENAME
            )
        else:
            logger.warning("Scheduler: No feeds configured to process.")
    except Exception as e:
        logger.exception(f"Scheduler: Error during scheduled RSS processing: {e}")

async def run_classification_job():
    """Wrapper for classification job - manages its own session."""
    logger.info("Scheduler: Triggering article classification job.")
    if async_session_factory is None:
         logger.error("Scheduler: Cannot run classification job, session factory not available.")
         return
    try:
        # Pass the factory
        await classify_pending_articles_in_db(
            session_factory=async_session_factory,
            limit=settings.CLASSIFICATION_BATCH_SIZE
        )
    except Exception as e:
        logger.exception(f"Scheduler: Error during scheduled classification: {e}")


# --- Lifespan Management (SQLAlchemy Engine, spaCy, Scheduler) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Store state on app instance if needed elsewhere (optional)
    app.state.db_engine = None
    app.state.session_factory = None
    app.state.spacy_loaded = False

    try:
        # 1. Setup DB Engine & Session Factory (use existing from database.py)
        if async_engine is None or async_session_factory is None:
             raise RuntimeError("Database engine/session factory failed to initialize.")
        app.state.db_engine = async_engine
        app.state.session_factory = async_session_factory
        logger.info("Database engine and session factory are ready.")

        # 2. Load spaCy Model
        logger.info("Application startup: Loading spaCy model and matchers...")
        load_spacy_model_and_matchers() # Assume this handles errors internally or raises
        app.state.spacy_loaded = True
        logger.info("spaCy model and matchers loaded successfully.")

        # --- Initial Runs (Optional - uncomment if needed) ---
        '''logger.info("Application startup: Running initial RSS feed processing...")
        await run_rss_processing_job()
        logger.info("Application startup: Initial RSS processing complete.")
        logger.info("Application startup: Running initial article classification...")
        await run_classification_job()
        logger.info("Application startup: Initial article classification complete.")'''
        # --- End Initial Runs ---

        # 3. Start Scheduler and Add Interval Jobs
        logger.info("Application startup: Starting scheduler and adding interval jobs...")
        scheduler.add_job(
            run_rss_processing_job,
            trigger=IntervalTrigger(minutes=settings.RSS_FETCH_INTERVAL_MINUTES, jitter=60),
            id="rss_processing_job", name="Periodic RSS Feed Processing", replace_existing=True
        )
        scheduler.add_job(
            run_classification_job,
            trigger=IntervalTrigger(minutes=settings.CLASSIFICATION_INTERVAL_MINUTES, jitter=60),
            id="classification_job", name="Periodic Article Classification", replace_existing=True
        )
        scheduler.start()
        logger.info(f"Scheduler started with periodic jobs: {scheduler.get_jobs()}")

    except Exception as startup_error:
         logger.critical(f"Application startup failed critically: {startup_error}", exc_info=True)
         # Ensure scheduler is stopped if it partially started
         if scheduler.running: scheduler.shutdown(wait=False)
         # Engine disposal happens in finally block
         raise # Re-raise to stop FastAPI

    # Yield control to the running application
    yield # App runs here

    # --- Shutdown ---
    logger.info("Application shutdown: Shutting down scheduler...")
    if scheduler.running:
        scheduler.shutdown()
    logger.info("Application shutdown: Disposing database engine...")
    if app.state.db_engine: # Use engine stored in state
        await app.state.db_engine.dispose()
    logger.info("Database engine disposed.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI News Classifier API (PostgreSQL)",
    description="Fetches, stores, classifies, and serves RSS feed articles using PostgreSQL.",
    version="0.2.0", # Bump version
    lifespan=lifespan
)

# --- CORS Middleware ---
origins = ["http://localhost:5173", "http://127.0.0.1:5173"] # Add others as needed
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API Routers ---
articles_router = APIRouter(prefix="/api/v1/articles", tags=["Articles"])
processing_router = APIRouter(prefix="/api/v1/processing", tags=["Processing"])

# --- Endpoints ---
# main.py - Health Check
@app.get("/health", tags=["Health"])
async def health_check(request: Request):
    scheduler_running = scheduler.running if scheduler else False
    spacy_loaded = request.app.state.spacy_loaded if hasattr(request.app.state, 'spacy_loaded') else False
    db_ok = False
    db_error_message = "No attempt made or factory not available" # Default message

    if request.app.state.session_factory:
        try:
            async with request.app.state.session_factory() as session:
                result = await session.execute(select(1))
                if result.scalar_one() == 1: # Ensure the query actually returned something
                    db_ok = True
                    db_error_message = "Successfully executed SELECT 1"
                else:
                    db_error_message = "SELECT 1 did not return expected result"
        except Exception as e:
            # Log the specific error encountered by the health check
            logger.error(f"Health check DB connection/query failed: {e}", exc_info=True)
            db_ok = False
            db_error_message = str(e) # Store the error message

    return {
        "status": "ok",
        "database_connected": db_ok,
        "database_check_message": db_error_message, # Add more detail
        "scheduler_running": scheduler_running,
        "spacy_model_loaded": spacy_loaded
    }


# Health Check Updated
'''@app.get("/health", tags=["Health"])
async def health_check(request: Request): # Inject Request to access app state
    # Check status from app state
    scheduler_running = scheduler.running if scheduler else False
    spacy_loaded = request.app.state.spacy_loaded if hasattr(request.app.state, 'spacy_loaded') else False
    db_ok = False
    # Check DB via session factory
    if request.app.state.session_factory:
        try:
            async with request.app.state.session_factory() as session:
                 # Use a simple query instead of ping command
                 await session.execute(select(1))
                 db_ok = True
        except Exception:
            db_ok = False
    return {
        "status": "ok",
        "database_connected": db_ok,
        "scheduler_running": scheduler_running,
        "spacy_model_loaded": spacy_loaded
    }'''


@articles_router.get("/", response_model=PaginatedArticleResponse)
async def get_articles_endpoint(
    page: int = Query(1, ge=1, description="Page number to retrieve (1-indexed)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of articles per page (1-100)"),
    # --- New Filter Parameters ---
    search: Optional[str] = Query(None, min_length=3, max_length=100, description="Search term for title and description"),
    source: Optional[str] = Query(None, max_length=100, description="Filter by news outlet (exact match, case-insensitive)"),
    category: Optional[str] = Query(None, max_length=100, description="Filter by feed category (exact match, case-insensitive)"),
    date_from: Optional[date] = Query(None, description="Filter articles published on or after this date (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="Filter articles published on or before this date (YYYY-MM-DD)"),
    # --- End New Filter Parameters ---
    db: AsyncSession = Depends(get_db_session)
):
    """
    Retrieve ONLY AI-related articles from the database with pagination
    and additional filtering options.
    """
    filter_criteria = {"is_ai_related": True} # Base filter

    if search:
        filter_criteria["search"] = search.strip()
    if source:
        filter_criteria["news_outlet"] = source.strip() # Pass as "news_outlet" to match ORM
    if category:
        filter_criteria["feed_category"] = category.strip() # Pass as "feed_category"
    if date_from:
        filter_criteria["pub_date_from"] = date_from # Use distinct key for "from"
    if date_to:
        # To make date_to inclusive, we might need to adjust it to end of day in database.py
        filter_criteria["pub_date_to"] = date_to # Use distinct key for "to"

    try:
        paginated_response = await get_articles_paginated(
            session=db,
            page=page,
            page_size=page_size,
            filter_criteria=filter_criteria
        )
        return paginated_response
    except SQLAlchemyError as e:
        logger.error(f"DB Error getting articles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query error")
    except Exception as e:
        logger.error(f"Unexpected Error getting articles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# Get Articles Endpoint (Uses Dependency Injection)
'''@articles_router.get("/", response_model=PaginatedArticleResponse)
async def get_articles_endpoint( # Renamed endpoint function for clarity
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    # session: AsyncSession = Depends(get_db_session) # Inject session
    db: AsyncSession = Depends(get_db_session) # Use 'db' consistent with tutorial examples
    # TODO: Add filtering params
):
    """Retrieve articles from the database with pagination."""
    # --- TODO: Implement filtering based on query params ---
    filter_criteria = {}
    # Example:
    # if is_ai is not None:
    #    filter_criteria["is_ai_related"] = is_ai # Match field name

    try:
        # Call the updated database function, passing the session
        paginated_response = await get_articles_paginated(
            session=db, # Pass the injected session
            page=page,
            page_size=page_size,
            filter_criteria=filter_criteria
        )
        return paginated_response
    except SQLAlchemyError as e:
        logger.error(f"DB Error getting articles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query error")
    except Exception as e:
        logger.error(f"Unexpected Error getting articles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")'''


# --- Processing Endpoints (No change needed here, they use the wrappers) ---
@processing_router.post("/trigger-rss", status_code=202)
async def trigger_rss_processing_endpoint():
     logger.info("API request to trigger RSS feed processing.")
     if not scheduler.running: raise HTTPException(status_code=503, detail="Scheduler is not running.")
     try:
         scheduler.add_job(run_rss_processing_job, id="manual_rss_trigger", name="Manual RSS Trigger", replace_existing=True, next_run_time=datetime.now(timezone.utc))
         logger.info("Manual RSS processing job triggered.")
         return {"message": "RSS feed processing job triggered successfully."}
     except Exception as e:
         logger.error(f"Failed to trigger manual RSS job: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Failed to trigger RSS processing job.")


@processing_router.post("/trigger-classification", status_code=202)
async def trigger_classification_endpoint():
    logger.info("API request to trigger article classification.")
    if not scheduler.running: raise HTTPException(status_code=503, detail="Scheduler is not running.")
    try:
        scheduler.add_job(run_classification_job, id="manual_classification_trigger", name="Manual Classification Trigger", replace_existing=True, next_run_time=datetime.now(timezone.utc))
        logger.info("Manual classification job triggered.")
        return {"message": "Article classification job triggered successfully."}
    except Exception as e:
         logger.error(f"Failed to trigger manual classification job: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Failed to trigger classification job.")


# --- Include Routers ---
app.include_router(articles_router)
app.include_router(processing_router)

# --- Root Endpoint ---
@app.get("/", include_in_schema=False)
async def root():
    return {"message": f"Welcome to the {app.title}!"}

# --- Run Command ---
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000