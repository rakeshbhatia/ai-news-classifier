# ai_news_classifier/app/main.py

import logging
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, Query, HTTPException, Depends, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import OperationFailure
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Import project modules
from .config import settings
from .database import db_handler, Database
from .models import PaginatedArticleResponse # Keep ArticleModel import if needed elsewhere
# Import necessary functions from processor and classifier
from .rss_processor import process_all_feeds, flatten_feed_config
# Import the ACTUAL functions needed from your classifier
from .ai_classifier import NLP_MODEL, load_spacy_model_and_matchers, classify_pending_articles_in_db

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- APScheduler Setup ---
scheduler = AsyncIOScheduler(timezone="UTC")

# --- Job Functions (Wrappers) ---
async def run_rss_processing_job():
    """Wrapper function for the RSS processing job."""
    logger.info("Scheduler: Triggering RSS feed processing job.")
    if db_handler.client is None or db_handler.articles_collection is None:
         logger.error("Scheduler: Cannot run RSS job, DB not connected.")
         return
    try:
        feed_list = flatten_feed_config(settings.rss_feeds)
        if feed_list:
            # Make sure the correct csv filename is used
            await process_all_feeds(db=db_handler, feed_list=feed_list, csv_filename="output_articles_added.csv")
        else:
            logger.warning("Scheduler: No feeds configured to process.")
    except Exception as e:
        logger.exception(f"Scheduler: Error during scheduled RSS processing: {e}")

async def run_classification_job():
    """Wrapper function for the classification job."""
    logger.info("Scheduler: Triggering article classification job.")
    if db_handler.client is None or db_handler.articles_collection is None:
         logger.error("Scheduler: Cannot run classification job, DB not connected.")
         return
    try:
        # Call the new batch classification function
        await classify_pending_articles_in_db(db=db_handler, limit=settings.CLASSIFICATION_BATCH_SIZE)
    except Exception as e:
        logger.exception(f"Scheduler: Error during scheduled classification: {e}")


# --- Database, spaCy Model & Scheduler Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan: DB connection, spaCy model load, Scheduler start/stop.
    """
    # 1. Connect DB
    logger.info("Application startup: Connecting to database...")
    await db_handler.connect_db()

    # 2. Load spaCy Model (Must happen before scheduler starts jobs using it)
    logger.info("Application startup: Loading spaCy model and matchers...")
    try:
        load_spacy_model_and_matchers() # This initializes the global variables in ai_classifier
        logger.info("spaCy model and matchers loaded successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load spaCy model during startup: {e}", exc_info=True)
        # Decide how to handle: maybe prevent app startup? For now, log critical error.
        raise # Or re-raise to potentially stop FastAPI startup

    # 3. Start Scheduler and Add Jobs
    logger.info("Application startup: Starting scheduler and adding jobs...")
    try:
        scheduler.add_job(
            run_rss_processing_job,
            trigger=IntervalTrigger(minutes=settings.RSS_FETCH_INTERVAL_MINUTES, jitter=60), # Add jitter
            id="rss_processing_job", name="Periodic RSS Feed Processing", replace_existing=True
        )
        scheduler.add_job(
            run_classification_job,
            trigger=IntervalTrigger(minutes=settings.CLASSIFICATION_INTERVAL_MINUTES, jitter=60), # Add jitter
            id="classification_job", name="Periodic Article Classification", replace_existing=True
        )
        scheduler.start()
        logger.info(f"Scheduler started with jobs: {scheduler.get_jobs()}")
    except Exception as e:
         logger.error(f"CRITICAL: Failed to start scheduler or add jobs: {e}", exc_info=True)
         # Consider implications if scheduler fails to start

    yield # Application runs here

    # --- Shutdown ---
    logger.info("Application shutdown: Shutting down scheduler...")
    if scheduler.running:
        scheduler.shutdown()
    logger.info("Application shutdown: Closing database connection...")
    await db_handler.close_db()
    logger.info("Database connection closed.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI News Classifier API",
    description="Fetches, stores, classifies, and serves RSS feed articles.",
    version="0.1.0",
    lifespan=lifespan # Use the updated lifespan manager
)

# --- CORS Middleware ---
# (Keep your existing CORS middleware setup here)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Routers ---
articles_router = APIRouter(prefix="/api/v1/articles", tags=["Articles"])
processing_router = APIRouter(prefix="/api/v1/processing", tags=["Processing"])


# --- Endpoints ---

# Health Check (Keep as before, maybe add spaCy model status)
@app.get("/health", tags=["Health"])
async def health_check():
    #from .ai_classifier import NLP_MODEL # Check the global variable state
    scheduler_running = scheduler.running if scheduler else False
    spacy_loaded = NLP_MODEL is not None
    db_ok = False
    if db_handler.client:
        try: await db_handler.client.admin.command('ping'); db_ok = True
        except Exception: db_ok = False
    return {
        "status": "ok",
        "database_connected": db_ok,
        "scheduler_running": scheduler_running,
        "spacy_model_loaded": spacy_loaded
    }

# Get Articles Endpoint (Keep as before on articles_router)
@articles_router.get("/", response_model=PaginatedArticleResponse)
async def get_articles(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
    # TODO: Add is_ai: Optional[bool] = Query(None) parameter
):
    if db_handler.articles_collection is None:
        raise HTTPException(status_code=503, detail="Database service not available")

    # --- TODO: Implement filtering based on query params ---
    filter_criteria = {}
    # Example:
    # if is_ai is not None:
    #    filter_criteria["isAIRelated"] = is_ai # Check alias if used

    try:
        paginated_response = await db_handler.get_articles_paginated(
            page=page, page_size=page_size, filter_criteria=filter_criteria
        )
        return paginated_response
    except OperationFailure as e:
        logger.error(f"DB Error getting articles: {e.details}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query error")
    except Exception as e:
        logger.error(f"Unexpected Error getting articles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# --- Processing Endpoints (Updated to use correct job wrappers) ---

@processing_router.post("/trigger-rss", status_code=202)
async def trigger_rss_processing_endpoint():
    """Manually triggers an immediate run of the RSS feed processing job."""
    logger.info("API request to trigger RSS feed processing.")
    if not scheduler.running:
         raise HTTPException(status_code=503, detail="Scheduler is not running.")
    try:
        scheduler.add_job(
            run_rss_processing_job, # Use the correct wrapper
            id="manual_rss_trigger", name="Manual RSS Trigger", replace_existing=True,
            next_run_time=datetime.now(timezone.utc)
        )
        logger.info("Manual RSS processing job triggered.")
        return {"message": "RSS feed processing job triggered successfully."}
    except Exception as e:
         logger.error(f"Failed to trigger manual RSS job: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Failed to trigger RSS processing job.")


@processing_router.post("/trigger-classification", status_code=202)
async def trigger_classification_endpoint():
    """Manually triggers an immediate run of the article classification job."""
    logger.info("API request to trigger article classification.")
    if not scheduler.running:
         raise HTTPException(status_code=503, detail="Scheduler is not running.")
    try:
        scheduler.add_job(
            run_classification_job, # Use the correct wrapper
            id="manual_classification_trigger", name="Manual Classification Trigger", replace_existing=True,
            next_run_time=datetime.now(timezone.utc)
        )
        logger.info("Manual classification job triggered.")
        return {"message": "Article classification job triggered successfully."}
    except Exception as e:
         logger.error(f"Failed to trigger manual classification job: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Failed to trigger classification job.")


# --- Include Routers in the App ---
app.include_router(articles_router)
app.include_router(processing_router)

# --- Root Endpoint (Keep as before) ---
@app.get("/", include_in_schema=False)
async def root():
    return {"message": f"Welcome to the {app.title}!"}

# --- Run Command Reminder ---
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000



# ai_news_classifier/app/main.py

'''import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, HTTPException, Depends, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import OperationFailure # To catch specific DB errors

# Import project modules
from .config import settings
from .database import db_handler, Database # Import the handler instance and class for type hinting
from .models import PaginatedArticleResponse, ArticleModel # Import response and base models
from .rss_processor import process_all_feeds, flatten_feed_config # For potential background tasks later
# from .ai_classifier import classify_articles # Placeholder for classifier task

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Database Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle application lifespan events:
    - Connect to database on startup.
    - Close database connection on shutdown.
    """
    logger.info("Application startup: Connecting to database...")
    try:
        await db_handler.connect_db()
        yield # Application runs here
    finally:
        logger.info("Application shutdown: Closing database connection...")
        await db_handler.close_db()
        logger.info("Database connection closed.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI News Classifier API",
    description="Fetches, stores, classifies, and serves RSS feed articles.",
    version="0.1.0",
    lifespan=lifespan # Use the lifespan context manager
)

# --- CORS Middleware ---
# Configure CORS to allow requests from your Svelte frontend
# Adjust origins as needed for development and production
origins = [
    "http://localhost:5173",  # Default Svelte dev server port
    "http://127.0.0.1:5173",
    # Add your production frontend URL here eventually
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True, # Allow cookies/auth headers
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)

# --- API Routers ---
# Using routers helps organize endpoints, especially as the app grows.
# Prefixing with /api/v1 for versioning.
articles_router = APIRouter(prefix="/api/v1/articles", tags=["Articles"])
# processing_router = APIRouter(prefix="/api/v1/processing", tags=["Processing"]) # Example for future


# --- Endpoints ---

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Simple health check endpoint.
    """
    # Could add a basic DB check here if needed, e.g., try pinging
    # try:
    #     await db_handler.client.admin.command('ping') # Requires auth
    # except Exception:
    #     raise HTTPException(status_code=503, detail="Database connection failed")
    return {"status": "ok"}


@articles_router.get("/", response_model=PaginatedArticleResponse)
async def get_articles(
    page: int = Query(1, ge=1, description="Page number to retrieve (1-indexed)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of articles per page (1-100)")
    # --- Placeholder for future filtering ---
    # is_ai: Optional[bool] = Query(None, description="Filter by AI classification status"),
    # search: Optional[str] = Query(None, description="Search term for title/description"),
    # Add more filters here (e.g., source, date range)
):
    """
    Retrieve articles from the database with pagination.

    Allows fetching articles in chunks (pages). Future enhancements will
    allow filtering by AI classification, search terms, date ranges, etc.
    """
    if db_handler.articles_collection is None:
        # This check should ideally not be needed if lifespan manages connection,
        # but serves as a safeguard.
        logger.error("get_articles endpoint called but database collection is not available.")
        raise HTTPException(status_code=503, detail="Database service not available")

    # --- Construct filter dictionary based on future query parameters ---
    filter_criteria = {}
    # if is_ai is not None:
    #     filter_criteria["isAIRelated"] = is_ai # Use correct field name/alias
    # if search:
    #     filter_criteria["$or"] = [
    #         {"title": {"$regex": search, "$options": "i"}},
    #         {"description": {"$regex": search, "$options": "i"}}
    #     ]
    # Add more filter logic here...

    try:
        logger.info(f"Fetching articles: page={page}, page_size={page_size}, filter={filter_criteria}")
        paginated_response = await db_handler.get_articles_paginated(
            page=page,
            page_size=page_size,
            filter_criteria=filter_criteria # Pass the constructed filter
        )
        logger.info(f"Found {paginated_response.total_count} total articles matching filter. Returning {len(paginated_response.articles)} for page {page}.")
        return paginated_response
    except OperationFailure as e:
        logger.error(f"Database operation failed during article retrieval: {e.details}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database query error: {e.details.get('errmsg', 'Unknown DB error')}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during article retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving articles.")


# --- Potential Future Endpoint for Triggering RSS Processing ---
# @processing_router.post("/trigger-rss", status_code=202, tags=["Processing"])
# async def trigger_rss_processing(background_tasks: BackgroundTasks):
#     """
#     Triggers the background task to fetch and process all configured RSS feeds.
#     Returns immediately with 202 Accepted.
#     """
#     logger.info("Received request to trigger RSS feed processing.")
#     feed_list_to_process = flatten_feed_config(settings.rss_feeds)
#     if not feed_list_to_process:
#          logger.warning("Cannot trigger RSS processing: No processable feed URLs found in configuration.")
#          raise HTTPException(status_code=400, detail="No valid feed URLs configured.")
#
#     # Add the long-running task to the background
#     background_tasks.add_task(
#         process_all_feeds,
#         db=db_handler, # Pass the connected db_handler
#         feed_list=feed_list_to_process,
#         csv_filename="output_headlines.csv" # Or configure filename
#     )
#     logger.info("RSS feed processing task added to background.")
#     return {"message": "RSS feed processing started in the background."}


# --- Include Routers in the App ---
app.include_router(articles_router)
# app.include_router(processing_router) # Include future routers here

# --- Root Endpoint (Optional) ---
@app.get("/", include_in_schema=False) # Hide from OpenAPI docs
async def root():
    return {"message": f"Welcome to the {app.title}!"}


# Note: To run this app, use uvicorn:
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000'''




'''import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from typing import List, Optional

from .config import settings # Load config
from .database import init_db # DB initializer
from .models import Article # DB Model
from .rss_processor import fetch_and_process_feeds # RSS logic
from .ai_classifier import load_spacy_model_and_matchers, classify_ai_text # Classifier logic

# Configure logging
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

# --- App State ---
# Store loaded NLP components in app state to avoid reloading
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    logger.info("Application startup...")
    # Initialize Database Connection
    await init_db()
    # Load NLP Model and Matchers
    logger.info("Loading NLP model and matchers...")
    try:
        nlp, ph_matcher, matcher = load_spacy_model_and_matchers()
        app_state["nlp_model"] = nlp
        app_state["phrase_matcher"] = ph_matcher
        app_state["matcher"] = matcher
        logger.info("NLP components loaded successfully.")
    except RuntimeError as e:
         logger.critical(f"Failed to initialize NLP components during startup: {e}")
         # Decide if the app should fail to start or run without classification ability
         # raise e # Option: Stop app startup
         app_state["nlp_model"] = None # Option: Continue without NLP
    yield
    # --- Shutdown ---
    logger.info("Application shutdown...")
    app_state.clear()
    # Add any other cleanup tasks here (e.g., closing external connections)

# Create FastAPI app instance with lifespan context manager
app = FastAPI(
    title="AI News Aggregator API",
    description="Fetches RSS feeds, classifies articles for AI relevance, and stores them.",
    version="0.1.0",
    lifespan=lifespan # Use the lifespan manager
)


# --- Background Task Functions ---

async def run_classification_task():
    """Fetches unclassified articles and classifies them."""
    logger.info("Starting background classification task...")
    if not app_state.get("nlp_model"):
        logger.error("NLP model not loaded. Skipping classification task.")
        return

    nlp = app_state["nlp_model"]
    ph_matcher = app_state["phrase_matcher"]
    matcher = app_state["matcher"]

    processed_count = 0
    try:
        # Fetch articles that haven't been classified yet
        unclassified_articles = await Article.find(Article.is_ai_related == None).to_list()
        logger.info(f"Found {len(unclassified_articles)} articles to classify.")

        for article in unclassified_articles:
            # Combine title and summary for better classification context
            text_to_classify = f"{article.title or ''}. {article.summary or ''}"

            if not text_to_classify.strip() or text_to_classify == ". ":
                 logger.warning(f"Skipping classification for article with no content: {article.link}")
                 # Optionally set to False or leave as None
                 article.is_ai_related = False
                 article.classification_score = 0.0
            else:
                 is_ai, score, _ = classify_ai_text(
                     text=text_to_classify,
                     nlp_model=nlp,
                     phrase_matcher=ph_matcher,
                     matcher=matcher,
                     threshold=settings.classifier_threshold # Use configured threshold
                 )
                 article.is_ai_related = is_ai
                 article.classification_score = score

            await article.save() # Update the article in the database
            processed_count += 1
            # Optional: Add a small sleep to prevent overwhelming DB or CPU if needed
            # await asyncio.sleep(0.01)

        logger.info(f"Finished classification task. Processed {processed_count} articles.")

    except Exception as e:
        logger.error(f"Error during classification task: {e}", exc_info=True)


async def run_rss_fetch_task():
    """Runs the RSS fetching and processing logic."""
    logger.info("Starting background RSS fetch task...")
    try:
        await fetch_and_process_feeds(settings.rss_feeds)
        logger.info("Finished background RSS fetch task.")
    except Exception as e:
        logger.error(f"Error during RSS fetch task: {e}", exc_info=True)


# --- API Endpoints ---

@app.post("/tasks/trigger-rss-fetch", status_code=202)
async def trigger_rss_fetch(background_tasks: BackgroundTasks):
    """
    Triggers a background task to fetch and process RSS feeds.
    """
    logger.info("Received request to trigger RSS fetch.")
    background_tasks.add_task(run_rss_fetch_task)
    return {"message": "RSS feed fetching task accepted and running in the background."}

@app.post("/tasks/trigger-classification", status_code=202)
async def trigger_classification(background_tasks: BackgroundTasks):
    """
    Triggers a background task to classify unclassified articles in the database.
    """
    if not app_state.get("nlp_model"):
         raise HTTPException(status_code=503, detail="Classifier service unavailable (NLP model not loaded).")

    logger.info("Received request to trigger classification.")
    background_tasks.add_task(run_classification_task)
    return {"message": "Article classification task accepted and running in the background."}


@app.get("/articles", response_model=List[Article])
async def get_articles(
    ai_related: Optional[bool] = Query(None, description="Filter by AI relevance (true/false). Leave blank for all."),
    limit: int = Query(50, ge=1, le=200, description="Number of articles to return."),
    skip: int = Query(0, ge=0, description="Number of articles to skip (for pagination).")
):
    """
    Retrieves articles from the database, optionally filtered by AI relevance.
    Sorted by publication date descending (newest first).
    """
    query = {}
    if ai_related is not None:
        query["is_ai_related"] = ai_related

    try:
        articles = await Article.find(query)\
            .sort(-Article.published_datetime) \
            .skip(skip)\
            .limit(limit)\
            .to_list()
        return articles
    except Exception as e:
         logger.error(f"Error retrieving articles: {e}")
         raise HTTPException(status_code=500, detail="Could not retrieve articles from database.")


@app.get("/")
async def read_root():
    """Root endpoint providing basic API info."""
    return {
        "message": "Welcome to the AI News Aggregator API",
        "docs": "/docs",
        "redoc": "/redoc"
        }

# --- Main execution (for running with uvicorn directly) ---
# Note: Typically you'd run via `uvicorn app.main:app --reload`
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly (for debugging)...")
    # Note: --reload is not easily available when run this way.
    # Use `uvicorn app.main:app --reload --log-level debug` from terminal for development.
    uvicorn.run(app, host="0.0.0.0", port=8000)'''