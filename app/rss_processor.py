# ai_news_classifier/app/rss_processor.py

import asyncio
import csv
import logging
import os
import httpx
import feedparser
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from time import mktime
from pydantic import ValidationError, HttpUrl
import pandas as pd

# --- Import SQLAlchemy specific components ---
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from .database import async_session_factory, create_article_if_not_exists # Import session factory and create function
# Keep models import for extract_article_data potentially
from .models import ArticleModel # Keep for type hints if needed, but primary interaction is dicts now
from .config import settings

# --- Logging, constants, flatten_feed_config, fetch_feed_content, parse_feed_data remain the same ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
header = ['source', 'headline', 'url'] # Define header row
unique_field = 'url' # Define the fieldname used for deduplication

DEFAULT_HTTP_TIMEOUT = 15.0 # Timeout for fetching feeds in seconds

# --- Helper Function to Transform Config ---
def flatten_feed_config(nested_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms the nested feed configuration from settings into a flat list
    where each item represents a single feed URL with its metadata.
    """
    flat_list = []
    if not nested_config:
        logger.warning("Feed configuration in settings is empty or not found.")
        return flat_list

    for category_group in nested_config:
        category_name = category_group.get("name")
        if not category_name:
            logger.warning(f"Skipping category group due to missing 'name': {category_group}")
            continue

        sources = category_group.get("sources", [])
        if not isinstance(sources, list):
             logger.warning(f"Skipping category '{category_name}' due to invalid 'sources' format (expected list).")
             continue

        for source in sources:
            source_name = source.get("name")
            source_logo = source.get("logo") # This is likely a path, not a full URL
            urls = source.get("urls", [])

            if not source_name:
                logger.warning(f"Skipping source in category '{category_name}' due to missing 'name': {source}")
                continue
            if not isinstance(urls, list):
                 logger.warning(f"Skipping source '{source_name}' due to invalid 'urls' format (expected list).")
                 continue

            for url in urls:
                if url and isinstance(url, str): # Ensure URL is a non-empty string
                    flat_list.append({
                        "url": url.strip(),
                        "news_outlet": source_name.strip(),
                        "feed_category": category_name.strip(),
                        "news_outlet_logo": source_logo # Keep as string path/identifier
                    })
                else:
                     logger.warning(f"Skipping invalid URL entry '{url}' for source '{source_name}'.")

    logger.info(f"Successfully transformed {len(flat_list)} feed URLs from configuration.")
    return flat_list


async def fetch_feed_content(url: str, client: httpx.AsyncClient) -> Optional[str]:
    """Fetches raw content from a given URL asynchronously."""
    try:
        response = await client.get(url, timeout=DEFAULT_HTTP_TIMEOUT, follow_redirects=True)
        response.raise_for_status() # Raise exception for 4xx/5xx status codes
        content_type = response.headers.get("content-type", "").lower()
        if not ("xml" in content_type or "rss" in content_type or "atom" in content_type):
             logger.warning(f"Unexpected content type '{content_type}' for feed {url}. May not parse correctly.")
             # Proceeding anyway, feedparser might handle it
        return response.text
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching {url}: {e.response.status_code} {e.response.reason_phrase}")
    except httpx.RequestError as e:
        logger.error(f"Network error fetching {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
    return None


def parse_feed_data(feed_content: str, feed_url: str) -> Optional[feedparser.FeedParserDict]:
    """Parses feed content using feedparser."""
    try:
        parsed_feed = feedparser.parse(feed_content)
        if parsed_feed.bozo:
            bozo_exception = parsed_feed.get('bozo_exception', 'Unknown parsing issue')
            logger.warning(f"Feed at {feed_url} might be ill-formed: {bozo_exception}")
        if not parsed_feed.entries:
            # This is common for feeds with temporary issues or empty feeds, log as info
            logger.info(f"No entries found in feed: {feed_url}")
            return None
        return parsed_feed
    except Exception as e:
        logger.error(f"Error parsing feed content from {feed_url}: {e}")
    return None


# --- extract_article_data remains the same ---
# It should return a dictionary compatible with create_article_if_not_exists
# IMPORTANT: Ensure ArticleModel's newsOutletLogo field accepts Optional[str]
# as the config provides paths like "/images/logos/...", not full URLs.
# If ArticleModel expects HttpUrl, Pydantic validation will fail here.
# Assuming ArticleModel.news_outlet_logo is Optional[str] based on config values.
def extract_article_data(entry: feedparser.FeedParserDict, feed_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extracts and maps data from a feedparser entry to a dictionary suitable for ArticleModel."""
    try:
        title = entry.get("title")
        link = entry.get("link")
        published_parsed = entry.get("published_parsed")

        if not title or not link:
            logger.debug(f"Skipping entry in {feed_config['url']} due to missing title or link.")
            return None

        if published_parsed:
            try:
                dt_obj = datetime.fromtimestamp(mktime(published_parsed), tz=timezone.utc)
            except (TypeError, ValueError) as e:
                 logger.warning(f"Could not parse date {published_parsed} for entry '{title}' from {feed_config['url']}. Using current time. Error: {e}")
                 dt_obj = datetime.now(timezone.utc)
        else:
            logger.debug(f"Missing publication date for entry '{title}' from {feed_config['url']}. Using current time.")
            dt_obj = datetime.now(timezone.utc)

        description = entry.get("summary", entry.get("description"))
        content_list = entry.get("content", [])
        content = content_list[0].get("value") if isinstance(content_list, list) and content_list and isinstance(content_list[0], dict) else None

        author = entry.get("author")
        image_url = None
        if "media_content" in entry:
            media = entry.media_content[0] if entry.media_content else {}
            image_url = media.get("url")
        elif "links" in entry:
            for l in entry.links:
                if "image" in l.get("type", ""):
                    image_url = l.get("href")
                    break

        # ... after extracting other fields ...
        categories_from_feed = None
        if hasattr(entry, 'tags') and entry.tags:
            # Get the 'term' from the first tag, if available
            first_tag = entry.tags[0]
            if isinstance(first_tag, dict) and 'term' in first_tag:
                # Get all tags
                categories_from_feed = ", ".join([tag.get('term') for tag in entry.tags if tag.get('term')])

        article_dict = {
            "title": title.strip(),
            "link": link.strip(),
            "pub_date": dt_obj,
            "category": categories_from_feed,
            "description": description.strip() if description else None,
            "content": content.strip() if content else None,
            "author": author.strip() if author else "Unknown",
            "news_outlet": feed_config["news_outlet"],
            "news_outlet_logo": feed_config.get("news_outlet_logo"), # Pass the string path/identifier
            "feed_category": feed_config["feed_category"],
            "image_url": image_url.strip() if image_url else None,
        }

        print(f"DEBUG: Extracted data: {article_dict}")

        return article_dict

    except Exception as e:
        entry_title = entry.get('title', 'N/A')
        logger.exception(f"Error processing entry '{entry_title}' from {feed_config['url']}: {e}") # Use logger.exception for traceback
        return None


async def process_single_feed(
    feed_config: Dict[str, Any],
    client: httpx.AsyncClient,
    # Session factory passed for background task session management
    session_factory: async_sessionmaker[AsyncSession]
) -> List[Dict[str, str]]:
    """
    Fetches, parses, and attempts to store articles for a single feed.
    Manages its own database session scope.
    Returns info for newly added articles.
    """
    feed_url = feed_config["url"]
    logger.info(f"Processing feed: {feed_url} ({feed_config['news_outlet']})")
    added_articles_info: List[Dict[str, str]] = []

    content = await fetch_feed_content(feed_url, client)
    if not content: return added_articles_info

    parsed_feed = parse_feed_data(content, feed_url)
    if not parsed_feed or not parsed_feed.entries: return added_articles_info

    logger.debug(f"Found {len(parsed_feed.entries)} entries in {feed_url}")

    added_count = 0
    existing_count = 0
    error_count = 0
    skipped_count = 0
    processed_count = 0

    # Create a session scope for this feed's processing
    async with session_factory() as session:
        for entry in parsed_feed.entries:
            processed_count += 1
            article_data = extract_article_data(entry, feed_config)
            if article_data:
                try:
                    # Use the function to create if not exists
                    # Pass the session and article data dictionary
                    created_article = await create_article_if_not_exists(session, article_data)

                    if created_article:
                        # Article was newly inserted
                        added_count += 1
                        added_articles_info.append({
                            "source": created_article.news_outlet,
                            "headline": created_article.title,
                            "author": created_article.author,
                            "description": created_article.description,
                            "content": created_article.content,
                            "url": created_article.link # URL is now string from DB model,
                        })
                    else:
                        # Article already existed
                        existing_count += 1
                except Exception as e:
                    error_count += 1
                    # logger.error(f"Error processing article '{article_data.get('title', 'N/A')}': {e}", exc_info=True)
                    # Make this error super visible
                    logger.error(f"!!!!! EXCEPTION caught in process_single_feed loop for article '{article_data.get('title', 'N/A')}': {e}", exc_info=True)
                    print(f"DEBUG !!!!! EXCEPTION in loop for '{article_data.get('title', 'N/A')}': {e}") # Temporary Print
            else:
                skipped_count += 1 # Extract failed

    # Logging outside the session block
    logger.info(f"Finished processing {feed_url}. Added: {added_count}, Existing: {existing_count}, Errors: {error_count}, Skipped/Invalid: {skipped_count}")
    return added_articles_info


# --- save_articles_info_to_csv using Pandas remains the same ---
# --- Audited pandas-based save_articles_info_to_csv ---
def save_articles_info_to_csv(articles_info: List[Dict[str, str]], filename: str = "rss_feed_data.csv"):
    """
    Appends information about newly processed articles (Source, Headline, URL)
    to a CSV file using pandas, avoiding duplicate entries based on the URL.
    Writes the header only if the file is new or empty.
    """
    # 0. Input Validation / Early Exit
    if not articles_info:
        logger.info("No new articles identified in this run, CSV file not modified.")
        return

    existing_urls = set()
    file_exists = os.path.exists(filename)
    # Determine initial state for header writing. Assume new/empty if it doesn't exist.
    is_empty_or_new = not file_exists

    # 1. Read existing URLs using pandas (if file exists)
    if file_exists:
        try:
            # Read only the URL column for efficiency
            df_existing = pd.read_csv(filename, usecols=[unique_field])
            if df_existing.empty:
                is_empty_or_new = True # File exists but is effectively empty
            else:
                # Get unique, non-null URLs as strings from the file
                existing_urls.update(df_existing[unique_field].dropna().astype(str).unique())
                is_empty_or_new = False # File exists and has content
            logger.debug(f"Read {len(existing_urls)} existing URLs from {filename}.")
        except FileNotFoundError:
             # This case is technically covered by os.path.exists, but handles race conditions
             is_empty_or_new = True
             logger.debug(f"CSV file '{filename}' not found during read attempt. Will create.")
        except (ValueError, KeyError) as e: # Catches empty file, missing URL col, other parse errors
            logger.warning(f"Could not parse existing CSV '{filename}' or find '{unique_field}' column: {e}. Assuming empty for header writing.")
            is_empty_or_new = True
        except Exception as e:
            logger.error(f"Error reading existing CSV '{filename}': {e}. Deduplication might be incomplete.", exc_info=True)
            # After unknown read error, stick with initial os.path.exists check for is_empty_or_new

    # 2. Create DataFrame for new articles and filter out duplicates
    df_new = pd.DataFrame(articles_info)

    # Check if the essential unique_field exists in the *new* data
    if unique_field not in df_new.columns:
        logger.error(f"Cannot perform deduplication: Unique field '{unique_field}' not found in new articles data. Appending all new data.")
        # Append without deduplication if URL is missing in the input dicts
        df_to_append = df_new
    else:
        # Make a copy to avoid SettingWithCopyWarning
        df_potential_append = df_new.copy()
        # Ensure URL column is string type for reliable comparison, handle None/NaN
        df_potential_append[unique_field] = df_potential_append[unique_field].astype(str)
        # Filter out rows with missing/empty/NaN-equivalent URLs *before* set comparison
        valid_url_mask = df_potential_append[unique_field].notna() & (df_potential_append[unique_field] != '') & (df_potential_append[unique_field].str.lower() != 'nan') & (df_potential_append[unique_field].str.lower() != 'none')
        df_potential_append = df_potential_append[valid_url_mask]

        # Filter out rows where the URL is already in the existing file's set
        new_mask = ~df_potential_append[unique_field].isin(existing_urls)
        df_to_append = df_potential_append[new_mask].copy()

        # Drop duplicates *within the new batch* itself, keeping the first occurrence
        # Check if df is not empty before dropping duplicates
        if not df_to_append.empty:
            df_to_append.drop_duplicates(subset=[unique_field], keep='first', inplace=True)

    if df_to_append.empty:
        logger.info("No new unique articles to append to CSV file after checking duplicates.")
        return

    # 3. Append the new unique articles to the CSV
    try:
        df_to_append.to_csv(
            filename,
            mode='a',                 # Append mode
            header=is_empty_or_new,   # Write header only if file is new/empty
            index=False,              # Don't write pandas index
            columns=header            # Ensure columns are written in the desired order
        )
        logger.info(f"Successfully appended info for {len(df_to_append)} new unique articles to {filename}")

    except IOError as e:
        logger.error(f"Failed to write article info to CSV file {filename}: {e}", exc_info=True)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during CSV writing: {e}")


# --- Updated process_all_feeds ---
async def process_all_feeds(
    # Removed 'db' argument, now uses session factory
    session_factory: async_sessionmaker[AsyncSession],
    feed_list: List[Dict[str, Any]],
    csv_filename: str = "rss_feed_data.csv"
):
    """
    Main orchestrator function to process all configured RSS feeds.
    Uses the provided session factory to manage sessions for sub-tasks.
    """
    if session_factory is None:
        logger.error("Session factory not available. Aborting feed processing.")
        return
    if not feed_list:
        logger.warning("Received an empty feed list to process. Aborting.")
        return

    all_new_articles_info: List[Dict[str, str]] = []
    total_feeds = len(feed_list)
    processed_feeds = 0
    failed_feeds = 0

    logger.info(f"Starting processing for {total_feeds} feed URLs...")

    async with httpx.AsyncClient() as client:
        # Pass the session factory to each task
        tasks = [process_single_feed(feed_config, client, session_factory) for feed_config in feed_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        feed_url = feed_list[i]['url']
        if isinstance(result, Exception):
            logger.error(f"An unexpected error occurred processing feed {feed_url}: {result}")
            failed_feeds += 1
        elif isinstance(result, list):
            # Extend the main list with the list of dicts from this feed
            all_new_articles_info.extend(result)
            processed_feeds += 1
        else:
            logger.error(f"Unexpected result type for feed {feed_url}: {type(result)}")
            failed_feeds += 1

    logger.info(f"--- Feed Processing Summary ---")
    logger.info(f"Total feed URLs processed: {processed_feeds + failed_feeds}/{total_feeds}")
    logger.info(f"Feeds completed successfully: {processed_feeds}")
    logger.info(f"Feeds failed entirely: {failed_feeds}")
    # Update log message to reflect article info count
    logger.info(f"Total new unique articles added across all feeds: {len(all_new_articles_info)}")

    # Call the renamed CSV function with the aggregated list of dictionaries
    save_articles_info_to_csv(all_new_articles_info, csv_filename)


# --- Standalone Execution Example (main function) ---
# This needs significant changes as it now relies on SQLAlchemy setup
async def main():
    """Main function for standalone execution - Primarily for setup/utility now."""
    logger.info("Starting standalone RSS processor execution...")
    logger.warning("Standalone mode may not have full application context (e.g., scheduler).")

    # We need the session factory for process_all_feeds
    if async_session_factory is None:
         logger.critical("Cannot run standalone: Database session factory not initialized.")
         return

    try:
        # Optionally create tables if in dev environment (USE WITH CAUTION)
        # await create_db_and_tables() # Uncomment carefully for initial setup

        logger.info("Loading and transforming feed configuration from settings...")
        feed_list_to_process = flatten_feed_config(settings.rss_feeds)

        if not feed_list_to_process:
            logger.warning("No processable feed URLs found in configuration. Exiting.")
            return

        # Run the feed processing, passing the session factory
        await process_all_feeds(
            session_factory=async_session_factory,
            feed_list=feed_list_to_process,
            csv_filename=settings.CSV_OUTPUT_FILENAME
        )

    except Exception as e:
        logger.exception(f"An error occurred during standalone execution: {e}")
    finally:
        # Clean up the engine if it was created (update: commented out due to error)
        # if async_engine:
        #     logger.info("Closing SQLAlchemy engine.")
        #     await async_engine.dispose()
        logger.info("Standalone RSS processor finished.")


if __name__ == "__main__":
    # Ensure environment variables are loaded if using .env for config
    from dotenv import load_dotenv
    import asyncio # Ensure asyncio is imported here if not already top-level
    # Make sure settings is imported if not already top-level in this file scope
    # (It should be imported at the top already)
    # from .config import settings

    load_dotenv()

    # --- CORRECTED CHECK: Use DATABASE_URL ---
    # Check if DATABASE_URL seems correctly set (basic check)
    if not settings.DATABASE_URL or "localhost" in settings.DATABASE_URL:
        print("\nWARNING: DATABASE_URL environment variable not set or points to localhost.")
        print("Ensure it's set in your .env file and points to your Supabase test instance.")
        # Depending on requirements, you might exit here or proceed cautiously
        # exit(1)

    # Important Check: Ensure the rss_feeds setting is actually loaded
    if not settings.rss_feeds:
         print("\nERROR: settings.rss_feeds is empty or not loaded correctly from config.py!")
         print("Please check your config.py and environment variables/defaults.")
         # exit(1) # Optionally exit if config is crucial

    # Run the async main function for standalone execution
    asyncio.run(main())
