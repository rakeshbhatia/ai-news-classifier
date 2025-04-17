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
import bson # Import bson to catch specific errors
import pandas as pd

from .database import db_handler, Database
from .models import ArticleModel
from .config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
header = ['Source', 'Headline', 'URL'] # Define header row
unique_field = 'URL' # Define the fieldname used for deduplication

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
                        "newsOutlet": source_name.strip(),
                        "feedCategory": category_name.strip(),
                        "newsOutletLogo": source_logo # Keep as string path/identifier
                    })
                else:
                     logger.warning(f"Skipping invalid URL entry '{url}' for source '{source_name}'.")

    logger.info(f"Successfully transformed {len(flat_list)} feed URLs from configuration.")
    return flat_list

# --- Functions fetch_feed_content, parse_feed_data remain the same ---
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


# --- Function extract_article_data remains mostly the same ---
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

        article_dict = {
            "title": title.strip(),
            "link": link.strip(),
            "pubDate": dt_obj,
            "description": description.strip() if description else None,
            "content": content.strip() if content else None,
            "author": author.strip() if author else "Unknown",
            "newsOutlet": feed_config["newsOutlet"],
            "newsOutletLogo": feed_config.get("newsOutletLogo"), # Pass the string path/identifier
            "feedCategory": feed_config["feedCategory"],
            "imageUrl": image_url.strip() if image_url else None,
        }
        return article_dict

    except Exception as e:
        entry_title = entry.get('title', 'N/A')
        logger.exception(f"Error processing entry '{entry_title}' from {feed_config['url']}: {e}") # Use logger.exception for traceback
        return None

# --- Updated store_article ---
async def store_article(article_data: Dict[str, Any], db: Database) -> Tuple[bool, Optional[Dict[str, str]]]:
    """
    Stores a single article in the database using upsert.
    Converts HttpUrl fields to strings before database operation.

    Returns:
        Tuple[bool, Optional[Dict[str, str]]]:
        - bool: True if article was newly inserted, False otherwise.
        - Optional[Dict]: If inserted, a dict containing {'source': ..., 'headline': ..., 'url': ...}, else None.
    """
    article_title = article_data.get("title", "Unknown Title")
    try:
        article = ArticleModel(**article_data)
        article_db_data = article.model_dump(by_alias=True, exclude={'id'})
        link_str = str(article.link) # Use string for query and potentially for return dict

        # Convert HttpUrl types to strings for BSON encoding
        link_key = ArticleModel.model_fields['link'].alias or 'link'
        if link_key in article_db_data and isinstance(article_db_data[link_key], HttpUrl):
             article_db_data[link_key] = str(article_db_data[link_key])
        img_url_key = ArticleModel.model_fields['image_url'].alias or 'imageUrl'
        if img_url_key in article_db_data and article_db_data[img_url_key] is not None and isinstance(article_db_data[img_url_key], HttpUrl):
             article_db_data[img_url_key] = str(article_db_data[img_url_key])
        logo_key = ArticleModel.model_fields['news_outlet_logo'].alias or 'newsOutletLogo'
        if logo_key in article_db_data and article_db_data[logo_key] is not None and isinstance(article_db_data[logo_key], HttpUrl):
            article_db_data[logo_key] = str(article_db_data[logo_key])

        if db.articles_collection is None:
             logger.error("Database collection not available. Cannot store article.")
             # Return False and None for the article info dict
             return False, None

        result = await db.articles_collection.update_one(
            {"link": link_str},
            {"$setOnInsert": article_db_data},
            upsert=True
        )

        if result.upserted_id:
            logger.info(f"Inserted new article: '{article_title}'")
            # Return True and the required info dict
            article_info = {
                "source": article.news_outlet, # Use field name from Pydantic model
                "headline": article.title,
                "url": link_str # Use the string version of the link
            }
            return True, article_info
        elif result.matched_count > 0:
            logger.debug(f"Article already exists (link match): '{article_title}'")
            return False, None
        else:
            logger.warning(f"Upsert operation reported no match and no upsert for: '{article_title}'")
            return False, None

    except ValidationError as e:
        logger.error(f"Data validation failed for article '{article_title}': {e}")
    except Exception as e:
        if isinstance(e, bson.errors.InvalidDocument):
             logger.exception(f"BSON Encoding Error storing article '{article_title}': {e} - Data: {article_db_data}")
        else:
             logger.exception(f"Database error storing article '{article_title}': {e}")

    # Return False and None in case of errors
    return False, None

# --- Updated process_single_feed ---
async def process_single_feed(feed_config: Dict[str, Any], client: httpx.AsyncClient, db: Database) -> List[Dict[str, str]]:
    """
    Fetches, parses, and stores articles for a single feed configuration.
    Returns a list of dictionaries, each containing info for newly added articles.
    """
    feed_url = feed_config["url"]
    logger.info(f"Processing feed: {feed_url} ({feed_config['newsOutlet']})")
    # Changed list name and type hint
    added_articles_info: List[Dict[str, str]] = []

    content = await fetch_feed_content(feed_url, client)
    if not content:
        return added_articles_info

    parsed_feed = parse_feed_data(content, feed_url)
    if not parsed_feed or not parsed_feed.entries:
        return added_articles_info

    logger.debug(f"Found {len(parsed_feed.entries)} entries in {feed_url}")

    added_count = 0
    skipped_count = 0
    error_count = 0
    processed_count = 0

    tasks = []
    for entry in parsed_feed.entries:
        article_data = extract_article_data(entry, feed_config)
        if article_data:
            tasks.append(store_article(article_data, db))
        else:
            skipped_count += 1

    storage_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in storage_results:
        processed_count += 1
        if isinstance(result, Exception):
            logger.error(f"Caught exception during article storage for feed {feed_url}: {result}")
            error_count += 1
        elif isinstance(result, tuple) and len(result) == 2: # Check tuple structure
            was_added, article_info = result
            if was_added and article_info is not None: # Check if info dict is returned
                added_count += 1
                # Append the dictionary, not just the headline
                added_articles_info.append(article_info)
        else:
             logger.warning(f"Unexpected result type/structure from store_article for feed {feed_url}: {type(result)} / {result}")
             error_count +=1

    existing_count = processed_count - added_count - error_count
    logger.info(f"Finished processing {feed_url}. Added: {added_count}, Existing: {existing_count}, Skipped/Invalid: {skipped_count}, Errors: {error_count}")
    # Return the list of info dictionaries
    return added_articles_info


# --- Audited pandas-based save_articles_info_to_csv ---
def save_articles_info_to_csv(articles_info: List[Dict[str, str]], filename: str = "rss_feed_entries.csv"):
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
async def process_all_feeds(db: Database, feed_list: List[Dict[str, Any]], csv_filename: str = "rss_feed_entries.csv"):
    """
    Main orchestrator function to process all configured RSS feeds.

    Args:
        db: Connected Database instance.
        feed_list: A FLAT list of feed configuration dictionaries,
                   where each dict represents one URL.
        csv_filename: Name of the output CSV file for headlines.
    """
    #if not db or not db.client or not db.articles_collection:
    if db is None or db.client is None or db.articles_collection is None:
        logger.error("Database connection is not available. Aborting feed processing.")
        return
    if not feed_list:
        logger.warning("Received an empty feed list to process. Aborting.")
        return

    # Changed list name and type hint
    all_new_articles_info: List[Dict[str, str]] = []
    total_feeds = len(feed_list)
    processed_feeds = 0
    failed_feeds = 0

    logger.info(f"Starting processing for {total_feeds} feed URLs...")

    async with httpx.AsyncClient() as client:
        tasks = [process_single_feed(feed_config, client, db) for feed_config in feed_list]
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
async def main():
    """Main function for standalone execution and testing."""
    logger.info("Starting standalone RSS processor execution...")
    try:
        await db_handler.connect_db()
        logger.info("Loading and transforming feed configuration from settings...")
        feed_list_to_process = flatten_feed_config(settings.rss_feeds)

        if not feed_list_to_process:
            logger.warning("No processable feed URLs found in configuration. Exiting.")
            return

        # Use the updated default CSV filename or pass explicitly
        await process_all_feeds(
            db=db_handler,
            feed_list=feed_list_to_process,
            csv_filename="output_articles_added.csv" # Example of explicit filename
        )

    except Exception as e:
        logger.exception(f"An error occurred during standalone execution: {e}")
    finally:
        await db_handler.close_db()
        logger.info("Standalone RSS processor finished.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    if not settings.MONGO_URI or "localhost" in settings.MONGO_URI:
        print("\nWARNING: MONGO_URI environment variable not set or points to localhost.")
        print("Ensure MongoDB is running and accessible, or set MONGO_URI in your environment/.env file.")

    # Important Check: Ensure the rss_feeds setting is actually loaded
    if not settings.rss_feeds:
         print("\nERROR: settings.rss_feeds is empty or not loaded correctly from config.py!")
         print("Please check your config.py and environment variables/defaults.")
         # exit(1) # Optionally exit if config is crucial

    asyncio.run(main())