# ai_news_classifier/app/ai_classifier.py
# ... (existing imports, constants, global variables) ...
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc, Span
import warnings
import logging
import asyncio
from .database import db_handler, Database # Needs db access
from .models import ArticleModel # For type hints potentially
from datetime import datetime, timezone
from pymongo.errors import OperationFailure

# Import keyword lists
from .ai_keywords import ( # Use relative import within app package
    ALL_PHRASE_KEYWORDS, AI_NOUNS_CONTEXT, AI_ADJECTIVES_CONTEXT,
    CONTEXT_VERBS, HIGH_CONFIDENCE_VERBS,
    CORE_PHRASES_FOR_HIGH_WEIGHT
)
# Import settings for model name and threshold
from .config import settings

logger = logging.getLogger(__name__)

# --- Constants for Scoring (now potentially loaded from config) ---
PHRASE_MATCH_WEIGHT = 1.0
CORE_PHRASE_WEIGHT = 2.0
CONTEXT_PATTERN_WEIGHT = 1.5
HIGH_CONF_VERB_WEIGHT = 2.0
# Use threshold from config file
SCORE_THRESHOLD = settings.classifier_threshold

# Make core phrases a set for efficient lookup
CORE_PHRASES_SET = set(CORE_PHRASES_FOR_HIGH_WEIGHT)

# --- Global Variables for spaCy model and matchers ---
# Load model and initialize matchers once when the module is loaded
# This is efficient for a server context
NLP_MODEL = None
PHRASE_MATCHER = None
MATCHER = None

def load_spacy_model_and_matchers():
    """Loads the spaCy model and initializes matchers."""
    global NLP_MODEL, PHRASE_MATCHER, MATCHER
    if NLP_MODEL is None: # Load only if not already loaded
        model_name = settings.spacy_model_name
        logger.info(f"Loading spaCy model: {model_name}...")
        try:
            NLP_MODEL = spacy.load(model_name)
            logger.info(f"Successfully loaded spaCy model: {model_name}")
            logger.info("Initializing spaCy matchers...")
            PHRASE_MATCHER, MATCHER = initialize_matchers(NLP_MODEL)
            logger.info("Successfully initialized spaCy matchers.")
        except OSError:
            logger.error(f"Could not load spaCy model '{model_name}'.")
            logger.error(f"Please ensure it's downloaded: python -m spacy download {model_name}")
            # Depending on requirements, might want to raise an exception or exit
            raise RuntimeError(f"Failed to load spaCy model: {model_name}")
        except Exception as e:
             logger.error(f"An unexpected error occurred during spaCy initialization: {e}")
             raise RuntimeError("Failed to initialize spaCy components")
    return NLP_MODEL, PHRASE_MATCHER, MATCHER


def initialize_matchers(nlp_model: spacy.language.Language) -> tuple[PhraseMatcher, Matcher]:
    """
    Create and configure spaCy PhraseMatcher and Matcher (modified slightly for logging).
    """
    vocab = nlp_model.vocab
    phrase_matcher = PhraseMatcher(vocab, attr="LOWER")
    valid_keywords = [kw for kw in ALL_PHRASE_KEYWORDS if kw and isinstance(kw, str)]
    try:
        patterns = list(nlp_model.pipe(valid_keywords))
        phrase_matcher.add("AI_PHRASES_CONCEPTS", patterns)
        logger.info(f"Initialized PhraseMatcher with {len(valid_keywords)} patterns.")
    except Exception as e:
        logger.error(f"Error initializing PhraseMatcher: {e}")
        # Handle error appropriately

    matcher = Matcher(vocab)
    try:
        # Pattern: CONTEXT_VERB + AI_NOUN_CONTEXT
        matcher.add("VERB_AI_NOUN", [[
                {"LEMMA": {"IN": CONTEXT_VERBS}, "POS": "VERB"},
                {"POS": {"IN": ["DET", "ADJ", "NOUN", "PROPN", "PRON", "NUM", "ADV", "PART"]}, "OP": "*"},
                {"LEMMA": {"IN": AI_NOUNS_CONTEXT}, "POS": {"IN": ["NOUN", "PROPN"]}}
            ]])
        # Pattern: AI_NOUN_CONTEXT + CONTEXT_VERB
        matcher.add("AI_NOUN_VERB", [[
                {"LEMMA": {"IN": AI_NOUNS_CONTEXT}, "POS": {"IN": ["NOUN", "PROPN"]}},
                {"POS": {"IN": ["AUX", "VERB", "ADJ", "ADV", "DET", "PART", "PRON", "SCONJ", "ADP"]}, "OP": "*", "IS_SENT_START": False},
                {"LEMMA": {"IN": CONTEXT_VERBS}, "POS": "VERB"}
            ]])
        # Pattern: AI_ADJECTIVE_CONTEXT + NOUN
        matcher.add("AI_ADJ_NOUN", [[
                {"LOWER": {"IN": AI_ADJECTIVES_CONTEXT}, "POS": {"IN": ["ADJ", "NOUN", "PROPN"]}},
                {"POS": {"IN": ["NOUN", "PROPN"]}}
            ]])
        # Pattern: High-confidence verbs
        matcher.add("HIGH_CONF_VERB", [[
                {"LEMMA": {"IN": HIGH_CONFIDENCE_VERBS}, "POS": "VERB"}
            ]])
        logger.info(f"Initialized Matcher with {len(matcher)} contextual/high-confidence rule patterns.")
    except Exception as e:
         logger.error(f"Error initializing Matcher patterns: {e}")
         # Handle error appropriately

    return phrase_matcher, matcher

def classify_ai_text(
    text: str,
    # Pass loaded components instead of reloading each time
    nlp_model: spacy.language.Language,
    phrase_matcher: PhraseMatcher,
    matcher: Matcher,
    threshold: float = SCORE_THRESHOLD
) -> tuple[bool, float, list[tuple[str, str, str]]]:
    """
    Classify if text is AI-related using weighted scoring based on matches.
    (This is your 95.65% accurate version's logic)
    """
    if not text or not isinstance(text, str): # Basic check for valid input
        return False, 0.0, []

    doc: Doc = nlp_model(text)
    total_score: float = 0.0
    match_details: list[tuple[str, str, str]] = []
    pattern_match_scores: dict[tuple[int, int], float] = {}

    # --- Scoring Step 1: Pattern Matcher ---
    try:
        pattern_matches = matcher(doc)
        spans_for_filtering = [doc[start:end] for _, start, end in pattern_matches]
        span_indices_to_match_id = {(doc[start:end].start, doc[start:end].end): match_id for match_id, start, end in pattern_matches}
        filtered_pattern_spans: list[Span] = spacy.util.filter_spans(spans_for_filtering)

        for span in filtered_pattern_spans:
            start, end = span.start, span.end
            original_match_id = span_indices_to_match_id.get((start, end))
            if original_match_id is not None:
                rule_id_str = nlp_model.vocab.strings[original_match_id]
                score_increase = 0.0
                if rule_id_str == "HIGH_CONF_VERB": score_increase = HIGH_CONF_VERB_WEIGHT
                elif rule_id_str in ["VERB_AI_NOUN", "AI_NOUN_VERB", "AI_ADJ_NOUN"]: score_increase = CONTEXT_PATTERN_WEIGHT
                else: score_increase = 1.0

                total_score += score_increase
                pattern_match_scores[(start, end)] = score_increase

                context_start = max(0, start - 5)
                context_end = min(len(doc), end + 5)
                context = doc[context_start:context_end].text.replace("\n", " ")
                match_details.append((rule_id_str, span.text, f"...{context}..."))
            else:
                # Warning already printed in earlier versions, maybe reduce noise
                # logger.warning(f"Could not find original match_id for filtered pattern span: '{span.text}' at ({start}, {end})")
                pass # Suppress warning for cleaner logs if frequent
    except Exception as e:
        logger.error(f"Error during pattern matching for text starting with '{text[:50]}...': {e}")


    # --- Scoring Step 2: Phrase Matcher ---
    try:
        phrase_matches = phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            matched_span = doc[start:end]
            matched_text_lower = matched_span.text.lower()
            phrase_score = CORE_PHRASE_WEIGHT if matched_text_lower in CORE_PHRASES_SET else PHRASE_MATCH_WEIGHT
            pattern_score_here = pattern_match_scores.get((start, end))
            score_to_add = 0.0

            if pattern_score_here is None:
                score_to_add = phrase_score
            elif phrase_score > pattern_score_here:
                 score_to_add = phrase_score - pattern_score_here

            if score_to_add > 0:
                total_score += score_to_add
                rule_id_str = nlp_model.vocab.strings[match_id]
                context_start = max(0, start - 5)
                context_end = min(len(doc), end + 5)
                context = doc[context_start:context_end].text.replace("\n", " ")
                display_rule_id = f"CORE_PHRASE ({rule_id_str})" if matched_text_lower in CORE_PHRASES_SET else rule_id_str
                match_details.append((display_rule_id, matched_span.text, f"...{context}..."))
    except Exception as e:
         logger.error(f"Error during phrase matching for text starting with '{text[:50]}...': {e}")


    # --- Final Classification ---
    is_ai_related = total_score >= threshold
    match_details.sort(key=lambda item: text.find(item[1]) if item[1] in text else 0) # Safer sort key

    return is_ai_related, round(total_score, 2), match_details


# --- NEW Batch Processing Function ---
async def classify_pending_articles_in_db(db: Database, limit: int = 100):
    """
    Fetches articles pending classification, classifies them using the loaded
    spaCy model, and updates the database.

    Args:
        db: The connected Database instance.
        limit: Maximum number of articles to process in one batch.
    """
    # Check if spaCy components are loaded (should be by lifespan)
    if NLP_MODEL is None or PHRASE_MATCHER is None or MATCHER is None:
        logger.error("Classifier Error: spaCy model or matchers not loaded. Skipping classification.")
        return

    if db.articles_collection is None:
        logger.error("Classifier Error: Database collection is not available.")
        return

    logger.info(f"Starting classification task for up to {limit} pending articles.")
    processed_count = 0
    updated_count = 0

    try:
        # Find articles where classification hasn't been run yet
        query = {"isAIRelated": None} # Based on updated ArticleModel default
        cursor = db.articles_collection.find(query).limit(limit)
        # Fetch only necessary fields for classification + _id
        articles_to_classify = await cursor.to_list(length=limit)

        if not articles_to_classify:
            logger.info("No articles found pending classification in this batch.")
            return

        logger.info(f"Found {len(articles_to_classify)} articles to classify.")

        # --- Prepare tasks for concurrent classification and update ---
        tasks = []
        for article_data in articles_to_classify:
            article_id = article_data.get("_id")
            if not article_id:
                 logger.warning("Skipping article with missing _id in classification batch.")
                 continue

            # Construct text for classification
            text_to_classify = f"{article_data.get('title', '')}\n{article_data.get('description', '')}\n{article_data.get('content', '')}"
            # Check if text is substantial enough (optional)
            '''if len(text_to_classify.strip()) < 5: # Arbitrary minimum length
                 logger.debug(f"Skipping article ID {article_id} due to insufficient text.")
                 # Optionally update status to 'skipped' or similar
                 # For now, just skip processing
                 continue'''

            # Create task for this article
            tasks.append(
                _classify_and_update_single_article(
                    db, article_id, text_to_classify
                )
            )

        # Run classification/update tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                 # Error logged within _classify_and_update_single_article
                 pass # Or add specific handling here if needed
            elif result is True: # Successfully updated
                 updated_count +=1
            # Increment processed count regardless of update success/failure (if task ran)
            processed_count += 1

        logger.info(f"Classification task finished. Attempted: {processed_count}, Successfully Updated: {updated_count}")

    except OperationFailure as e:
        logger.error(f"Classifier DB Error: {e.details}", exc_info=True)
    except Exception as e:
        logger.error(f"Classifier Error: An unexpected error occurred: {e}", exc_info=True)


# --- Helper for single article processing ---
async def _classify_and_update_single_article(db: Database, article_id: Any, text: str) -> bool:
    """
    Helper function to classify text and update the corresponding DB record.
    Returns True if update was successful, False otherwise.
    """
    try:
        # Ensure model is loaded (redundant check if called from main func, but safe)
        if NLP_MODEL is None: raise RuntimeError("spaCy model not loaded")

        # Call the core classification function
        is_ai, score, _ = classify_ai_text(
            text,
            NLP_MODEL,
            PHRASE_MATCHER,
            MATCHER,
            settings.classifier_threshold
        )

        logger.debug(f"Classified article ID {article_id}: is_ai={is_ai}, score={score}")

        # Update the document in the database
        update_result = await db.articles_collection.update_one(
            {"_id": article_id},
            {"$set": {
                "isAIRelated": is_ai, # Matches model alias
                "classificationConfidence": score, # Matches model alias
                "updatedAt": datetime.now(timezone.utc) # Update timestamp
            }}
        )

        if update_result.modified_count > 0:
            logger.debug(f"Successfully updated classification for article ID {article_id}")
            return True
        elif update_result.matched_count == 0:
             logger.warning(f"Attempted to update article ID {article_id}, but it was not found.")
             return False
        else: # Matched but not modified
             logger.warning(f"Article ID {article_id} was matched but not updated (modified_count=0). Status might have changed or values were the same.")
             return False

    except Exception as e:
        logger.error(f"Failed during classification/update for article ID {article_id}: {e}", exc_info=True)
        # Propagate exception to be caught by asyncio.gather
        raise e


# Guard the original main execution block if you want to keep it for standalone testing
if __name__ == "__main__":
     # You could potentially add a small test here that loads the model
     # and runs classify_ai_text on a sample string, but the main
     # testing should ideally be in a separate test file (e.g., test_classifier.py)
     logging.basicConfig(level=logging.INFO)
     print("Testing classifier standalone...")
     try:
         # Load model and matchers
         nlp, ph_matcher, matcher = load_spacy_model_and_matchers()
         # Test text
         test_text = "New study uses deep learning for medical image recognition."
         is_ai, score, details = classify_ai_text(test_text, nlp, ph_matcher, matcher)
         print(f"Test Text: {test_text}")
         print(f"Is AI: {is_ai}, Score: {score}")
         print("Details:", details)

         test_text_2 = "The weather forecast predicts rain tomorrow."
         is_ai_2, score_2, details_2 = classify_ai_text(test_text_2, nlp, ph_matcher, matcher)
         print(f"\nTest Text: {test_text_2}")
         print(f"Is AI: {is_ai_2}, Score: {score_2}")
         print("Details:", details_2)

     except Exception as e:
         print(f"Standalone test failed: {e}")