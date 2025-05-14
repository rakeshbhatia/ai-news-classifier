# ai_news_classifier/app/ai_classifier.py

import math # For abs
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc, Span
# import warnings # Not used, can remove
import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional # Added Optional

# --- SQLAlchemy Imports ---
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select # update # No direct update statement needed if updating ORM objects
from sqlalchemy.exc import SQLAlchemyError

from datetime import datetime, timezone

# --- Project Imports ---
from .models import Article # Import SQLAlchemy ORM Article model
from .config import settings
from .ai_keywords import (
    ALL_PHRASE_KEYWORDS, AI_NOUNS_CONTEXT, AI_ADJECTIVES_CONTEXT,
    CONTEXT_VERBS, HIGH_CONFIDENCE_VERBS,
    CORE_PHRASES_FOR_HIGH_WEIGHT
)

logger = logging.getLogger(__name__)

# --- Constants for Scoring ---
# These might come from settings, but defining here for clarity if not already in config
PHRASE_MATCH_WEIGHT = settings.PHRASE_MATCH_WEIGHT if hasattr(settings, 'PHRASE_MATCH_WEIGHT') else 1.0
CORE_PHRASE_WEIGHT = settings.CORE_PHRASE_WEIGHT if hasattr(settings, 'CORE_PHRASE_WEIGHT') else 2.0
CONTEXT_PATTERN_WEIGHT = settings.CONTEXT_PATTERN_WEIGHT if hasattr(settings, 'CONTEXT_PATTERN_WEIGHT') else 1.5
HIGH_CONF_VERB_WEIGHT = settings.HIGH_CONF_VERB_WEIGHT if hasattr(settings, 'HIGH_CONF_VERB_WEIGHT') else 2.0
SCORE_THRESHOLD = settings.classifier_threshold

CORE_PHRASES_SET = set(CORE_PHRASES_FOR_HIGH_WEIGHT)
# AI_NOUNS_CONTEXT_SET = set(AI_NOUNS_CONTEXT) # If used in refined classify_ai_text

# Define a "confidence spread" parameter.
# This determines how far the total_score needs to be from the threshold
# to reach maximum confidence. A good starting point might be the threshold value itself.
# If CONFIDENCE_SPREAD = SCORE_THRESHOLD, then:
# - a score of 0 gives confidence ~1.0 (for the "not AI" decision)
# - a score of SCORE_THRESHOLD * 2 gives confidence ~1.0 (for the "is AI" decision)
CONFIDENCE_SPREAD = SCORE_THRESHOLD # Or settings.CONFIDENCE_SPREAD if you make it configurable

# --- Global Variables for spaCy model and matchers ---
NLP_MODEL: Optional[spacy.language.Language] = None
PHRASE_MATCHER: Optional[PhraseMatcher] = None
MATCHER: Optional[Matcher] = None

def load_spacy_model_and_matchers():
    """Loads the spaCy model and initializes matchers."""
    global NLP_MODEL, PHRASE_MATCHER, MATCHER
    if NLP_MODEL is None:
        model_name = settings.spacy_model_name
        logger.info(f"Loading spaCy model: {model_name}...")
        try:
            NLP_MODEL = spacy.load(model_name)
            logger.info(f"Successfully loaded spaCy model: {model_name}")
            logger.info("Initializing spaCy matchers...")
            PHRASE_MATCHER, MATCHER = initialize_matchers(NLP_MODEL)
            logger.info("Successfully initialized spaCy matchers.")
        except OSError:
            logger.error(f"Could not load spaCy model '{model_name}'. Download: python -m spacy download {model_name}", exc_info=True)
            raise RuntimeError(f"Failed to load spaCy model: {model_name}")
        except Exception as e:
             logger.error(f"An unexpected error occurred during spaCy initialization: {e}", exc_info=True)
             raise RuntimeError("Failed to initialize spaCy components")
    return NLP_MODEL, PHRASE_MATCHER, MATCHER


def initialize_matchers(nlp_model: spacy.language.Language) -> tuple[PhraseMatcher, Matcher]:
    """Create and configure spaCy PhraseMatcher and Matcher."""
    # ... (Your existing initialize_matchers logic - seems fine) ...
    vocab = nlp_model.vocab
    phrase_matcher = PhraseMatcher(vocab, attr="LOWER")
    valid_keywords = [kw for kw in ALL_PHRASE_KEYWORDS if kw and isinstance(kw, str)]
    try:
        patterns = list(nlp_model.pipe(valid_keywords))
        phrase_matcher.add("AI_PHRASES_CONCEPTS", patterns)
        logger.info(f"Initialized PhraseMatcher with {len(valid_keywords)} patterns.")
    except Exception as e:
        logger.error(f"Error initializing PhraseMatcher: {e}")

    matcher = Matcher(vocab)
    try:
        matcher.add("VERB_AI_NOUN", [[{"LEMMA": {"IN": CONTEXT_VERBS}, "POS": "VERB"}, {"POS": {"IN": ["DET", "ADJ", "NOUN", "PROPN", "PRON", "NUM", "ADV", "PART"]}, "OP": "*"}, {"LEMMA": {"IN": AI_NOUNS_CONTEXT}, "POS": {"IN": ["NOUN", "PROPN"]}}]])
        matcher.add("AI_NOUN_VERB", [[{"LEMMA": {"IN": AI_NOUNS_CONTEXT}, "POS": {"IN": ["NOUN", "PROPN"]}}, {"POS": {"IN": ["AUX", "VERB", "ADJ", "ADV", "DET", "PART", "PRON", "SCONJ", "ADP"]}, "OP": "*", "IS_SENT_START": False}, {"LEMMA": {"IN": CONTEXT_VERBS}, "POS": "VERB"}]])
        matcher.add("AI_ADJ_NOUN", [[{"LOWER": {"IN": AI_ADJECTIVES_CONTEXT}, "POS": {"IN": ["ADJ", "NOUN", "PROPN"]}}, {"POS": {"IN": ["NOUN", "PROPN"]}}]])
        matcher.add("HIGH_CONF_VERB", [[{"LEMMA": {"IN": HIGH_CONFIDENCE_VERBS}, "POS": "VERB"}]])
        logger.info(f"Initialized Matcher with {len(matcher)} contextual/high-confidence rule patterns.")
    except Exception as e:
         logger.error(f"Error initializing Matcher patterns: {e}")
    return phrase_matcher, matcher


def classify_ai_text(
    text: str,
    nlp_model: spacy.language.Language,
    phrase_matcher: PhraseMatcher,
    matcher: Matcher,
    threshold: float = SCORE_THRESHOLD
) -> tuple[bool, float, list[tuple[str, str, str]]]:
    """
    Classify if text is AI-related using weighted scoring.
    Calculates a confidence score (0-1) reflecting certainty in the
    is_ai_related prediction, regardless of whether it's True or False.
    """
    if not text or not isinstance(text, str):
        return False, 0.5, [] # Default to uncertain confidence for invalid input

    doc: Doc = nlp_model(text)
    total_score: float = 0.0
    match_details: list[tuple[str, str, str]] = []
    span_scores: dict[tuple[int, int], tuple[str, float]] = {}

    # --- add_score helper (as defined previously) ---
    def add_score(start, end, rule_id_str, score_val, matched_text):
        nonlocal total_score
        span_key = (start, end)
        current_best_score_tuple = span_scores.get(span_key, (None, -1.0))
        current_best_score = current_best_score_tuple[1]
        if score_val > current_best_score:
            if current_best_score > 0: total_score -= current_best_score
            total_score += score_val
            span_scores[span_key] = (rule_id_str, score_val)
            context_start = max(0, start - 5); context_end = min(len(doc), end + 5)
            context = doc[context_start:context_end].text.replace("\n", " ")
            match_details.append((rule_id_str, matched_text, f"...{context}..."))
    # --- End add_score helper ---

    # --- Scoring Steps 1 & 2 (Pattern and Phrase Matchers - as defined previously) ---
    try: # Pattern Matcher
        pattern_matches = matcher(doc)
        for match_id, start, end in pattern_matches:
            span = doc[start:end]; rule_id_str = nlp_model.vocab.strings[match_id]
            score_increase = 0.0
            if rule_id_str == "HIGH_CONF_VERB": score_increase = HIGH_CONF_VERB_WEIGHT
            elif rule_id_str in ["VERB_AI_NOUN", "AI_NOUN_VERB", "AI_ADJ_NOUN"]: score_increase = CONTEXT_PATTERN_WEIGHT
            else: score_increase = 0.1
            if score_increase > 0: add_score(start, end, rule_id_str, score_increase, span.text)
    except Exception as e: logger.error(f"Pattern matching error for text '{text[:50]}...': {e}")

    try: # Phrase Matcher
        phrase_matches = phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            span = doc[start:end]; matched_text_lower = span.text.lower()
            is_core = matched_text_lower in CORE_PHRASES_SET
            phrase_score = CORE_PHRASE_WEIGHT if is_core else PHRASE_MATCH_WEIGHT
            rule_id = f"CORE_PHRASE" if is_core else "AI_PHRASE"
            add_score(start, end, rule_id, phrase_score, span.text)
    except Exception as e: logger.error(f"Phrase matching error for text '{text[:50]}...': {e}")
    # --- End Scoring Steps ---

    is_ai_related = total_score >= threshold
    confidence_value = 0.5 # Default to uncertain (mid-point)

    # Use the defined CONFIDENCE_SPREAD for normalization
    # Ensure CONFIDENCE_SPREAD is not zero to avoid division errors
    effective_spread = max(CONFIDENCE_SPREAD, 1e-6) # Use a very small number if CONFIDENCE_SPREAD is 0 or less

    # Calculate normalized distance from the threshold
    # abs(total_score - threshold) is how far the score is from the decision boundary
    normalized_distance = abs(total_score - threshold) / effective_spread

    # Map this normalized distance to a confidence:
    # 0.5 when at threshold, increasing to 1.0 as distance reaches `effective_spread`
    confidence_value = 0.5 + (0.5 * min(normalized_distance, 1.0))

    # Ensure confidence is strictly within [0,1] due to potential floating point nuances
    final_confidence = round(min(max(confidence_value, 0.0), 1.0), 4)

    match_details.sort(key=lambda item: text.find(item[1]) if item[1] in text else float('inf'))
    return is_ai_related, final_confidence, match_details


'''def classify_ai_text(
    text: str,
    nlp_model: spacy.language.Language,
    phrase_matcher: PhraseMatcher,
    matcher: Matcher,
    threshold: float = SCORE_THRESHOLD # Use the threshold from settings
) -> tuple[bool, float, list[tuple[str, str, str]]]: # Returns: is_ai, confidence_score, details
    """
    Classify if text is AI-related using weighted scoring.
    Calculates a separate confidence score based on the margin from the threshold.
    """
    if not text or not isinstance(text, str):
        return False, 0.0, [] # is_ai=False, confidence=0.0

    doc: Doc = nlp_model(text)
    total_score: float = 0.0 # This is the internal rule-based score
    match_details: list[tuple[str, str, str]] = []
    span_scores: dict[tuple[int, int], tuple[str, float]] = {}

    # --- add_score helper function (as defined previously, it calculates total_score) ---
    def add_score(start, end, rule_id_str, score_val, matched_text):
        nonlocal total_score
        span_key = (start, end)
        current_best_score_tuple = span_scores.get(span_key, (None, -1.0))
        current_best_score = current_best_score_tuple[1]
        if score_val > current_best_score:
            if current_best_score > 0: total_score -= current_best_score
            total_score += score_val
            span_scores[span_key] = (rule_id_str, score_val)
            context_start = max(0, start - 5)
            context_end = min(len(doc), end + 5)
            context = doc[context_start:context_end].text.replace("\n", " ")
            match_details.append((rule_id_str, matched_text, f"...{context}..."))
    # --- End add_score helper ---

    # --- Scoring Step 1: Pattern Matcher (as defined previously) ---
    try:
        pattern_matches = matcher(doc)
        for match_id, start, end in pattern_matches:
            span = doc[start:end]
            rule_id_str = nlp_model.vocab.strings[match_id]
            score_increase = 0.0
            if rule_id_str == "HIGH_CONF_VERB": score_increase = HIGH_CONF_VERB_WEIGHT
            elif rule_id_str in ["VERB_AI_NOUN", "AI_NOUN_VERB", "AI_ADJ_NOUN"]:
                score_increase = CONTEXT_PATTERN_WEIGHT # Or refined logic here
            else: score_increase = 0.1
            if score_increase > 0: add_score(start, end, rule_id_str, score_increase, span.text)
    except Exception as e: logger.error(f"Pattern matching error: {e}")

    # --- Scoring Step 2: Phrase Matcher (as defined previously) ---
    try:
        phrase_matches = phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            span = doc[start:end]
            matched_text_lower = span.text.lower()
            is_core = matched_text_lower in CORE_PHRASES_SET
            phrase_score = CORE_PHRASE_WEIGHT if is_core else PHRASE_MATCH_WEIGHT
            rule_id = f"CORE_PHRASE" if is_core else "AI_PHRASE"
            add_score(start, end, rule_id, phrase_score, span.text)
    except Exception as e: logger.error(f"Phrase matching error: {e}")

    # --- Binary Classification based on total_score vs threshold ---
    is_ai_related = total_score >= threshold

    # --- Calculate Heuristic Confidence Score (0.0 to 1.0) ---
    confidence_value = 0.0
    if is_ai_related:
        # Article classified as AI-related
        if total_score >= threshold * 1.8:  # Significantly above threshold
            confidence_value = 0.95
        elif total_score >= threshold * 1.3: # Moderately above threshold
            confidence_value = 0.75
        else:  # Just met or slightly above threshold
            confidence_value = 0.55
    else:
        # Article classified as NOT AI-related
        # Confidence here means "confidence that it IS AI" (so it will be low)
        # If score is very low, it's confidently NOT AI.
        if total_score <= threshold * 0.2:   # Very low score
            confidence_value = 0.05
        elif total_score <= threshold * 0.6: # Moderately low score
            confidence_value = 0.25
        else:  # Score is close to threshold but below
            confidence_value = 0.45

    match_details.sort(key=lambda item: text.find(item[1]) if item[1] in text else float('inf'))

    # Return:
    # 1. is_ai_related (boolean: True or False)
    # 2. confidence_value (float: 0.0-1.0, our heuristic confidence)
    # 3. match_details (list for debugging)
    return is_ai_related, round(confidence_value, 2), match_details'''


'''def classify_ai_text(
    text: str,
    nlp_model: spacy.language.Language,
    phrase_matcher: PhraseMatcher,
    matcher: Matcher,
    threshold: float = SCORE_THRESHOLD # Use directly from global/settings
) -> tuple[bool, float, list[tuple[str, str, str]]]:
    """Classify if text is AI-related using weighted scoring."""
    # ... (Your existing classify_ai_text logic - keep the latest refined version) ...
    # This function *only* classifies, it does not touch the DB.
    if not text or not isinstance(text, str): return False, 0.0, []
    doc: Doc = nlp_model(text)
    total_score: float = 0.0
    match_details: list[tuple[str, str, str]] = []
    span_scores: dict[tuple[int, int], tuple[str, float]] = {}

    def add_score(start, end, rule_id_str, score, matched_text):
        nonlocal total_score # Ensure total_score is modifiable
        span_key = (start, end)
        current_best_score_tuple = span_scores.get(span_key, (None, -1.0))
        current_best_score = current_best_score_tuple[1]

        if score > current_best_score:
            if current_best_score > 0: # If there was a previous score for this span
                total_score -= current_best_score
            total_score += score
            span_scores[span_key] = (rule_id_str, score)
            # For match_details, we might want to replace if a better score for the same span is found
            # Simple append for now, can refine later if details are too noisy
            context_start = max(0, start - 5)
            context_end = min(len(doc), end + 5)
            context = doc[context_start:context_end].text.replace("\n", " ")
            match_details.append((rule_id_str, matched_text, f"...{context}..."))

    # Scoring Step 1: Pattern Matcher
    try:
        pattern_matches = matcher(doc)
        for match_id, start, end in pattern_matches:
            span = doc[start:end]
            rule_id_str = nlp_model.vocab.strings[match_id]
            score_increase = 0.0
            if rule_id_str == "HIGH_CONF_VERB": score_increase = HIGH_CONF_VERB_WEIGHT
            elif rule_id_str in ["VERB_AI_NOUN", "AI_NOUN_VERB", "AI_ADJ_NOUN"]:
                score_increase = CONTEXT_PATTERN_WEIGHT # Using simplified weight for now
            else: score_increase = 0.1
            if score_increase > 0: add_score(start, end, rule_id_str, score_increase, span.text)
    except Exception as e: logger.error(f"Pattern matching error: {e}")

    # Scoring Step 2: Phrase Matcher
    try:
        phrase_matches = phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            span = doc[start:end]
            matched_text_lower = span.text.lower()
            is_core = matched_text_lower in CORE_PHRASES_SET
            phrase_score = CORE_PHRASE_WEIGHT if is_core else PHRASE_MATCH_WEIGHT
            rule_id = f"CORE_PHRASE" if is_core else "AI_PHRASE"
            add_score(start, end, rule_id, phrase_score, span.text)
    except Exception as e: logger.error(f"Phrase matching error: {e}")

    is_ai_related = total_score >= threshold
    # Ensure details are sorted by appearance
    match_details.sort(key=lambda item: text.find(item[1]) if item[1] in text else float('inf'))
    return is_ai_related, round(total_score, 2), match_details'''


# --- Batch Processing and DB Update Function ---
async def classify_pending_articles_in_db(
    session_factory: async_sessionmaker[AsyncSession],
    limit: int = settings.CLASSIFICATION_BATCH_SIZE # Use from settings
):
    """
    Fetches articles pending classification, classifies them using the loaded
    spaCy model, and updates the database using SQLAlchemy sessions.
    """
    if NLP_MODEL is None or PHRASE_MATCHER is None or MATCHER is None:
        logger.error("Classifier Error: spaCy model or matchers not loaded. Skipping classification.")
        return
    if session_factory is None:
        logger.error("Classifier Error: Database session factory not available.")
        return

    logger.info(f"Starting classification task for up to {limit} pending articles.")
    processed_count = 0
    updated_count = 0

    async with session_factory() as session: # Create a new session for this batch
        try:
            # Query for pending articles (where is_ai_related is NULL)
            stmt_select = select(Article).where(Article.is_ai_related.is_(None)).limit(limit)
            results = await session.execute(stmt_select)
            articles_to_classify: List[Article] = results.scalars().all()

            if not articles_to_classify:
                logger.info("No articles found pending classification in this batch.")
                await session.commit() # Commit even if no articles, to close transaction
                return

            logger.info(f"Found {len(articles_to_classify)} articles to classify.")

            for article_orm in articles_to_classify: # Iterate over ORM objects
                processed_count += 1
                text_to_classify = f"{article_orm.title or ''}\n{article_orm.description or ''}\n{article_orm.content or ''}"

                if len(text_to_classify.strip()) < 50: # Arbitrary min length for meaningful classification
                    logger.debug(f"Skipping article ID {article_orm.id} (Title: '{article_orm.title[:30]}...') due to insufficient text.")
                    # Optionally update to a 'skipped_classification' status here if needed
                    continue

                try:
                    # Call the core classification function
                    # It now returns (is_ai, confidence, details)
                    is_ai, confidence, _ = classify_ai_text( # Unpack the third value (details) if not used
                        text_to_classify,
                        NLP_MODEL,
                        PHRASE_MATCHER,
                        MATCHER,
                        settings.classifier_threshold
                    )
                    logger.debug(f"Classified article ID {article_orm.id} (Title: '{article_orm.title[:30]}...'): is_ai={is_ai}, confidence={confidence}")

                    # Update the ORM object directly
                    article_orm.is_ai_related = is_ai
                    article_orm.classification_confidence = confidence # Use the new confidence value
                    article_orm.updated_at = datetime.now(timezone.utc)
                    session.add(article_orm)
                    updated_count += 1
                except Exception as class_err:
                    logger.error(f"Failed during classification for article ID {article_orm.id} (Title: '{article_orm.title[:30]}...'): {class_err}", exc_info=True)
                    # This specific article won't be updated, but others will proceed

            await session.commit() # Commit all changes for the batch
            logger.info(f"Classification batch finished. Attempted: {len(articles_to_classify)}, Successfully prepared for DB update: {updated_count}")

        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Classifier DB Error during batch processing: {e}", exc_info=True)
        except Exception as e:
            await session.rollback()
            logger.error(f"Classifier Error: An unexpected error occurred during batch processing: {e}", exc_info=True)


# Standalone test block (ensure settings and models are available)
if __name__ == "__main__":
     logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
     logger_main = logging.getLogger(__name__) # Use a distinct logger if needed
     logger_main.info("Testing classifier standalone...")
     try:
         nlp, ph_matcher, matcher = load_spacy_model_and_matchers()
         current_threshold = settings.classifier_threshold
         logger_main.info(f"Using Threshold: {current_threshold}")

         test_text = "New study uses deep learning for medical image recognition."
         is_ai, score, details = classify_ai_text(test_text, nlp, ph_matcher, matcher, current_threshold)
         logger_main.info(f"\nTest Text: {test_text}\nIs AI: {is_ai}, Score: {score}\nDetails: {details}")

         test_text_2 = "The weather forecast predicts rain tomorrow."
         is_ai_2, score_2, details_2 = classify_ai_text(test_text_2, nlp, ph_matcher, matcher, current_threshold)
         logger_main.info(f"\nTest Text: {test_text_2}\nIs AI: {is_ai_2}, Score: {score_2}\nDetails: {details_2}")

     except Exception as e:
         logger_main.error(f"Standalone test failed: {e}", exc_info=True)