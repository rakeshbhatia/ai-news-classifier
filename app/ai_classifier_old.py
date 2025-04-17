# ai-news-classifier/ai_classifier.py
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc, Span
import warnings

# Attempt to load a medium model for potentially better accuracy, fall back to small
try:
    nlp = spacy.load("en_core_web_md")
    print("Loaded spaCy model: en_core_web_md")
except OSError:
    print("en_core_web_md not found. Falling back to en_core_web_sm.")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model: en_core_web_sm")
    except OSError:
        print("Error: Neither en_core_web_md nor en_core_web_sm found.")
        print("Please install a spaCy model: python -m spacy download en_core_web_sm")
        exit()

from ai_keywords import (
    ALL_PHRASE_KEYWORDS, AI_NOUNS_CONTEXT, AI_ADJECTIVES_CONTEXT,
    CONTEXT_VERBS, HIGH_CONFIDENCE_VERBS,
    CORE_PHRASES_FOR_HIGH_WEIGHT # Import the new set/list
)

# --- Constants for Scoring ---
PHRASE_MATCH_WEIGHT = 1.0       # Default weight
CORE_PHRASE_WEIGHT = 2.0      # *** Weight for core phrases ***
CONTEXT_PATTERN_WEIGHT = 1.5
HIGH_CONF_VERB_WEIGHT = 2.0
SCORE_THRESHOLD = 1.9         # Keep threshold at 1.9

# Make core phrases a set for efficient lookup
CORE_PHRASES_SET = set(CORE_PHRASES_FOR_HIGH_WEIGHT)

def initialize_matchers(nlp_model: spacy.language.Language) -> tuple[PhraseMatcher, Matcher]:
    """
    Create and configure spaCy PhraseMatcher and Matcher
    with AI-related terms and contextual patterns.
    """
    vocab = nlp_model.vocab

    # 1. Phrase Matcher for specific terms, phrases, companies (case-insensitive)
    phrase_matcher = PhraseMatcher(vocab, attr="LOWER")
    # Use nlp.pipe for efficiency with many patterns
    # Filter out any empty strings or potential None values from lists
    valid_keywords = [kw for kw in ALL_PHRASE_KEYWORDS if kw and isinstance(kw, str)]
    patterns = list(nlp_model.pipe(valid_keywords))
    phrase_matcher.add("AI_PHRASES_CONCEPTS", patterns)
    print(f"Initialized PhraseMatcher with {len(valid_keywords)} patterns.") # Use count of valid keywords

    # 2. Matcher for contextual and high-confidence patterns
    matcher = Matcher(vocab)

    # --- Add Contextual Patterns ---

    # Pattern: CONTEXT_VERB + AI_NOUN_CONTEXT (e.g., "train model", "optimize parameters")
    # Allows for optional determiners/adjectives/etc. between verb and noun
    matcher.add("VERB_AI_NOUN", [
        [
            {"LEMMA": {"IN": CONTEXT_VERBS}, "POS": "VERB"},
            {"POS": {"IN": ["DET", "ADJ", "NOUN", "PROPN", "PRON", "NUM", "ADV", "PART"]}, "OP": "*"}, # Allow modifiers/pronouns etc. (PART for 'to')
            {"LEMMA": {"IN": AI_NOUNS_CONTEXT}, "POS": {"IN": ["NOUN", "PROPN"]}}
        ]
    ])

    # Pattern: AI_NOUN_CONTEXT + CONTEXT_VERB (e.g., "model predicts", "data is processed")
    # Allows for optional auxiliary verbs or modifiers, including pronouns/conjunctions
    matcher.add("AI_NOUN_VERB", [
         [
            {"LEMMA": {"IN": AI_NOUNS_CONTEXT}, "POS": {"IN": ["NOUN", "PROPN"]}},
            # Expanded list of allowed intermediate POS tags
            {"POS": {"IN": ["AUX", "VERB", "ADJ", "ADV", "DET", "PART", "PRON", "SCONJ", "ADP"]}, "OP": "*", "IS_SENT_START": False},
            {"LEMMA": {"IN": CONTEXT_VERBS}, "POS": "VERB"}
        ]
    ])

    # Pattern: AI_ADJECTIVE_CONTEXT + NOUN (e.g., "generative model", "autonomous system")
    # Catches adjective modifying *any* noun, as context often clarifies
    matcher.add("AI_ADJ_NOUN", [
        [
            {"LOWER": {"IN": AI_ADJECTIVES_CONTEXT}, "POS": {"IN": ["ADJ", "NOUN", "PROPN"]}}, # Allow compound nouns/proper nouns used adjectivally
            {"POS": {"IN": ["NOUN", "PROPN"]}} # Match common or proper nouns
        ]
    ])

    # Pattern: High-confidence verbs (relatively unambiguous)
    matcher.add("HIGH_CONF_VERB", [
        [{"LEMMA": {"IN": HIGH_CONFIDENCE_VERBS}, "POS": "VERB"}]
    ])

    print(f"Initialized Matcher with {len(matcher)} contextual/high-confidence rule patterns.")

    return phrase_matcher, matcher

def classify_ai_text(
    text: str,
    nlp_model: spacy.language.Language,
    phrase_matcher: PhraseMatcher,
    matcher: Matcher,
    threshold: float = SCORE_THRESHOLD
) -> tuple[bool, float, list[tuple[str, str, str]]]:
    """
    Classify if text is AI-related using weighted scoring based on matches.
    Uses higher weight for CORE_PHRASES and simplified overlap handling.
    """
    doc: Doc = nlp_model(text)
    total_score: float = 0.0
    match_details: list[tuple[str, str, str]] = []
    # Store spans from pattern matches to check against phrase matches later
    # Store the score associated with the pattern match
    pattern_match_scores: dict[tuple[int, int], float] = {}

    # --- Scoring Step 1: Pattern Matcher ---
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
            else: score_increase = 1.0 # Fallback

            total_score += score_increase
            # Store the score achieved by the pattern for this span
            pattern_match_scores[(start, end)] = score_increase

            # Add details for reporting
            context_start = max(0, start - 5)
            context_end = min(len(doc), end + 5)
            context = doc[context_start:context_end].text.replace("\n", " ")
            match_details.append((rule_id_str, span.text, f"...{context}..."))
        else:
            print(f"Warning: Could not find original match_id for filtered pattern span: '{span.text}' at ({start}, {end})")


    # --- Scoring Step 2: Phrase Matcher ---
    phrase_matches = phrase_matcher(doc)
    for match_id, start, end in phrase_matches:
        matched_span = doc[start:end]
        matched_text_lower = matched_span.text.lower()

        # Determine the potential score for this phrase
        phrase_score = CORE_PHRASE_WEIGHT if matched_text_lower in CORE_PHRASES_SET else PHRASE_MATCH_WEIGHT

        # *** Overlap Handling Logic ***
        pattern_score_here = pattern_match_scores.get((start, end))

        score_to_add = 0.0
        if pattern_score_here is None:
            # No exact pattern overlap, add the full phrase score
            score_to_add = phrase_score
        else:
            # Exact pattern overlap exists. Add phrase score only if it's HIGHER
            # than the pattern score already added. Add the difference.
            if phrase_score > pattern_score_here:
                 score_to_add = phrase_score - pattern_score_here
                 # Optional: Could update match_details to reflect the higher-scoring rule?

        if score_to_add > 0:
            total_score += score_to_add
            rule_id_str = nlp_model.vocab.strings[match_id] # "AI_PHRASES_CONCEPTS"

            # Add details for reporting (only if score was added)
            context_start = max(0, start - 5)
            context_end = min(len(doc), end + 5)
            context = doc[context_start:context_end].text.replace("\n", " ")
            # Use a generic Rule ID like 'PHRASE' or maybe indicate if it was Core?
            display_rule_id = f"CORE_PHRASE ({rule_id_str})" if matched_text_lower in CORE_PHRASES_SET else rule_id_str
            match_details.append((display_rule_id, matched_span.text, f"...{context}..."))


    # --- Final Classification ---
    is_ai_related = total_score >= threshold

    # Sort match details by appearance in text for cleaner output
    match_details.sort(key=lambda item: text.find(item[1])) # Simple find should be okay

    return is_ai_related, round(total_score, 2), match_details

def main():
    """Example usage and testing with improved test case structure"""
    print("\nInitializing matchers...")
    phrase_matcher, matcher = initialize_matchers(nlp)
    print("\n--- Classification Examples ---")

    # Define test cases as a single list of tuples: (text, expected_outcome_bool)
    # Grouped for readability: AI-related first (True), then non-AI (False)
    test_cases = [
        # --- Expected AI-Related (True) ---
        # Original Clearly AI
        ("Researchers developed a new deep learning model for natural language processing.", True),
        ("The company uses machine learning algorithms to predict customer churn.", True),
        ("This paper discusses advances in reinforcement learning agents for robotics.", True),
        ("We need to fine-tune the transformer model (like BERT or GPT-4) using our training data.", True),
        ("OpenAI released its latest generative AI, showcasing impressive text synthesis.", True),
        ("Autonomous vehicles rely heavily on computer vision and sensor fusion.", True),
        ("They are training a large language model on a massive text corpus.", True),
        ("Backpropagation is essential for training deep neural networks.", True),
        # Original Borderline/Fixed Cases Expected True
        ("His research is about network optimization algorithms.", True), # Fixed FN
        ("Using gradient descent helps find the minimum of a function.", True), # Fixed FN? (Check score)
        ("The new processor features an AI engine.", True), # Fixed FN
        ("Machine translation has improved significantly.", True),
        ("Build a model that predicts sales.", True), # Fixed FN
        # New Article Headlines (all True)
        ("DeepMind's AlphaFold revolutionizes protein structure prediction.", True),
        ("New AI model generates realistic images from text prompts.", True),
        ("Machine learning algorithm detects fraudulent transactions with high accuracy.", True),
        ("Researchers develop a novel neural network architecture for image recognition.", True),
        ("The chatbot provides instant customer support using natural language processing.", True),
        ("Autonomous vehicles are being tested in California.", True),
        ("Generative AI is transforming the creative industry.", True),
        ("Large language models show impressive capabilities in text generation and understanding.", True),
        ("This article discusses the ethical implications of artificial intelligence.", True),
        ("Computer vision technology is used in facial recognition systems.", True),
        ("The company announced a breakthrough in quantum machine learning.", True),
        ("Explainable AI aims to make AI decisions more transparent.", True),
        ("Reinforcement learning agent masters complex video games.", True),
        ("AI-powered robots are being used in manufacturing.", True),
        ("The latest advancements in natural language generation.", True),
        ("Transformer models are the foundation of many modern NLP applications.", True),
        ("The use of deep learning in medical diagnosis is rapidly increasing.", True),
        ("A new study explores the bias in machine learning algorithms.", True),
        ("Researchers are working on neurosymbolic AI approaches.", True),

        # --- Expected NOT AI-Related (False) ---
        # Original Clearly NOT AI
        ("The train was delayed due to technical issues with the engine.", False),
        ("Birds build nests in spring using twigs and leaves.", False),
        ("The chef decided to process the vegetables for the soup.", False),
        ("Financial analysts analyze market trends daily.", False),
        ("Let's learn the basics of Python programming.", False),
        ("The electrical transformer needs maintenance.", False), # Note: Score might be 1.0, but below threshold
        ("Company policy requires managers to evaluate employee performance quarterly.", False),
        # Original Borderline Cases Expected False
        ("The system learns user preferences over time.", False), # 'system learns' is weak context
        ("Can we optimize the supply chain using data analysis?", False), # 'optimize' needs AI noun context
        ("The study involves statistical regression analysis.", False), # Fixed FP (removed 'regression')
        ("We need to process the images before uploading.", False), # 'process' needs AI noun context
        ("The team will generate a report based on the findings.", False), # 'generate' needs AI context
        ("Analyze the data to find patterns.", False), # Too general
        ("Data processing is the first step.", False), # Too general
    ]

    # Run classification
    correct_count = 0
    total_count = len(test_cases)

    for i, (text, expected_is_ai) in enumerate(test_cases):
        actual_is_ai, score, details = classify_ai_text(text, nlp, phrase_matcher, matcher)

        # Determine outcome
        expected_str = "AI" if expected_is_ai else "Non-AI"
        outcome = "Correct"
        if actual_is_ai != expected_is_ai:
            outcome = f"MISCLASSIFIED (Expected {expected_str})"
        else:
            correct_count += 1

        print(f"\n[{i+1}/{total_count}] Text: {text[:90]}...")
        print(f"   Is AI-related: {actual_is_ai} (Score: {score:.2f} / Threshold: {SCORE_THRESHOLD:.2f}) - {outcome}")
        if details:
            print("   Matches Found:")
            # Match details are now pre-sorted by classify_ai_text
            for rule, match_text, context in details:
                # Pad rule name for alignment
                print(f"     - Rule: {rule:<20} Match: '{match_text}' ({context})")
        else:
            print("   Matches Found: None")

    # Print summary
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("\n--- Classification Summary ---")
    print(f"Total Cases: {total_count}")
    print(f"Correctly Classified: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")


    print("\n--- Tuning Suggestions ---")
    print(f"Current threshold: {SCORE_THRESHOLD}")
    print("Review MISCLASSIFIED examples above.")
    print("If you see too many False Positives (non-AI text classified as AI), INCREASE the threshold or REMOVE/REFINE ambiguous keywords/patterns.")
    print("If you see too many False Negatives (AI text classified as non-AI), DECREASE the threshold or ADD more specific keywords/patterns/rules, or INCREASE weights.")
    print("Consider using a larger model ('en_core_web_lg') for potentially better accuracy if performance allows.")

if __name__ == "__main__":
    main()