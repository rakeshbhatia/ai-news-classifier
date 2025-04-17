import spacy
from ai_classifier import initialize_matchers, classify_ai_text
from typing import List, Tuple

TEST_HEADLINES_BATCH_1 = [
    # ================= AI-RELATED (TRUE) =================
    ("Microsoft's Phi-3 mini outperforms Llama 3 in language benchmarks", True),
    ("Hugging Face releases fine-tuned Mistral-7B variant for coding tasks", True),
    ("Study: 78% of enterprises now use AI-powered data analytics tools", True),
    ("NVIDIA's new AI chips achieve 30% faster training speeds", True),
    ("Meta open-sources multi-modal Llama 3 vision-language model", True),
    ("GitHub Copilot X introduces AI-powered command line interface", True),
    ("Researchers develop brain-inspired neural network for edge devices", True),
    ("Stability AI announces Stable Diffusion 3 with improved text rendering", True),
    ("EU passes landmark AI Act regulating generative systems", True),
    ("Amazon Bedrock now supports Anthropic's Claude 3 Opus model", True),
    ("AI-generated synthetic data could solve privacy challenges, experts claim", True),
    ("DeepMind's new AlphaFold 3 predicts protein interactions with 80% accuracy", True),
    ("Tesla Optimus robot demonstrates improved AI-based object manipulation", True),
    ("Apple's MM1 model brings multimodal AI to iPhones", True),
    ("AI winter or spring? Venture capital trends show 40% YoY growth", True),
    ("Lawmakers propose requiring watermarks for AI-generated political ads", True),
    ("OpenAI's Sora generates 1-minute videos from text prompts", True),
    ("AI-assisted drug discovery reduces trial phases by 6 months", True),
    ("ChatGPT passes U.S. medical licensing exam with 85% accuracy", True),
    ("Computer vision system detects manufacturing defects with 99.2% precision", True),
    ("Generative AI tools now used in 65% of game studios", True),
    ("AI ethics debate intensifies over copyright infringement cases", True),
    ("Baidu's ERNIE 4.0 shows 20% improvement in Chinese NLP tasks", True),
    ("AI-powered trading algorithms now handle 35% of NASDAQ volume", True),
    ("MIT researchers develop AI that explains its reasoning", True),
    
    # ================= NON-AI (FALSE) =================
    ("Federal Reserve signals potential rate cuts amid inflation concerns", False),
    ("Python 3.12 introduces significant performance improvements", False),
    ("Tesla recalls 125,000 vehicles over seat belt warning issue", False),
    ("CRISPR gene editing shows promise for rare genetic disorders", False),
    ("Global wheat prices surge after Russian export restrictions", False),
    ("New study links Mediterranean diet to reduced dementia risk", False),
    ("Apple fixes iOS zero-day vulnerability exploited in the wild", False),
    ("Quantum computing breakthrough achieves 128-qubit entanglement", False),
    ("UN report warns of accelerating biodiversity loss worldwide", False),
    ("Spotify announces price hike for premium subscriptions", False),
    ("Neuroscience study reveals new insights about memory formation", False),
    ("Boeing Starliner launch delayed due to technical issues", False),
    ("FDA approves new weight-loss drug with 15% efficacy rate", False),
    ("Solar eclipse draws millions of viewers across North America", False),
    ("VinFast unveils new electric SUV with 300-mile range", False),
    ("James Webb Telescope detects earliest known galaxy", False),
    ("Bitcoin surges past $70,000 amid ETF approval rumors", False),
    ("COVID-19 variant KP.2 shows increased immune evasion", False),
    ("Major earthquake strikes Taiwan, tsunami warning issued", False),
    ("Taylor Swift's Eras Tour breaks $1 billion revenue mark", False),
    ("Scientists discover new octopus species in Pacific depths", False),
    ("Netflix ad-tier subscriptions grow by 8 million in Q2", False),
    ("Ford recalls 2023 F-150 trucks over brake system flaw", False),
    ("Global coffee shortage expected after Brazilian drought", False),
    ("Quantum computing breakthrough achieves 128-qubit entanglement", False),  # Sci-tech but not AI
    ("AI (Artificial Insemination) tech transforms cattle breeding", False),  # Term collision
    ("Generative art exhibit features traditional mediums only", False)  # Term misuse
]

def run_tests(test_cases: List[Tuple[str, bool]]) -> dict:
    """Execute classification tests and return performance metrics."""
    nlp = spacy.load("en_core_web_sm")
    phrase_matcher, verb_matcher = initialize_matchers(nlp)
    
    results = {
        'true_positives': 0,
        'true_negatives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'errors': []
    }
    
    for text, expected in test_cases:
        try:
            prediction = classify_ai_text(text, nlp, phrase_matcher, verb_matcher)
            if expected and prediction:
                results['true_positives'] += 1
            elif not expected and not prediction:
                results['true_negatives'] += 1
            elif not expected and prediction:
                results['false_positives'] += 1
                results['errors'].append(f"FP: {text}")
            else:
                results['false_negatives'] += 1
                results['errors'].append(f"FN: {text}")
        except Exception as e:
            print(f"Error processing: '{text}' - {str(e)}")
    
    total = len(test_cases)
    results.update({
        'accuracy': (results['true_positives'] + results['true_negatives']) / total,
        'precision': results['true_positives'] / (results['true_positives'] + results['false_positives'] + 1e-10),
        'recall': results['true_positives'] / (results['true_positives'] + results['false_negatives'] + 1e-10),
        'f1': 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'] + 1e-10)
    })
    
    return results

def print_results(results: dict):
    """Display test results in human-readable format."""
    print("\n=== Test Results ===")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall: {results['recall']:.2%}")
    print(f"F1 Score: {results['f1']:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"True Positives: {results['true_positives']}")
    print(f"True Negatives: {results['true_negatives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")
    
    if results['errors']:
        print("\n=== Error Cases ===")
        for error in results['errors']:
            print(error)

if __name__ == "__main__":
    test_results = run_tests(TEST_HEADLINES_BATCH_1)
    print_results(test_results)