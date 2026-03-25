"""
TurboQuant Validation Benchmark — Needle in a Haystack

Tests retrieval accuracy by inserting unique facts at varying depths
in filler text and asking the model to recall them.
Target: >95% accuracy (paper reports 0.997 at 4K+).

Usage (standalone, server already running):
    python eval_needle.py --port 30000
"""

import argparse
import re

from tqdm import tqdm

from common import (
    DEFAULT_PORT,
    MODEL_PATH,
    generate_text,
    print_comparison_table,
    save_results,
)

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

NEEDLE_FACTS = [
    ("The secret code for project alpha is 847293.",
     "What is the secret code for project alpha?", "847293"),
    ("The headquarters of Zenith Corp is located in Springfield.",
     "Where is the headquarters of Zenith Corp?", "Springfield"),
    ("The launch date for Operation Mercury is March 15th 2027.",
     "What is the launch date for Operation Mercury?", "March 15th 2027"),
    ("Dr. Elena Vasquez won the Turing Award in 2024.",
     "Who won the Turing Award in 2024?", "Elena Vasquez"),
    ("The maximum capacity of warehouse delta is 42000 units.",
     "What is the maximum capacity of warehouse delta?", "42000"),
]

CONTEXT_LENGTHS = [1024, 2048, 3072, 4096]
NEEDLE_DEPTHS = [0.10, 0.25, 0.50, 0.75, 0.90]

# Generic filler paragraph (~80 tokens when tokenized)
FILLER_PARAGRAPH = (
    "The principles of thermodynamics govern the behavior of energy in physical "
    "systems. Heat naturally flows from regions of higher temperature to regions "
    "of lower temperature until thermal equilibrium is reached. This fundamental "
    "law has profound implications for engineering, chemistry, and cosmology alike. "
    "Researchers continue to explore novel materials and methods to improve energy "
    "efficiency in industrial processes and everyday applications around the globe. "
)


def _build_filler_text(num_paragraphs: int) -> str:
    """Generate filler text of approximately *num_paragraphs* paragraphs."""
    return "\n\n".join([FILLER_PARAGRAPH] * num_paragraphs)


_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer as _get_tok
        _tokenizer = _get_tok(MODEL_PATH)
    return _tokenizer


def _token_length(text: str) -> int:
    return len(_get_tokenizer().encode(text))


def build_test_case(
    needle_fact: str,
    question: str,
    target_context_tokens: int,
    depth: float,
) -> str:
    """Build a prompt with filler text, a needle inserted at *depth*, and a question."""
    # Estimate tokens for the question suffix
    question_suffix = f"\n\nAnswer the following question with ONLY the answer value, nothing else. Do not explain.\nQuestion: {question}\nAnswer:"
    question_tokens = _token_length(question_suffix)
    needle_tokens = _token_length(needle_fact)

    # Available tokens for filler
    available = target_context_tokens - question_tokens - needle_tokens - 20  # margin
    if available < 50:
        available = 50

    # Build filler — each paragraph is ~80 tokens
    num_paragraphs = max(1, available // 80)
    filler = _build_filler_text(num_paragraphs)

    # Trim filler to approximate target
    paragraphs = filler.split("\n\n")
    while _token_length("\n\n".join(paragraphs)) > available and len(paragraphs) > 1:
        paragraphs.pop()

    # Insert needle at the given depth
    insert_idx = max(1, int(len(paragraphs) * depth))
    insert_idx = min(insert_idx, len(paragraphs))
    paragraphs.insert(insert_idx, needle_fact)

    context = "\n\n".join(paragraphs)
    prompt = context + question_suffix
    return prompt


def _check_answer(response: str, expected: str) -> bool:
    """Check if the expected value appears in the response."""
    # Normalize whitespace/case for comparison
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()
    if expected_lower in response_lower:
        return True
    # Try numeric extraction for numeric answers
    numbers = re.findall(r"\d+", expected)
    if numbers:
        return all(n in response for n in numbers)
    return False


def run_needle_benchmark(base_url: str) -> dict:
    """Run needle-in-haystack across all context lengths and depths."""
    total = 0
    correct = 0
    results_detail = []

    test_cases = []
    for ctx_len in CONTEXT_LENGTHS:
        for depth in NEEDLE_DEPTHS:
            fact_idx = (len(test_cases)) % len(NEEDLE_FACTS)
            needle_fact, question, expected = NEEDLE_FACTS[fact_idx]
            test_cases.append((ctx_len, depth, needle_fact, question, expected))

    for ctx_len, depth, needle_fact, question, expected in tqdm(
        test_cases, desc="Needle tests"
    ):
        prompt = build_test_case(needle_fact, question, ctx_len, depth)
        response = generate_text(base_url, prompt, max_tokens=128, temperature=0.0)
        # Strip <think>...</think> reasoning if present
        if "</think>" in response:
            response = response.split("</think>")[-1]
        hit = _check_answer(response, expected)
        total += 1
        if hit:
            correct += 1
        results_detail.append({
            "context_length": ctx_len,
            "depth": depth,
            "expected": expected,
            "response": response.strip()[:100],
            "correct": hit,
        })

    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "details": results_detail,
    }
    print(f"[needle] Accuracy: {correct}/{total} = {accuracy:.3f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Needle-in-haystack benchmark")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    metrics = run_needle_benchmark(base_url)

    if args.config_name:
        save_results("needle", args.config_name, metrics, args.output_dir)

    return metrics


if __name__ == "__main__":
    main()
