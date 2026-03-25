"""
TurboQuant Validation Benchmark — GSM8K Arithmetic Reasoning

10-shot prompted evaluation on GSM8K test set.
Target: <2% accuracy drop vs BF16 baseline (arXiv:2504.19874).

Usage (standalone, server already running):
    python eval_gsm8k.py --port 30000 [--num-examples 50]
"""

import argparse
import ast
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from common import (
    DEFAULT_PORT,
    generate_text,
    save_results,
)

from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

NUM_SHOTS = 10
NUM_EXAMPLES = 50
START_INDEX = 10  # skip first 10 (used as few-shot examples)
NUM_WORKERS = 16


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def run_gsm8k_benchmark(
    base_url: str,
    num_examples: int = NUM_EXAMPLES,
    num_workers: int = NUM_WORKERS,
) -> dict:
    """Run GSM8K 10-shot evaluation."""
    # Download data
    filename = download_and_cache_file(GSM8K_URL)
    lines = list(read_jsonl(filename))

    # Build few-shot prefix
    few_shot = get_few_shot_examples(lines, NUM_SHOTS)

    # Build test questions (indices START_INDEX .. START_INDEX + num_examples)
    end_index = START_INDEX + num_examples
    questions = []
    labels = []
    for i in range(START_INDEX, min(end_index, len(lines))):
        questions.append(get_one_example(lines, i, include_answer=False))
        labels.append(get_answer_value(lines[i]["answer"]))

    assert all(l != INVALID for l in labels), "Some labels could not be parsed"

    predictions = [None] * len(questions)

    def get_one_answer(idx):
        prompt = few_shot + questions[idx]
        response = generate_text(
            base_url,
            prompt,
            max_tokens=256,
            temperature=0.0,
            stop=["Question", "Assistant:", "<|separator|>"],
        )
        predictions[idx] = response

    # Parallel dispatch
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(get_one_answer, range(len(questions))),
            total=len(questions),
            desc="GSM8K",
        ))

    # Score
    preds = [get_answer_value(p) if p else INVALID for p in predictions]
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    invalid_count = sum(1 for p in preds if p == INVALID)
    accuracy = correct / len(labels) if labels else 0.0

    metrics = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(labels),
        "invalid": invalid_count,
    }
    print(f"[gsm8k] Accuracy: {correct}/{len(labels)} = {accuracy:.3f}")
    print(f"[gsm8k] Invalid responses: {invalid_count}/{len(labels)}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="GSM8K 10-shot benchmark")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    metrics = run_gsm8k_benchmark(base_url, num_examples=args.num_examples)

    if args.config_name:
        save_results("gsm8k", args.config_name, metrics, args.output_dir)

    return metrics


if __name__ == "__main__":
    main()
