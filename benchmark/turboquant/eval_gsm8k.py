"""
TurboQuant Validation Benchmark — GSM8K Arithmetic Reasoning

10-shot prompted evaluation on GSM8K test set.
Target: <2% accuracy drop vs BF16 baseline (arXiv:2504.19874).

Usage (standalone, server already running):
    python eval_gsm8k.py --port 30000 [--num-examples 50]

Usage (multi-config, manages server lifecycle):
    python eval_gsm8k.py --configs bf16 turboquant_3.5bit --model /path/to/model
"""

import argparse
import ast
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from common import (
    DEFAULT_PORT,
    SERVER_CONFIGS,
    check_target,
    generate_text,
    launch_server,
    print_comparison_table,
    save_results,
    shutdown_server,
)

from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

NUM_SHOTS = 10
NUM_EXAMPLES = 500
START_INDEX = 10  # skip first 10 (used as few-shot examples)
NUM_WORKERS = 8


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
        # Strip <think>...</think> reasoning if present (Qwen3.5)
        if "</think>" in response:
            response = response.split("</think>")[-1]
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


def run_multi_config(args) -> int:
    """Run GSM8K eval across multiple configs, managing server lifecycle."""
    all_results: dict[str, dict] = {}

    for config_name in args.configs:
        print(f"\n{'#' * 60}")
        print(f"  GSM8K — {SERVER_CONFIGS[config_name]['name']}")
        print(f"{'#' * 60}")

        proc = None
        try:
            proc = launch_server(
                config_name,
                port=args.port,
                context_length=args.context_length,
                model_path=args.model,
            )
            base_url = f"http://127.0.0.1:{args.port}"

            t0 = time.time()
            metrics = run_gsm8k_benchmark(
                base_url, num_examples=args.num_examples,
            )
            metrics["elapsed_seconds"] = round(time.time() - t0, 1)
            save_results("gsm8k", config_name, metrics, args.output_dir)
            all_results[config_name] = metrics

        except Exception as exc:
            print(f"  ERROR running {config_name}: {exc}")
        finally:
            if proc is not None:
                shutdown_server(proc)
            time.sleep(5)

    # Print comparison table
    if all_results:
        display = {}
        for cfg, m in all_results.items():
            display[cfg] = {
                "accuracy": m["accuracy"],
                "correct": f"{m['correct']}/{m['total']}",
                "invalid": m["invalid"],
            }
        print_comparison_table("gsm8k", display)

        # PASS/FAIL check
        if "bf16" in all_results:
            baseline = all_results["bf16"]
            for cfg in all_results:
                if cfg == "bf16":
                    continue
                passed, msg = check_target("gsm8k", baseline, all_results[cfg])
                status = "PASS" if passed else "FAIL"
                name = SERVER_CONFIGS.get(cfg, {}).get("name", cfg)
                print(f"  [{status}] {name}: {msg}")
            print()

    return 0


def main():
    parser = argparse.ArgumentParser(description="GSM8K 10-shot benchmark")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES)
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--model", type=str,
                        default="/home/keko/AI/image-prep/models/qwen2.5-3b-instruct",
                        help="Model path for server")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Multi-config mode: launch server per config (e.g. bf16 turboquant_3.5bit)")
    parser.add_argument("--context-length", type=int, default=4096)
    args = parser.parse_args()

    # Multi-config mode: manage server lifecycle internally
    if args.configs:
        for c in args.configs:
            if c not in SERVER_CONFIGS:
                print(f"Unknown config: {c}. Choose from: {list(SERVER_CONFIGS)}")
                return 1
        return run_multi_config(args)

    # Single-config mode: server already running
    base_url = f"http://127.0.0.1:{args.port}"
    metrics = run_gsm8k_benchmark(base_url, num_examples=args.num_examples)

    if args.config_name:
        save_results("gsm8k", args.config_name, metrics, args.output_dir)

    return metrics


if __name__ == "__main__":
    main()
