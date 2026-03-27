"""
TurboQuant Validation Benchmark — LongBench-E Quality Evaluation

Replicates the paper's Section 4.3 / Table 1 (arXiv:2504.19874).
Evaluates KV cache quantization quality across 6 task categories using
F1 score as the universal metric.

Usage (multi-config, launches servers automatically):
    python eval_longbench.py --port 30000

Usage (against running server):
    python eval_longbench.py --base-url http://localhost:30000 --config bf16

Usage (custom model and configs):
    python eval_longbench.py --model /path/to/model --configs bf16 turboquant_3.5bit
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

# Add parent to path for common module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    SERVER_CONFIGS,
    DEFAULT_PORT,
    launch_server,
    shutdown_server,
    generate_text,
    save_results,
)

# ---------------------------------------------------------------------------
# LongBench-E configuration
# ---------------------------------------------------------------------------

LONGBENCH_SUBSETS = {
    "SingleQA": ["qasper", "multifieldqa_en"],
    "MultiQA": ["hotpotqa", "2wikimqa"],
    "Summarization": ["gov_report", "multi_news"],
    "Few-shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}

MAX_EXAMPLES_PER_SUBSET = 25
MAX_GEN_TOKENS = 256
NUM_WORKERS = 8

# Default to pure transformer model for maximum TurboQuant benefit
DEFAULT_MODEL = "/home/keko/AI/image-prep/models/qwen2.5-3b-instruct"


# ---------------------------------------------------------------------------
# F1 scoring
# ---------------------------------------------------------------------------

def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def score_example(prediction: str, answers: list[str]) -> float:
    """Max F1 across all gold answers for a single example."""
    if not answers:
        return 0.0
    return max(f1_score(prediction, ans) for ans in answers)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_longbench_subset(subset_name: str, max_examples: int) -> list[dict] | None:
    """Load a LongBench subset from HuggingFace. Returns None on failure."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[longbench] ERROR: `datasets` package not installed. Run: pip install datasets")
        return None

    try:
        ds = load_dataset("THUDM/LongBench", subset_name, split="test",
                          trust_remote_code=True)
    except Exception as e:
        print(f"[longbench] WARNING: Failed to load subset '{subset_name}': {e}")
        return None

    examples = []
    for i, item in enumerate(ds):
        if i >= max_examples:
            break
        # Parse answers — may be a list or JSON string
        raw_answers = item.get("answers", item.get("answer", ""))
        if isinstance(raw_answers, str):
            try:
                answers = json.loads(raw_answers)
                if isinstance(answers, str):
                    answers = [answers]
            except (json.JSONDecodeError, TypeError):
                answers = [raw_answers] if raw_answers else []
        elif isinstance(raw_answers, list):
            answers = raw_answers
        else:
            answers = [str(raw_answers)]

        examples.append({
            "context": item.get("context", ""),
            "input": item.get("input", ""),
            "answers": answers,
            "dataset": item.get("dataset", subset_name),
            "length": item.get("length", 0),
        })

    return examples


def build_prompt(example: dict) -> str:
    """Build the evaluation prompt for a LongBench example."""
    context = example["context"]
    question = example["input"]

    # Truncate very long contexts to avoid exceeding context window
    # (~6K tokens budget for context, rough 4 chars/token estimate)
    max_context_chars = 24000
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n[...truncated...]"

    return f"Context: {context}\n\n{question}\n\nAnswer concisely."


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_subset(
    base_url: str,
    subset_name: str,
    examples: list[dict],
    num_workers: int = NUM_WORKERS,
) -> dict:
    """Evaluate a single subset. Returns metrics dict with per-example details."""
    predictions = [None] * len(examples)
    prompts = [build_prompt(ex) for ex in examples]

    def generate_one(idx):
        try:
            text = generate_text(
                base_url,
                prompts[idx],
                max_tokens=MAX_GEN_TOKENS,
                temperature=0.0,
            )
            # Strip <think>...</think> reasoning if present (Qwen3.5)
            if "</think>" in text:
                text = text.split("</think>")[-1]
            predictions[idx] = text.strip()
        except Exception as e:
            print(f"[longbench] WARNING: Generation failed for {subset_name}[{idx}]: {e}")
            predictions[idx] = ""

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(generate_one, range(len(examples))),
            total=len(examples),
            desc=f"  {subset_name}",
        ))

    # Score
    scores = []
    per_example = []
    for i, (ex, pred) in enumerate(zip(examples, predictions)):
        f1 = score_example(pred or "", ex["answers"])
        scores.append(f1)
        per_example.append({
            "idx": i,
            "input_preview": ex["input"][:100],
            "prediction": pred or "",
            "answers": ex["answers"],
            "f1": round(f1, 4),
        })

    avg_f1 = sum(scores) / len(scores) if scores else 0.0

    return {
        "subset": subset_name,
        "num_examples": len(examples),
        "avg_f1": round(avg_f1, 4),
        "per_example": per_example,
    }


def run_longbench(
    base_url: str,
    max_examples: int = MAX_EXAMPLES_PER_SUBSET,
    num_workers: int = NUM_WORKERS,
) -> dict:
    """Run LongBench-E evaluation across all categories.

    Returns dict with category scores and overall average.
    """
    category_scores = {}
    all_subset_results = {}

    for category, subsets in LONGBENCH_SUBSETS.items():
        subset_f1s = []
        for subset_name in subsets:
            print(f"\n[longbench] Loading {subset_name}...")
            examples = load_longbench_subset(subset_name, max_examples)
            if examples is None or len(examples) == 0:
                print(f"[longbench] Skipping {subset_name} (no data)")
                continue

            print(f"[longbench] Evaluating {subset_name} ({len(examples)} examples)")
            result = evaluate_subset(base_url, subset_name, examples, num_workers)
            all_subset_results[subset_name] = result
            subset_f1s.append(result["avg_f1"])
            print(f"[longbench] {subset_name}: F1 = {result['avg_f1']:.4f}")

        if subset_f1s:
            category_avg = sum(subset_f1s) / len(subset_f1s)
        else:
            category_avg = 0.0
        category_scores[category] = round(category_avg, 4)
        print(f"[longbench] {category} avg F1: {category_avg:.4f}")

    # Overall average
    valid_scores = [v for v in category_scores.values() if v > 0]
    overall_avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    metrics = {
        "category_scores": category_scores,
        "overall_avg_f1": round(overall_avg, 4),
        "subset_results": all_subset_results,
    }

    return metrics


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_longbench_table(all_metrics: dict[str, dict]):
    """Print the paper's Table 1 format comparison."""
    categories = list(LONGBENCH_SUBSETS.keys())
    short_names = {
        "SingleQA": "SingleQA",
        "MultiQA": "MultiQA",
        "Summarization": "Summ.",
        "Few-shot": "Few-shot",
        "Synthetic": "Synth.",
        "Code": "Code",
    }

    print(f"\n{'=' * 90}")
    print(f"  LONGBENCH-E — Quality Comparison (F1 Score × 100)")
    print(f"{'=' * 90}")

    # Header
    header = f"{'Config':<28}"
    for cat in categories:
        header += f"{short_names.get(cat, cat):>10}"
    header += f"{'Avg':>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for config_name, metrics in all_metrics.items():
        name = SERVER_CONFIGS.get(config_name, {}).get("name", config_name)
        cat_scores = metrics.get("category_scores", {})
        row = f"{name:<28}"
        for cat in categories:
            score = cat_scores.get(cat, 0.0)
            row += f"{score * 100:>10.1f}"
        row += f"{metrics.get('overall_avg_f1', 0.0) * 100:>10.1f}"
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant LongBench-E quality benchmark (paper Section 4.3 / Table 1)"
    )
    parser.add_argument("--base-url", type=str, default=None,
                        help="Use an already-running server instead of launching one")
    parser.add_argument("--config", type=str, default=None,
                        choices=list(SERVER_CONFIGS.keys()),
                        help="Run only this config (requires --base-url)")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=["bf16", "turboquant_3.5bit"],
                        choices=list(SERVER_CONFIGS.keys()),
                        help="Configs to benchmark (default: bf16 turboquant_3.5bit)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES_PER_SUBSET,
                        help=f"Max examples per subset (default: {MAX_EXAMPLES_PER_SUBSET})")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help=f"Concurrent generation workers (default: {NUM_WORKERS})")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    all_metrics = {}

    if args.base_url and args.config:
        # Single run against existing server
        print(f"\n[longbench] Evaluating {args.config} at {args.base_url}")
        metrics = run_longbench(args.base_url, args.max_examples, args.num_workers)
        save_results("longbench", args.config, metrics, args.output_dir)
        all_metrics[args.config] = metrics
        print_longbench_table(all_metrics)
        return

    # Multi-config: launch servers sequentially
    configs = [args.config] if args.config else args.configs
    for config_name in configs:
        cfg = SERVER_CONFIGS[config_name]
        print(f"\n{'#' * 60}")
        print(f"# Config: {cfg['name']}")
        print(f"{'#' * 60}")

        proc = None
        try:
            proc = launch_server(
                config_name,
                port=args.port,
                timeout=300,
                context_length=8192,
                model_path=args.model,
            )
            base_url = f"http://127.0.0.1:{args.port}"
            metrics = run_longbench(base_url, args.max_examples, args.num_workers)
            save_results("longbench", config_name, metrics, args.output_dir)
            all_metrics[config_name] = metrics
        except Exception as e:
            print(f"ERROR: {config_name} failed: {e}")
        finally:
            if proc:
                shutdown_server(proc)

    # Print comparison table
    if all_metrics:
        print_longbench_table(all_metrics)

        # Print delta vs baseline if we have both
        if len(all_metrics) > 1:
            baseline_cfg = configs[0]
            if baseline_cfg in all_metrics:
                base_avg = all_metrics[baseline_cfg].get("overall_avg_f1", 0)
                for cfg in configs[1:]:
                    if cfg in all_metrics:
                        tq_avg = all_metrics[cfg].get("overall_avg_f1", 0)
                        if base_avg > 0:
                            delta = (tq_avg - base_avg) / base_avg * 100
                            sign = "+" if delta >= 0 else ""
                            print(f"  {SERVER_CONFIGS[cfg]['name']} vs "
                                  f"{SERVER_CONFIGS[baseline_cfg]['name']}: "
                                  f"{sign}{delta:.1f}% avg F1")
                print()


if __name__ == "__main__":
    main()
