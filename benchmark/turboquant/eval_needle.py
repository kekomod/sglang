"""
TurboQuant Validation Benchmark — Needle in a Haystack

Tests retrieval accuracy by inserting unique facts at varying depths
in filler text and asking the model to recall them.
Target: >95% accuracy (paper reports 0.997 at 4K+).

Usage (standalone, server already running):
    python eval_needle.py --port 30000

Multi-config mode (launches/kills servers automatically):
    python eval_needle.py --configs bf16 turboquant_3.5bit --context-lengths 4096 8192 16384 32768

Long-context comparison (Qwen2.5-3B):
    python eval_needle.py --model /home/keko/AI/image-prep/models/qwen2.5-3b-instruct \
        --configs bf16 turboquant_3.5bit --context-lengths 4096 8192 16384 32768
"""

import argparse
import re
import subprocess as _subprocess
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DEFAULT_PORT,
    MODEL_PATH,
    SERVER_CONFIGS,
    generate_text,
    launch_server,
    print_comparison_table,
    save_results,
    shutdown_server,
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

NEEDLE_DEPTHS = [0.10, 0.25, 0.50, 0.75, 0.90]

# Diverse filler paragraphs (~80 tokens each) — different topics
FILLER_PARAGRAPHS = [
    (
        "The principles of thermodynamics govern the behavior of energy in physical "
        "systems. Heat naturally flows from regions of higher temperature to regions "
        "of lower temperature until thermal equilibrium is reached. This fundamental "
        "law has profound implications for engineering, chemistry, and cosmology alike. "
        "Researchers continue to explore novel materials and methods to improve energy "
        "efficiency in industrial processes and everyday applications around the globe."
    ),
    (
        "The Amazon rainforest is the largest tropical rainforest in the world, covering "
        "approximately 5.5 million square kilometers across nine countries. It is home "
        "to an estimated 10 percent of all species on Earth, including jaguars, river "
        "dolphins, and over 40,000 plant species. The forest plays a critical role in "
        "regulating the global climate by absorbing vast quantities of carbon dioxide "
        "and releasing oxygen through photosynthesis every day of the year."
    ),
    (
        "The Renaissance was a period of cultural and intellectual transformation that "
        "began in Italy during the 14th century and spread across Europe over the next "
        "three hundred years. Artists like Leonardo da Vinci and Michelangelo produced "
        "masterpieces that continue to influence art and architecture today. The era also "
        "saw significant advances in science, philosophy, and literature, reshaping how "
        "people understood the natural world and their place within it."
    ),
    (
        "Modern computing relies heavily on semiconductor technology, where silicon "
        "wafers are etched with billions of transistors to create integrated circuits. "
        "These chips power everything from smartphones and laptops to data centers and "
        "autonomous vehicles. The relentless pace of miniaturization, described by "
        "Moore's Law, has driven exponential improvements in processing power while "
        "reducing costs over the past five decades of technological advancement."
    ),
    (
        "The human immune system is a complex network of cells, tissues, and organs "
        "that work together to defend the body against harmful pathogens. White blood "
        "cells, including T-cells and B-cells, identify and neutralize bacteria, viruses, "
        "and other foreign invaders. Vaccines train the immune system to recognize specific "
        "threats without causing disease, providing long-lasting protection through the "
        "production of memory cells that remain active for years or even decades."
    ),
    (
        "Coral reefs are among the most biodiverse ecosystems on the planet, supporting "
        "roughly 25 percent of all marine species despite covering less than one percent "
        "of the ocean floor. These underwater structures are built by tiny organisms "
        "called coral polyps, which secrete calcium carbonate to form their skeletons. "
        "Rising ocean temperatures and acidification threaten reef health worldwide, "
        "prompting conservation efforts that include reef restoration and marine reserves."
    ),
    (
        "The ancient Silk Road was a network of trade routes connecting East Asia to the "
        "Mediterranean, facilitating the exchange of goods, ideas, and cultures for over "
        "1,500 years. Merchants transported silk, spices, precious metals, and gemstones "
        "across deserts and mountain ranges, often traveling in large caravans for safety. "
        "The routes also enabled the spread of religions, technologies, and scientific "
        "knowledge between civilizations that might otherwise have remained isolated."
    ),
    (
        "Plate tectonics is the scientific theory explaining the movement of Earth's "
        "lithospheric plates, which float on the semi-fluid asthenosphere beneath them. "
        "These massive plates shift, collide, and separate over millions of years, driving "
        "the formation of mountains, ocean trenches, and volcanic island chains. Earthquakes "
        "and volcanic eruptions are concentrated along plate boundaries where the mechanical "
        "stresses of plate interactions are greatest and most frequently released."
    ),
    (
        "Jazz music originated in the early 20th century in New Orleans, drawing on "
        "African American musical traditions including blues, ragtime, and spirituals. "
        "Characterized by improvisation, syncopated rhythms, and complex harmonies, jazz "
        "quickly spread across the United States and around the world. Legendary musicians "
        "like Louis Armstrong, Duke Ellington, and Miles Davis pushed the boundaries of "
        "the genre, influencing virtually every form of popular music that followed."
    ),
    (
        "The International Space Station orbits Earth approximately 400 kilometers above "
        "the surface, traveling at roughly 28,000 kilometers per hour. This collaborative "
        "project involves space agencies from the United States, Russia, Europe, Japan, "
        "and Canada. Astronauts aboard the station conduct experiments in microgravity, "
        "studying everything from crystal growth and fluid dynamics to human physiology "
        "and plant biology in preparation for future long-duration missions to Mars."
    ),
    (
        "Photosynthesis is the process by which green plants and certain other organisms "
        "convert light energy into chemical energy stored in glucose molecules. This "
        "reaction occurs primarily in the chloroplasts of plant cells, where chlorophyll "
        "absorbs sunlight and uses it to split water molecules, releasing oxygen as a "
        "byproduct. The glucose produced serves as the primary energy source for the "
        "plant and forms the base of nearly all food chains on Earth."
    ),
    (
        "The field of artificial intelligence has experienced remarkable progress in "
        "recent years, driven by advances in deep learning and the availability of large "
        "datasets. Neural networks can now recognize images, translate languages, and "
        "generate human-like text with impressive accuracy. However, challenges remain "
        "in areas such as reasoning, common-sense understanding, and ensuring that AI "
        "systems behave safely and fairly across diverse populations and use cases."
    ),
]


def _build_filler_text(num_paragraphs: int) -> str:
    """Generate filler text using diverse paragraphs, cycling through topics."""
    paragraphs = []
    for i in range(num_paragraphs):
        paragraphs.append(FILLER_PARAGRAPHS[i % len(FILLER_PARAGRAPHS)])
    return "\n\n".join(paragraphs)


_tokenizer = None
_tokenizer_model = None


def _get_tokenizer(model_path: str = MODEL_PATH):
    global _tokenizer, _tokenizer_model
    if _tokenizer is None or _tokenizer_model != model_path:
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer as _get_tok
        _tokenizer = _get_tok(model_path)
        _tokenizer_model = model_path
    return _tokenizer


def _token_length(text: str, model_path: str = MODEL_PATH) -> int:
    return len(_get_tokenizer(model_path).encode(text))


def build_test_case(
    needle_fact: str,
    question: str,
    target_context_tokens: int,
    depth: float,
    model_path: str = MODEL_PATH,
) -> str:
    """Build a prompt with filler text, a needle inserted at *depth*, and a question."""
    question_suffix = (
        f"\n\nAnswer the following question with ONLY the answer value, nothing else. "
        f"Do not explain.\nQuestion: {question}\nAnswer:"
    )
    question_tokens = _token_length(question_suffix, model_path)
    needle_tokens = _token_length(needle_fact, model_path)

    # Available tokens for filler
    available = target_context_tokens - question_tokens - needle_tokens - 20
    if available < 50:
        available = 50

    # Build filler — each paragraph is ~80 tokens
    num_paragraphs = max(1, available // 80)
    filler = _build_filler_text(num_paragraphs)

    # Trim filler to approximate target
    paragraphs = filler.split("\n\n")
    while _token_length("\n\n".join(paragraphs), model_path) > available and len(paragraphs) > 1:
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
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()
    if expected_lower in response_lower:
        return True
    # Try numeric extraction for numeric answers
    numbers = re.findall(r"\d+", expected)
    if numbers:
        return all(n in response for n in numbers)
    return False


def get_gpu_memory_mb() -> int:
    """Query current GPU memory usage via nvidia-smi."""
    result = _subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(result.stdout.strip().split("\n")[0])


def run_needle_benchmark(
    base_url: str,
    context_lengths: list[int],
    model_path: str = MODEL_PATH,
) -> dict:
    """Run needle-in-haystack across all context lengths and depths.

    Returns metrics dict with accuracy per context length and overall.
    """
    total = 0
    correct = 0
    results_detail = []
    accuracy_by_ctx = {}

    test_cases = []
    for ctx_len in context_lengths:
        for depth in NEEDLE_DEPTHS:
            fact_idx = len(test_cases) % len(NEEDLE_FACTS)
            needle_fact, question, expected = NEEDLE_FACTS[fact_idx]
            test_cases.append((ctx_len, depth, needle_fact, question, expected))

    for ctx_len, depth, needle_fact, question, expected in tqdm(
        test_cases, desc="Needle tests"
    ):
        prompt = build_test_case(needle_fact, question, ctx_len, depth, model_path)
        response = generate_text(base_url, prompt, max_tokens=128, temperature=0.0)
        # Strip <think>...</think> reasoning if present
        if "</think>" in response:
            response = response.split("</think>")[-1]
        hit = _check_answer(response, expected)
        total += 1
        if hit:
            correct += 1

        # Track per context length
        if ctx_len not in accuracy_by_ctx:
            accuracy_by_ctx[ctx_len] = {"correct": 0, "total": 0}
        accuracy_by_ctx[ctx_len]["total"] += 1
        if hit:
            accuracy_by_ctx[ctx_len]["correct"] += 1

        results_detail.append({
            "context_length": ctx_len,
            "depth": depth,
            "expected": expected,
            "response": response.strip()[:100],
            "correct": hit,
        })

    accuracy = correct / total if total > 0 else 0.0
    per_ctx = {}
    for ctx_len, counts in accuracy_by_ctx.items():
        per_ctx[ctx_len] = (
            round(counts["correct"] / counts["total"], 3) if counts["total"] > 0 else 0.0
        )

    metrics = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "accuracy_by_context_length": per_ctx,
        "details": results_detail,
    }
    print(f"[needle] Accuracy: {correct}/{total} = {accuracy:.3f}")
    for ctx_len in context_lengths:
        acc = per_ctx.get(ctx_len, 0.0)
        print(f"  {ctx_len:>6} tokens: {acc:.3f}")
    return metrics


def run_single_config(
    config_name: str,
    port: int,
    context_lengths: list[int],
    model_path: str,
    output_dir: str,
) -> dict:
    """Run needle benchmark for a single config, managing server lifecycle.

    Returns a dict with metrics, gpu memory, and max context reached.
    If the server fails to launch (OOM), returns partial results.
    """
    result = {
        "config_name": config_name,
        "display_name": SERVER_CONFIGS.get(config_name, {}).get("name", config_name),
        "accuracy_by_ctx": {},
        "mem_launch_mb": None,
        "mem_steady_mb": None,
        "max_ctx_reached": 0,
        "oom": False,
        "metrics": None,
    }

    # Determine the maximum context length needed for server launch
    max_ctx = max(context_lengths)

    print(f"\n{'='*60}")
    print(f"  Config: {result['display_name']}")
    print(f"  Context lengths: {context_lengths}")
    print(f"{'='*60}")

    try:
        proc = launch_server(
            config_name, port=port, context_length=max_ctx, model_path=model_path,
        )
    except RuntimeError as e:
        print(f"[needle] Server launch failed for {config_name}: {e}")
        result["oom"] = True
        for ctx_len in context_lengths:
            result["accuracy_by_ctx"][ctx_len] = "OOM"
        return result

    try:
        result["mem_launch_mb"] = get_gpu_memory_mb()

        base_url = f"http://127.0.0.1:{port}"

        # Run benchmark incrementally by context length so we can detect OOM mid-run
        all_details = []
        total_correct = 0
        total_count = 0
        oom_hit = False

        for ctx_len in context_lengths:
            if oom_hit:
                result["accuracy_by_ctx"][ctx_len] = "OOM"
                continue

            try:
                sub_metrics = run_needle_benchmark(
                    base_url, [ctx_len], model_path=model_path,
                )
                acc = sub_metrics["accuracy_by_context_length"].get(ctx_len, 0.0)
                result["accuracy_by_ctx"][ctx_len] = acc
                result["max_ctx_reached"] = ctx_len
                total_correct += sub_metrics["correct"]
                total_count += sub_metrics["total"]
                all_details.extend(sub_metrics.get("details", []))
            except Exception as e:
                print(f"[needle] Error at context {ctx_len} for {config_name}: {e}")
                result["accuracy_by_ctx"][ctx_len] = "OOM"
                oom_hit = True
                # Mark remaining as OOM
                continue

        result["mem_steady_mb"] = get_gpu_memory_mb()

        # Build aggregate metrics
        overall_acc = total_correct / total_count if total_count > 0 else 0.0
        result["metrics"] = {
            "accuracy": round(overall_acc, 4),
            "correct": total_correct,
            "total": total_count,
            "accuracy_by_context_length": {
                k: v for k, v in result["accuracy_by_ctx"].items()
            },
            "mem_launch_mb": result["mem_launch_mb"],
            "mem_steady_mb": result["mem_steady_mb"],
            "details": all_details,
        }

        # Save per-config results
        save_results("needle", config_name, result["metrics"], output_dir)

    finally:
        shutdown_server(proc)

    return result


def print_comparison_grid(
    results: list[dict], context_lengths: list[int],
) -> None:
    """Print a formatted comparison grid across configs and context lengths."""
    print(f"\n{'='*80}")
    print(f"  NEEDLE-IN-HAYSTACK — Comparison Grid")
    print(f"{'='*80}")

    # Header
    ctx_cols = "".join(f"{ctx//1024:>6}K" for ctx in context_lengths)
    header = f"{'Config':<28}{ctx_cols}{'MaxCtx':>9}{'Mem(launch)':>13}{'Mem(steady)':>13}"
    print(header)
    print("-" * len(header))

    for r in results:
        name = r["display_name"]
        row = f"{name:<28}"
        for ctx_len in context_lengths:
            val = r["accuracy_by_ctx"].get(ctx_len, "N/A")
            if isinstance(val, str):
                row += f"{val:>7}"
            else:
                row += f"{val:>7.3f}"
        max_ctx = r["max_ctx_reached"]
        row += f"{max_ctx:>9}"
        mem_l = r.get("mem_launch_mb")
        mem_s = r.get("mem_steady_mb")
        row += f"{(str(mem_l) + ' MB') if mem_l else 'N/A':>13}"
        row += f"{(str(mem_s) + ' MB') if mem_s else 'N/A':>13}"
        print(row)

    print()


def main():
    parser = argparse.ArgumentParser(description="Needle-in-haystack benchmark")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--config-name", type=str, default=None,
                        help="Config name for single-server mode (server already running)")
    parser.add_argument("--model", type=str,
                        default="/home/keko/AI/image-prep/models/qwen2.5-3b-instruct",
                        help="Model path for tokenizer and server launch")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Multi-config mode: config names to compare (launches servers)")
    parser.add_argument("--context-lengths", nargs="+", type=int,
                        default=[4096, 8192, 16384, 32768],
                        help="Context lengths to test (default: 4096 8192 16384 32768)")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    context_lengths = sorted(args.context_lengths)

    # Multi-config mode: launch/kill servers per config
    if args.configs:
        all_results = []
        for config_name in args.configs:
            if config_name not in SERVER_CONFIGS:
                print(f"[needle] WARNING: Unknown config '{config_name}', skipping")
                continue
            r = run_single_config(
                config_name, args.port, context_lengths,
                model_path=args.model, output_dir=args.output_dir,
            )
            all_results.append(r)

        print_comparison_grid(all_results, context_lengths)
        return all_results

    # Single-server mode (server already running)
    base_url = f"http://127.0.0.1:{args.port}"
    metrics = run_needle_benchmark(
        base_url, context_lengths, model_path=args.model,
    )

    if args.config_name:
        save_results("needle", args.config_name, metrics, args.output_dir)

    return metrics


if __name__ == "__main__":
    main()
