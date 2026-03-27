"""
TurboQuant Throughput Benchmark

Supports image captioning (multimodal models) and text completion (text-only models).
Measures end-to-end throughput, GPU memory, output quality (cosine sim vs baseline).

Usage:
    # Image benchmark (Qwen3.5-9B, 4 configs, concurrency=16):
    python benchmark/turboquant/eval_throughput.py --num-requests 256 --concurrency 16

    # Text-only benchmark (Qwen2.5-3B):
    python benchmark/turboquant/eval_throughput.py --text-only --num-requests 256 --concurrency 16 \
        --model /home/keko/AI/image-prep/models/qwen2.5-3b-instruct \
        --configs bf16 turboquant_3.5bit

    # Against a running server:
    python benchmark/turboquant/eval_throughput.py --base-url http://localhost:30000 --config turboquant_3.5bit
"""

import argparse
import base64
import json
import math
import random
import subprocess as _subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Add parent to path for common module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    SERVER_CONFIGS,
    launch_server,
    shutdown_server,
    wait_for_health,
    save_results,
    print_comparison_table,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_DIR = Path("/home/keko/AI/image-prep/data/images/processed")
IMAGE_PROMPT = "Describe this image in one detailed sentence."
TEXT_PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a short story about a robot learning to cook.",
    "What are the main differences between Python and Rust?",
    "Describe the process of photosynthesis step by step.",
    "What were the key causes of World War I?",
    "Explain how a neural network learns from data.",
    "Write a poem about the ocean at sunset.",
    "What is the significance of the Turing test?",
    "Describe the water cycle in detail.",
    "What are the pros and cons of nuclear energy?",
    "Explain the concept of supply and demand in economics.",
    "Write a recipe for a simple pasta dish.",
    "What is the difference between weather and climate?",
    "Describe the structure of a eukaryotic cell.",
    "What are the major programming paradigms?",
    "Explain how vaccines work to prevent disease.",
]
MAX_TOKENS = 128
TEMPERATURE = 0.1
REQUEST_TIMEOUT = 120
SEED = 42


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

def get_gpu_memory_mb():
    """Query current GPU memory usage via nvidia-smi."""
    result = _subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(result.stdout.strip().split("\n")[0])


# ---------------------------------------------------------------------------
# Output quality — cosine similarity
# ---------------------------------------------------------------------------

def cosine_sim_bow(text_a, text_b):
    """Cosine similarity on token-level bag-of-words."""
    words_a = Counter(text_a.lower().split())
    words_b = Counter(text_b.lower().split())
    all_words = set(words_a) | set(words_b)
    dot = sum(words_a.get(w, 0) * words_b.get(w, 0) for w in all_words)
    mag_a = math.sqrt(sum(v ** 2 for v in words_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in words_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Image sampling
# ---------------------------------------------------------------------------

def sample_images(image_dir: Path, num_images: int, seed: int = SEED) -> list[Path]:
    """Sample num_images random images from the directory."""
    all_images = sorted(image_dir.glob("*.webp"))
    if not all_images:
        all_images = sorted(image_dir.iterdir())
    if len(all_images) < num_images:
        print(f"Warning: only {len(all_images)} images available, using all")
        num_images = len(all_images)
    rng = random.Random(seed)
    return rng.sample(all_images, num_images)


def image_to_data_url(image_path: Path) -> str:
    """Convert image file to base64 data URL."""
    data = image_path.read_bytes()
    suffix = image_path.suffix.lstrip(".")
    mime = {"webp": "image/webp", "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/webp")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def send_image_request(base_url: str, data_url: str, idx: int) -> dict:
    """Send a single image captioning request. Returns timing info."""
    payload = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": IMAGE_PROMPT},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        elapsed = time.perf_counter() - t0

        usage = result.get("usage", {})
        output_text = result["choices"][0]["message"]["content"]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return {
            "idx": idx,
            "ok": True,
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "output_len": len(output_text),
            "output": output_text,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "idx": idx,
            "ok": False,
            "elapsed": elapsed,
            "error": str(e),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "output_len": 0,
            "output": "",
        }


def send_text_request(base_url: str, prompt: str, idx: int) -> dict:
    """Send a single text completion request. Returns timing info."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        elapsed = time.perf_counter() - t0

        usage = result.get("usage", {})
        output_text = result["choices"][0]["message"]["content"]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return {
            "idx": idx,
            "ok": True,
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "output_len": len(output_text),
            "output": output_text,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "idx": idx,
            "ok": False,
            "elapsed": elapsed,
            "error": str(e),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "output_len": 0,
            "output": "",
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    base_url: str,
    images: list[Path] | None = None,
    text_prompts: list[str] | None = None,
    concurrency: int = 16,
    warmup: int = 2,
    mem_launch: int | None = None,
) -> dict:
    """Run the throughput benchmark, return metrics dict.

    Provide either images (multimodal) or text_prompts (text-only).
    """
    text_only = text_prompts is not None

    if text_only:
        num_requests = len(text_prompts)
        print(f"  Text-only mode: {num_requests} requests")
        send_fn = lambda idx: send_text_request(base_url, text_prompts[idx], idx)
    else:
        print(f"  Encoding {len(images)} images to base64...")
        data_urls = [image_to_data_url(img) for img in images]
        num_requests = len(data_urls)
        send_fn = lambda idx: send_image_request(base_url, data_urls[idx], idx)

    # Warmup
    if warmup > 0:
        print(f"  Warmup: {warmup} requests...")
        for i in range(min(warmup, num_requests)):
            send_fn(i)

    mem_steady = get_gpu_memory_mb()

    # Benchmark
    print(f"  Running {num_requests} requests (concurrency={concurrency})...")
    results = []
    wall_start = time.perf_counter()

    if concurrency == 1:
        for i in range(num_requests):
            r = send_fn(i)
            results.append(r)
            if (i + 1) % 20 == 0:
                ok_count = sum(1 for r in results if r["ok"])
                avg_lat = sum(r["elapsed"] for r in results if r["ok"]) / max(ok_count, 1)
                print(f"    {i+1}/{num_requests} done, {ok_count} ok, avg latency={avg_lat:.2f}s")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(send_fn, i): i
                for i in range(num_requests)
            }
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                done = len(results)
                if done % 20 == 0:
                    ok_count = sum(1 for r in results if r["ok"])
                    print(f"    {done}/{num_requests} done, {ok_count} ok")

    wall_elapsed = time.perf_counter() - wall_start

    # Compute metrics
    ok_results = [r for r in results if r["ok"]]
    fail_count = len(results) - len(ok_results)

    total_prompt_tokens = sum(r["prompt_tokens"] for r in ok_results)
    total_completion_tokens = sum(r["completion_tokens"] for r in ok_results)
    total_tokens = total_prompt_tokens + total_completion_tokens

    avg_latency = sum(r["elapsed"] for r in ok_results) / max(len(ok_results), 1)
    p50_latency = sorted(r["elapsed"] for r in ok_results)[len(ok_results) // 2] if ok_results else 0
    p95_idx = int(len(ok_results) * 0.95)
    p95_latency = sorted(r["elapsed"] for r in ok_results)[min(p95_idx, len(ok_results) - 1)] if ok_results else 0

    metrics = {
        "num_requests": len(results),
        "num_ok": len(ok_results),
        "num_failed": fail_count,
        "wall_time_s": round(wall_elapsed, 2),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "throughput_req_per_s": round(len(ok_results) / wall_elapsed, 2),
        "throughput_tok_per_s": round(total_tokens / wall_elapsed, 2),
        "throughput_gen_tok_per_s": round(total_completion_tokens / wall_elapsed, 2),
        "avg_latency_s": round(avg_latency, 3),
        "p50_latency_s": round(p50_latency, 3),
        "p95_latency_s": round(p95_latency, 3),
        "avg_prompt_tokens": round(total_prompt_tokens / max(len(ok_results), 1), 1),
        "avg_completion_tokens": round(total_completion_tokens / max(len(ok_results), 1), 1),
        "gpu_mem_launch_mb": mem_launch,
        "gpu_mem_steady_mb": mem_steady,
        "per_request": results,
    }

    return metrics


def print_metrics(config_name: str, metrics: dict):
    """Print a formatted metrics summary."""
    print(f"\n{'=' * 55}")
    print(f"  {config_name}")
    print(f"{'=' * 55}")
    print(f"  Requests:    {metrics['num_ok']}/{metrics['num_requests']} ok ({metrics['num_failed']} failed)")
    print(f"  Wall time:   {metrics['wall_time_s']:.1f}s")
    print(f"  Throughput:  {metrics['throughput_req_per_s']:.2f} req/s")
    print(f"               {metrics['throughput_tok_per_s']:.1f} total tok/s")
    print(f"               {metrics['throughput_gen_tok_per_s']:.1f} gen tok/s")
    print(f"  Latency:     avg={metrics['avg_latency_s']:.2f}s  p50={metrics['p50_latency_s']:.2f}s  p95={metrics['p95_latency_s']:.2f}s")
    print(f"  Tokens/req:  prompt={metrics['avg_prompt_tokens']:.0f}  gen={metrics['avg_completion_tokens']:.0f}")
    mem_launch = metrics.get("gpu_mem_launch_mb")
    mem_steady = metrics.get("gpu_mem_steady_mb")
    if mem_launch is not None or mem_steady is not None:
        launch_str = f"{mem_launch} MB" if mem_launch is not None else "N/A"
        steady_str = f"{mem_steady} MB" if mem_steady is not None else "N/A"
        print(f"  GPU memory:  launch={launch_str}  steady={steady_str}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TurboQuant throughput benchmark")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Use an already-running server instead of launching one")
    parser.add_argument("--config", type=str, default=None,
                        choices=list(SERVER_CONFIGS.keys()),
                        help="Run only this config (requires --base-url)")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=["bf16", "turboquant_3.5bit", "turboquant_3bit", "turboquant_3.5bit_fused"],
                        choices=list(SERVER_CONFIGS.keys()),
                        help="Configs to benchmark")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path (overrides common.py MODEL_PATH)")
    parser.add_argument("--text-only", action="store_true",
                        help="Text-only mode (no images, for text-only models like Qwen2.5)")
    parser.add_argument("--num-requests", type=int, default=256,
                        help="Total number of requests (default: 256)")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Number of images to sample (default: same as --num-requests)")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Concurrent requests (default: 16)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup requests before timing (default: 2)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # Prepare request data
    text_only = args.text_only
    images = None
    text_prompts = None

    if text_only:
        # Cycle through TEXT_PROMPTS to fill num_requests
        rng = random.Random(args.seed)
        text_prompts = [rng.choice(TEXT_PROMPTS) for _ in range(args.num_requests)]
        print(f"Text-only mode: {args.num_requests} requests from {len(TEXT_PROMPTS)} prompt templates")
    else:
        num_images = args.num_images or args.num_requests
        images = sample_images(IMAGE_DIR, num_images, args.seed)
        print(f"Sampled {len(images)} images from {IMAGE_DIR}")

    all_metrics = {}

    if args.base_url and args.config:
        # Single run against existing server
        print(f"\nBenchmarking {args.config} at {args.base_url}")
        mem_launch = get_gpu_memory_mb()
        metrics = run_benchmark(args.base_url, images=images, text_prompts=text_prompts,
                                concurrency=args.concurrency, warmup=args.warmup,
                                mem_launch=mem_launch)
        print_metrics(SERVER_CONFIGS[args.config]["name"], metrics)
        save_results("throughput", args.config, metrics, args.output_dir)
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
            proc = launch_server(config_name, port=args.port, timeout=300,
                                 context_length=4096, model_path=args.model)
            base_url = f"http://127.0.0.1:{args.port}"
            mem_launch = get_gpu_memory_mb()
            metrics = run_benchmark(base_url, images=images, text_prompts=text_prompts,
                                    concurrency=args.concurrency, warmup=args.warmup,
                                    mem_launch=mem_launch)
            print_metrics(cfg["name"], metrics)
            save_results("throughput", config_name, metrics, args.output_dir)
            all_metrics[config_name] = metrics
        except Exception as e:
            print(f"ERROR: {config_name} failed: {e}")
        finally:
            if proc:
                shutdown_server(proc)

    # Print comparison
    if len(all_metrics) > 1:
        print(f"\n{'#' * 60}")
        print(f"# COMPARISON")
        print(f"{'#' * 60}")
        summary = {}
        for cfg, m in all_metrics.items():
            summary[cfg] = {
                "gen_tok/s": m["throughput_gen_tok_per_s"],
                "req/s": m["throughput_req_per_s"],
                "avg_lat": m["avg_latency_s"],
                "p95_lat": m["p95_latency_s"],
                "mem_launch": m.get("gpu_mem_launch_mb", "N/A"),
                "mem_steady": m.get("gpu_mem_steady_mb", "N/A"),
            }
        print_comparison_table("throughput", summary)

        # Compute speedup/regression
        baseline_cfg = configs[0]
        if baseline_cfg in all_metrics:
            base_tps = all_metrics[baseline_cfg]["throughput_gen_tok_per_s"]
            for cfg in configs[1:]:
                if cfg in all_metrics:
                    tq_tps = all_metrics[cfg]["throughput_gen_tok_per_s"]
                    if base_tps > 0:
                        ratio = tq_tps / base_tps
                        label = "speedup" if ratio >= 1 else "regression"
                        print(f"  {SERVER_CONFIGS[cfg]['name']} vs {SERVER_CONFIGS[baseline_cfg]['name']}: "
                              f"{ratio:.2f}x ({label})")

        # Output quality — cosine similarity vs baseline
        if baseline_cfg in all_metrics:
            baseline_results = all_metrics[baseline_cfg].get("per_request", [])
            # Build idx -> output map for baseline
            baseline_outputs = {}
            for r in baseline_results:
                if r["ok"] and r.get("output"):
                    baseline_outputs[r["idx"]] = r["output"]

            quality_rows = []
            for cfg in configs[1:]:
                if cfg not in all_metrics:
                    continue
                tq_results = all_metrics[cfg].get("per_request", [])
                sims = []
                for r in tq_results:
                    if r["ok"] and r.get("output") and r["idx"] in baseline_outputs:
                        sim = cosine_sim_bow(baseline_outputs[r["idx"]], r["output"])
                        sims.append(sim)
                if sims:
                    mean_sim = sum(sims) / len(sims)
                    min_sim = min(sims)
                    above_09 = sum(1 for s in sims if s > 0.9)
                    quality_rows.append((
                        SERVER_CONFIGS[cfg]["name"],
                        mean_sim,
                        min_sim,
                        above_09,
                        len(sims),
                    ))

            if quality_rows:
                print(f"\n{'=' * 60}")
                print(f"  OUTPUT QUALITY -- Cosine Similarity vs BF16 Baseline")
                print(f"{'=' * 60}")
                print(f"{'Config':<35} {'mean_sim':>10} {'min_sim':>10} {'   >0.9':>10}")
                print("-" * 60)
                for name, mean_s, min_s, above, total in quality_rows:
                    print(f"{name:<35} {mean_s:>10.2f} {min_s:>10.2f} {above:>5}/{total}")
                print()


if __name__ == "__main__":
    main()
