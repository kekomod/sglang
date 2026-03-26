"""
TurboQuant Throughput Benchmark — Image Captioning

Sends 200 random images from the processed image dataset through the model
with and without TurboQuant, measuring end-to-end throughput.

Usage:
    # Run both configs (launches servers automatically):
    python benchmark/turboquant/eval_throughput.py

    # Run against an already-running server:
    python benchmark/turboquant/eval_throughput.py --base-url http://localhost:30000 --config turboquant_3.5bit

    # Customize image count:
    python benchmark/turboquant/eval_throughput.py --num-images 50
"""

import argparse
import base64
import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
PROMPT = "Describe this image in one detailed sentence."
MAX_TOKENS = 128
TEMPERATURE = 0.1
REQUEST_TIMEOUT = 120
SEED = 42


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
                    {"type": "text", "text": PROMPT},
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
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    base_url: str,
    images: list[Path],
    concurrency: int = 1,
    warmup: int = 2,
) -> dict:
    """Run the throughput benchmark, return metrics dict."""

    # Pre-encode all images
    print(f"  Encoding {len(images)} images to base64...")
    data_urls = [image_to_data_url(img) for img in images]

    # Warmup
    if warmup > 0:
        print(f"  Warmup: {warmup} requests...")
        for i in range(min(warmup, len(data_urls))):
            send_image_request(base_url, data_urls[i], -1)

    # Benchmark
    print(f"  Running {len(data_urls)} requests (concurrency={concurrency})...")
    results = []
    wall_start = time.perf_counter()

    if concurrency == 1:
        for i, data_url in enumerate(data_urls):
            r = send_image_request(base_url, data_url, i)
            results.append(r)
            if (i + 1) % 20 == 0:
                ok_count = sum(1 for r in results if r["ok"])
                avg_lat = sum(r["elapsed"] for r in results if r["ok"]) / max(ok_count, 1)
                print(f"    {i+1}/{len(data_urls)} done, {ok_count} ok, avg latency={avg_lat:.2f}s")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(send_image_request, base_url, data_url, i): i
                for i, data_url in enumerate(data_urls)
            }
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                done = len(results)
                if done % 20 == 0:
                    ok_count = sum(1 for r in results if r["ok"])
                    print(f"    {done}/{len(data_urls)} done, {ok_count} ok")

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
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TurboQuant throughput benchmark (image captioning)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Use an already-running server instead of launching one")
    parser.add_argument("--config", type=str, default=None,
                        choices=list(SERVER_CONFIGS.keys()),
                        help="Run only this config (requires --base-url)")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=["bf16", "turboquant_3.5bit"],
                        choices=list(SERVER_CONFIGS.keys()),
                        help="Configs to benchmark (default: bf16 + turboquant_3.5bit)")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Number of images to sample (default: 100)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent requests (default: 1, sequential)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup requests before timing (default: 2)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # Sample images (same set for all configs)
    images = sample_images(IMAGE_DIR, args.num_images, args.seed)
    print(f"Sampled {len(images)} images from {IMAGE_DIR}")

    all_metrics = {}

    if args.base_url and args.config:
        # Single run against existing server
        print(f"\nBenchmarking {args.config} at {args.base_url}")
        metrics = run_benchmark(args.base_url, images, args.concurrency, args.warmup)
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
                                 context_length=4096)
            base_url = f"http://127.0.0.1:{args.port}"
            metrics = run_benchmark(base_url, images, args.concurrency, args.warmup)
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


if __name__ == "__main__":
    main()
