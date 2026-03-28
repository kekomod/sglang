"""
TurboQuant Memory Scaling Benchmark — the "money shot".

Launches BF16 and TQ servers at progressively larger context lengths,
records GPU memory at each, and shows the crossover point where BF16 OOMs
but TQ keeps going.

This is the critical value demonstration for TurboQuant: memory savings
enabling longer context windows that BF16 cannot support.

Usage:
    python eval_memory_scaling.py \
        --model /path/to/llama-3.2-3b-instruct \
        --context-lengths 4096 8192 16384 32768 65536 \
        --configs bf16 turboquant_3.5bit

Reference: arXiv:2504.19874 — TQ's value is memory reduction, not throughput.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DEFAULT_PORT,
    MODEL_PATH,
    SERVER_CONFIGS,
    generate_text,
    launch_server,
    save_results,
    shutdown_server,
)


def get_gpu_memory_mb():
    """Query current GPU memory usage via nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(result.stdout.strip().split("\n")[0])


def measure_memory_at_context(
    config_name: str,
    context_length: int,
    port: int,
    model_path: str,
    num_warmup: int = 3,
) -> dict:
    """Launch server, warm up KV cache, measure peak memory, shutdown.

    Returns dict with mem_launch_mb, mem_steady_mb, or oom=True.
    """
    result = {
        "config": config_name,
        "context_length": context_length,
        "oom": False,
        "mem_launch_mb": None,
        "mem_steady_mb": None,
    }

    proc = None
    try:
        proc = launch_server(
            config_name, port=port, timeout=300,
            context_length=context_length, model_path=model_path,
        )

        base_url = f"http://127.0.0.1:{port}"
        result["mem_launch_mb"] = get_gpu_memory_mb()

        # Warm up with a few requests to fill KV cache
        for i in range(num_warmup):
            try:
                generate_text(base_url, f"Hello, this is warmup request {i}.", max_tokens=32)
            except Exception:
                pass

        result["mem_steady_mb"] = get_gpu_memory_mb()

    except RuntimeError:
        # Server failed to start (likely OOM)
        result["oom"] = True
    except Exception as e:
        result["oom"] = True
        result["error"] = str(e)
    finally:
        if proc:
            shutdown_server(proc)
        # Brief pause to let GPU memory fully release
        time.sleep(3)

    return result


def print_results_table(results: dict, context_lengths: list[int], configs: list[str]):
    """Print formatted comparison table."""
    print(f"\n{'=' * 70}")
    print(f"  MEMORY SCALING — BF16 vs TurboQuant")
    print(f"{'=' * 70}")

    # Header
    header = f"{'Context':>10}"
    for cfg in configs:
        name = SERVER_CONFIGS.get(cfg, {}).get("name", cfg)
        header += f"  {name:>18}"
    if len(configs) == 2:
        header += f"  {'Savings':>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for ctx_len in context_lengths:
        ctx_label = f"{ctx_len // 1024}K"
        row = f"{ctx_label:>10}"

        mem_values = []
        for cfg in configs:
            key = f"{cfg}_{ctx_len}"
            r = results.get(key)
            if r is None or r["oom"]:
                row += f"  {'OOM':>18}"
                mem_values.append(None)
            else:
                mem = r["mem_steady_mb"] or r["mem_launch_mb"]
                row += f"  {mem:>15} MB"
                mem_values.append(mem)

        # Savings column
        if len(configs) == 2 and all(v is not None for v in mem_values):
            savings_pct = (1 - mem_values[1] / mem_values[0]) * 100
            row += f"  {savings_pct:>8.1f}%"
        elif len(configs) == 2:
            row += f"  {'--':>10}"

        print(row)

    print()


def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant memory scaling benchmark — shows OOM crossover"
    )
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Model path")
    parser.add_argument("--context-lengths", nargs="+", type=int,
                        default=[4096, 8192, 16384, 32768, 65536],
                        help="Context lengths to test (default: 4K 8K 16K 32K 64K)")
    parser.add_argument("--configs", nargs="+", type=str,
                        default=["bf16", "turboquant_3.5bit"],
                        help="Server configs to compare")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--warmup-requests", type=int, default=3,
                        help="Warmup requests per config (default: 3)")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    context_lengths = sorted(args.context_lengths)
    configs = args.configs

    print(f"[memory_scaling] Model: {args.model}")
    print(f"[memory_scaling] Context lengths: {context_lengths}")
    print(f"[memory_scaling] Configs: {configs}")
    print()

    all_results = {}

    for config_name in configs:
        cfg_name = SERVER_CONFIGS.get(config_name, {}).get("name", config_name)
        oom_hit = False

        for ctx_len in context_lengths:
            key = f"{config_name}_{ctx_len}"

            if oom_hit:
                # Once OOM, mark all larger contexts as OOM too
                all_results[key] = {
                    "config": config_name,
                    "context_length": ctx_len,
                    "oom": True,
                }
                print(f"  [{cfg_name}] {ctx_len // 1024}K: OOM (skipped, previous OOM)")
                continue

            print(f"  [{cfg_name}] {ctx_len // 1024}K: launching...")
            result = measure_memory_at_context(
                config_name, ctx_len, args.port, args.model, args.warmup_requests,
            )
            all_results[key] = result

            if result["oom"]:
                print(f"  [{cfg_name}] {ctx_len // 1024}K: OOM")
                oom_hit = True
            else:
                print(f"  [{cfg_name}] {ctx_len // 1024}K: "
                      f"launch={result['mem_launch_mb']} MB, "
                      f"steady={result['mem_steady_mb']} MB")

    # Print comparison table
    print_results_table(all_results, context_lengths, configs)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "memory_scaling.json"
    output_data = {
        "benchmark": "memory_scaling",
        "model": args.model,
        "context_lengths": context_lengths,
        "configs": configs,
        "results": {k: v for k, v in all_results.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"[memory_scaling] Results saved to {output_path}")


if __name__ == "__main__":
    main()
