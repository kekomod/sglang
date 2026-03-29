"""
TurboQuant Validation Benchmark — Master Orchestrator

Launches servers with each config, runs all benchmarks, saves results,
and prints comparison tables with PASS/FAIL vs paper targets.

Usage:
    python run_all.py
    python run_all.py --configs bf16,turboquant_3.5bit --benchmarks needle,gsm8k
    python run_all.py --resume --output-dir results/
"""

import argparse
import sys
import time

from common import (
    DEFAULT_PORT,
    SERVER_CONFIGS,
    check_target,
    launch_server,
    load_results,
    print_comparison_table,
    save_results,
    shutdown_server,
)

# Benchmark runners — imported here so run_all.py is the single entry point
from eval_gsm8k import run_gsm8k_benchmark
from eval_needle import run_needle_benchmark

BENCHMARK_RUNNERS = {
    "needle": lambda base_url, **kw: run_needle_benchmark(base_url, context_lengths=[1024, 2048, 3072, 4096]),
    "gsm8k": lambda base_url, **kw: run_gsm8k_benchmark(base_url),
}

ALL_BENCHMARKS = list(BENCHMARK_RUNNERS.keys())
ALL_CONFIGS = list(SERVER_CONFIGS.keys())


def run_suite(
    configs: list[str],
    benchmarks: list[str],
    output_dir: str,
    resume: bool,
    port: int,
    max_chunks: int | None = None,
):
    """Run the full benchmark suite."""
    # Collect all results: benchmark_name -> {config_name -> metrics}
    all_results: dict[str, dict[str, dict]] = {b: {} for b in benchmarks}

    for config_name in configs:
        print(f"\n{'#' * 60}")
        print(f"  Config: {SERVER_CONFIGS[config_name]['name']}")
        print(f"{'#' * 60}")

        # Check resume — skip if all benchmarks already have results
        if resume:
            all_cached = True
            for bench in benchmarks:
                cached = load_results(bench, config_name, output_dir)
                if cached is None:
                    all_cached = False
                else:
                    all_results[bench][config_name] = cached["metrics"]
                    print(f"  [resume] Loaded cached {bench} results for {config_name}")
            if all_cached:
                print(f"  [resume] All benchmarks cached for {config_name}, skipping.")
                continue

        # Launch server
        proc = None
        try:
            proc = launch_server(config_name, port=port)
            base_url = f"http://127.0.0.1:{port}"

            for bench in benchmarks:
                # Check resume for individual benchmark
                if resume:
                    cached = load_results(bench, config_name, output_dir)
                    if cached is not None:
                        all_results[bench][config_name] = cached["metrics"]
                        print(f"  [resume] Using cached {bench} for {config_name}")
                        continue

                print(f"\n--- Running {bench} ({config_name}) ---")
                t0 = time.time()
                runner = BENCHMARK_RUNNERS[bench]
                metrics = runner(base_url, max_chunks=max_chunks)
                elapsed = time.time() - t0
                metrics["elapsed_seconds"] = round(elapsed, 1)

                save_results(bench, config_name, metrics, output_dir)
                all_results[bench][config_name] = metrics

        except Exception as exc:
            print(f"  ERROR running {config_name}: {exc}")
        finally:
            if proc is not None:
                shutdown_server(proc)
            # Brief pause before next server launch
            time.sleep(5)

    # --- Print comparison tables and PASS/FAIL ---
    print(f"\n{'=' * 60}")
    print("  FINAL RESULTS")
    print(f"{'=' * 60}")

    any_fail = False
    for bench in benchmarks:
        results = all_results[bench]
        if not results:
            continue

        # Print table (strip detail fields for display)
        display = {}
        for cfg, m in results.items():
            display[cfg] = {k: v for k, v in m.items() if k != "details"}
        print_comparison_table(bench, display)

        # PASS/FAIL checks against baseline
        if "bf16" in results:
            baseline = results["bf16"]
            for cfg in results:
                if cfg == "bf16":
                    continue
                passed, msg = check_target(bench, baseline, results[cfg])
                status = "PASS" if passed else "FAIL"
                if not passed:
                    any_fail = True
                cfg_label = SERVER_CONFIGS.get(cfg, {}).get("name", cfg)
                print(f"  [{status}] {cfg_label}: {msg}")
        print()

    if any_fail:
        print("Some benchmarks FAILED to meet paper targets.")
        return 1
    else:
        print("All benchmarks PASSED paper targets.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="TurboQuant validation benchmark suite")
    parser.add_argument(
        "--configs", type=str, default=",".join(ALL_CONFIGS),
        help=f"Comma-separated config names (default: {','.join(ALL_CONFIGS)})",
    )
    parser.add_argument(
        "--benchmarks", type=str, default=",".join(ALL_BENCHMARKS),
        help=f"Comma-separated benchmark names (default: {','.join(ALL_BENCHMARKS)})",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--resume", action="store_true",
                        help="Skip benchmarks that already have saved results")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit perplexity chunks (for quick tests)")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    # Validate
    for c in configs:
        if c not in SERVER_CONFIGS:
            print(f"Unknown config: {c}. Choose from: {ALL_CONFIGS}")
            sys.exit(1)
    for b in benchmarks:
        if b not in BENCHMARK_RUNNERS:
            print(f"Unknown benchmark: {b}. Choose from: {ALL_BENCHMARKS}")
            sys.exit(1)

    rc = run_suite(configs, benchmarks, args.output_dir, args.resume, args.port, args.max_chunks)
    sys.exit(rc)


if __name__ == "__main__":
    main()
