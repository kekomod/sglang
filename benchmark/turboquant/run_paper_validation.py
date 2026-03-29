"""
TurboQuant Paper Validation — Run All Benchmarks

Replicates the experimental results from arXiv:2504.19874.
Runs: distortion, NIAH, LongBench-E, GSM8K.

Each sub-benchmark handles its own server lifecycle when in multi-config
mode. This master script orchestrates the order and aggregates results.

Usage:
    python run_paper_validation.py
    python run_paper_validation.py --model /path/to/model --configs bf16 turboquant_3.5bit
    python run_paper_validation.py --benchmarks needle gsm8k  # run subset
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_PORT, SERVER_CONFIGS, TARGETS

# Default model for paper validation (pure transformer = 100% KV cache layers)
DEFAULT_MODEL = "/home/keko/AI/image-prep/models/qwen2.5-3b-instruct"
DEFAULT_CONFIGS = ["bf16", "turboquant_3.5bit"]

# Benchmarks in paper order, with their CLI construction
BENCHMARK_SPECS = {
    "distortion": {
        "script": "eval_distortion.py",
        "description": "Quantization Distortion (Section 4.1)",
        "needs_server": False,
    },
    "needle": {
        "script": "eval_needle.py",
        "description": "Needle-in-a-Haystack (Section 4.2)",
        "needs_server": True,
    },
    "longbench": {
        "script": "eval_longbench.py",
        "description": "LongBench-E (Section 4.3)",
        "needs_server": True,
    },
    "gsm8k": {
        "script": "eval_gsm8k.py",
        "description": "GSM8K Arithmetic Reasoning",
        "needs_server": True,
    },
}

ALL_BENCHMARKS = list(BENCHMARK_SPECS.keys())


def run_benchmark(
    benchmark_name: str,
    model: str,
    configs: list[str],
    port: int,
    output_dir: str,
    context_lengths: list[int],
    extra_args: list[str] | None = None,
) -> tuple[bool, float]:
    """Run a single benchmark via subprocess.

    Returns (success, elapsed_seconds).
    """
    spec = BENCHMARK_SPECS[benchmark_name]
    script_path = Path(__file__).resolve().parent / spec["script"]

    if not script_path.exists():
        print(f"  [SKIP] {spec['script']} not found — skipping {benchmark_name}")
        return False, 0.0

    cmd = [sys.executable, str(script_path)]

    if spec["needs_server"]:
        # Multi-config mode: sub-script manages server lifecycle
        cmd += ["--configs"] + configs
        cmd += ["--model", model]
        cmd += ["--port", str(port)]

        # Benchmark-specific args
        if benchmark_name == "needle":
            cmd += ["--context-lengths"] + [str(c) for c in context_lengths]
    else:
        # Standalone benchmarks (e.g. distortion) — no server needed
        pass

    cmd += ["--output-dir", output_dir]

    if extra_args:
        cmd += extra_args

    print(f"\n{'#' * 70}")
    print(f"  Running: {spec['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'#' * 70}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(script_path.parent))
    elapsed = time.time() - t0

    success = result.returncode == 0
    status = "OK" if success else f"FAILED (exit code {result.returncode})"
    print(f"\n  [{status}] {benchmark_name} completed in {elapsed:.1f}s")

    return success, elapsed


def load_result_json(output_dir: str, benchmark: str, config: str) -> dict | None:
    """Load a result JSON for a benchmark+config pair."""
    path = Path(output_dir) / f"{benchmark}_{config}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def format_gsm8k_row(config: str, metrics: dict, baseline_acc: float | None) -> str:
    """Format a GSM8K result row. GSM8K is NOT in the paper — informational only."""
    name = SERVER_CONFIGS.get(config, {}).get("name", config)
    acc = metrics.get("accuracy", "N/A")
    if isinstance(acc, (int, float)):
        acc_pct = acc * 100
        if baseline_acc and config != "bf16":
            drop = (baseline_acc - acc) / baseline_acc * 100
            return f"  {name:<25} Accuracy = {acc_pct:.1f}% ({drop:+.1f}%)"
        return f"  {name:<25} Accuracy = {acc_pct:.1f}%"
    return f"  {name:<25} Accuracy = {acc}"


def format_needle_table(configs: list[str], output_dir: str, context_lengths: list[int]) -> list[str]:
    """Format the NIAH comparison table."""
    lines = []
    ctx_header = "".join(f"{c // 1024:>7}K" for c in context_lengths)
    lines.append(f"  {'Config':<25}{ctx_header}")
    lines.append("  " + "-" * (25 + 7 * len(context_lengths)))

    for config in configs:
        data = load_result_json(output_dir, "needle", config)
        name = SERVER_CONFIGS.get(config, {}).get("name", config)
        if data is None:
            lines.append(f"  {name:<25}" + "    N/A" * len(context_lengths))
            continue
        acc_by_ctx = data.get("metrics", {}).get("accuracy_by_context_length", {})
        row = f"  {name:<25}"
        for ctx in context_lengths:
            val = acc_by_ctx.get(str(ctx), acc_by_ctx.get(ctx, "N/A"))
            if isinstance(val, str) and val == "OOM":
                row += f"{'OOM':>7}"
            elif isinstance(val, (int, float)):
                row += f"{val:>7.3f}"
            else:
                row += f"{'N/A':>7}"
        lines.append(row)

    return lines


def print_unified_summary(
    configs: list[str],
    output_dir: str,
    model: str,
    benchmarks_run: list[str],
    context_lengths: list[int],
    timings: dict[str, float],
):
    """Print the unified paper validation summary."""
    print(f"\n{'=' * 70}")
    print(f"  TurboQuant Paper Validation Summary")
    print(f"  Model: {Path(model).name}")
    print(f"  Configs: {', '.join(configs)}")
    print(f"{'=' * 70}")

    # --- Distortion ---
    if "distortion" in benchmarks_run:
        print(f"\nDistortion (Section 4.1):")
        data = load_result_json(output_dir, "distortion", "standalone")
        if data:
            metrics = data.get("metrics", {})
            for bits in ["2", "3", "4"]:
                mse = metrics.get(f"{bits}bit_mse", "N/A")
                theo = metrics.get(f"{bits}bit_theoretical", "N/A")
                if isinstance(mse, (int, float)) and isinstance(theo, (int, float)):
                    print(f"  {bits}-bit MSE: {mse:.4f} (theoretical: {theo:.4f})")
                else:
                    print(f"  {bits}-bit MSE: {mse}")
        else:
            print("  (no results)")

    # --- Needle ---
    if "needle" in benchmarks_run:
        print(f"\nNIAH (Section 4.2):")
        for line in format_needle_table(configs, output_dir, context_lengths):
            print(line)

    # --- LongBench ---
    if "longbench" in benchmarks_run:
        print(f"\nLongBench-E (Section 4.3):")
        any_data = False
        for config in configs:
            data = load_result_json(output_dir, "longbench", config)
            if data:
                any_data = True
                name = SERVER_CONFIGS.get(config, {}).get("name", config)
                metrics = data.get("metrics", {})
                avg = metrics.get("average", "N/A")
                print(f"  {name:<25} Avg = {avg}")
        if not any_data:
            print("  (no results)")

    # --- GSM8K ---
    if "gsm8k" in benchmarks_run:
        print(f"\nGSM8K (Arithmetic Reasoning):")
        baseline_acc = None
        bf16_data = load_result_json(output_dir, "gsm8k", "bf16")
        if bf16_data:
            baseline_acc = bf16_data.get("metrics", {}).get("accuracy")

        for config in configs:
            data = load_result_json(output_dir, "gsm8k", config)
            if data:
                print(format_gsm8k_row(config, data["metrics"], baseline_acc))
            else:
                name = SERVER_CONFIGS.get(config, {}).get("name", config)
                print(f"  {name:<25} (no results)")

    # --- Timings ---
    print(f"\nTimings:")
    total = 0.0
    for bench, elapsed in timings.items():
        print(f"  {bench:<20} {elapsed:>8.1f}s")
        total += elapsed
    print(f"  {'TOTAL':<20} {total:>8.1f}s")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant Paper Validation — Run All Benchmarks"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--configs", nargs="+", default=DEFAULT_CONFIGS,
        help=f"Server configs to compare (default: {' '.join(DEFAULT_CONFIGS)})",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--benchmarks", nargs="+", default=None,
        help=f"Benchmarks to run (default: all). Choose from: {', '.join(ALL_BENCHMARKS)}",
    )
    parser.add_argument(
        "--context-lengths", nargs="+", type=int,
        default=[4096, 8192, 16384, 32768],
        help="Context lengths for NIAH benchmark",
    )
    args = parser.parse_args()

    # Validate configs
    for c in args.configs:
        if c not in SERVER_CONFIGS:
            print(f"Unknown config: {c}. Choose from: {list(SERVER_CONFIGS)}")
            sys.exit(1)

    benchmarks = args.benchmarks or ALL_BENCHMARKS
    for b in benchmarks:
        if b not in BENCHMARK_SPECS:
            print(f"Unknown benchmark: {b}. Choose from: {ALL_BENCHMARKS}")
            sys.exit(1)

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  TurboQuant Paper Validation")
    print(f"  Model:      {args.model}")
    print(f"  Configs:    {', '.join(args.configs)}")
    print(f"  Benchmarks: {', '.join(benchmarks)}")
    print(f"  Output:     {args.output_dir}")
    print(f"{'=' * 70}")

    timings = {}
    benchmarks_run = []

    for bench_name in benchmarks:
        extra_args = []

        success, elapsed = run_benchmark(
            bench_name,
            model=args.model,
            configs=args.configs,
            port=args.port,
            output_dir=args.output_dir,
            context_lengths=args.context_lengths,
            extra_args=extra_args,
        )
        timings[bench_name] = elapsed
        if success:
            benchmarks_run.append(bench_name)

    # Print unified summary
    print_unified_summary(
        configs=args.configs,
        output_dir=args.output_dir,
        model=args.model,
        benchmarks_run=benchmarks_run,
        context_lengths=args.context_lengths,
        timings=timings,
    )


if __name__ == "__main__":
    main()
