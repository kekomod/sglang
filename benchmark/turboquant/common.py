"""
TurboQuant Validation Benchmark — Shared Infrastructure

Provides server lifecycle management, HTTP helpers, and result I/O
for the TurboQuant benchmark suite.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ALWAYS use the local model path — NEVER use HuggingFace model IDs
MODEL_PATH = "/home/keko/AI/image-prep/models/uncensored-9b-bf16-hf"

SERVER_CONFIGS = {
    "bf16": {
        "name": "BF16 Baseline",
        "args": [],
    },
    "turboquant_3.5bit": {
        "name": "TurboQuant 3.5-bit",
        "args": [
            "--kv-cache-quantization", "turboquant",
            "--turboquant-bits", "3.5",
        ],
    },
}

DEFAULT_PORT = 30000
DEFAULT_TIMEOUT = 300
REQUEST_TIMEOUT = 300  # 5 min — Qwen3.5 <think> mode can produce long reasoning
MAX_RETRIES = 3

# Paper targets (arXiv:2504.19874, Section 4)
# The paper only evaluates: distortion (4.1), NIAH (4.2), LongBench-E (4.3),
# near neighbor search (4.4). Models: Llama-3.1-8B-Instruct, Ministral-7B-Instruct.
# GSM8K, perplexity, throughput are NOT in the paper.
TARGETS = {
    "needle": {"min_accuracy": 0.95},  # Paper: TQ=0.997 (matches FP baseline)
    "longbench": {"max_avg_f1_drop_pct": 2.0},  # Paper: TQ@3.5bit=50.06 (matches FP 50.06)
}


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def wait_for_health(base_url: str, timeout: float = DEFAULT_TIMEOUT) -> bool:
    """Poll /health_generate until the server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/health_generate", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def launch_server(
    config_name: str,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    context_length: int = 4096,
    model_path: str | None = None,
) -> subprocess.Popen:
    """Launch an SGLang server with the given config.

    Returns the Popen handle. Raises RuntimeError if the server
    doesn't become healthy within *timeout* seconds.
    """
    if config_name not in SERVER_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(SERVER_CONFIGS)}")

    cfg = SERVER_CONFIGS[config_name]
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path or MODEL_PATH,
        "--port", str(port),
        "--context-length", str(context_length),
        *cfg["args"],
    ]

    print(f"[common] Launching server ({cfg['name']}): {' '.join(cmd)}")

    # Write server logs to file instead of PIPE to prevent buffer deadlock.
    # PIPE buffers fill up (~64KB) and block the server's stdout writes.
    log_dir = Path(os.environ.get("TURBOQUANT_LOG_DIR", "/tmp"))
    log_path = log_dir / f"sglang_server_{config_name}_{port}.log"
    log_file = open(log_path, "w")
    print(f"[common] Server log: {log_path}")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    proc._log_file = log_file  # attach for cleanup

    base_url = f"http://127.0.0.1:{port}"
    if not wait_for_health(base_url, timeout):
        shutdown_server(proc)
        raise RuntimeError(
            f"Server ({cfg['name']}) failed to become healthy within {timeout}s"
        )

    print(f"[common] Server ({cfg['name']}) healthy on port {port}")
    return proc


def shutdown_server(proc: subprocess.Popen) -> None:
    """Kill the server process tree."""
    # Close log file if attached
    log_file = getattr(proc, '_log_file', None)
    if log_file:
        try:
            log_file.close()
        except Exception:
            pass

    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass
    print("[common] Server shut down.")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _request_with_retry(method: str, url: str, **kwargs):
    """HTTP request with exponential-backoff retry."""
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    raise last_exc


def generate_with_logprobs(
    base_url: str,
    input_ids: list[int],
    max_new_tokens: int = 0,
) -> dict:
    """POST /generate with return_logprob=True, logprob_start_len=0.

    Returns the parsed JSON response (single request, not batched).
    """
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": 0,
    }
    resp = _request_with_retry("POST", f"{base_url}/generate", json=payload)
    return resp.json()


def generate_text(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> str:
    """POST /generate for text completion. Returns the generated text."""
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
        },
    }
    if stop:
        payload["sampling_params"]["stop"] = stop
    resp = _request_with_retry("POST", f"{base_url}/generate", json=payload)
    return resp.json()["text"]


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _results_path(benchmark_name: str, config_name: str, output_dir: str) -> Path:
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{benchmark_name}_{config_name}.json"


def save_results(
    benchmark_name: str,
    config_name: str,
    metrics: dict,
    output_dir: str = "results",
) -> Path:
    """Save benchmark metrics to JSON."""
    path = _results_path(benchmark_name, config_name, output_dir)
    data = {
        "benchmark": benchmark_name,
        "config": config_name,
        "config_name": SERVER_CONFIGS.get(config_name, {}).get("name", config_name),
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"[common] Results saved to {path}")
    return path


def load_results(
    benchmark_name: str,
    config_name: str,
    output_dir: str = "results",
) -> dict | None:
    """Load benchmark metrics from JSON, or None if not found."""
    path = _results_path(benchmark_name, config_name, output_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def print_comparison_table(benchmark_name: str, all_results: dict[str, dict]) -> None:
    """Print a formatted comparison table for a benchmark.

    *all_results* maps config_name -> metrics dict.
    """
    print(f"\n{'=' * 60}")
    print(f"  {benchmark_name.upper()} — Comparison")
    print(f"{'=' * 60}")

    # Determine columns from first entry
    if not all_results:
        print("  (no results)")
        return

    metric_keys = list(next(iter(all_results.values())).keys())
    header = f"{'Config':<25}" + "".join(f"{k:>15}" for k in metric_keys)
    print(header)
    print("-" * len(header))
    for cfg, metrics in all_results.items():
        name = SERVER_CONFIGS.get(cfg, {}).get("name", cfg)
        row = f"{name:<25}" + "".join(f"{metrics.get(k, 'N/A'):>15}" for k in metric_keys)
        print(row)
    print()


def check_target(
    benchmark_name: str,
    baseline_metrics: dict,
    quant_metrics: dict,
) -> tuple[bool, str]:
    """Check whether quant_metrics meets the paper target relative to baseline.

    Returns (passed, message).
    """
    targets = TARGETS.get(benchmark_name)
    if targets is None:
        return True, "No target defined."

    if benchmark_name == "needle":
        acc = quant_metrics["accuracy"]
        threshold = targets["min_accuracy"]
        passed = acc >= threshold
        msg = f"Accuracy: {acc:.3f} (threshold: {threshold})"
        return passed, msg

    if benchmark_name == "longbench":
        base_f1 = baseline_metrics.get("overall_avg_f1", 0)
        quant_f1 = quant_metrics.get("overall_avg_f1", 0)
        drop_pct = (base_f1 - quant_f1) / base_f1 * 100 if base_f1 > 0 else 0
        threshold = targets["max_avg_f1_drop_pct"]
        passed = drop_pct <= threshold
        msg = f"Avg F1 drop: {drop_pct:.2f}% (threshold: {threshold}%)"
        return passed, msg

    return True, "No paper target for this benchmark."
