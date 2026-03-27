"""
TurboQuant piecewise CUDA graph test — Qwen2.5-3B-Instruct.

Launches a Qwen2.5-3B server with TurboQuant 3.5-bit and piecewise CUDA
graphs enabled, runs text-only validation tests, then tears down the server.

Uses Qwen2.5-3B (pure transformer) because piecewise graphs require a model
where 100% of layers produce KV cache.  Qwen3.5 is hybrid (GatedDeltaNet)
and is NOT suitable for this test.

Usage:
    python test/test_turboquant_piecewise.py
"""

import concurrent.futures
import os
import signal
import subprocess
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = "/home/keko/AI/image-prep/models/qwen2.5-3b-instruct"
PORT = 30001
BASE_URL = f"http://127.0.0.1:{PORT}"
TIMEOUT = 300  # 5 min for generation requests
SERVER_STARTUP_TIMEOUT = 300  # 5 min for server to become healthy


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def launch_server():
    """Launch SGLang server with TurboQuant + piecewise CUDA graphs."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--kv-cache-quantization", "turboquant",
        "--turboquant-bits", "3.5",
        "--enforce-piecewise-cuda-graph",
        "--mem-fraction-static", "0.5",
        "--port", str(PORT),
        "--context-length", "4096",
    ]
    print(f"[piecewise] Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_health(timeout=SERVER_STARTUP_TIMEOUT):
    """Poll /health_generate until the server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/health_generate", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def shutdown_server(proc):
    """Kill the server process tree."""
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
    print("[piecewise] Server shut down.")


# ---------------------------------------------------------------------------
# Request helper
# ---------------------------------------------------------------------------

def generate(prompt, max_new_tokens=20, temperature=0.1):
    """Send a request to the /generate endpoint."""
    resp = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["text"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_short_generation():
    """Short generation: expect 'Paris' in response."""
    output = generate("The capital of France is", max_new_tokens=20, temperature=0.1)
    assert len(output.strip()) > 0, f"Empty response: {output!r}"
    assert "Paris" in output, f"Expected 'Paris' in output, got: {output!r}"
    return output.strip()[:80]


def test_consistency():
    """Same prompt twice with temperature=0 should produce identical output."""
    prompt = "The chemical formula for water is"
    out1 = generate(prompt, max_new_tokens=20, temperature=0.0)
    out2 = generate(prompt, max_new_tokens=20, temperature=0.0)
    assert out1 == out2, f"Outputs differ:\n  1: {out1!r}\n  2: {out2!r}"
    return out1.strip()[:80]


def test_concurrent():
    """10 concurrent requests, all return non-empty output."""
    prompt = "Count from 1 to 5:"
    num_requests = 10

    def send_request(_):
        return generate(prompt, max_new_tokens=30, temperature=0.1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        results = list(executor.map(send_request, range(num_requests)))

    assert all(len(r.strip()) > 0 for r in results), "Some requests returned empty output"
    return f"All {num_requests} concurrent requests returned output"


ALL_TESTS = [
    test_short_generation,
    test_consistency,
    test_concurrent,
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    proc = None
    try:
        proc = launch_server()
        print(f"[piecewise] Waiting for server to become healthy...")
        if not wait_for_health():
            print("FAIL: Server did not become healthy within timeout")
            sys.exit(1)
        print(f"[piecewise] Server healthy on port {PORT}")

        # Warmup: single request + concurrent burst to stabilize caches
        generate("Warmup", max_new_tokens=5, temperature=0.0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(
                lambda _: generate("Warmup", max_new_tokens=5, temperature=0.0),
                range(5),
            ))

        passed = 0
        failed = 0
        for test_fn in ALL_TESTS:
            name = test_fn.__name__
            try:
                summary = test_fn()
                print(f"  PASS  {name}: {summary}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1

        print(f"\n{passed}/{passed + failed} tests passed")
        sys.exit(0 if failed == 0 else 1)

    finally:
        if proc:
            shutdown_server(proc)


if __name__ == "__main__":
    main()
