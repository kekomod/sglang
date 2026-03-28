"""
TurboQuant server tests — Llama-3.2-3B-Instruct.

Runs against a LIVE Llama-3.2-3B server with TurboQuant enabled.
Does NOT launch its own server — start one first:

    python -m sglang.launch_server \
        --model /home/keko/AI/image-prep/models/llama-3.2-3b-instruct \
        --kv-cache-quantization turboquant \
        --turboquant-bits 3.5 \
        --port 30000 --context-length 4096

Then run:
    python test/test_turboquant_llama.py [--port 30000]

Llama-3.2-3B is the ideal TQ test model: pure transformer, 100% KV layers,
no QKV attention bias, H_kv=8, 28 layers.
"""

import argparse
import concurrent.futures
import sys

import requests

TIMEOUT = 120


def generate(base_url, prompt, max_new_tokens=20, temperature=0.1):
    """Send a request to the /generate endpoint."""
    resp = requests.post(
        f"{base_url}/generate",
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


def test_short_generation(base_url):
    """Short generation should produce coherent text."""
    output = generate(base_url, "The capital of France is", max_new_tokens=20, temperature=0.1)
    assert len(output.strip()) > 0, f"Empty response: {output!r}"
    assert "Paris" in output, f"Expected 'Paris' in output, got: {output!r}"
    return output.strip()[:80]


def test_long_generation(base_url):
    """Generate 200 tokens — should not degrade or stutter."""
    output = generate(base_url, "Explain how photosynthesis works in detail:", max_new_tokens=200, temperature=0.1)
    words = output.split()
    assert len(words) >= 30, f"Expected 30+ words, got {len(words)}: {output[:100]!r}"
    return f"{len(words)} words"


def test_consistency(base_url):
    """Same prompt twice with temperature=0 should produce identical output."""
    prompt = "The chemical formula for water is"
    out1 = generate(base_url, prompt, max_new_tokens=20, temperature=0.0)
    out2 = generate(base_url, prompt, max_new_tokens=20, temperature=0.0)
    assert out1 == out2, f"Outputs differ:\n  1: {out1!r}\n  2: {out2!r}"
    return out1.strip()[:80]


def test_concurrent(base_url):
    """10 concurrent requests, all return non-empty output."""
    prompt = "Count from 1 to 5:"
    num_requests = 10

    def send_request(_):
        return generate(base_url, prompt, max_new_tokens=30, temperature=0.1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        results = list(executor.map(send_request, range(num_requests)))

    assert all(len(r.strip()) > 0 for r in results), "Some requests returned empty output"
    return f"All {num_requests} concurrent requests returned output"


ALL_TESTS = [
    test_short_generation,
    test_long_generation,
    test_consistency,
    test_concurrent,
]


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Llama-3.2-3B server tests")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"

    # Check server health
    try:
        resp = requests.get(f"{base_url}/health_generate", timeout=5)
        if resp.status_code != 200:
            print(f"Server not healthy (status {resp.status_code}). Start a Llama-3.2-3B TQ server first.")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"Cannot connect to {base_url}. Start a Llama-3.2-3B TQ server first.")
        sys.exit(1)

    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            summary = test_fn(base_url)
            print(f"  PASS  {name}: {summary}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
