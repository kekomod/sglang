"""
TurboQuant server integration tests.

Runs against a live SGLang server at http://localhost:30000.
Usage: python test/test_turboquant_server.py
"""

import sys
import requests

BASE_URL = "http://localhost:30000"
TIMEOUT = 300  # 5 min — dequant-on-read path is slower than fused kernels


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


def chat_completions(messages, max_tokens=50, temperature=0.1):
    """Send a request to the /v1/chat/completions endpoint."""
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def check_server():
    """Check if the server is running."""
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
        return True
    except requests.ConnectionError:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_short_generation():
    """Short generation: expect 'Paris' in response."""
    output = generate("The capital of France is", max_new_tokens=20, temperature=0.1)
    assert "Paris" in output, f"Expected 'Paris' in output, got: {output!r}"
    return output.strip()[:80]


def test_long_generation():
    """Long generation: 256 tokens, check length and no degenerate patterns."""
    output = generate(
        "Write a detailed explanation of how neural networks learn through backpropagation.",
        max_new_tokens=256,
        temperature=0.0,  # deterministic for reproducible quality checks
    )
    assert len(output) > 200, f"Output too short ({len(output)} chars): {output[:100]!r}"

    words = output.split()

    # Check 1: no degenerate same-token repetition (e.g. "the the the the")
    import re
    degen = re.search(r"(\b\w+\b)(?:\s+\1){3,}", output)
    assert degen is None, f"Degenerate repetition: {degen.group()!r}"

    # Check 2: no repeated 5+ word phrases (copy-paste loops)
    ngram_size = 5
    if len(words) >= ngram_size:
        ngrams = [" ".join(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
        from collections import Counter
        ngram_counts = Counter(ngrams)
        most_common_ng, ng_count = ngram_counts.most_common(1)[0]
        assert ng_count <= 3, f"5-gram repeated {ng_count} times: {most_common_ng!r}"

    return f"{len(output)} chars, {len(words)} words"


def test_long_prompt():
    """Long prompt (~500 words) then short generation."""
    long_prompt = (
        "The history of computing is a fascinating journey spanning centuries. "
        "The earliest computing devices were mechanical calculators, such as the "
        "abacus, which dates back thousands of years to ancient civilizations. "
        "In the 17th century, Blaise Pascal invented the Pascaline, a mechanical "
        "calculator capable of addition and subtraction. Around the same time, "
        "Gottfried Wilhelm Leibniz developed the Step Reckoner, which could "
        "perform all four arithmetic operations.\n\n"
        "The 19th century saw Charles Babbage design the Analytical Engine, "
        "which is considered the first general-purpose computer concept. Ada "
        "Lovelace wrote what is recognized as the first computer program for "
        "this machine. Babbage's designs were remarkably prescient, incorporating "
        "concepts like memory, processing, and input/output that would later "
        "become fundamental to modern computers.\n\n"
        "The 20th century brought electronic computing. Alan Turing's theoretical "
        "work on the Turing machine provided the mathematical foundation for "
        "computation. During World War II, machines like Colossus and ENIAC were "
        "built to perform calculations at unprecedented speed. ENIAC, completed "
        "in 1945, is often cited as the first general-purpose electronic computer. "
        "It weighed 30 tons and filled an entire room.\n\n"
        "The invention of the transistor at Bell Labs in 1947 by John Bardeen, "
        "Walter Brattain, and William Shockley revolutionized electronics. "
        "Transistors replaced vacuum tubes, making computers smaller, faster, "
        "and more reliable. The integrated circuit, developed independently by "
        "Jack Kilby and Robert Noyce in the late 1950s, allowed multiple "
        "transistors to be placed on a single chip.\n\n"
        "The personal computer revolution began in the 1970s with machines like "
        "the Altair 8800, Apple II, and IBM PC. The development of operating "
        "systems like Unix, MS-DOS, and later Windows and macOS made computers "
        "accessible to ordinary people. The World Wide Web, invented by Tim "
        "Berners-Lee in 1989, transformed computing from a standalone activity "
        "into a globally connected experience.\n\n"
        "What was the most significant invention mentioned above?"
    )
    output = generate(long_prompt, max_new_tokens=50, temperature=0.1)
    assert len(output.strip()) > 0, "Empty response to long prompt"
    return output.strip()[:80]


def test_multiple_requests():
    """Send 5 different prompts sequentially, all should succeed."""
    prompts = [
        "List three colors: ",
        "2 + 2 = ",
        "The largest planet in our solar system is ",
        "Water boils at ",
        "The speed of light is approximately ",
    ]
    results = []
    for i, prompt in enumerate(prompts):
        output = generate(prompt, max_new_tokens=30, temperature=0.1)
        assert len(output.strip()) > 0, f"Empty response for prompt {i}: {prompt!r}"
        results.append(output.strip()[:40])
    return f"All 5 returned output"


def test_image_prompt():
    """Multimodal: image + text via chat completions."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"
                    },
                },
                {
                    "type": "text",
                    "text": "What is unusual about this image? Describe in one sentence.",
                },
            ],
        }
    ]
    output = chat_completions(messages, max_tokens=50, temperature=0.1)
    assert len(output) > 10, f"Response too short: {output!r}"
    return output.strip()[:80]


def test_consistency():
    """Same prompt twice with temperature=0 should produce identical output."""
    prompt = "The chemical formula for water is"
    out1 = generate(prompt, max_new_tokens=20, temperature=0.0)
    out2 = generate(prompt, max_new_tokens=20, temperature=0.0)
    assert out1 == out2, f"Outputs differ:\n  1: {out1!r}\n  2: {out2!r}"
    return out1.strip()[:80]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_short_generation,
    test_long_generation,
    test_long_prompt,
    test_multiple_requests,
    test_image_prompt,
    test_consistency,
]


def main():
    if not check_server():
        print(f"SERVER NOT RUNNING at {BASE_URL} — skipping all tests")
        sys.exit(1)

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


if __name__ == "__main__":
    main()
