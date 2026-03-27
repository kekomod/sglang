"""
TurboQuant Validation Benchmark — Wikitext-2 Perplexity

Measures perplexity on Wikitext-2 using a sliding window approach.
Target: <5% PPL increase vs BF16 baseline (arXiv:2504.19874).

Usage (standalone, server already running):
    python eval_perplexity.py --port 30000 [--max-chunks 50]

Usage (multi-config, manages server lifecycle):
    python eval_perplexity.py --configs bf16 turboquant_3.5bit --model /path/to/model
"""

import argparse
import math
import time

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from common import (
    DEFAULT_PORT,
    MODEL_PATH,
    SERVER_CONFIGS,
    check_target,
    generate_with_logprobs,
    launch_server,
    print_comparison_table,
    save_results,
    shutdown_server,
)

# Lazy import — only needed when running, not when imported
_tokenizer = None
_tokenizer_model = None


def get_tokenizer(model_path: str | None = None):
    global _tokenizer, _tokenizer_model
    path = model_path or MODEL_PATH
    if _tokenizer is None or _tokenizer_model != path:
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer as _get_tok
        _tokenizer = _get_tok(path)
        _tokenizer_model = path
    return _tokenizer


def load_wikitext2_tokens(model_path: str | None = None) -> list[int]:
    """Load and tokenize the Wikitext-2 test split."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    tokenizer = get_tokenizer(model_path)
    tokens = tokenizer.encode(text)
    return tokens


def compute_perplexity(
    base_url: str,
    max_chunks: int | None = None,
    window_size: int = 2048,
    stride: int = 1536,  # window_size - overlap(512)
    model_path: str | None = None,
) -> dict:
    """Compute perplexity using sliding-window logprob extraction.

    For each chunk of *window_size* tokens we send a prefill-only request
    (max_new_tokens=0) with return_logprob=True.  We accumulate NLL only
    for the non-overlapping portion (the last *stride* tokens) except for
    the first chunk where we use all tokens.
    """
    tokens = load_wikitext2_tokens(model_path)
    total_tokens = len(tokens)
    print(f"[perplexity] Total tokens: {total_tokens}")

    # Build chunk start positions
    starts = list(range(0, total_tokens - window_size + 1, stride))
    if max_chunks is not None:
        starts = starts[:max_chunks]

    total_nll = 0.0
    total_count = 0

    for idx, start in enumerate(tqdm(starts, desc="Perplexity chunks")):
        end = start + window_size
        chunk_ids = tokens[start:end]

        result = generate_with_logprobs(base_url, chunk_ids, max_new_tokens=0)
        logprobs_raw = result["meta_info"]["input_token_logprobs"]
        # Each entry is [logprob, token_id] — extract logprob values
        logprobs = [lp[0] for lp in logprobs_raw]

        # First token has no logprob (conditioned on nothing) — logprobs
        # list is length (window_size - 1).  For the first chunk, use all
        # of them; for subsequent chunks, use only the last *stride* entries
        # (the non-overlapping part).
        if idx == 0:
            valid_lps = logprobs
        else:
            overlap = window_size - stride
            valid_lps = logprobs[overlap - 1:]  # -1 because logprobs is already len-1

        nll = -sum(lp for lp in valid_lps if lp is not None)
        count = sum(1 for lp in valid_lps if lp is not None)
        total_nll += nll
        total_count += count

    ppl = math.exp(total_nll / total_count) if total_count > 0 else float("inf")
    metrics = {
        "perplexity": round(ppl, 4),
        "total_nll": round(total_nll, 4),
        "total_tokens_scored": total_count,
        "num_chunks": len(starts),
    }
    print(f"[perplexity] PPL = {ppl:.4f}  (scored {total_count} tokens over {len(starts)} chunks)")
    return metrics


def run_multi_config(args) -> int:
    """Run perplexity eval across multiple configs, managing server lifecycle."""
    all_results: dict[str, dict] = {}

    for config_name in args.configs:
        print(f"\n{'#' * 60}")
        print(f"  Perplexity — {SERVER_CONFIGS[config_name]['name']}")
        print(f"{'#' * 60}")

        proc = None
        try:
            proc = launch_server(
                config_name,
                port=args.port,
                context_length=args.context_length,
                model_path=args.model,
            )
            base_url = f"http://127.0.0.1:{args.port}"

            t0 = time.time()
            metrics = compute_perplexity(
                base_url,
                max_chunks=args.max_chunks,
                model_path=args.model,
            )
            metrics["elapsed_seconds"] = round(time.time() - t0, 1)
            save_results("perplexity", config_name, metrics, args.output_dir)
            all_results[config_name] = metrics

        except Exception as exc:
            print(f"  ERROR running {config_name}: {exc}")
        finally:
            if proc is not None:
                shutdown_server(proc)
            time.sleep(5)

    # Print comparison table
    if all_results:
        display = {}
        for cfg, m in all_results.items():
            display[cfg] = {
                "perplexity": m["perplexity"],
                "tokens_scored": m["total_tokens_scored"],
                "chunks": m["num_chunks"],
            }
        print_comparison_table("perplexity", display)

        # PASS/FAIL check
        if "bf16" in all_results:
            baseline = all_results["bf16"]
            for cfg in all_results:
                if cfg == "bf16":
                    continue
                passed, msg = check_target("perplexity", baseline, all_results[cfg])
                status = "PASS" if passed else "FAIL"
                name = SERVER_CONFIGS.get(cfg, {}).get("name", cfg)
                print(f"  [{status}] {name}: {msg}")
            print()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Wikitext-2 perplexity benchmark")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit number of sliding-window chunks (for quick tests)")
    parser.add_argument("--config-name", type=str, default=None,
                        help="Config name for saving results (single-config mode)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--model", type=str,
                        default="/home/keko/AI/image-prep/models/qwen2.5-3b-instruct",
                        help="Model path for tokenizer and server")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Multi-config mode: launch server per config (e.g. bf16 turboquant_3.5bit)")
    parser.add_argument("--context-length", type=int, default=4096)
    args = parser.parse_args()

    # Multi-config mode: manage server lifecycle internally
    if args.configs:
        for c in args.configs:
            if c not in SERVER_CONFIGS:
                print(f"Unknown config: {c}. Choose from: {list(SERVER_CONFIGS)}")
                return 1
        return run_multi_config(args)

    # Single-config mode: server already running
    base_url = f"http://127.0.0.1:{args.port}"
    metrics = compute_perplexity(base_url, max_chunks=args.max_chunks, model_path=args.model)

    if args.config_name:
        save_results("perplexity", args.config_name, metrics, args.output_dir)

    return metrics


if __name__ == "__main__":
    main()
