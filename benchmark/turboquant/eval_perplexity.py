"""
TurboQuant Validation Benchmark — Wikitext-2 Perplexity

Measures perplexity on Wikitext-2 using a sliding window approach.
Target: <5% PPL increase vs BF16 baseline (arXiv:2504.19874).

Usage (standalone, server already running):
    python eval_perplexity.py --port 30000 [--max-chunks 50]
"""

import argparse
import math

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from common import (
    DEFAULT_PORT,
    MODEL_PATH,
    generate_with_logprobs,
    print_comparison_table,
    save_results,
)

# Lazy import — only needed when running, not when imported
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer as _get_tok
        _tokenizer = _get_tok(MODEL_PATH)
    return _tokenizer


def load_wikitext2_tokens() -> list[int]:
    """Load and tokenize the Wikitext-2 test split."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    return tokens


def compute_perplexity(
    base_url: str,
    max_chunks: int | None = None,
    window_size: int = 2048,
    stride: int = 1536,  # window_size - overlap(512)
) -> dict:
    """Compute perplexity using sliding-window logprob extraction.

    For each chunk of *window_size* tokens we send a prefill-only request
    (max_new_tokens=0) with return_logprob=True.  We accumulate NLL only
    for the non-overlapping portion (the last *stride* tokens) except for
    the first chunk where we use all tokens.
    """
    tokens = load_wikitext2_tokens()
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


def main():
    parser = argparse.ArgumentParser(description="Wikitext-2 perplexity benchmark")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit number of sliding-window chunks (for quick tests)")
    parser.add_argument("--config-name", type=str, default=None,
                        help="Config name for saving results (e.g. bf16, turboquant_3bit)")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    metrics = compute_perplexity(base_url, max_chunks=args.max_chunks)

    if args.config_name:
        save_results("perplexity", args.config_name, metrics, args.output_dir)

    return metrics


if __name__ == "__main__":
    main()
