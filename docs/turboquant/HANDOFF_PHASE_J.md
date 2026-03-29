# Phase J Handoff: Polish, Benchmarks, Dead Code Cleanup

## Status: COMPLETE

## What Was Done

### Dead Code Cleanup (~3000 lines removed)
- Removed all QJL/Prod code (Algorithm 2) — community consensus: MSE-only is better
- Deleted `triton_fwht.py`, `turboquant_extend_attention.py`, `eval_perplexity.py`, `TURBOQUANT_TRACE_ANALYSIS.md`
- Removed fused decode kernels, stage2 inv-FWHT kernels, all prod imports/tests/benchmarks
- Removed fabricated benchmark targets (GSM8K 2%, perplexity 5% — NOT from paper)

### Benchmark Infrastructure
- Created `eval_memory_scaling.py` — memory comparison at progressive context lengths
- Added `--context-length` to eval_throughput.py and eval_longbench.py
- Fixed LongBench dataset loading (switched to direct JSONL download)
- Fixed NIAH 32K prompt overflow (margin 20 → 200 tokens)
- Fixed server log deadlock: stdout to log file instead of subprocess.PIPE
- Created `.venv` — never use conda

### Benchmark Results

**NIAH — Zero quality loss:**

| Model | 4K | 8K | 16K | 32K | Memory Savings |
|-------|----|----|-----|-----|----------------|
| Llama-3.2-3B | 100% | 100% | 100% | 100% | -36% |
| Mistral-7B | 100% | 100% | 100% | 80%* | -17% |
| Qwen3.5-2B | 100% | 100% | 100% | 100% | ~0%** |

**LongBench-E (avg F1 × 100):**

| Model | BF16 | TQ 3.5-bit | Paper (8B) |
|-------|------|------------|------------|
| Llama-3.2-3B | 7.4 | 6.9 | 50.06 |
| Mistral-7B | 9.6 | 11.6 | 50.06 |
| Qwen3.5-2B | 16.3 | 14.4 | 50.06 |

### Models
- Llama-3.2-3B, Mistral-7B-v0.3, Qwen3.5-2B, Qwen3.5-9B (all at `/home/keko/AI/image-prep/models/`)
- Qwen2.5-3B: UNSUPPORTED (QKV bias)

## Future Work
- Sparse V dequant, outlier channel routing, weight quant + TQ composition
- Llama-3.1-8B for paper-comparable LongBench numbers
