# Phase H3 Handoff: Paper Validation & Correctness Fix

## Status: CRITICAL BUG — TQ Produces Garbage on Qwen2.5-3B

> **Blocker before Phase I.** Paper validation benchmarks revealed that TurboQuant produces completely wrong outputs on Qwen2.5-3B-Instruct (0% accuracy on NIAH, 0% on GSM8K) while BF16 works perfectly (100% NIAH, 78% GSM8K). Memory compression works (11.3 GB TQ vs 21.9 GB BF16) but attention output is incorrect.

## What Was Done This Session

### Phase G: Fused FWHT Decode Kernels
- Inline butterfly helper `_fwht_butterfly_inplace` for use inside Triton kernels
- Fused Stage 2 + inverse FWHT kernel (`_turboquant_stage2_inv_fwht_kernel`) — merges kv_splits and applies inverse Hadamard in one kernel launch
- Both integer-bit and split-channel variants
- **Reverted in H2.4:** The initial Phase G approach inlined forward FWHT into Stage 1, causing 8x redundant computation (each kv_split program recomputed the same FWHT). Reverted to use the original Stage 1 kernel with pre-rotated queries. Only the Stage 2 inverse FWHT fusion remains.
- Net effect: 4 kernel launches per decode layer (was 5) — the inverse FWHT is fused into Stage 2

### Phase H: Rotated-Space Accumulation (Verified)
- Both decode and extend kernels already accumulate values in rotated space (codebook lookup without per-token inverse Hadamard)
- Inverse Hadamard applied ONCE per layer in the backend — mathematically correct due to linearity of H_inv
- Added `test_rotated_space_equivalence` proving this
- Added Phase H comments to kernel code

### Phase H2: Critical Fixes
- **H2.1: Auto-fused backend default** — When `--kv-cache-quantization turboquant` is set, automatically selects `--attention-backend turboquant`. Eliminates the BF16 workspace that was doubling KV cache memory. Launch memory dropped from 20321 MB to 19505 MB (vs 20063 MB BF16 baseline).
- **H2.4: Redundant FWHT fix** — Reverted inline forward FWHT in fused Stage 1. Original Stage 1 kernel + fused Stage 2 inverse.
- **H2.8: PAPER_SUMMARY.md fix** — Section 6.5 clarified as Google blog addendum, not paper section.
- **H2.9: Extend kernel fusion** — Fused extend wrappers handle inverse FWHT internally. Backend no longer calls `hadamard.inverse()` for extend path.

### Deep Research: Comparison Against Paper & Other Implementations
- Read actual paper (arXiv:2504.19874) — TQ is about memory savings for longer context, NOT throughput improvement
- Analyzed SGLang PR #21419 (same workspace problem), vLLM PRs #38273/#38280 (pre-dequant only, no savings), MLX-VLM (gold standard with fused Metal kernels)
- RotorQuant (Clifford algebra rotors) — interesting math but not useful (FWHT is better)
- TheTom/turboquant_plus — has sparse V dequant optimization worth investigating
- Our codebook already uses Beta distribution (was thought to be Gaussian but is correct)

### Paper Validation Benchmark Suite (NEW)
Built 6 benchmark scripts replicating the paper's experiments:
- `eval_distortion.py` — MSE/inner-product distortion (standalone, no server)
- `eval_needle.py` — NIAH at 4K-32K, multi-config, OOM detection
- `eval_longbench.py` — 6 LongBench-E categories (BROKEN — HF dataset loading fails)
- `eval_perplexity.py` — Wikitext-2 perplexity, multi-config
- `eval_gsm8k.py` — GSM8K arithmetic reasoning, multi-config
- `run_paper_validation.py` — Master script running all benchmarks

## Paper Validation Results (Qwen2.5-3B-Instruct)

| Benchmark | BF16 Baseline | TQ 3.5-bit | Status |
|-----------|--------------|------------|--------|
| **Distortion** | — | MSE tracks theoretical bound (0.121 vs 0.116 at 2-bit) | ✅ PASS (standalone, no server) |
| **NIAH (4K-16K)** | 1.000 | **0.000** | ❌ CRITICAL — TQ outputs wrong answers |
| **NIAH (32K)** | OOM | OOM (400 error) | Both fail at 32K |
| **Perplexity** | Connection error | Timeout | ⚠️ Script issues (server dying) |
| **LongBench-E** | 0.0 | 0.0 | ⚠️ Dataset loading broken (trust_remote_code) |
| **GSM8K** | 78% (39/50) | **0% (0/50)** | ❌ CRITICAL — TQ outputs garbage |
| **Memory (launch)** | 21,900 MB | **11,302 MB** | ✅ Memory compression works (48% less) |

### Key Observation
The distortion benchmark (standalone, no server) shows TQ math is correct — quantization and dequantization work properly in isolation. The bug is in the **server integration path** — somewhere between `set_kv_buffer()` and attention output, the data gets corrupted for Qwen2.5-3B.

On Qwen3.5-9B, everything works (8/8 server tests pass, coherent outputs). The bug is **model-specific** — likely related to how the TQ pool is configured for Qwen2.5-3B's architecture (pure transformer, 36 layers, 2 KV heads, head_dim=128).

## Bug Investigation Leads

### 1. Head dimension mismatch
Qwen2.5-3B: head_dim=128, KV heads=2, Q heads=16 (GQA ratio 8:1)
Qwen3.5-9B: head_dim=256, KV heads=4, Q heads=32 (GQA ratio 8:1)

The TQ pool initializes buffers based on `head_dim` and `head_num`. If there's an off-by-one or dimension mismatch in the packed buffer stride calculations for head_dim=128 vs 256, it would corrupt data.

### 2. Piecewise vs standard CUDA graphs
Qwen2.5-3B uses piecewise CUDA graphs (pure transformer). Qwen3.5-9B uses standard CUDA graphs (hybrid model with DeltaNet layers). The graph capture/replay path differs. TQ quantization inside piecewise graphs may have different timing/ordering.

### 3. Layer mapping
Qwen3.5-9B uses `HybridLinearKVPool` which wraps `TurboQuantTokenToKVPool` and maps layer IDs via `_transfer_full_attention_id()`. Qwen2.5-3B uses `TurboQuantTokenToKVPool` directly (all 36 layers). The `start_layer`/`end_layer` mapping may be wrong for the direct path.

### 4. The 3/3 piecewise tests pass but are superficial
`test_turboquant_piecewise.py` tests: short generation ("Paris"), consistency (temperature=0), concurrent (10 requests). These pass but don't verify factual accuracy — "Paris" could match by chance in a garbled output. The test may be passing with degraded but not completely broken output.

### 5. Fused extend wrapper may corrupt output
The new `turboquant_extend_attention_fused_fwd` wrapper calls `triton_fwht_inverse` and then `o.copy_(o_inv)`. If the shapes don't match (padded_dim vs original_dim) or the copy is wrong, it would corrupt the output during prefill.

## Benchmark Script Issues to Fix

### eval_distortion.py
- Exit code 2 when run from master script (works standalone) — likely CLI arg parsing issue with `--output-dir`

### eval_perplexity.py
- Server dies mid-benchmark (connection refused for BF16, timeout for TQ)
- The perplexity eval sends many sequential requests — server may OOM or crash under load
- Need to investigate why server becomes unhealthy

### eval_longbench.py
- `THUDM/LongBench` dataset uses a loading script that newer HuggingFace `datasets` library no longer supports
- Fix: use `THUDM/LongBench-v2` or download the dataset manually and load from disk

### eval_needle.py
- Works correctly for BF16 (1.000 at 4K-16K)
- 32K context fails with 400 Bad Request (context too long for server's allocation)
- Need to launch server with matching `--context-length` per test

### run_paper_validation.py
- Summary section shows "no results" for most benchmarks — result JSON parsing may not match the format written by sub-scripts

## Files Changed This Session

| File | Changes |
|------|---------|
| `turboquant_decode_attention.py` | +1222 lines: butterfly helper, fused Stage 1 (reverted to use original), fused Stage 2 + inv FWHT, split-channel variants, fused wrappers |
| `turboquant_extend_attention.py` | +141 lines: fused extend wrappers (integer + split) with post-kernel inverse FWHT |
| `turboquant_backend.py` | +166/-116: imports fused variants, forward_decode uses fused Stage 2, forward_extend uses fused wrappers, q_rot pre-computed externally |
| `server_args.py` | +24/-14: auto-select fused backend when TQ enabled, warning for non-fused |
| `test_turboquant_kernel.py` | +272 lines: test_fused_decode_vs_nonfused, test_fused_split_decode_vs_nonfused, test_rotated_space_equivalence |
| `eval_throughput.py` | +8/-2: VLM prompt from file, context_length 8192 |
| `PAPER_SUMMARY.md` | +6/-2: Section 6.5 clarified as blog addendum |
| `eval_distortion.py` | NEW: standalone distortion benchmark |
| `eval_needle.py` | REWRITTEN: multi-config, long context, OOM detection |
| `eval_longbench.py` | NEW: LongBench-E 6 categories |
| `eval_perplexity.py` | EXTENDED: multi-config, --model arg |
| `eval_gsm8k.py` | EXTENDED: multi-config, --model arg |
| `run_paper_validation.py` | NEW: master validation runner |
| `HANDOFF_PHASE_I.md` | +74 lines: I7-I12 (long-context, concurrency, metrics, optimizations) |

## Test Results

- **15/15 kernel tests pass** (including 3 new: fused decode, fused split, rotated-space)
- **8/8 server tests pass** on Qwen3.5-9B with auto-fused backend
- **3/3 piecewise tests pass** on Qwen2.5-3B (but may be superficially correct)

## Throughput & Memory (Qwen3.5-9B, 5K-token prompts)

| Config | gen tok/s | vs BF16 | Launch MB | Steady MB |
|--------|-----------|---------|-----------|-----------|
| BF16 | 39.0 | 1.00x | 20,063 | 20,571 |
| TQ 3.5-bit | 32.5 | 0.83x | 19,505 | 20,075 |
| TQ 3-bit | 32.6 | 0.84x | 19,647 | 20,217 |
| TQ fused | 32.4 | 0.83x | 19,505 | 20,075 |

- 0.83x throughput is expected (paper doesn't claim speedup)
- 558 MB launch memory savings with fused backend (no BF16 workspace)
- All TQ configs now auto-select fused backend

## What Must Happen Next (Priority Order)

### P0: Fix TQ on Qwen2.5-3B (CRITICAL)
The 0% accuracy on a pure transformer model means TQ is broken for the primary use case. Investigate:
1. Check if the bug is in `set_kv_buffer` (quantization path) or attention (dequant/scoring path)
2. Run the kernel tests with Qwen2.5-3B dimensions (head_dim=128, H_kv=2) — they pass at head_dim=256, do they pass at 128?
3. Check the fused extend wrapper's `o.copy_(o_inv)` — shape mismatch?
4. Test with the NON-fused path on Qwen2.5-3B (disable auto-fused, force FlashInfer+workspace)

### P1: Fix Benchmark Scripts
- eval_distortion.py: CLI arg fix for master script
- eval_longbench.py: Fix HF dataset loading (use LongBench-v2 or manual download)
- eval_perplexity.py: Fix server dying mid-benchmark
- run_paper_validation.py: Fix result JSON parsing for summary

### P2: Run Full Validation
Once P0 and P1 are fixed, re-run the complete paper validation suite and verify:
- NIAH >0.95 at all context lengths
- Perplexity <5% increase
- GSM8K <2% accuracy drop
- LongBench-E within 1 point of BF16

### P3: Long-Context Demo (Phase I7)
- Show BF16 OOMs at 32K+ while TQ keeps going
- This is the "money shot" that proves TQ's value

## Architecture Overview

```
Server Launch
  └─ --kv-cache-quantization turboquant
     └─ Auto-selects --attention-backend turboquant (H2.1)
        └─ No BF16 workspace allocated (saves ~500MB)

Per-Layer Forward Pass (Decode):
  1. set_kv_buffer() → quantize K,V into packed uint8 buffers
  2. hadamard.forward(q) → standalone FWHT kernel (tiny, fast)
  3. torch.matmul(q, S.T) → cuBLAS QJL projection
  4. _turboquant_fwd_kernel_stage1 → original Triton kernel (pre-rotated q)
  5. _turboquant_stage2_inv_fwht_kernel → fused merge + inline inverse FWHT
  → Output is de-rotated, ready to return

Per-Layer Forward Pass (Extend/Prefill):
  1. set_kv_buffer() → quantize K,V
  2. hadamard.forward(q) → FWHT
  3. matmul → QJL projection
  4. turboquant_extend_attention_fused_fwd → extend kernel + post-kernel inv FWHT
  → Output is de-rotated

Memory Pool (TurboQuantTokenToKVPool):
  - Packed K buffers: uint8 (MSE indices + QJL signs + norms)
  - Packed V buffers: uint8 (MSE indices + norms)
  - NO BF16 workspace (fused backend default)
  - Split-channel variant for 3.5-bit (lo/hi groups)
```

## Key Insights From Research

1. **TQ = memory savings for longer context, NOT faster inference.** 0.83x throughput is the expected cost.
2. **FlashInfer + TQ wastes memory** (BF16 workspace + packed = more than vanilla BF16). Fused backend is the only path with real savings.
3. **MLX-VLM is the gold standard** — fused Metal kernels reading directly from compressed data.
4. **Our fused Triton path is architecturally correct** — same approach as MLX-VLM but for CUDA.
5. **The paper uses Beta-distribution codebooks** — our implementation is correct (already uses Beta, not Gaussian as initially thought).
6. **Sparse V dequant** (from TheTom/turboquant_plus) could save ~20% decode throughput at long context — worth implementing in Phase I.
7. **Bag-of-words cosine similarity is the wrong quality metric** — use task accuracy (NIAH, GSM8K, LongBench-E).
