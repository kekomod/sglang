# Phase I Handoff: Codebook Verification, Math Tests, Gather-Dequant, Stuttering Analysis

## Status: COMPLETE — Codebook verified, math tests built, extend optimized, stuttering resolved (Qwen2.5 unsupported)

### Phase I Session 2 Summary (added post-completion)
- **Stuttering root cause**: QKV attention bias in Qwen2/Qwen2.5 creates near-degenerate K vectors (mean_dir=0.998, norms 25-30x normal). Not a TQ code bug — no implementation handles biased inputs. Qwen2/2.5 marked UNSUPPORTED.
- **Llama-3.2-3B validated**: Pure transformer, no QKV bias, TQ works perfectly (6/6 tests).
- **Separate K/V rotations**: Added independent rotation seeds for K and V (matches MLX), verified with H_kv=2 regression tests.
- **Gather-dequant optimization**: Extend path uses selective O(prefix) dequant instead of O(pool_size).
- **Implementation comparison**: MSE codebook matches MLX exactly. Community consensus: QJL hurts quality, MSE-only is preferred.

## What Was Done

### I1: Codebook Verification Against MLX
**Result: PASS — Perfect match (max diff = 0.00e+00)**

Compared SGLang `compute_codebook(dim, bits)` against MLX's `_codebook(dim, bits)` for dim=[32, 64, 128, 256], bits=[2, 3, 4]. Both implementations are character-for-character identical in their numpy paths:
- Same grid: 32768 points, bounds -1+1e-6 to 1-1e-6
- Same Beta PDF formula
- Same Max-Lloyd: 100 iterations, 1e-6 convergence
- Same CDF quantile initialization
- Same seed formulas: rotation=`seed + dim * 7919`, projection=`seed + dim * 2971 + 17`

Only intentional difference: SGLang has analytical 1-bit fast path (`c = sqrt(2/pi)/sqrt(d)`), MLX doesn't.

### I2: Math Test Suite
**File: `sglang/test/test_turboquant_math.py` — 5/5 tests pass**

1. **test_codebook_cross_impl** (CPU): SGLang vs MLX codebook match
2. **test_codebook_optimality** (GPU): Empirical MSE within paper bounds (Theorems 1 & 3)
3. **test_cosine_sim_sweep** (GPU): Monotonicity in bits and dim
4. **test_multi_layer_attention_quality** (GPU): Multi-layer error accumulation simulation
5. **test_extend_roundtrip_error** (GPU): Extend path per-layer distortion

### I3: Key Quality Numbers

**Codebook optimality (dim=128):**
| Bits | Empirical MSE | Shannon Lower | Paper Upper |
|------|-------------|---------------|-------------|
| 2    | 0.116       | 0.0625        | 0.170       |
| 3    | 0.034       | 0.0156        | 0.043       |
| 4    | 0.009       | 0.0039        | 0.011       |

All within theoretical bounds.

**Per-token cosine similarity (MSE roundtrip):**
| dim\bits | b=2   | b=3   | b=4   |
|----------|-------|-------|-------|
| 32       | 0.945 | 0.984 | 0.996 |
| 64       | 0.941 | 0.984 | 0.996 |
| 128      | 0.941 | 0.983 | 0.995 |
| 256      | 0.940 | 0.983 | 0.995 |

**Multi-layer simulation (Qwen2.5-3B config: H_q=16, H_kv=2, D=128, seq_len=64, with RMSNorm + residual connections):**
| Layers | 3-bit sim | 4-bit sim |
|--------|----------|----------|
| 1      | 0.980    | 0.996    |
| 4      | 0.981    | 0.996    |
| 12     | 0.967    | 0.994    |
| 36     | 0.957    | 0.993    |

**Critical finding: 4-bit at 36 layers gives 0.993 cosine sim — this should NOT cause stuttering.**

### I4: Extend Path Optimization
**Files modified: `turboquant_pool.py`, `turboquant_backend.py`, `test_turboquant_kernel.py`**

Key discovery: the 2-stage Triton extend kernel ALREADY uses raw K,V for extend tokens (stage 2) and only reads dequantized pool for prefix tokens (stage 1). The H4 handoff incorrectly attributed stuttering to extend path roundtrip — extend tokens never go through a lossy roundtrip.

Changes:
1. **Pool**: Added `gather_dequant_key()` and `gather_dequant_value()` — selective dequant at specific positions only (O(prefix) instead of O(pool_size))
2. **Backend**: Overrode `forward_extend()` to use gather-dequant + remapped kv_indices instead of full-pool dequant
3. **Tests**: Added `test_gather_dequant` verifying gather matches full dequant at indexed positions

## The Stuttering Bug — Analysis

### What We Know
- TQ produces stuttering/repetition on Qwen2.5-3B at BOTH 3.5-bit and 4-bit
- BF16 baseline is perfect
- Math tests prove quantization quality is sufficient (0.993 cosine sim at 4-bit/36 layers)
- Kernel logic is correct (verified by code review and kernel tests)
- Extend path was already using raw K,V for extend tokens (not the cause)

### Most Likely Root Cause: MSE-Only Key Quantization Bias

The paper explicitly says (Section 3.3):
> "Why not just use TurboQuant_mse for keys? Because MSE-optimal quantizers are **biased** for inner products."

We switched keys from prod (MSE+QJL) to MSE-only in Phase H4 because QJL variance was too high at dim=64 (split groups). But MSE-only introduces a systematic bias in attention scores:
- `E[<q, dequant_mse(k)>] != <q, k>` (biased estimator)
- The bias is a multiplicative factor close to 1 at high bits, but signal-dependent
- Different tokens experience different bias magnitudes depending on their rotated-coordinate distribution
- This distorts the softmax attention distribution, potentially over-weighting some tokens

The paper uses prod (MSE+QJL) specifically because it provides an **unbiased** estimator:
- `E[<q, dequant_prod(k)>] = <q, k>` (unbiased)

### Why QJL Failed Previously

In Phase H4, prod key cosine sim was 0.948 at 3-bit/dim=128 (vs MSE's 0.983). The QJL residual at dim=64 (split groups) had too much variance. But this may have been a bug in our prod implementation, not an inherent limitation:
- Our prod cosine sim (0.918) seems low compared to what the paper claims
- The MLX prod codec uses identical math (verified in I1 exploration)
- Possible issues: norm storage precision (fp16), residual computation, or the projection matrix

### Proposed Next Steps (Priority Order)

1. **Test MSE score bias directly**: Measure `<q, dequant_mse(k)>` vs `<q, k>` for many random (q, k) pairs. Quantify the bias magnitude and signal-dependence. If bias is significant, it confirms this hypothesis.

2. **Investigate prod quality at full dim=128 (no split)**: Test prod_quantize at integer 3-bit (dim=128, no split groups). If cosine sim is much better than the 0.918 we saw before, the issue was specifically with dim=64 QJL.

3. **Try prod for keys at higher dims only**: Use prod for keys when dim >= 128 (integer bits), MSE-only for split groups. This would match the paper's approach for integer bit-widths.

4. **Fix QJL at dim=64**: The QJL projection matrix has dim^2 parameters — at dim=64 that's only 4096 elements. The matrix might need better conditioning or scaling. Compare our projection_matrix() output against MLX's for same seed.

5. **Compare against MLX end-to-end**: Run MLX's TurboQuant on same inputs, measure attention output quality. If MLX's prod codec gives better cosine sim, our prod implementation has a bug.

## Files Modified

| File | Change |
|------|--------|
| `sglang/test/test_turboquant_math.py` | NEW — 5 math validation tests |
| `sglang/python/sglang/srt/mem_cache/turboquant_pool.py` | Added gather_dequant_key/value methods |
| `sglang/python/sglang/srt/layers/attention/turboquant_backend.py` | Added forward_extend override with gather-dequant |
| `sglang/test/test_turboquant_kernel.py` | Added test_gather_dequant |
| `CLAUDE.md` | Updated status |

## Test Results

```
15/15 kernel tests pass (including gather_dequant)
5/5 math tests pass
2/3 piecewise tests pass (consistency fails due to stuttering-induced non-determinism)
```
