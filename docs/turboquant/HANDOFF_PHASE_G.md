# Phase G Handoff: Fused Attention Kernel Optimization

## Context

We have working fused Triton attention kernels in `python/sglang/srt/layers/attention/turboquant_backend.py` that read directly from packed uint8 buffers. These are opt-in via `--attention-backend turboquant`. Both decode and extend kernels work correctly (11/11 tests pass).

However, the surrounding Python code dominates runtime:
1. **FWHT for query rotation** — `hadamard.forward(q)` runs before the attention kernel
2. **QJL projection** — `q_projected = q @ S.T` for key scoring
3. **FWHT for output inverse-rotation** — `hadamard.inverse(output)` runs after the attention kernel
4. **Pack/unpack overhead** — `set_kv_buffer()` quantization for fresh KV during extend

The attention kernels themselves (score computation, softmax, weighted sum) are fast — the bottleneck is the Python glue between kernel launches.

## What to Optimize

### Goal: Zero Python overhead between raw query and attention output

Fuse the entire attention pipeline into minimal kernel launches:

```
Current flow (5+ kernel launches per forward):
  1. hadamard.forward(q)          # FWHT kernel
  2. q_proj = q @ S.T             # matmul kernel
  3. attention_kernel(q, q_proj, packed_kv)  # attention kernel
  4. hadamard.inverse(output)     # FWHT kernel
  5. [extend only] quantize fresh KV  # multiple kernels

Target flow (2-3 kernel launches):
  1. fused_attention(q, signs, S, packed_kv) → output
     - Inline: sign flip + FWHT + projection + scoring + softmax + weighted sum + inverse FWHT
  2. [extend only] fused_quantize(fresh_kv) → packed
```

### Decode kernel fusion

Modify `turboquant_decode_attention_fwd` to accept raw (unrotated) queries and perform rotation inline:

```python
@triton.jit
def turboquant_decode_fused_kernel(
    Q,              # [B, H, D] raw queries (not yet rotated)
    signs,          # [H, D] Hadamard sign vectors (per layer)
    S,              # [D, D] QJL projection matrix
    packed_k,       # packed key cache
    packed_v,       # packed value cache
    ...
):
    # Step 1: Load query
    q = tl.load(Q + ...)

    # Step 2: Forward Hadamard on query (for key scoring)
    q_signs = tl.load(signs + ...)
    q_rotated = fwht_inline(q * q_signs)  # inline butterfly stages

    # Step 3: QJL projection (for prod scoring)
    # q_proj = q_rotated @ S^T — this is a D×D matmul per query
    # For D=128, this fits in registers with tiling

    # Step 4: Score against all keys (existing kernel logic)
    # ... online softmax over sequence blocks ...

    # Step 5: Weighted sum of values (existing kernel logic)
    # ... accumulate codebook values weighted by softmax ...

    # Step 6: Inverse Hadamard on output
    output = fwht_inline(weighted_sum) * q_signs

    tl.store(output_ptr + ..., output)
```

**Challenge:** The QJL projection is a D×D matmul (128×128). This is 16K multiply-add operations per query per head. In the current two-stage decode kernel, each program handles one head across multiple sequence blocks. The matmul needs to happen once per head, before the sequence-block loop.

**Options for the matmul:**
1. **Inline in Triton** — Load S matrix tiles, accumulate in registers. For D=128 with BLOCK_SIZE=32, this is 4×4=16 tiles. Feasible but register-heavy.
2. **Separate kernel** — Keep the matmul as a separate cuBLAS call. Accept 2 kernel launches instead of 1. This is pragmatic since cuBLAS is highly optimized for small matmuls.
3. **Skip projection for value scoring** — Values use MSE mode which doesn't need QJL projection. Only keys need it. So the projection is only needed for the scoring half, not the weighted-sum half.

### Extend kernel fusion

The extend kernel is more complex because it also quantizes fresh KV. The fusion opportunity is:

```
Current: quantize(fresh_kv) → store → load_all_kv → attend
Fused:   attend_with_inline_quantize(fresh_kv, cached_packed_kv)
```

This is harder because the extend kernel processes both prefix (already quantized) and extend (fresh BF16) tokens. The current quantize-first approach simplifies the kernel by making all KV uniform. Fusing quantization into the kernel would require handling two data formats inline.

**Recommendation:** Keep quantize-first for extend. Focus fusion efforts on the rotation/projection steps only.

### Inline FWHT helper

Create a Triton inline function for the 7-stage butterfly:

```python
@triton.jit
def fwht_inline(x, D: tl.constexpr, LOG_D: tl.constexpr):
    """Inline FWHT for use inside other Triton kernels.
    x: a register-resident vector of D elements.
    Returns: FWHT(x) (not yet scaled — caller handles scaling).
    """
    # This needs careful implementation since x is a 1D register block
    # and we need to do butterfly operations on pairs within it.
    # Use the XOR-partner trick from Phase E.
    offs = tl.arange(0, D)
    stride = 1
    for _ in tl.static_range(LOG_D):
        top_mask = ((offs & stride) == 0)
        partner_offs = offs ^ stride
        # Gather partner values — this is the tricky part in registers
        # May need to use tl.where with shifted copies
        ...
        stride *= 2
    return x
```

**Note:** Inline FWHT within a larger kernel is tricky because Triton doesn't have native shuffle/permute within a register vector. The standalone FWHT kernel from Phase E uses `tl.load`/`tl.store` to memory, but inline we want to stay in registers. This may require creative use of `tl.where` and redundant computation. Investigate Triton's support for register-level permutations.

## Reference: MLX PR #858

MLX has 9 specialized Metal kernels that demonstrate full fusion:

1. `score_packed_mse` — Score Q against packed K using MSE codebook
2. `score_packed_prod` — Score Q against packed K using product codebook + QJL signs
3. `weighted_sum_packed_mse` — Weighted sum of packed V using MSE codebook
4. `fast_wht_forward` / `fast_wht_inverse` — Standalone FWHT
5. Various helper kernels for quantization

The key insight from MLX: the score computation and weighted sum are separate kernels that each read packed data inline. The FWHT is separate. This suggests that full single-kernel fusion may not be necessary — the main win is avoiding BF16 materialization of dequantized KV, which our kernels already do.

## Key Optimization: Value Weighted-Sum in Rotated Space

(See HANDOFF_PHASE_H.md for full details)

The biggest algorithmic optimization is computing the value weighted sum in rotated space, reducing inverse Hadamard calls from O(T) to O(1). This should be implemented alongside the kernel fusion work since it changes the value accumulation logic.

## Files to Modify

### `python/sglang/srt/layers/attention/turboquant_backend.py`
- Modify `forward_decode()`: pass raw queries + signs to fused kernel, remove Python-level rotation
- Modify `forward_extend()`: same for extend path (rotation only, keep quantize-first)
- Update kernel launch parameters

### `python/sglang/srt/layers/attention/triton_ops/turboquant_decode_attention.py`
- Add fused variant `turboquant_decode_fused_fwd` with inline rotation
- Keep existing `turboquant_decode_attention_fwd` as fallback
- Add inline FWHT helper function

### `python/sglang/srt/layers/attention/triton_ops/turboquant_extend_attention.py`
- Add fused variant with inline rotation for queries
- Keep quantize-first approach for KV

## Verification

```bash
conda activate turboquant

# All 11 kernel tests must still pass
python -m pytest test/test_turboquant_kernel.py -v

# Profile before and after to measure speedup
# Focus on per-token decode latency (most sensitive to kernel launch overhead)
python -m sglang.launch_server \
    --model /home/keko/AI/image-prep/models/uncensored-9b-bf16-hf \
    --kv-cache-quantization turboquant \
    --attention-backend turboquant \
    --turboquant-bits 3.5 \
    --port 30000 \
    --disable-cuda-graph \
    --context-length 4096

# Benchmark decode latency with long context
python benchmark/turboquant/eval_needle.py --port 30000 --context-lengths 2048 4096
```

## Performance Expectations

The main expected wins:
1. **Eliminating 2 FWHT kernel launches per layer per forward** — saves ~2 × 36 = 72 kernel launches per forward pass
2. **Eliminating Python overhead** between kernel launches — PyTorch dispatch + Python interpreter overhead for each call
3. **Better GPU utilization** — fewer kernel launch gaps means better SM occupancy

The actual latency improvement depends on how much of the total time is kernel-launch overhead vs. actual compute. Profile with `nsys` to measure.
