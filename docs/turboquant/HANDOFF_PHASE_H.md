# Phase H Handoff: Value Weighted-Sum in Rotated Space

## Context (Paper Section 3)

After softmax, the attention output for one head is:

```
output = sum_t(alpha_t * V_t)
```

where `alpha_t` are softmax weights and `V_t` is the dequantized value vector for token `t`.

The TurboQuant MSE dequantization of a value vector is:

```
V_t = norm_t * H_inv(codebook[indices_t])
```

where:
- `norm_t` is the scalar norm stored per token
- `H_inv` is the inverse Hadamard transform (sign flip + FWHT + scale)
- `codebook[indices_t]` looks up the codebook entry for each of the D dimensions

Currently, we dequantize every value vector first (applying `H_inv` to each), then compute the weighted sum. For a sequence of length T, this means T inverse Hadamard transforms of D-dimensional vectors: **O(T * D * log D)** total work.

## The Optimization

Since the inverse Hadamard transform is **linear**, we can factor it out:

```
output = sum_t(alpha_t * norm_t * H_inv(codebook[indices_t]))
       = H_inv( sum_t(alpha_t * norm_t * codebook[indices_t]) )
```

**Proof:**
- `H_inv(x) = signs * FWHT(x)` where FWHT and element-wise sign multiplication are both linear operations
- For linear operator L: `sum_t(a_t * L(x_t)) = L(sum_t(a_t * x_t))`
- Therefore: `sum_t(alpha_t * norm_t * H_inv(cb_t)) = H_inv(sum_t(alpha_t * norm_t * cb_t))`

### Complexity reduction

| Operation | Before | After |
|-----------|--------|-------|
| Codebook lookups | T * D | T * D (same) |
| Scalar multiplications | T * D (norms) | T * D (norms, same) |
| Weighted accumulation | T * D | T * D (same) |
| Inverse Hadamard | **T * D * log(D)** | **1 * D * log(D)** |
| **Total** | **O(T * D * (1 + log D))** | **O(T * D + D * log D)** |

For D=128 (log D = 7), the inverse Hadamard goes from being ~87.5% of the work (7T/(T+7T)) to negligible (7/(T+7)). For T=1000, this is roughly a **7x reduction** in the value-side compute.

## Implementation

### In the fused Triton decode kernel

Modify the value accumulation loop to work in rotated space:

```python
# Current implementation (turboquant_decode_attention.py):
# For each sequence block:
#   1. Compute scores, update running softmax
#   2. Unpack value indices
#   3. Look up codebook: v_dequant = norm * codebook[idx]
#   4. NOTE: Currently does NOT apply H_inv here (values stored post-rotation)
#   5. Accumulate: weighted_sum += softmax_weight * v_dequant

# New implementation:
# For each sequence block:
#   1. Compute scores, update running softmax
#   2. Unpack value indices
#   3. Look up codebook entries directly: cb_val = codebook[idx]
#   4. Accumulate in rotated space: rot_sum += softmax_weight * norm * cb_val
# After all blocks:
#   5. Apply inverse Hadamard ONCE: output = H_inv(rot_sum)
```

The key change: the per-token accumulation uses raw codebook values (which are already in rotated space since values were quantized after forward Hadamard). The inverse Hadamard is only applied once to the final accumulated result.

### Important: Current kernel already works in rotated space

Looking at the current implementation, the decode kernel already accumulates codebook values directly without inverse rotation inside the loop. The inverse rotation happens after the kernel returns, in `turboquant_backend.py`:

```python
# In TurboQuantAttentionBackend.forward_decode():
output = turboquant_decode_attention_fwd(...)  # returns weighted sum in rotated space
output = self.hadamard.inverse(output)         # apply inverse rotation once
```

**This means the rotated-space optimization is ALREADY partially implemented for decode.** The remaining work is:

1. **Extend kernel** — verify the extend kernel also accumulates in rotated space
2. **Non-fused path** — the dequant-on-read path (`get_value_buffer()`) currently applies full dequantization (including inverse Hadamard) for every token, then the standard attention backend computes the weighted sum. For the non-fused path, we can't easily do rotated-space accumulation since the standard backend expects BF16 values.
3. **Norm handling** — ensure norms are properly factored into the accumulation

### Extend kernel changes

In the extend kernel, the value accumulation loop processes all tokens (prefix + extend). Verify that:

```python
# In turboquant_extend_attention_fwd:
# The inner loop should accumulate:
#   rot_sum += alpha_t * norm_t * codebook[indices_t]
# NOT:
#   rot_sum += alpha_t * norm_t * H_inv(codebook[indices_t])
```

If the extend kernel already works this way (likely, since it was modeled after the decode kernel), then the optimization is already in place. The inverse Hadamard should be applied once after the kernel returns.

### Non-fused path optimization (optional)

For the dequant-on-read path, we could add a `get_value_buffer_rotated()` method to the pool:

```python
class TurboQuantTokenToKVPool:
    def get_value_buffer_rotated(self, layer_id):
        """Return dequantized values WITHOUT inverse Hadamard.

        Values are in rotated space: norm * codebook[indices].
        Caller must apply inverse Hadamard to the attention output.
        """
        indices = self._unpack_v(layer_id)
        cb = self.codebook_v[layer_id]
        norms = self.v_norm_buffer[layer_id]
        return norms.unsqueeze(-1) * cb[indices]  # [pool_size, H, D]
```

This would require modifying the standard attention backends to apply inverse Hadamard after the weighted sum, which is invasive. **Recommendation: Only implement for the fused path. The non-fused path is a correctness fallback, not a performance target.**

## Files to Modify

### `python/sglang/srt/layers/attention/triton_ops/turboquant_decode_attention.py`
- Verify accumulation is in rotated space (likely already correct)
- Add comments documenting the rotated-space optimization
- Ensure norm multiplication happens inside the accumulation loop

### `python/sglang/srt/layers/attention/triton_ops/turboquant_extend_attention.py`
- Same verification and changes as decode kernel

### `python/sglang/srt/layers/attention/turboquant_backend.py`
- Verify that `hadamard.inverse()` is called ONCE on the output, not per-token
- Add comments explaining why this is correct (linearity of H_inv)

### `python/sglang/srt/mem_cache/turboquant_pool.py` (optional)
- Add `get_value_buffer_rotated()` if we want to support rotated-space accumulation in the non-fused path

## Verification

### Correctness test

Create a test that compares the two approaches:

```python
def test_rotated_space_equivalence():
    """Verify rotated-space accumulation equals standard accumulation."""
    # Generate random attention weights, packed values, norms
    # Method 1 (standard): dequant each V, then weighted sum
    for t in range(T):
        v_t = dequant(packed_v[t])  # includes inverse Hadamard
        output_standard += alpha[t] * v_t

    # Method 2 (rotated): accumulate codebook values, then one inverse Hadamard
    for t in range(T):
        cb_t = norm[t] * codebook[indices[t]]  # NO inverse Hadamard
        rot_sum += alpha[t] * cb_t
    output_rotated = hadamard.inverse(rot_sum)

    assert torch.allclose(output_standard, output_rotated, atol=1e-3)
```

### Existing tests

All 11 kernel tests must continue to pass. The optimization doesn't change the mathematical result, only the order of operations.

### Performance benchmark

```bash
# Profile decode latency to measure improvement
# The improvement scales with sequence length — longer sequences benefit more
python benchmark/turboquant/eval_needle.py --port 30000 --context-lengths 1024 2048 4096
```

## Relationship to Phase G

Phase H (rotated-space accumulation) and Phase G (kernel fusion) are complementary:
- Phase G fuses rotation into the attention kernel to eliminate kernel launches
- Phase H reduces the amount of rotation work from O(T) to O(1)

They can be implemented independently, but the combination gives the best performance. If implementing both:
1. The fused kernel does forward Hadamard on the query (inline)
2. The kernel accumulates values in rotated space (codebook lookups only)
3. The kernel applies inverse Hadamard once on the accumulated output (inline)
4. Total Hadamard operations per head per layer: 2 (forward on Q, inverse on output) regardless of sequence length
