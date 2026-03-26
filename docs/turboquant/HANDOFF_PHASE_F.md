# Phase F Handoff: CUDA Graph Support — COMPLETED (2026-03-27, F2 approach)

> **Status:** F2 (BF16 workspace) implemented. CUDA graphs enabled by default for FlashInfer backend. Throughput: 35.2 tok/s (0.76x baseline). F1 (full graph-safe) remains as future optimization to eliminate BF16 workspace memory overhead.

## Context

TurboQuant currently auto-disables CUDA graphs in `server_args.py`. This is necessary because the quantize/dequantize operations run during `set_kv_buffer()` / `get_key_buffer()` / `get_value_buffer()`, which execute inside the CUDA graph capture context. CUDA graph capture requires all tensor shapes to be deterministic and prohibits certain dynamic operations.

After Phase 5B fixes (vectorized pack/unpack, tensors pre-placed on GPU, torch.compile FWHT), the remaining blockers are:

1. **`torch.zeros()` allocations in pack/unpack** — `quant_ops.py` creates temporary tensors during quantization. CUDA graph capture does not allow `torch.zeros()` / `torch.empty()` calls.
2. **`torch.compile` FWHT** — The compiled FWHT kernel may or may not be captured correctly by CUDA graphs. torch.compile + CUDA graphs interaction is version-dependent.
3. **`F.pad()` in Hadamard forward** — Padding to next power-of-2 uses `F.pad()`, which may allocate internally.
4. **Dynamic shapes** — Batch sizes and sequence lengths vary across calls, but CUDA graphs require fixed shapes per captured graph.

## Approach F1: Full Graph-Safe Path

Make the entire `set_kv_buffer()` → quantize → pack path work inside CUDA graph capture.

### Pre-allocate workspace buffers

At pool initialization time (`TurboQuantTokenToKVPool.__init__`), pre-allocate all temporary tensors needed by quantize/dequantize:

```python
# In turboquant_pool.py __init__:
max_bs = server_args.max_batch_size  # or a reasonable upper bound
H = num_kv_heads
D = head_dim
padded_D = next_power_of_2(D)

# Workspace for quantization (per layer)
self.workspace = {
    'rotated': torch.empty(max_bs, H, padded_D, dtype=torch.bfloat16, device=device),
    'norms': torch.empty(max_bs, H, dtype=torch.float32, device=device),
    'normalized': torch.empty(max_bs, H, padded_D, dtype=torch.bfloat16, device=device),
    'indices': torch.empty(max_bs, H, D, dtype=torch.int64, device=device),
    'packed': torch.empty(max_bs, H, packed_dim, dtype=torch.uint8, device=device),
}
```

### Replace dynamic allocations

In `quant_ops.py`, change all functions to accept workspace tensors:

```python
# Before (not graph-safe):
def turboquant_mse_quantize(x, hadamard, codebook):
    norms = torch.zeros(B, H, ...)
    ...

# After (graph-safe):
def turboquant_mse_quantize(x, hadamard, codebook, workspace=None):
    if workspace is not None:
        norms = workspace['norms'][:B, :H]
        norms.zero_()  # in-place zero is graph-safe
    else:
        norms = torch.zeros(B, H, ...)
    ...
```

### FWHT graph safety

If using the Triton FWHT from Phase E, Triton kernels are natively CUDA-graph-safe (they're just kernel launches with fixed grid/block dims). If still using `torch.compile`, test whether the compiled kernel works inside `torch.cuda.CUDAGraph()` capture.

### Padding

Replace `F.pad()` with writes to a pre-allocated padded buffer:
```python
# Before:
x_padded = F.pad(x, (0, padded_dim - dim))

# After:
x_padded = workspace['rotated'][:B, :H, :padded_D]
x_padded[:, :, :D] = x
x_padded[:, :, D:] = 0
```

### Shape handling

CUDA graphs capture specific tensor shapes. SGLang typically captures multiple graphs for different batch sizes (powers of 2). The workspace approach handles this because:
- Workspaces are allocated at max size
- Slicing (`workspace[:B]`) doesn't allocate — it's a view
- The kernel launch grid varies by batch size, which SGLang's graph infrastructure already handles

## Approach F2: Dtype-Level Integration (Simpler)

Move quantize/dequantize OUTSIDE the CUDA graph capture region, following the pattern from SGLang PR #21419.

### Architecture

```
[Pre-graph] Quantize fresh KV → write to packed pool
[CUDA Graph] Attention kernel reads BF16 workspace ← dequantized from pool
[Post-graph] No cleanup needed
```

### Implementation

1. **Add pre/post hooks to the model runner:**
   - Before CUDA graph replay: dequantize needed KV from packed pool into BF16 workspace buffers
   - The captured graph sees standard BF16 KV buffers
   - After graph replay: quantize any newly generated KV back into the packed pool

2. **Pool changes:**
   - `TurboQuantTokenToKVPool` maintains both packed storage AND BF16 workspace buffers
   - The BF16 workspace is sized for the max tokens needed in one forward pass (not the full pool)
   - `get_key_buffer()` / `get_value_buffer()` return views into the BF16 workspace

### Tradeoffs

| Aspect | F1 (Full graph-safe) | F2 (Dtype-level) |
|--------|----------------------|-------------------|
| Complexity | Higher — must make every op graph-safe | Lower — isolate quant/dequant outside graph |
| Performance | Better — no extra BF16 materialization | Slower — extra dequant + BF16 workspace memory |
| Memory | Minimal workspace | BF16 workspace for active tokens |
| Fused kernels | Compatible with `--attention-backend turboquant` | Fused kernels NOT inside graph (standard backend only) |
| Risk | Higher — any missed dynamic op breaks graph capture | Lower — well-understood pattern |

### Recommendation

Start with F2 for correctness, then optimize to F1 once Phase E (Triton FWHT) is done. F2 gets CUDA graphs working immediately with the default (non-fused) attention path. F1 is needed later for maximum performance with fused kernels.

## Files to Modify

### `python/sglang/srt/server_args.py`
- Remove the auto-disable: `if server_args.kv_cache_quantization == "turboquant": server_args.disable_cuda_graph = True`
- Gate on whether the graph-safe path is available

### `python/sglang/srt/mem_cache/turboquant_pool.py`
- **F1:** Add workspace pre-allocation in `__init__`, pass workspace through to quant_ops
- **F2:** Add BF16 workspace buffers, implement `materialize_for_graph()` / `commit_from_graph()` methods

### `python/sglang/srt/layers/quantization/turboquant/quant_ops.py`
- **F1:** All quantize/dequantize functions accept optional workspace dict, use in-place ops when workspace provided
- **F2:** No changes needed (dequant happens outside graph)

### `python/sglang/srt/layers/quantization/turboquant/rotation.py`
- **F1:** Ensure FWHT works inside graph capture (Triton kernel from Phase E, or pre-allocated padded buffers)
- **F2:** No changes needed

## Verification

```bash
conda activate turboquant

# Test 1: Server starts without --disable-cuda-graph
python -m sglang.launch_server \
    --model /home/keko/AI/image-prep/models/uncensored-9b-bf16-hf \
    --kv-cache-quantization turboquant \
    --turboquant-bits 3 \
    --port 30000 \
    --context-length 4096

# Test 2: Run inference (both decode and prefill)
curl http://localhost:30000/v1/completions \
    -d '{"model": "default", "prompt": "Hello", "max_tokens": 50}'

# Test 3: Multiple concurrent requests (exercises CUDA graph replay)
for i in $(seq 1 10); do
    curl -s http://localhost:30000/v1/completions \
        -d '{"model": "default", "prompt": "Count to 5:", "max_tokens": 20}' &
done
wait

# Test 4: Kernel tests still pass
python -m pytest test/test_turboquant_kernel.py -v
```

## Notes

- SGLang captures CUDA graphs lazily on first forward pass of each batch-size bucket. The first few requests will be slower (graph capture), then subsequent requests replay the graph.
- The `--disable-cuda-graph` flag should remain available as a fallback.
- Profile with `nsys` to verify graph capture/replay is actually happening: look for `cudaGraphLaunch` calls in the trace.
