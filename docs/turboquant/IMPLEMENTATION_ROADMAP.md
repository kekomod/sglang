# TurboQuant Implementation Roadmap

This document outlines the phased implementation plan for integrating TurboQuant KV cache quantization into SGLang.

---

## Current Status (2026-03-25)

| Phase | Status | Notes |
|---|---|---|
| Phase 1: Core Library | DONE | codebook, rotation, quant_ops, packing all working |
| Phase 2: SGLang Integration | DONE | Config, KVCacheMethod, TurboQuantTokenToKVPool, CLI args, Qwen3.5 model fix, HybridLinearKVPool integration |
| Phase 3: Triton Kernels | NOT STARTED | |
| Phase 4: CUDA Kernels | NOT STARTED | |
| Phase 5: Validation | IN PROGRESS | Server starts, TurboQuant pool initializes (3.1x savings), but garbled output due to dequant buffer invalidation bug |

### Known Bug: Shared Dequant Buffer

The `TurboQuantTokenToKVPool` uses a single shared dequant buffer pair for all layers. The `_ensure_dequant()` method skips re-dequantization when `_dequant_layer_id == layer_id`, but `set_kv_buffer()` only writes dequantized data for the newly stored `loc` positions. When a different code path calls `get_kv_buffer()` for the same layer, the buffer contains stale/zero data at positions not recently written. This causes garbled attention output.

**Fix options:**
1. Always dequant the full buffer on `get_kv_buffer` (simple but slow — O(pool_size * head_dim) per layer per forward)
2. Track dirty positions per layer and selectively dequant (complex bookkeeping)
3. Remove the shared buffer optimization and use per-layer dequant buffers (uses more memory but simpler)

### Hybrid Architecture Integration

`HybridLinearKVPool` (for Qwen3.5's DeltaNet + full attention hybrid) now accepts `kv_cache_quantization` parameter and creates `TurboQuantTokenToKVPool` as its inner `full_kv_pool` when `kv_cache_quantization == "turboquant"`. Modified in both `memory_pool.py` and `model_runner_kv_cache_mixin.py`.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Key quantization | TurboQuant_prod | Unbiased inner-product estimation for QK^T attention scores (paper Theorem 2) |
| Value quantization | TurboQuant_mse | Optimal MSE reconstruction for weighted sum after softmax (paper Theorem 1) |
| Rotation matrix scope | Per-layer, shared across heads | Heads have different learned projections; sharing is theoretically sound and memory-efficient (256KB/layer) |
| QJL projection scope | Per-layer, shared across heads | Same rationale as rotation matrix |
| Default bit-width | 3.5 bits (configurable) | Matches full-precision quality per paper Table 1; 2.5 bits also supported |
| Outlier fraction | Configurable (default 12.5%) | Paper's example; selectable via channel magnitude statistics |
| Phase 1 attention | Dequant-then-FlashAttention | Correct, simple, leverages existing backends with no kernel work |
| Phase 2 attention | Fused Triton decode kernel | Avoids materializing full dequantized KV; key performance optimization |
| Rotation matrix init | QR + sign(diag(R)) correction | Haar-uniform orthogonal matrix per paper and confirmed by MLX implementation |

---

## Phase 1: Core Library + Correctness

### 1.1 Codebook Computation

**New file:** `python/sglang/srt/layers/quantization/turboquant/codebook.py`

- Implement Beta distribution PDF: `f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1-x^2)^{(d-3)/2}`
- Max-Lloyd iterative algorithm (100 iterations, convergence 1e-6)
- Fine grid integration (32768 points on [-1+eps, 1+eps])
- Analytical codebooks for b=1,2 as fast path
- Cache codebooks by (dimension, bits) — one-time computation
- Unit tests: verify centroids match paper values, verify distortion bounds

### 1.2 Rotation Matrix Generation

**New file:** `python/sglang/srt/layers/quantization/turboquant/rotation.py`

- QR decomposition on Gaussian random matrix
- Sign correction: `Q *= sign(diag(R))` for Haar-uniform distribution
- Deterministic seeding per layer
- Pre-compute and store both Pi and Pi^T (for dequantization)
- Unit tests: verify orthogonality, verify rotation preserves norms

### 1.3 Quantization/Dequantization Ops

**New file:** `python/sglang/srt/layers/quantization/turboquant/quant_ops.py`

Python-level orchestration using PyTorch ops:
- `turboquant_mse_quantize(x, pi, codebook)` → packed_indices, norms
- `turboquant_mse_dequantize(packed_indices, norms, pi_t, codebook)` → x_reconstructed
- `turboquant_prod_quantize(x, pi, s, codebook)` → packed_indices, qjl_signs, norms, residual_norms
- `turboquant_prod_dequantize(...)` → x_reconstructed
- `turboquant_prod_score(q, state, pi, s, codebook)` → attention scores (fused inner product)
- Use `torch.matmul` for rotation (batch GEMM for prefill, GEMV for decode)

### 1.4 Bit-Packing Utilities

**New file:** `python/sglang/srt/layers/quantization/turboquant/packing.py`

- Pack b-bit indices into uint32 words
- Unpack uint32 words to b-bit indices
- Pack 1-bit QJL signs into uint32 bitmasks
- Support arbitrary bit-widths (2, 3, 4)
- Efficient vectorized implementations using torch bitwise ops

### 1.5 Outlier Channel Handling

**New file:** `python/sglang/srt/layers/quantization/turboquant/outlier.py`

- Split channels by magnitude statistics (mean absolute activation)
- Create two independent codecs (different bit-widths)
- Manage channel permutation indices for split/merge
- Support for fractional bit-widths (e.g., 2.5, 3.5)

---

## Phase 2: SGLang Integration

### 2.1 TurboQuantConfig

**New file:** `python/sglang/srt/layers/quantization/turboquant/config.py`

```python
class TurboQuantConfig(QuantizationConfig):
    def __init__(self, key_bits=3.5, value_bits=3.5, seed=42, ...)
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RadixAttention):
            return TurboQuantKVCacheMethod(self)
        return None  # No weight quantization
```

### 2.2 Registration

**Modify:** `python/sglang/srt/layers/quantization/__init__.py`
- Add `"turboquant": TurboQuantConfig` to `BASE_QUANTIZATION_METHODS`

### 2.3 TurboQuantTokenToKVPool

**New file:** `python/sglang/srt/mem_cache/turboquant_pool.py`

Custom memory pool that stores quantized KV cache:
- `k_buffer`: packed indices (uint8/uint32) per layer
- `k_norm_buffer`: norms (fp16) per layer
- `k_qjl_buffer`: packed QJL signs (uint8) per layer (for keys)
- `k_residual_norm_buffer`: residual norms (fp16) per layer
- `v_buffer`: packed indices per layer
- `v_norm_buffer`: norms per layer
- Implements `set_kv_buffer()` with quantization on write
- Implements `get_key_buffer()` / `get_value_buffer()` with dequantization on read

### 2.4 Model Fix

**Modify:** `python/sglang/srt/models/qwen3_5.py`
- Pass `quant_config` to `RadixAttention` constructor in `Qwen3_5AttentionDecoderLayer`

### 2.5 Server Configuration

**Modify:** `python/sglang/srt/server_args.py`
- Add `--kv-cache-quantization` or extend `--quantization` to support `turboquant`
- Add sub-options: `--turboquant-bits`, `--turboquant-seed`

### 2.6 End-to-End Test

- Load Qwen3.5-9B with TurboQuant enabled
- Run inference, verify correct output shapes
- Compare perplexity against FP16 baseline

---

## Phase 3: Triton Kernels

### 3.1 Fused Quantization Kernel

**New file:** `python/sglang/srt/layers/quantization/turboquant/triton_kernels.py`

Triton kernel that fuses: normalize → rotate → quantize → pack
- Grid: (num_tokens * num_heads,)
- Each program handles one token, one head
- Use `torch.matmul` for the rotation (d=256 is small enough for cuBLAS)
- Main benefit: fuse the quantize+pack step to avoid intermediate tensors

### 3.2 Fused Decode Attention Kernel

Triton kernel for decode-step attention on quantized KV:
- Input: query, packed K cache, packed V cache, page table
- Operation: project query → score against all keys → softmax → weighted sum of values
- Grid: (batch * num_kv_heads, num_seq_blocks)
- Online softmax across sequence blocks
- Reference: PolarQuant's `kernel4group.py` for Triton structure, MLX's Metal kernels for logic

### 3.3 Fused Score Kernel (for keys)

Triton kernel for computing Q*K scores directly from packed data:
- Load query (already rotated/projected)
- Load packed indices, unpack on-the-fly
- Lookup codebook values in shared memory
- Accumulate dot product
- Add QJL correction term

---

## Phase 4: CUDA Kernels + Optimization

### 4.1 CUDA Quantization Kernel

Port critical Triton kernels to CUDA for maximum performance:
- Reference: QJL's `qjl_quant_kernel.cu` for CUDA patterns
- L2 persistent cache hints for codebook and rotation matrix
- Warp-level reductions for dot products

### 4.2 Register in sgl-kernel

**New files:** `sgl-kernel/csrc/turboquant/*.cu`
**Modify:** `sgl-kernel/csrc/common_extension.cc` — register ops
**Modify:** `sgl-kernel/CMakeLists.txt` — add source files

### 4.3 Performance Optimization

- Profile decode latency vs FP16 baseline
- Optimize memory access patterns for paged KV cache
- Consider code-generated kernels for specific GQA repeat counts (MLX pattern)
- CUDA graph compatibility

---

## Phase 5: Validation + Documentation

### 5.1 Reproduce Paper Results

- Run Llama-3.1-8B-Instruct on LongBench with TurboQuant 2.5-bit and 3.5-bit
- Compare against Table 1 values
- Run needle-in-a-haystack test

### 5.2 Qwen3.5 Evaluation

- Run Qwen3.5-9B, 27B, 35B-A3B with TurboQuant
- Benchmark memory savings and latency
- Compare quality against FP16 baseline

### 5.3 Documentation

- Usage guide with example commands
- Performance benchmarks
- Troubleshooting guide

---

## New Files Summary

| File | Phase | Purpose |
|---|---|---|
| `quantization/turboquant/__init__.py` | 2 | Package init |
| `quantization/turboquant/config.py` | 2 | TurboQuantConfig |
| `quantization/turboquant/kv_cache_method.py` | 2 | KV cache quantization method |
| `quantization/turboquant/codebook.py` | 1 | Codebook computation |
| `quantization/turboquant/rotation.py` | 1 | Rotation matrix management |
| `quantization/turboquant/quant_ops.py` | 1 | Quant/dequant operations |
| `quantization/turboquant/packing.py` | 1 | Bit-packing utilities |
| `quantization/turboquant/outlier.py` | 1 | Outlier channel handling |
| `quantization/turboquant/triton_kernels.py` | 3 | Triton GPU kernels |
| `mem_cache/turboquant_pool.py` | 2 | Quantized KV memory pool |
| `sgl-kernel/csrc/turboquant/*.cu` | 4 | CUDA kernels |
| `tests/test_turboquant_*.py` | 1-5 | Test suite |

## Modified Files Summary

| File | Phase | Change |
|---|---|---|
| `quantization/__init__.py` | 2 | Register TurboQuantConfig |
| `models/qwen3_5.py` | 2 | Pass quant_config to RadixAttention |
| `server_args.py` | 2 | CLI arguments |
| `model_executor/model_runner.py` | 2 | Memory pool instantiation |
| `sgl-kernel/csrc/common_extension.cc` | 4 | Kernel registration |
| `sgl-kernel/CMakeLists.txt` | 4 | Build configuration |
