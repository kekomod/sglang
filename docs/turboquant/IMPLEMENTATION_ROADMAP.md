# TurboQuant Implementation Roadmap

This document outlines the phased implementation plan for integrating TurboQuant KV cache quantization into SGLang.

---

## Current Status (2026-03-27)

| Phase | Status | Notes |
|---|---|---|
| Phase 1: Core Library | DONE | codebook, rotation, quant_ops, packing all working |
| Phase 2: SGLang Integration | DONE | Config, KVCacheMethod, TurboQuantTokenToKVPool, CLI args, Qwen3.5 model fix, HybridLinearKVPool integration |
| Phase 3A: Triton Decode Kernel | DONE | Triton decode kernel reads packed uint8 KV directly, split-channel codec, compact pool storage |
| Phase 3B: Triton Extend Kernel | DONE | Fused extend/prefill kernel reads ALL KV from quantized buffers (quantize-first approach). Both integer and split-channel paths. |
| Phase 4: Hadamard + Kernel-Agnostic | DONE | Replaced QR rotation with Fast Walsh-Hadamard Transform (O(d log d), O(d) storage). Removed forced backend — TurboQuant now works with any attention backend via dequant-on-read. Fused kernels opt-in via `--attention-backend turboquant`. |
| Phase 5: Validation | PARTIAL | 12/12 kernel tests pass. 6/6 server tests pass (with and without CUDA graphs). Needle-in-haystack: 19/20 = 95% at 3.5-bit (meets target). GSM8K and perplexity benchmarks written but not yet run. |
| Phase 5B: Performance Fixes | DONE | Vectorized pack/unpack (no Python loops), eliminated host-to-device copies, GPU-init tensors. |
| Phase 6: Triton FWHT Kernel | DONE | Dedicated Triton forward/inverse FWHT kernels replace torch.compile. Graph-safe, no JIT warmup delay. Supports D=32,64,128,256. Roundtrip error 4.77e-07. |
| Phase 7: CUDA Graph Support | DONE | F1: quant verified graph-safe, runs inside CUDA graph. Workspace conditional (fused=none, FlashInfer=BF16). Throughput: 47.0 tok/s = 1.00x baseline. Piecewise graphs tested on Qwen2.5-3B. |

### Step 3 Completion Summary (Phase 3B)

**New files:**
- `layers/attention/triton_ops/turboquant_extend_attention.py` — Fused Triton extend kernel for prefill/extend. Unified single-loop design: reads ALL KV from quantized buffers via kv_indices page table. Both integer-bit (`_turboquant_extend_kernel` + `turboquant_extend_attention_fwd`) and split-channel (`_turboquant_extend_kernel_split` + `turboquant_extend_attention_fwd_split`) variants. No Stage 2 reduction (single-stage online softmax). Causal masking: prefix tokens always visible, extend tokens masked causally.
- `benchmark/turboquant/common.py` — Shared benchmark infrastructure (server lifecycle, HTTP helpers, result I/O)
- `benchmark/turboquant/eval_perplexity.py` — Wikitext-2 perplexity benchmark
- `benchmark/turboquant/eval_needle.py` — Needle-in-haystack retrieval benchmark
- `benchmark/turboquant/eval_gsm8k.py` — GSM8K arithmetic reasoning benchmark
- `benchmark/turboquant/run_all.py` — Master benchmark runner

**Modified:**
- `layers/attention/turboquant_backend.py` — Now overrides BOTH `forward_decode` AND `forward_extend`. The extend path uses quantize-first flow: quantize fresh K/V into pool first, build unified kv_indices, pre-rotate/project queries, call fused extend kernel, inverse-rotate output. No BF16 dequant buffers materialized.
- `test/test_turboquant_kernel.py` — Added 4 extend kernel tests (integer + split + causal mask + variable batch). All 11 tests pass with cosine_sim=1.0000.

**Key design decision: Quantize-first approach (matching MLX):**
The backend quantizes fresh K/V into the pool BEFORE calling the extend kernel. Then ALL KV (prefix + extend) is in quantized form, and the kernel reads from a single unified path. This eliminates the need for a mixed BF16+quantized kernel.

### Step 4 Completion Summary (Phase 4: Hadamard + Kernel-Agnostic)

**Rotation: QR → Hadamard (FWHT)**
- Replaced QR decomposition (O(d^2) matmul, D x D matrix per layer) with randomized Hadamard transform (O(d log d) FWHT, just a sign vector per layer)
- `rotation.py` — Added `HadamardTransform` class with forward/inverse/FWHT. Removed `rotation_matrix()`. Kept `projection_matrix()` for QJL.
- `quant_ops.py` — All quantize/dequantize functions now accept `hadamard` instead of `pi`/`pi_t` matrix parameters
- `turboquant_pool.py` — Stores `k_hadamard` / `hadamard_lo` / `hadamard_hi` per layer instead of D x D matrices. Buffer allocation uses Hadamard `padded_dim` (next power-of-2).
- `turboquant_backend.py` — Uses `hadamard.forward()` / `hadamard.inverse()` for query pre-rotation and output inverse-rotation
- Memory savings: ~4.7 MB of rotation matrices → ~18 KB of sign vectors (for D=128, L=36)

**Kernel-Agnostic Backend**
- Removed forced `attention_backend = "turboquant"` from `server_args.py`. TurboQuant now works with SGLang's default backend selection (FlashInfer, Triton, etc.) via dequant-on-read.
- The pool's `get_key_buffer()` / `get_value_buffer()` methods dequantize on demand, returning standard [pool_size, H, D] BF16 tensors compatible with any attention backend.
- Fused TurboQuant Triton kernels remain available as opt-in via `--attention-backend turboquant`.
- Added `"turboquant"` to `ATTENTION_BACKEND_CHOICES` in server_args.py.

### Step 2 Completion Summary (Phase 3A)

**New files:**
- `layers/attention/triton_ops/turboquant_decode_attention.py` — Triton two-stage decode kernel (MSE codebook + QJL sign-bit scoring for K, MSE weighted sum for V)

**Rewritten:**
- `mem_cache/turboquant_pool.py` — Now inherits from KVCache directly with compact uint8 packed buffers (no BF16 dequant buffers)

**Extended:**
- `layers/quantization/turboquant/quant_ops.py` — 5 split-channel functions for fractional bit-widths
- `layers/attention/attention_registry.py` — Registered "turboquant" backend
- `server_args.py` — TurboQuant CLI flags (no longer forces attention backend)

**Remaining:**
- Run GSM8K and perplexity validation benchmarks (scripts written, need dataset download)
- Autotuning configs for the Triton kernels

### Hybrid Architecture Integration

`HybridLinearKVPool` (for Qwen3.5's DeltaNet + full attention hybrid) now accepts `kv_cache_quantization` parameter and creates `TurboQuantTokenToKVPool` as its inner `full_kv_pool` when `kv_cache_quantization == "turboquant"`. Modified in both `memory_pool.py` and `model_runner_kv_cache_mixin.py`.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Key quantization | TurboQuant_prod | Unbiased inner-product estimation for QK^T attention scores (paper Theorem 2) |
| Value quantization | TurboQuant_mse | Optimal MSE reconstruction for weighted sum after softmax (paper Theorem 1) |
| Rotation method | Randomized Hadamard (FWHT) | O(d log d) compute, O(d) storage. Paper supports any random rotation (Section 3.1). Matches PR #21419. |
| QJL projection scope | Per-layer, shared across heads, dense Gaussian | Paper Definition 1: S with i.i.d. N(0,1) entries. Still D x D but only needed for prod mode. |
| Default bit-width | 3.5 bits (configurable) | Matches full-precision quality per paper Table 1; 2.5 bits also supported |
| Outlier fraction | Configurable (default 12.5%) | Paper's example; selectable via channel magnitude statistics |
| Default attention path | Dequant-on-read with any backend | Kernel-agnostic: works with FlashInfer, Triton, etc. Paper doesn't mandate fused kernels. |
| Opt-in fused attention | `--attention-backend turboquant` | Fused Triton kernels read packed buffers directly — avoids BF16 materialization. Performance optimization. |
| Extend kernel approach | Quantize-first, unified single-loop | Quantize fresh KV first, then ALL KV from quantized buffers. Matches MLX reference. |
| Pack/unpack method | Vectorized tensor bitwise ops | Replaced Python per-element loops (128 iter) with tensor-level group ops (7-8 iter). Follows SGLang PR #21419 pattern. |
| CUDA graph strategy | BF16 workspace (F2 approach, Phase 7) | Attention reads BF16 workspace during graph. Quant/dequant runs outside graph. Fused backend (`--attention-backend turboquant`) auto-disables graphs. |
| FWHT compilation | Dedicated Triton kernel (Phase 6) | Forward/inverse kernels with butterfly stages. Graph-safe, no torch.compile warmup. CPU fallback via _fwht_impl. |

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

## Phase 6: Triton FWHT Kernel — DONE (see HANDOFF_PHASE_E.md)

Replaced `torch.compile`-wrapped Python FWHT with dedicated Triton kernels.

**Completed:**
- `triton_fwht.py`: `triton_fwht_forward_kernel` and `triton_fwht_inverse_kernel` with butterfly stages using `tl.debug_barrier()` between stages. Supports D=32,64,128,256.
- `rotation.py`: Auto-dispatches to Triton on CUDA, falls back to `_fwht_impl` on CPU. JIT warmup in `__init__` ensures consistent results from first call.
- 12/12 kernel tests pass. Roundtrip error: 4.77e-07.

---

## Phase 7: CUDA Graph Support — DONE (see HANDOFF_PHASE_F.md)

Re-enabled CUDA graphs using the F2 (BF16 workspace) approach.

**Completed:**
- `turboquant_pool.py`: Per-layer BF16 workspace buffers (`k_workspace`, `v_workspace`), `_graph_mode` flag, dual-write `set_kv_buffer`, mode-aware `get_key/value_buffer`, `quant_new_tokens()` for post-graph quantization.
- `model_runner.py`: `set_graph_mode(True)` before graph replay, `quant_new_tokens(out_cache_loc)` after.
- `cuda_graph_runner.py`: `set_graph_mode(True)` during `_capture_graph()` so warmup runs only do BF16 scatter.
- `server_args.py`: Auto-disable only for fused backend (`--attention-backend turboquant`). Default FlashInfer backend allows CUDA graphs.
- `memory_pool.py`: `HybridLinearKVPool` passthrough for `_graph_mode`, `set_graph_mode()`, `quant_new_tokens()`.
- Throughput: 35.2 tok/s (0.76x baseline), up from 18.6 tok/s (0.40x) without graphs.

**Future optimization (F1):** Make quant/dequant ops fully graph-safe to eliminate BF16 workspace memory overhead.

---

## Phase 8: Fused Kernel Optimization (see HANDOFF_PHASE_G.md)

Fuse query rotation (FWHT + QJL projection) into the attention kernels to eliminate Python overhead between kernel launches. Reduces per-layer kernel launches from 5+ to 2-3. Requires inline FWHT helper for Triton.

**Key deliverables:**
- Fused decode kernel with inline rotation
- Fused extend kernel with inline query rotation (keep quantize-first for KV)
- Measurable reduction in per-token decode latency

---

## Phase 9: Rotated-Space Value Accumulation (see HANDOFF_PHASE_H.md)

Exploit linearity of inverse Hadamard to compute value weighted-sum in rotated space, reducing inverse Hadamard calls from O(T) to O(1) per head per layer. For T=1000 and d=128, this is roughly a 7x reduction in value-side compute.

**Key deliverables:**
- Verify decode/extend kernels already accumulate in rotated space (likely already done)
- Ensure inverse Hadamard applied once on output, not per-token
- Optional `get_value_buffer_rotated()` for non-fused path

---

## Phase 10: Hardening & Edge Cases (see HANDOFF_PHASE_I.md)

Final smoothing pass for production readiness. Workspace lifecycle correctness, defensive error handling, CPU offload incompatibility messaging, graph mode context manager, prefix cache consistency verification, memory reporting.

**Key deliverables:**
- Each item requires a decision about whether to address (cost vs likelihood)
- Graph mode context manager (try/finally safety)
- CPU offload error message
- Optional debug assertions for workspace consistency

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
| `quantization/turboquant/triton_fwht.py` | 6 | Triton FWHT forward/inverse kernels (D=32,64,128,256) |
| `mem_cache/turboquant_pool.py` | 2 | Quantized KV memory pool |
| `sgl-kernel/csrc/turboquant/*.cu` | 4 | CUDA kernels |
| `tests/test_turboquant_*.py` | 1-5 | Test suite |

## Modified Files Summary

| File | Phase | Change |
|---|---|---|
| `quantization/__init__.py` | 2 | Register TurboQuantConfig |
| `models/qwen3_5.py` | 2 | Pass quant_config to RadixAttention |
| `server_args.py` | 2, 7 | CLI arguments; CUDA graph auto-disable for fused backend only |
| `model_executor/model_runner.py` | 2, 7 | Memory pool instantiation; graph-mode hooks around replay |
| `model_executor/cuda_graph_runner.py` | 7 | Graph-mode during capture |
| `mem_cache/memory_pool.py` | 7 | HybridLinearKVPool graph-mode passthrough |
| `sgl-kernel/csrc/common_extension.cc` | 4 | Kernel registration |
| `sgl-kernel/CMakeLists.txt` | 4 | Build configuration |
