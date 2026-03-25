# Component References: QJL, PolarQuant, and MLX Implementation

This document maps TurboQuant paper concepts to existing code for reference. **These are reference implementations, NOT to be copied blindly.** Every implementation decision must be validated against the paper (arXiv:2504.19874).

---

## 1. QJL Repository

**Location:** `/home/keko/AI/turboquant/QJL/`
**License:** Apache-2.0
**Paper:** "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead" (AAAI 2025)

### 1.1 CUDA Kernels

| File | Purpose | Paper Mapping |
|---|---|---|
| `qjl_kernel/csrc/qjl_quant_kernel.cu` | Quantize keys with random projection + sign | Algorithm 2 line 7: `qjl = sign(S * r)` |
| `qjl_kernel/csrc/qjl_score_kernel.cu` | Compute attention scores from quantized keys | Fused inner product: `<S*y, qjl>` |
| `qjl_kernel/csrc/qjl_gqa_score_kernel.cu` | GQA-aware score computation | Same as above, handles multiple Q heads per KV head |
| `qjl_kernel/csrc/qjl_quant_values_kernel.cu` | Value quantization | Not directly used (TurboQuant uses MSE codec for values) |
| `qjl_kernel/csrc/quantization.cu` | Batched dequant+matmul | Reference for fused dequant patterns |

**Key implementation details:**
- `EMB_DIM=128` hardcoded — we need 256 for Qwen3.5
- Uses L2 persistent cache hints for projection matrix access
- WARP_SIZE=32, WARPS_PER_BLOCK=32
- Shared memory for query vectors and outlier indices

### 1.2 Python Integration

| File | Purpose | Paper Mapping |
|---|---|---|
| `models/llama2_utils_qjl.py` | `QJLSketch` class | Manages projection matrices, dispatches kernels |
| `models/llama2_qjl.py` | `QJLCache`, `LlamaAttention_QJL` | Cache structure, attention integration |
| `qjl_kernel/qjl_kernel.py` | Python kernel dispatch | Type-based dispatch (fp16/fp32/bf16) |
| `qjl_kernel/new_pack.py` | Triton bit-packing kernels | Reference for efficient bit-packing |
| `qjl_kernel/matmul.py` | Quantized batched matmul | `cuda_quantized_bmm_dynamic()` |

**Key patterns:**
- `QJLSketch.__init__()`: Creates projection matrices via QR decomposition (for scoring) and random Gaussian (for quantization)
- Mixed-precision cache: recent tokens in full precision, older tokens quantized
- Group-based quantization: tokens grouped for amortized overhead

### 1.3 Paper vs QJL Code

| Concept | Paper (TurboQuant) | QJL Code |
|---|---|---|
| Rotation matrix | QR decomposition of Gaussian, Haar-uniform | QR decomposition, used for scoring projections |
| Sign projection | sign(S * r) where S ~ N(0,1) | sign(S * x) — applies to full vector, not residual |
| Scale factor | sqrt(pi/2) / d | Embedded in score computation |
| Bit budget | (b-1) bits MSE + 1 bit QJL | 1 bit QJL only (no MSE component) |

**Critical difference:** QJL uses ONLY 1-bit sign quantization. TurboQuant adds the multi-bit MSE component for much better quality at the cost of more storage.

---

## 2. PolarQuant Repository

**Location:** `/home/keko/AI/turboquant/PolarQuant/`
**Paper:** "PolarQuant: Quantizing KV Cache via Polar Coordinate Decomposition" (AISTATS 2026)

### 2.1 Triton Kernels

| File | Purpose | Paper Mapping |
|---|---|---|
| `models/kernel4group.py` | Fused polar quantization + attention scoring | Reference for Triton kernel structure |
| `models/modeling_llama_polar.py` | Model integration with polar cache | Cache management pattern |

**Key implementation details:**
- Triton kernel structure: grid=(B*Nk, Nb), one program per (batch*head, block)
- Polar encoding: `indices = (rho << tbits) + theta`
- Per-group min/max scaling for angle and radius
- Directly computes `attention = sum(query * [cos(phi), sin(phi)] * radius)`

### 2.2 Paper vs PolarQuant Code

| Concept | Paper (TurboQuant) | PolarQuant Code |
|---|---|---|
| Coordinate system | Rotated Cartesian → nearest centroid | Polar (angle + radius) |
| Codebook | Max-Lloyd optimal for Beta distribution | Uniform angle/radius grid |
| Quantization | Per-coordinate scalar quantization | Per-pair polar encoding |
| Kernel | Not specified in paper | Triton, fused with attention |

**Critical difference:** PolarQuant uses a fundamentally different quantization approach (polar coordinates). TurboQuant uses scalar quantization in a rotated basis. The PolarQuant kernel structure is useful as a Triton reference, but the math is different.

---

## 3. MLX-VLM Implementation (PR #858)

**PR:** [Blaizzy/mlx-vlm#858](https://github.com/Blaizzy/mlx-vlm/pull/858)
**Status:** Draft PR, open, no reviews
**Author's note:** "This implementation is far from optimal... I don't see the prefill and decode performance matching up to the claimed 8x speedup."
**Tested on:** Qwen3.5-35B-A3B (bf16) on M3 Max 96GB

### 3.1 File Structure

| File | Lines | Purpose |
|---|---|---|
| `mlx_vlm/turboquant.py` | 2121 | Complete TurboQuant implementation |
| `mlx_vlm/tests/test_turboquant.py` | 319 | Unit tests |
| `mlx_vlm/generate.py` | +61/-6 | Hook into generation pipeline |
| `mlx_vlm/models/base.py` | +43/-1 | Wrap attention dispatch |
| `mlx_vlm/server.py` | +18/-3 | CLI args for kv_quant_scheme |

### 3.2 Core Components

**Codebook computation (`_codebook`):**
- Beta distribution PDF on fine grid (32768 points)
- Max-Lloyd iteration (100 iterations, convergence threshold 1e-6)
- LRU cached by (dim, bits)
- Matches paper exactly

**Rotation matrix (`_rotation_matrix`):**
- QR decomposition on Gaussian random matrix
- **Sign correction:** `q *= np.sign(np.diag(r))` for Haar-uniform distribution
- Deterministic seed: `seed + dim * 7919`
- LRU cached by (dim, seed)

**QJL projection (`_projection_matrix`):**
- Dense Gaussian random matrix (NOT orthogonalized)
- Different seed: `seed + dim * 2971 + 17`

**MSE codec (`_TurboQuantMSECodec`):**
- Used for **values**
- Implements Algorithm 1 exactly
- Bit-packs indices into uint32

**Prod codec (`_TurboQuantProdCodec`):**
- Used for **keys**
- Wraps MSE codec at (bits-1) + 1-bit QJL
- Stores: mse_indices, qjl_signs, norms, residual_norms
- Scale factor: `sqrt(pi/2) / dim`
- Implements Algorithm 2 exactly

**Split codec (`_SplitCodec`):**
- For fractional bit-widths (e.g., 3.5)
- Splits channels by mean absolute activation magnitude
- High-importance channels get ceil(bits), rest get floor(bits)
- **Not in the paper** — engineering extension

### 3.3 Metal Kernels (9 total)

| Kernel | Purpose |
|---|---|
| `_mse_score_kernel` | Q*codebook[indices] dot product from packed bits |
| `_qjl_score_kernel` | Q*signs dot product from packed sign bits |
| `_prod_score_kernel` | Fused MSE + QJL scoring |
| `_prod_score_multi_kernel` | Multi-GQA-repeat variant |
| `_prod_score_repeat_kernel(R)` | Code-generated unrolled variant per repeat count |
| `_mse_weighted_rot_kernel` | Weighted sum in rotated space |
| `_mse_weighted_rot_multi_kernel` | Multi-repeat variant |
| `_mse_weighted_rot_repeat_kernel(R)` | Code-generated unrolled variant |
| `_mse_scores_weighted_rot_repeat_kernel(R)` | Fused softmax + weighted sum |

### 3.4 Attention Integration

**Decode (L=1):** Operates on quantized data directly:
1. `q_rot = q @ Pi^T`, `q_proj = q @ Projection^T`
2. Score against all cached keys using Metal kernels
3. Softmax
4. Weighted sum of values using Metal kernels

**Prefill (L>1):** Falls back to dequantization:
1. `dequantized_keys, dequantized_values = cache.dequantize()`
2. Standard `scaled_dot_product_attention`

**Chunked attention:** Online softmax (log-sum-exp streaming) for long sequences.

### 3.5 Confirmed Faithful to Paper

- Beta-distributed codebook via Max-Lloyd: **correct**
- Haar-uniform rotation via QR + sign(diag(R)): **correct**
- (b-1) bits MSE + 1 bit QJL for Prod: **correct**
- Scale factor sqrt(pi/2)/d: **correct**
- Keys use Prod, Values use MSE: **correct**
- Dense Gaussian S matrix for QJL: **correct**

### 3.6 MLX vs SGLang Differences

| Aspect | MLX | SGLang |
|---|---|---|
| GPU framework | Metal (Apple Silicon) | CUDA/Triton (NVIDIA) |
| KV cache | Simple dynamic cache object | Paged KV cache with RadixAttention prefix caching |
| Attention backends | Single `scaled_dot_product_attention` | Multiple backends (FlashInfer, FlashAttention, Triton) |
| Memory management | Dynamic allocation with doubling | Fixed-size pool with page-based allocation |
| Tensor parallelism | Not applicable (single device) | Multi-GPU TP support |
| Cache eviction | Not handled | RadixCache with LRU eviction |
| Compilation | `mx.compile` graph compilation | CUDA graphs, Triton JIT |

**Key implication:** SGLang's paged memory pool requires that quantized data be stored in fixed-size slots that can be independently addressed. The MLX approach of growing arrays doesn't directly translate — we need a pool-based design.
