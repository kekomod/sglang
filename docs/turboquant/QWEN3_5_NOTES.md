# Qwen3.5 Architecture Notes for TurboQuant

## Overview

Qwen3.5 is a **hybrid architecture** that interleaves two attention mechanisms. TurboQuant KV cache quantization only applies to the full attention layers.

All Qwen3.5 models are natively multimodal (image-text-to-text) using early fusion.

---

## 1. Hybrid Layer Architecture

Qwen3.5 alternates between two layer types in a 3:1 pattern:

```
["linear_attention", "linear_attention", "linear_attention", "full_attention",
 "linear_attention", "linear_attention", "linear_attention", "full_attention",
 ...]
```

Controlled by config field `full_attention_interval: 4`.

### Gated DeltaNet Layers (75% — Linear Attention)

- **NO KV cache** — uses internal recurrent state (delta-rule based)
- Implemented in `Qwen3_5GatedDeltaNet` (qwen3_5.py line ~107)
- Uses `RadixLinearAttention` (NOT `RadixAttention`)
- State parameters: `A_log`, `dt_bias`, conv1d weights
- Has separate head counts: `linear_num_key_heads` and `linear_num_value_heads`
- Key/Value head dim: 128

### Full Attention Layers (25% — Standard Attention)

- **USES KV cache** via `RadixAttention`
- Implemented in `Qwen3_5AttentionDecoderLayer` (qwen3_5.py line ~620)
- Standard GQA with RoPE
- **This is where TurboQuant applies**

---

## 2. Model Parameters

| Parameter | 9B | 27B | 35B-A3B |
|---|---|---|---|
| Architecture class | `qwen3_5` (dense) | `qwen3_5` (dense) | `qwen3_5_moe` (MoE) |
| Total parameters | 9B | 27B | 35B total, 3B active |
| Hidden size | 4096 | 5120 | 2048 |
| Total layers | 32 | 64 | 40 |
| **Full attention layers** | **8** | **16** | **10** |
| DeltaNet layers | 24 | 48 | 30 |
| Vocab size | 248,320 | 248,320 | 248,320 |
| Native context | 262,144 | 262,144 | 262,144 |
| Dtype | bfloat16 | bfloat16 | bfloat16 |

### Full Attention Parameters (TurboQuant-relevant)

| Parameter | 9B | 27B | 35B-A3B |
|---|---|---|---|
| **head_dim** | **256** | **256** | **256** |
| Q heads | 16 | 24 | 16 |
| KV heads | 4 | 4 | 2 |
| **GQA ratio** | **4:1** | **6:1** | **8:1** |
| partial_rotary_factor | 0.25 | 0.25 | 0.25 |
| rope_theta | 10,000,000 | 10,000,000 | 10,000,000 |

### DeltaNet Parameters (NOT TurboQuant-relevant)

| Parameter | 9B | 27B | 35B-A3B |
|---|---|---|---|
| K/V head dim | 128 | 128 | 128 |
| V heads | 32 | 48 | 32 |
| Q/K heads | 16 | 16 | 16 |
| conv_kernel_dim | 4 | 4 | 4 |

---

## 3. MoE Specifics (35B-A3B)

| Parameter | Value |
|---|---|
| Total experts | 256 |
| Routed experts per token | 8 |
| Shared experts (always active) | 1 |
| Total active per token | 9 (8 + 1) |
| Expert intermediate dim | 512 |

Every layer in 35B-A3B uses MoE (including both DeltaNet and full attention layers). MoE is orthogonal to KV cache quantization — the FFN/MoE block runs after the attention block.

---

## 4. TurboQuant Implications

### 4.1 Only Full Attention Layers Need Quantization

Since DeltaNet layers have no KV cache, TurboQuant only affects:
- 9B: 8 of 32 layers (25%)
- 27B: 16 of 64 layers (25%)
- 35B-A3B: 10 of 40 layers (25%)

SGLang handles this via `HybridTokenToKVPool`, which maintains separate pools for full-attention and linear-attention layers. The `TurboQuantTokenToKVPool` only needs to handle full attention layers.

### 4.2 head_dim = 256

Qwen3.5 uses head_dim=256 for full attention, larger than typical (128). This affects:

**Rotation matrix:** Pi is 256x256 = 65,536 elements = 256KB per layer in FP32.
- Memory cost: 8 layers (9B) * 256KB = 2MB total — negligible
- Matrix-vector multiply: 256*256 = 65,536 FMA ops per token per head — small relative to attention

**Codebook:** Centroids depend on d=256. The Beta distribution at d=256 is well-concentrated around 0, so centroids cluster near zero. Must compute for d=256 specifically, not reuse d=128 codebooks.

**Compression ratio:** For 2.5-bit quantization:
- Unquantized: 256 dims * 2 bytes (FP16) = 512 bytes per head per token
- Quantized: ~80 bytes (indices + norms) = 6.4x compression

### 4.3 GQA Considerations

With high GQA ratios (up to 8:1 for 35B-A3B), each KV head serves multiple Q heads. TurboQuant quantizes per KV head. During attention:
- For decode: project query once per KV head, compute score against all cached keys for that KV head
- For GQA: multiple Q heads share the same quantized KV head — the quantization overhead is amortized

### 4.4 Partial Rotary Embeddings

Qwen3.5 applies RoPE to only 25% of the head dimension (64 of 256 dims). The remaining 192 dims are position-independent. This does NOT affect TurboQuant — we quantize the full post-RoPE K/V vectors. RoPE is applied before KV cache storage.

### 4.5 Q/K Normalization

Qwen3.5 applies RMSNorm to Q and K projections before RoPE (qwen3_5.py line ~680):
```python
q, k = self._apply_qk_norm(q, k)
```
This normalizes the magnitude of K vectors before they enter the cache. This is actually beneficial for TurboQuant since the vectors will have more uniform norms.

---

## 5. Required Code Changes in Qwen3.5 Model

### 5.1 Pass quant_config to RadixAttention

**File:** `python/sglang/srt/models/qwen3_5.py`
**Location:** `Qwen3_5AttentionDecoderLayer.__init__()` (~line 704)

**Current code:**
```python
self.attn = RadixAttention(
    self.num_heads,
    self.head_dim,
    self.scaling,
    num_kv_heads=self.num_kv_heads,
    layer_id=layer_id,
    prefix=f"{prefix}.attn",
)
```

**Required change:** Add `quant_config=quant_config` parameter:
```python
self.attn = RadixAttention(
    self.num_heads,
    self.head_dim,
    self.scaling,
    num_kv_heads=self.num_kv_heads,
    layer_id=layer_id,
    quant_config=quant_config,
    prefix=f"{prefix}.attn",
)
```

This follows the pattern used by other models (e.g., `llama.py`).

### 5.2 DeltaNet Layers: No Changes Needed

`Qwen3_5GatedDeltaNet` uses `RadixLinearAttention`, not `RadixAttention`. Since `TurboQuantConfig.get_quant_method()` will only match `RadixAttention` instances, DeltaNet layers are automatically excluded.

---

## 6. VRAM Estimates with TurboQuant

KV cache memory per token (full attention layers only, per head):

| Precision | Bytes per K+V per head | Total for 9B (4 KV heads, 8 layers) |
|---|---|---|
| FP16 | 1024 (256*2*2) | 32,768 bytes |
| FP8 | 512 (256*1*2) | 16,384 bytes |
| TurboQuant 3.5-bit | ~224 bytes | ~7,168 bytes |
| TurboQuant 2.5-bit | ~160 bytes | ~5,120 bytes |

For 100K tokens on Qwen3.5-9B:
- FP16: 100K * 32,768 = 3.1 GB
- TurboQuant 2.5-bit: 100K * 5,120 = 488 MB (~6.4x reduction)
