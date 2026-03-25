# SGLang Architecture: TurboQuant Integration Points

This document maps the SGLang codebase components relevant to TurboQuant KV cache quantization integration. All paths are relative to the SGLang repository root.

---

## 1. Quantization Registration System

**File:** `python/sglang/srt/layers/quantization/__init__.py`

The `BASE_QUANTIZATION_METHODS` dictionary maps string identifiers to `QuantizationConfig` subclasses:

```python
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "blockwise_int8": BlockInt8Config,
    "w8a8_int8": W8A8Int8Config,
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "gguf": GGUFConfig,
    # ... 20+ methods total
}
```

**To add TurboQuant:** Add `"turboquant": TurboQuantConfig` to this dictionary.

Lookup function: `get_quantization_config(quantization: str) -> Type[QuantizationConfig]`

---

## 2. Base Classes

**File:** `python/sglang/srt/layers/quantization/base_config.py`

### Class Hierarchy

```
QuantizeMethodBase (ABC)
  create_weights(layer, ...)
  apply(layer, ...) -> Tensor     [abstract]
  process_weights_after_loading(layer)

LinearMethodBase(QuantizeMethodBase)    # for dense linear layers
FusedMoEMethodBase(QuantizeMethodBase)  # for MoE layers

QuantizationConfig (ABC)
  get_name() -> str               [abstract]
  get_supported_act_dtypes()      [abstract]
  get_min_capability() -> int     [abstract]
  get_config_filenames() -> List  [abstract]
  from_config(config) -> Self     [abstract, classmethod]
  get_quant_method(layer, prefix) -> Optional[QuantizeMethodBase]  [abstract]
  get_scaled_act_names() -> List  [abstract]
  override_quantization_method(hf_quant_cfg, user_quant) -> Optional[str]
```

### Key Pattern

Each `QuantizationConfig` subclass inspects the layer type in `get_quant_method()` and returns the appropriate method:
- `LinearBase` → `LinearMethodBase` subclass (weight quantization)
- `FusedMoE` → `FusedMoEMethodBase` subclass (MoE weight quantization)
- `RadixAttention` → `BaseKVCacheMethod` subclass (KV cache quantization)

**For TurboQuant:** We only need the `RadixAttention` → KV cache method path. No weight quantization.

---

## 3. KV Cache Quantization

**File:** `python/sglang/srt/layers/quantization/kv_cache.py`

```python
class BaseKVCacheMethod(QuantizeMethodBase):
    def create_weights(self, layer):
        # Adds k_scale and v_scale parameters to the layer
        layer.k_scale = Parameter(tensor(-1.0, float32), requires_grad=False)
        layer.v_scale = Parameter(tensor(-1.0, float32), requires_grad=False)

    def process_weights_after_loading(self, layer):
        # Validates and normalizes k_scale/v_scale values
        # Handles: both loaded, neither loaded, one loaded
```

### Reference: FP8 KV Cache (in `fp8.py`)

```python
class Fp8KVCacheMethod(BaseKVCacheMethod):
    """Loads kv-cache scaling factors from FP8 checkpoints."""
    pass  # Inherits everything from BaseKVCacheMethod
```

The FP8 KV cache is simple — it just stores scale factors. The actual FP8 conversion is handled by setting `store_dtype` to `torch.uint8` within the regular `MHATokenToKVPool` and applying type casting during storage.

**TurboQuant is more complex:** We need to store rotation matrices, codebooks, and manage quantized storage buffers. This likely requires a custom KV cache method that goes beyond `BaseKVCacheMethod`.

---

## 4. RadixAttention (KV Cache Consumer)

**File:** `python/sglang/srt/layers/radix_attention.py`

```python
class RadixAttention(nn.Module):
    def __init__(self, ..., quant_config=None, ...):
        self.quant_method = None
        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)

    def forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
        v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
        return forward_batch.attn_backend.forward(
            q, k, v, self, forward_batch, save_kv_cache, **kwargs
        )
```

**Key points:**
- K and V arrive as post-RoPE, post-projection tensors
- The `attn_backend` handles actual KV cache storage and attention computation
- `quant_config` is passed at construction time — **Qwen3.5 currently does NOT pass it** (needs fix)

---

## 5. KV Cache Memory Pool

**File:** `python/sglang/srt/mem_cache/memory_pool.py`

### Base Class
```python
class KVCache(ABC):
    def set_kv_buffer(self, layer, loc, cache_k, cache_v, ...)
    def get_key_buffer(self, layer_id) -> torch.Tensor
    def get_value_buffer(self, layer_id) -> torch.Tensor
    def get_kv_buffer(self, layer_id) -> Tuple[torch.Tensor, torch.Tensor]
```

### MHA Implementation
```python
class MHATokenToKVPool(KVCache):
    # Stores K and V as dense tensors:
    # k_buffer[layer]: [pool_size, num_heads, head_dim] in model dtype (FP16/BF16)
    # v_buffer[layer]: [pool_size, num_heads, head_dim] in model dtype
```

### FP4 Variant
- `MHATokenToKVPoolFP4`: Compact 4-bit KV storage
- FP8 is handled within the regular `MHATokenToKVPool` by setting `store_dtype` to `torch.uint8`

### Hybrid Pool (for Qwen3.5)
`HybridLinearKVPool` wraps separate pools for full-attention and linear-attention layers, handling the layer ID mapping between global layer IDs and per-type indices.

**TurboQuant integration (DONE):** `HybridLinearKVPool` now accepts `kv_cache_quantization` parameter. When set to `"turboquant"`, it creates `TurboQuantTokenToKVPool` as its inner `full_kv_pool` instead of `MHATokenToKVPool`. Modified in `memory_pool.py` line ~1254 and `model_runner_kv_cache_mixin.py` line ~590.

---

## 6. Attention Backends

**Directory:** `python/sglang/srt/layers/attention/`

Key backends:
- `flashinfer_backend.py` — Primary backend, uses FlashInfer library
- `triton_backend.py` — Triton-based alternative
- `torch_native_backend.py` — Pure PyTorch fallback

### Backend Flow (FlashInfer example)

**Prefill/Extend (`forward_extend`):**
1. Save KV to cache: `token_to_kv_pool.set_kv_buffer(layer, loc, cache_k, cache_v)`
2. Run FlashInfer prefill attention kernel

**Decode (`forward_decode`):**
1. Save current token's KV to cache: `token_to_kv_pool.set_kv_buffer(...)`
2. Run FlashInfer decode attention kernel (reads from KV cache)

**TurboQuant integration (DONE):**

TurboQuant is **kernel-agnostic** — it works with any attention backend via dequant-on-read:
- `TurboQuantTokenToKVPool.get_key_buffer()` / `get_value_buffer()` dequantize on demand, returning standard [pool_size, H, D] BF16 tensors
- Any backend (FlashInfer, Triton, etc.) can use these without modification
- `--kv-cache-quantization turboquant` does NOT force a specific attention backend

**Opt-in fused kernels** (`--attention-backend turboquant`):
- `TurboQuantAttnBackend` inherits `TritonAttnBackend`, overrides `forward_decode` and `forward_extend`
- Reads packed uint8 KV directly — no BF16 materialization
- Decode: two-stage Triton kernel (score + reduce across KV splits)
- Extend: single-stage quantize-first kernel (all KV from quantized buffers, online softmax)
- Both support integer bits (3-bit) and split-channel fractional bits (3.5-bit)

**Rotation:** Randomized Hadamard transform (FWHT) — O(d log d), stores only a sign vector per layer. Query pre-rotation via `hadamard.forward(q)`, output inverse via `hadamard.inverse(o_rot)`. QJL projection (`q @ S^T`) uses a separate dense Gaussian matrix (paper Definition 1).

---

## 7. Kernel Registration

**File:** `sgl-kernel/csrc/common_extension.cc`

Pattern:
```cpp
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("int8_scaled_mm(...) -> Tensor");
    m.impl("int8_scaled_mm", torch::kCUDA, &int8_scaled_mm);
    // ...
}
```

All ops accessible via `torch.ops.sgl_kernel.*` namespace.

**Build system:** `sgl-kernel/CMakeLists.txt` — add new CUDA source files here.

**For TurboQuant:** Register quantization, dequantization, and fused attention kernels.

---

## 8. Model Loading Pipeline

**File:** `python/sglang/srt/model_loader/loader.py`

Three-phase pipeline:
1. `_get_quantization_config()` — resolves quant method from model config or CLI args
2. `_get_weights_iterator()` — loads weights from safetensors/pt/gguf
3. Model's `load_weights()` + `process_weights_after_loading()` per module

**File:** `python/sglang/srt/configs/model_config.py`

`_parse_quant_hf_config()` detects quantization from:
1. `config.json` → `quantization_config` field
2. `compression_config` field
3. `hf_quant_config.json`
4. `quant_model_description.json`

**For TurboQuant:** Since TurboQuant is a runtime KV cache method (not checkpoint-based), it's configured via CLI args, not model config. No weight loading changes needed.

---

## 9. Server Configuration

**File:** `python/sglang/srt/server_args.py`

Relevant args:
- `--quantization` — selects weight quantization method
- `--kv-cache-dtype` — selects KV cache dtype (auto, fp8_e5m2, fp8_e4m3fn)

**For TurboQuant:** Add either:
- A new `--kv-cache-quantization turboquant` arg, or
- Add `"turboquant"` to `--kv-cache-dtype` choices with additional sub-options (`--turboquant-bits`, etc.)

---

## 10. Critical Integration Summary

| Step | File | Status | Change |
|---|---|---|---|
| Register method | `quantization/__init__.py` | DONE | Added `"turboquant": TurboQuantConfig` |
| Config class | `quantization/turboquant/config.py` | DONE | `TurboQuantConfig(QuantizationConfig)` with fractional bit support |
| KV cache method | `quantization/turboquant/kv_cache_method.py` | DONE | Attaches config to RadixAttention layers |
| Memory pool | `mem_cache/turboquant_pool.py` | DONE | `TurboQuantTokenToKVPool(KVCache)` with compact uint8 packed buffers |
| Decode kernel | `attention/triton_ops/turboquant_decode_attention.py` | DONE | Two-stage Triton kernel (integer + split-channel) |
| Extend kernel | `attention/triton_ops/turboquant_extend_attention.py` | DONE | Single-stage quantize-first Triton kernel (integer + split-channel) |
| Backend | `attention/turboquant_backend.py` | DONE | Overrides both forward_decode and forward_extend |
| Qwen3.5 model | `models/qwen3_5.py` | DONE | Passes `quant_config` to `RadixAttention` |
| Server args | `server_args.py` | DONE | `--kv-cache-quantization turboquant --turboquant-bits N` |
| Benchmarks | `benchmark/turboquant/` | DONE | perplexity, needle-in-haystack, GSM8K, master runner |
| Model runner | `model_executor/model_runner.py` | Instantiate TurboQuant pool |
| Kernels | `sgl-kernel/csrc/turboquant/` (NEW) | CUDA quantization/attention kernels |
| Kernel registration | `sgl-kernel/csrc/common_extension.cc` | Register new ops |
| Build | `sgl-kernel/CMakeLists.txt` | Add CUDA sources |
