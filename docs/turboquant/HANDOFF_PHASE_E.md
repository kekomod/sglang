# Phase E Handoff: Triton FWHT Kernel — COMPLETED (2026-03-27)

> **Status:** Implemented. `triton_fwht.py` created with forward/inverse kernels. `rotation.py` auto-dispatches to Triton on CUDA. 12/12 kernel tests pass. Roundtrip error 4.77e-07.

## Context

Currently `rotation.py:_fwht_impl()` uses a Python while-loop with in-place butterfly operations, wrapped in `torch.compile(dynamic=True)`. This works and fuses into a single GPU kernel via the compiler, but may still be suboptimal due to torch.compile overhead (first-call compilation latency, limited control over memory access patterns, and inability to exploit Triton-specific features like shared memory tiling).

The Fast Walsh-Hadamard Transform for d=128 is 7 butterfly stages (`log2(128) = 7`). Each stage processes pairs of elements at increasing strides: for stride `s` in `{1, 2, 4, 8, 16, 32, 64}`, each pair `(a, b)` at distance `s` apart is replaced by `(a + b, a - b)`. After all stages, the result is scaled by `1/sqrt(128)`.

## What to Build

### Triton kernel: `fwht_kernel`

A single Triton kernel that performs all 7 butterfly stages in one kernel launch.

**Grid and block design:**
- Grid: `(num_tokens * num_heads,)` — one program per (token, head) pair
- Each program processes exactly one 128-element vector
- The 128 elements fit entirely in registers (no shared memory needed for d=128)

**Kernel pseudocode:**
```python
@triton.jit
def fwht_kernel(
    x_ptr,           # [N, H, D] input/output tensor (in-place)
    N,               # number of tokens
    H,               # number of heads
    D: tl.constexpr, # head dimension (128)
    LOG_D: tl.constexpr,  # log2(D) = 7
    scale: tl.constexpr,  # 1/sqrt(D)
):
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    # Base offset for this (token, head) pair
    base = token_idx * H * D + head_idx * D
    offs = tl.arange(0, D)

    # Load entire vector into registers
    x = tl.load(x_ptr + base + offs)

    # 7 butterfly stages
    # Stage 0: stride=1 — pairs (0,1), (2,3), (4,5), ...
    # Stage 1: stride=2 — pairs (0,2), (1,3), (4,6), (5,7), ...
    # ...
    # Stage 6: stride=64 — pairs (0,64), (1,65), ..., (63,127)
    #
    # For each stage with stride s:
    #   For each index i in [0, D):
    #     If (i & s) == 0:  # i is in the "top" half of the pair
    #       a = x[i], b = x[i + s]
    #       x[i] = a + b, x[i + s] = a - b
    #
    # In Triton, use masking to select top/bottom halves:
    stride = 1
    for _ in range(LOG_D):  # tl.static_range(LOG_D)
        top_mask = ((offs & stride) == 0)  # True for "top" elements
        partner = offs ^ stride              # XOR gives partner index
        x_partner = tl.load(x_ptr + base + partner)  # or use shuffle
        x_new = tl.where(top_mask, x + x_partner, x_partner - x)
        x = x_new
        stride *= 2

    # Scale
    x = x * scale

    # Store back
    tl.store(x_ptr + base + offs, x)
```

**Important implementation notes:**
- The `offs ^ stride` trick maps each element to its butterfly partner. For stride=1: `0<->1, 2<->3, ...`. For stride=2: `0<->2, 1<->3, 4<->6, ...`. This is a standard technique for parallel butterfly networks.
- `tl.static_range` should be used instead of Python `range` if the loop count is a `tl.constexpr`, to ensure full unrolling.
- Since all 128 elements are in registers, the partner load can potentially be optimized with warp shuffles, but Triton's `tl.load` from the same program's store buffer may already handle this efficiently.

### Forward and inverse variants

The forward Hadamard transform (for quantization) applies sign flips BEFORE the FWHT:
```
forward(x, signs) = FWHT(signs * x)
```

The inverse Hadamard transform (for dequantization) applies sign flips AFTER the FWHT:
```
inverse(x, signs) = signs * FWHT(x)
```

Implement these as two wrapper kernels or a single kernel with a `mode` flag:

```python
@triton.jit
def fwht_forward_kernel(x_ptr, signs_ptr, ...):
    # Load x and signs
    x = tl.load(x_ptr + base + offs)
    signs = tl.load(signs_ptr + head_idx * D + offs)  # signs are per-layer, shared across tokens
    # Apply signs first
    x = x * signs
    # Do FWHT butterfly stages
    ...
    # Scale and store

@triton.jit
def fwht_inverse_kernel(x_ptr, signs_ptr, ...):
    # Load x
    x = tl.load(x_ptr + base + offs)
    # Do FWHT butterfly stages
    ...
    # Scale
    x = x * scale
    # Apply signs after
    signs = tl.load(signs_ptr + head_idx * D + offs)
    x = x * signs
    # Store
```

## Reference Implementations

### MLX PR #860 (Blaizzy/mlx-vlm)
The Metal FWHT kernels (`fast_wht_forward`, `fast_wht_inverse`) use:
- Threadgroup shared memory for the butterfly network
- SIMD warp reductions for intra-warp communication
- Separate forward/inverse kernels with sign multiplication before/after

### Current Python implementation (`rotation.py:_fwht_impl`)
```python
def _fwht_impl(x):
    d = x.shape[-1]
    h = 1
    while h < d:
        # butterfly: add/subtract pairs at distance h
        x1 = x[..., 0::2*h]  # even indices at this stride
        x2 = x[..., h::2*h]  # odd indices at this stride
        x[..., 0::2*h] = x1 + x2
        x[..., h::2*h] = x1 - x2
        h *= 2
    return x * (1.0 / math.sqrt(d))
```

## File Changes

### Create: `python/sglang/srt/layers/quantization/turboquant/triton_fwht.py`
- `fwht_kernel` — core Triton FWHT kernel
- `fwht_forward_kernel` — FWHT with pre-sign-flip (for quantization)
- `fwht_inverse_kernel` — FWHT with post-sign-flip (for dequantization)
- Python wrapper functions: `triton_fwht(x)`, `triton_fwht_forward(x, signs)`, `triton_fwht_inverse(x, signs)`

### Modify: `python/sglang/srt/layers/quantization/turboquant/rotation.py`
- In `HadamardTransform.forward()`: replace `_fwht_impl` call with `triton_fwht_forward(x, self.signs)`
- In `HadamardTransform.inverse()`: replace `_fwht_impl` call with `triton_fwht_inverse(x, self.signs)`
- Keep `_fwht_impl` as a fallback for CPU tensors or testing
- Remove `@torch.compile` decorator from `_fwht_impl` (no longer needed on the hot path)

## Verification

Run the existing test suite:
```bash
conda activate turboquant
python -m pytest test/test_turboquant_kernel.py -v
```

All 11 tests must pass with the same cosine similarities (1.0000). The Triton FWHT should be a drop-in replacement — the mathematical operation is identical, only the execution strategy changes.

### Additional verification:
- Compare `triton_fwht(x)` output against `_fwht_impl(x)` for random inputs — should match to float precision
- Benchmark: time `_fwht_impl` (torch.compile) vs `triton_fwht` over 1000 iterations to confirm speedup
- Test with d=64, d=128, d=256 to ensure the kernel generalizes (adjust LOG_D accordingly)

## Performance Expectations

For d=128, the FWHT is 7 stages x 128 elements = 896 add/subtract operations per vector. This is a very small workload — the kernel will be memory-bandwidth-bound, not compute-bound. The main benefit of a Triton kernel over torch.compile is:
- Eliminating torch.compile warmup latency (~seconds on first call)
- More predictable performance (no compiler heuristics)
- Easier to fuse with surrounding operations in future phases (e.g., fusing sign multiplication + FWHT + quantization into a single kernel)
