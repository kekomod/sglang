# Phase I Handoff: Hardening & Edge Cases

## Context

After Phase E (Triton FWHT) and Phase F (CUDA graph support via F2 workspace), the core TurboQuant implementation is functionally complete. 12/12 kernel tests and 6/6 server tests pass. Throughput is 0.76x baseline with CUDA graphs enabled.

This phase covers hardening: edge cases, defensive error handling, and workspace lifecycle correctness under less-common SGLang configurations. None of these are blocking issues today — they are smoothing items for production readiness.

Each item requires a decision about whether it's worth addressing, given the tradeoff between code complexity and the likelihood of the scenario occurring.

## Items

### I1. Workspace Reset on Pool Reuse

**Scenario:** If `TurboQuantTokenToKVPool` is reused after a cache clear (e.g., all requests finish and slots are freed), the BF16 workspace retains stale data in freed slots.

**Current behavior:** Stale workspace data is harmless — freed slots are never indexed by `kv_indices` until they're re-allocated, at which point `set_kv_buffer` overwrites them with fresh data.

**Decision to consider:** Is a `clear()` method needed for defensive safety, or does the slot allocator's invariant (freed slots are never read) make this unnecessary overhead? Adding `clear()` would zero all workspace buffers — expensive for large pools (28 layers × 130K slots × 4 heads × 128 dim × 2 bytes × 2 (K+V) ≈ 7 GB of writes).

**If implementing:** Add a `clear()` method to `TurboQuantTokenToKVPool` that zeros workspace buffers. Only call it when the pool is explicitly reset, not on every free operation.

**Files:** `python/sglang/srt/mem_cache/turboquant_pool.py`

### I2. CPU Offloading Incompatibility

**Scenario:** SGLang supports CPU offloading of KV cache via `get_cpu_copy()` / `load_cpu_copy()` on the base `KVCache` class. `TurboQuantTokenToKVPool` does not override these methods.

**Current behavior:** If CPU offloading is enabled with TurboQuant, calling `get_cpu_copy()` would hit the base class `NotImplementedError`. The failure is clear but not informative.

**Decision to consider:** Should TurboQuant support CPU offloading (offload packed buffers + workspace), or is an explicit error message sufficient? Supporting it would require serializing both packed storage and BF16 workspace, which is more complex but could be valuable for very long context scenarios. Alternatively, offloading only packed storage and dequanting on reload would save CPU memory.

**If implementing (error only):** Override `get_cpu_copy()` and `load_cpu_copy()` to raise `NotImplementedError("CPU offloading not supported with TurboQuant KV cache")`.

**If implementing (full support):** Override both methods to transfer packed buffers + norms + workspace. Consider whether workspace needs to be offloaded at all — it can be reconstructed from packed storage via dequantization on reload.

**Files:** `python/sglang/srt/mem_cache/turboquant_pool.py`

### I3. Workspace Behavior During CUDA Graph Capture Warmup

**Scenario:** During CUDA graph capture (`_capture_graph` in `cuda_graph_runner.py`), the model runs warmup forward passes with `_graph_mode=True`. During these warmups, `set_kv_buffer` writes BF16 to workspace but does NOT quantize to packed storage. Meanwhile, `get_key_buffer` / `get_value_buffer` in graph mode returns the workspace directly.

**Current behavior:** This is correct for graph capture — the warmup data is throwaway and the captured graph only records the kernel launch pattern, not the data. After capture, real inference uses graph replay with proper dual-write and post-graph quantization.

**Decision to consider:** Should there be an assertion or debug check that `_graph_mode` is never accidentally left on? If a code path sets `_graph_mode=True` but crashes before the `finally` block (if one existed), packed storage would never be updated. Currently the set/unset is not wrapped in try/finally.

**If implementing:** Wrap graph mode in a context manager:
```python
@contextmanager
def graph_mode(self):
    self.set_graph_mode(True)
    try:
        yield
    finally:
        self.set_graph_mode(False)
```
Then use `with pool.graph_mode():` in model_runner.py and cuda_graph_runner.py instead of manual set/unset.

**Files:** `python/sglang/srt/mem_cache/turboquant_pool.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`

### I4. Prefix Cache Consistency Under Eviction

**Scenario:** SGLang's RadixCache evicts old KV entries and reuses their slots. When a slot is evicted and reallocated, `move_kv_cache()` copies data between slots. Both packed buffers and workspace are copied (verified in audit).

**Current behavior:** Correct. `move_kv_cache()` copies workspace in both split and integer code paths. When slots are freed (not moved), the stale workspace data is irrelevant per I1.

**Decision to consider:** Under heavy prefix cache churn with CUDA graphs, is there a window where a graph replay reads from a workspace slot that was freed but not yet overwritten? The slot allocator should prevent this — freed slots are removed from `kv_indices` before any graph replay. Verify this invariant by tracing the eviction→reallocation→graph replay sequence.

**If implementing:** Add a debug assertion in `set_kv_buffer` (graph mode) that verifies `loc` values are currently allocated slots. Only enable under `SGLANG_DEBUG=1` to avoid overhead.

**Files:** `python/sglang/srt/mem_cache/turboquant_pool.py`

### I5. Memory Reporting Accuracy

**Scenario:** `get_kv_size_bytes()` currently sums packed buffers + norms + workspace. This is used by SGLang for memory accounting and scheduling decisions.

**Current behavior:** Correct — workspace is included in the total. However, the reported size is the theoretical maximum (full pool × all layers). In practice, only allocated slots contain meaningful data.

**Decision to consider:** Should memory reporting distinguish between "capacity" (total allocated) and "used" (slots with valid data)? This would help users understand the memory overhead of the F2 workspace approach. Alternatively, this level of detail may not be useful until F1 eliminates the workspace.

**If implementing:** Add a `get_workspace_overhead_bytes()` method that reports workspace size separately from packed storage, for logging/diagnostics.

**Files:** `python/sglang/srt/mem_cache/turboquant_pool.py`

## Verification

```bash
conda activate turboquant

# Kernel tests (should be unaffected by hardening changes)
python test/test_turboquant_kernel.py

# Server tests with CUDA graphs
python -m sglang.launch_server \
    --model /home/keko/AI/image-prep/models/uncensored-9b-bf16-hf \
    --kv-cache-quantization turboquant \
    --turboquant-bits 3.5 \
    --port 30000 \
    --context-length 4096

python test/test_turboquant_server.py

# Stress test: many concurrent requests to exercise prefix cache + workspace
for i in $(seq 1 50); do
    curl -s http://localhost:30000/v1/completions \
        -d '{"model": "default", "prompt": "Count to 10:", "max_tokens": 30}' &
done
wait
```

## Priority Assessment

| Item | Risk if ignored | Complexity | Recommendation |
|------|----------------|------------|----------------|
| I1. Workspace reset | Very low (slot allocator prevents reads of stale data) | Low | Skip unless debugging shows stale reads |
| I2. CPU offload error | Low (feature not commonly used with quantized KV) | Low (error msg) / Medium (full support) | Add error message; defer full support |
| I3. Graph mode context manager | Low (current code is correct, just not crash-safe) | Low | Worth doing — small change, defensive |
| I4. Prefix cache consistency | Very low (slot allocator invariant protects this) | Medium (debug assertions + tracing) | Verify once manually; skip assertions unless issues arise |
| I5. Memory reporting | None (functional correctness unaffected) | Low | Nice-to-have for diagnostics |
