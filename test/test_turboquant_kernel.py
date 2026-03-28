"""
TurboQuant kernel correctness tests.

Direct unit tests of Triton kernels and quant ops with synthetic data.
No server needed — runs on GPU directly.

Usage: python test/test_turboquant_kernel.py
"""

import math
import sys

import torch
import triton

from sglang.srt.layers.quantization.turboquant.codebook import (
    compute_codebook,
    packed_width,
)
from sglang.srt.layers.quantization.turboquant.quant_ops import (
    mse_dequantize,
    mse_quantize,
    pack_bits,
    select_outlier_indices,
    split_channel_mse_dequantize,
    split_channel_mse_quantize,
    unpack_bits,
)
from sglang.srt.layers.quantization.turboquant.rotation import (
    HadamardTransform,
)


def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


# ---------------------------------------------------------------------------
# Test: bit-packing roundtrip
# ---------------------------------------------------------------------------

def test_bit_packing_roundtrip():
    """pack_bits -> unpack_bits should be identity for all supported bit-widths."""
    for bits in [1, 2, 3, 4]:
        length = 256
        max_val = (1 << bits) - 1
        values = torch.randint(0, max_val + 1, (4, 8, length), dtype=torch.int32)
        packed = pack_bits(values, bits)
        unpacked = unpack_bits(packed, bits, length)
        assert torch.equal(values, unpacked), (
            f"{bits}-bit: roundtrip failed, max diff={torch.abs(values - unpacked).max()}"
        )
    return "1,2,3,4-bit all pass"


# ---------------------------------------------------------------------------
# Test: select_outlier_indices
# ---------------------------------------------------------------------------

def test_select_outlier_indices():
    """Verify channel split logic for various bit-widths."""
    # 3.5-bit: split 50/50
    lo, hi = select_outlier_indices(256, 3.5)
    assert lo.shape[0] == 128 and hi.shape[0] == 128, f"3.5-bit: lo={lo.shape[0]}, hi={hi.shape[0]}"

    # 3.0-bit: no split
    lo, hi = select_outlier_indices(256, 3.0)
    assert lo.shape[0] == 256 and hi.shape[0] == 0, f"3.0-bit: lo={lo.shape[0]}, hi={hi.shape[0]}"

    # 2.5-bit: split 50/50
    lo, hi = select_outlier_indices(256, 2.5)
    assert lo.shape[0] == 128 and hi.shape[0] == 128, f"2.5-bit: lo={lo.shape[0]}, hi={hi.shape[0]}"

    # Coverage: all indices present
    lo, hi = select_outlier_indices(256, 3.5)
    all_idx = torch.cat([lo, hi]).sort()[0]
    assert torch.equal(all_idx, torch.arange(256, dtype=torch.int64)), "indices don't cover full range"

    return "3.5, 3.0, 2.5-bit all correct"


# ---------------------------------------------------------------------------
# Test: MSE quantize/dequantize roundtrip
# ---------------------------------------------------------------------------

def test_mse_roundtrip():
    """MSE quantize -> dequantize should have high cosine similarity."""
    device = "cuda"
    N, H, D = 8, 4, 256
    bits = 3
    seed = 42

    x = torch.randn(N, H, D, device=device)
    hadamard = HadamardTransform(D, seed, device)
    cb = compute_codebook(D, bits).to(device)

    packed, norms = mse_quantize(x, hadamard, cb, bits)
    recon = mse_dequantize(packed, norms, hadamard, cb, bits, D)

    sim = cosine_sim(x, recon)
    assert sim > 0.98, f"MSE roundtrip cosine sim {sim:.4f} < 0.98"
    return f"cosine_sim={sim:.4f}"


# ---------------------------------------------------------------------------
# Test: Integer-bit Triton kernel vs reference
# ---------------------------------------------------------------------------

def test_integer_kernel_vs_reference():
    """Compare Triton decode kernel output against dequant+matmul reference (MSE-only keys)."""
    from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
        turboquant_decode_attention_fwd,
    )

    device = "cuda"
    torch.manual_seed(42)

    B, H_q, H_kv, D = 1, 8, 4, 256
    seq_len = 64
    bits = 3
    seed = 42
    kv_group_num = H_q // H_kv

    # Setup codebooks, matrices — keys and values both use full bits MSE
    hadamard = HadamardTransform(D, seed, device)
    k_cb = compute_codebook(D, bits).to(device)
    v_cb = compute_codebook(D, bits).to(device)

    # Random Q, K, V
    q = torch.randn(B, H_q, D, device=device, dtype=torch.float32)
    k_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)
    v_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)

    # Quantize K and V (both MSE)
    k_packed, k_norms = mse_quantize(k_raw, hadamard, k_cb, bits)
    v_packed, v_norms = mse_quantize(v_raw, hadamard, v_cb, bits)

    # === REFERENCE: dequant + standard attention ===
    k_deq = mse_dequantize(k_packed, k_norms, hadamard, k_cb, bits, D)
    v_deq = mse_dequantize(v_packed, v_norms, hadamard, v_cb, bits, D)

    # GQA expand: [seq_len, H_kv, D] -> [seq_len, H_q, D]
    k_expanded = k_deq.repeat_interleave(kv_group_num, dim=1)
    v_expanded = v_deq.repeat_interleave(kv_group_num, dim=1)

    sm_scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("bhd,shd->bhs", q, k_expanded) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("bhs,shd->bhd", weights, v_expanded)

    # === TRITON KERNEL ===
    q_rot = hadamard.forward(q)

    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device=device)

    max_kv_splits = 8
    num_kv_splits = torch.tensor([max_kv_splits], dtype=torch.int32, device=device)

    BLOCK_DV = triton.next_power_of_2(hadamard.padded_dim)
    attn_logits = torch.empty(B, H_q, max_kv_splits, BLOCK_DV, dtype=torch.float32, device=device)
    attn_lse = attn_logits[:, :, :, 0].contiguous()

    o_rot = torch.empty(B, H_q, hadamard.padded_dim, dtype=torch.float32, device=device)

    turboquant_decode_attention_fwd(
        q_rot,
        k_packed, k_norms,
        v_packed, v_norms,
        k_cb, v_cb,
        o_rot,
        kv_indptr, kv_indices, num_kv_splits,
        max_kv_splits, sm_scale,
        bits, bits, hadamard.padded_dim,
        attn_logits, attn_lse,
    )

    # Inverse rotate
    kernel_out = hadamard.inverse(o_rot)

    sim = cosine_sim(ref_out, kernel_out)
    assert sim > 0.95, f"Integer kernel vs reference cosine sim {sim:.4f} < 0.95"
    return f"cosine_sim={sim:.4f}"


# ---------------------------------------------------------------------------
# Test: Split-channel Triton kernel vs reference
# ---------------------------------------------------------------------------

def test_split_kernel_vs_reference():
    """Compare split-channel Triton decode kernel against dequant+matmul reference (MSE-only keys)."""
    from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
        turboquant_decode_attention_fwd_split,
    )

    device = "cuda"
    torch.manual_seed(42)

    B, H_q, H_kv, D = 1, 8, 4, 256
    seq_len = 64
    avg_bits = 3.5
    seed = 42
    kv_group_num = H_q // H_kv

    lo_bits = math.floor(avg_bits)  # 3
    hi_bits = math.ceil(avg_bits)   # 4
    lo_indices, hi_indices = select_outlier_indices(D, avg_bits)
    restore_order = torch.argsort(torch.cat([lo_indices, hi_indices]))
    d_lo, d_hi = lo_indices.shape[0], hi_indices.shape[0]

    hadamard_lo = HadamardTransform(d_lo, seed, device)
    hadamard_hi = HadamardTransform(d_hi, seed + 97, device)

    # Keys and values both use full bits MSE per group
    k_cb_lo = compute_codebook(d_lo, lo_bits).to(device)
    k_cb_hi = compute_codebook(d_hi, hi_bits).to(device)
    v_cb_lo = compute_codebook(d_lo, lo_bits).to(device)
    v_cb_hi = compute_codebook(d_hi, hi_bits).to(device)

    # Random Q, K, V
    q = torch.randn(B, H_q, D, device=device, dtype=torch.float32)
    k_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)
    v_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)

    # Quantize K and V (both split MSE)
    k_lo_p, k_lo_n, k_hi_p, k_hi_n = split_channel_mse_quantize(
        k_raw, lo_indices, hi_indices, hadamard_lo, hadamard_hi,
        k_cb_lo, k_cb_hi, lo_bits, hi_bits,
    )
    v_lo_p, v_lo_n, v_hi_p, v_hi_n = split_channel_mse_quantize(
        v_raw, lo_indices, hi_indices, hadamard_lo, hadamard_hi,
        v_cb_lo, v_cb_hi, lo_bits, hi_bits,
    )

    # === REFERENCE: dequant + standard attention ===
    k_deq = split_channel_mse_dequantize(
        k_lo_p, k_lo_n, k_hi_p, k_hi_n,
        lo_indices, hi_indices, restore_order,
        hadamard_lo, hadamard_hi,
        k_cb_lo, k_cb_hi, lo_bits, hi_bits, D,
    )
    v_deq = split_channel_mse_dequantize(
        v_lo_p, v_lo_n, v_hi_p, v_hi_n,
        lo_indices, hi_indices, restore_order,
        hadamard_lo, hadamard_hi,
        v_cb_lo, v_cb_hi, lo_bits, hi_bits, D,
    )

    k_expanded = k_deq.repeat_interleave(kv_group_num, dim=1)
    v_expanded = v_deq.repeat_interleave(kv_group_num, dim=1)

    sm_scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("bhd,shd->bhs", q, k_expanded) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("bhs,shd->bhd", weights, v_expanded)

    # === TRITON SPLIT KERNEL ===
    q_lo = q.index_select(-1, lo_indices.to(device))
    q_hi = q.index_select(-1, hi_indices.to(device))

    q_rot_lo = hadamard_lo.forward(q_lo)
    q_rot_hi = hadamard_hi.forward(q_hi)

    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device=device)

    max_kv_splits = 8
    num_kv_splits = torch.tensor([max_kv_splits], dtype=torch.int32, device=device)

    padded_d_lo = hadamard_lo.padded_dim
    padded_d_hi = hadamard_hi.padded_dim
    head_dim = padded_d_lo + padded_d_hi
    BLOCK_DV = triton.next_power_of_2(head_dim)
    attn_logits = torch.empty(B, H_q, max_kv_splits, BLOCK_DV, dtype=torch.float32, device=device)
    attn_lse = attn_logits[:, :, :, 0].contiguous()

    o_rot_split = torch.empty(B, H_q, head_dim, dtype=torch.float32, device=device)

    turboquant_decode_attention_fwd_split(
        q_rot_lo, q_rot_hi,
        k_lo_p, k_lo_n,
        k_hi_p, k_hi_n,
        v_lo_p, v_lo_n,
        v_hi_p, v_hi_n,
        k_cb_lo, k_cb_hi, v_cb_lo, v_cb_hi,
        o_rot_split,
        kv_indptr, kv_indices, num_kv_splits,
        max_kv_splits, sm_scale,
        lo_bits, hi_bits,
        lo_bits, hi_bits,
        padded_d_lo, padded_d_hi, head_dim,
        attn_logits, attn_lse,
    )

    # Inverse rotate per group, reassemble
    o_lo = hadamard_lo.inverse(o_rot_split[..., :padded_d_lo])
    o_hi = hadamard_hi.inverse(o_rot_split[..., padded_d_lo:])
    output_split = torch.cat([o_lo, o_hi], dim=-1)
    kernel_out = output_split.index_select(-1, restore_order.to(device))

    sim = cosine_sim(ref_out, kernel_out)
    assert sim > 0.95, f"Split kernel vs reference cosine sim {sim:.4f} < 0.95"
    return f"cosine_sim={sim:.4f}"


# ---------------------------------------------------------------------------
# Test: Integer-bit extend kernel vs reference
# ---------------------------------------------------------------------------

def test_integer_extend_vs_reference():
    """Compare Triton extend kernel output against dequant+matmul reference."""
    from sglang.srt.layers.attention.triton_ops.turboquant_extend_attention import (
        turboquant_extend_attention_fwd,
    )

    device = "cuda"
    torch.manual_seed(42)

    B, H_q, H_kv, D = 2, 8, 4, 128
    prefix_len, extend_len = 32, 16
    total_kv = prefix_len + extend_len
    bits = 3
    seed = 42
    kv_group_num = H_q // H_kv

    # Setup codebooks, matrices
    hadamard = HadamardTransform(D, seed, device)
    k_cb = compute_codebook(D, bits).to(device)
    v_cb = compute_codebook(D, bits).to(device)

    # Random Q (extend queries), K, V (all tokens including prefix)
    q = torch.randn(B, extend_len, H_q, D, device=device, dtype=torch.float32)
    k_raw = torch.randn(B, total_kv, H_kv, D, device=device, dtype=torch.float32)
    v_raw = torch.randn(B, total_kv, H_kv, D, device=device, dtype=torch.float32)

    # Quantize K and V (both MSE) — flatten to [B*total_kv, H_kv, D]
    k_flat = k_raw.view(B * total_kv, H_kv, D)
    v_flat = v_raw.view(B * total_kv, H_kv, D)

    k_packed, k_norms = mse_quantize(k_flat, hadamard, k_cb, bits)
    v_packed, v_norms = mse_quantize(v_flat, hadamard, v_cb, bits)

    # === REFERENCE: dequant + standard attention with causal mask ===
    k_deq = mse_dequantize(k_packed, k_norms, hadamard, k_cb, bits, D)
    v_deq = mse_dequantize(v_packed, v_norms, hadamard, v_cb, bits, D)

    k_deq = k_deq.view(B, total_kv, H_kv, D)
    v_deq = v_deq.view(B, total_kv, H_kv, D)

    # GQA expand
    k_expanded = k_deq.repeat_interleave(kv_group_num, dim=2)
    v_expanded = v_deq.repeat_interleave(kv_group_num, dim=2)

    sm_scale = 1.0 / math.sqrt(D)

    # Reference attention: Q[B,extend,H_q,D] @ K[B,total_kv,H_q,D]
    q_t = q.permute(0, 2, 1, 3)           # [B, H_q, extend_len, D]
    k_t = k_expanded.permute(0, 2, 1, 3)  # [B, H_q, total_kv, D]
    v_t = v_expanded.permute(0, 2, 1, 3)  # [B, H_q, total_kv, D]

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale

    # Causal mask: prefix always visible, extend token k visible to query q iff q >= k
    causal_mask = torch.ones(extend_len, total_kv, dtype=torch.bool, device=device)
    for qi in range(extend_len):
        for ki in range(total_kv):
            if ki >= prefix_len and (ki - prefix_len) > qi:
                causal_mask[qi, ki] = False

    scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(weights, v_t).permute(0, 2, 1, 3)  # [B, extend_len, H_q, D]

    # === TRITON KERNEL ===
    q_flat = q.view(B * extend_len, H_q, D)
    q_rot = hadamard.forward(q_flat)

    total_q = B * extend_len
    total_kv_tokens = B * total_kv
    qo_indptr = torch.tensor(
        [i * extend_len for i in range(B + 1)], dtype=torch.int32, device=device
    )
    kv_indptr = torch.tensor(
        [i * total_kv for i in range(B + 1)], dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(total_kv_tokens, dtype=torch.int64, device=device)
    prefix_lens_t = torch.full((B,), prefix_len, dtype=torch.int32, device=device)

    o_rot = torch.empty(total_q, H_q, hadamard.padded_dim, dtype=torch.float32, device=device)

    turboquant_extend_attention_fwd(
        q_rot,
        k_packed, k_norms,
        v_packed, v_norms,
        k_cb, v_cb,
        o_rot,
        qo_indptr, kv_indptr, kv_indices, prefix_lens_t,
        max_extend_len=extend_len,
        sm_scale=sm_scale,
        mse_bits=bits,
        v_bits=bits,
        head_dim=hadamard.padded_dim,
        is_causal=True,
    )

    # Inverse rotate
    kernel_out = hadamard.inverse(o_rot).view(B, extend_len, H_q, D)

    sim = cosine_sim(ref_out, kernel_out)
    assert sim > 0.90, f"Integer extend vs reference cosine sim {sim:.4f} < 0.90"
    return f"cosine_sim={sim:.4f}"


# ---------------------------------------------------------------------------
# Test: Extend causal mask correctness
# ---------------------------------------------------------------------------

def test_extend_causal_mask():
    """Verify causal masking: prefix always visible, extend region causal."""
    from sglang.srt.layers.attention.triton_ops.turboquant_extend_attention import (
        turboquant_extend_attention_fwd,
    )

    device = "cuda"
    torch.manual_seed(42)

    B, H_q, H_kv, D = 1, 4, 4, 128
    prefix_len, extend_len = 16, 8
    total_kv = prefix_len + extend_len
    bits = 3
    seed = 42

    hadamard = HadamardTransform(D, seed, device)
    k_cb = compute_codebook(D, bits).to(device)
    v_cb = compute_codebook(D, bits).to(device)

    # Create K/V with one "hot" token that dominates attention
    k_raw = torch.randn(total_kv, H_kv, D, device=device, dtype=torch.float32) * 0.01
    v_raw = torch.zeros(total_kv, H_kv, D, device=device, dtype=torch.float32)

    # Place a "hot" V at the LAST extend position (should only be seen by last query)
    hot_pos = total_kv - 1  # last extend token
    hot_extend_idx = extend_len - 1
    v_raw[hot_pos, :, 0] = 100.0  # huge value in dim 0
    k_raw[hot_pos, :, :] = torch.randn(H_kv, D, device=device) * 10.0  # strong key

    # Quantize
    k_packed, k_norms = mse_quantize(k_raw, hadamard, k_cb, bits)
    v_packed, v_norms = mse_quantize(v_raw, hadamard, v_cb, bits)

    # Run kernel
    q = torch.randn(extend_len, H_q, D, device=device, dtype=torch.float32)
    q_rot = hadamard.forward(q)

    qo_indptr = torch.tensor([0, extend_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, total_kv], dtype=torch.int32, device=device)
    kv_indices = torch.arange(total_kv, dtype=torch.int64, device=device)
    prefix_lens_t = torch.tensor([prefix_len], dtype=torch.int32, device=device)

    sm_scale = 1.0 / math.sqrt(D)
    o_rot = torch.empty(extend_len, H_q, hadamard.padded_dim, dtype=torch.float32, device=device)

    turboquant_extend_attention_fwd(
        q_rot,
        k_packed, k_norms,
        v_packed, v_norms,
        k_cb, v_cb,
        o_rot,
        qo_indptr, kv_indptr, kv_indices, prefix_lens_t,
        max_extend_len=extend_len,
        sm_scale=sm_scale,
        mse_bits=bits,
        v_bits=bits,
        head_dim=hadamard.padded_dim,
        is_causal=True,
    )

    kernel_out = hadamard.inverse(o_rot)  # [extend_len, H_q, D]

    # The hot token is at extend position (extend_len-1).
    # Only the LAST query (position extend_len-1) should see it.
    # Earlier queries should have near-zero output norm (since V is ~zero except hot).
    last_query_norm = kernel_out[hot_extend_idx].norm().item()
    earlier_norms = kernel_out[:hot_extend_idx].norm(dim=-1).mean().item()

    # Last query should have much larger output than earlier queries
    assert last_query_norm > earlier_norms * 3.0, (
        f"Causal mask failed: last_query_norm={last_query_norm:.4f}, "
        f"earlier_mean_norm={earlier_norms:.4f} (expected last >> earlier)"
    )
    return f"last_norm={last_query_norm:.4f}, earlier_mean={earlier_norms:.4f}"


# ---------------------------------------------------------------------------
# Test: Extend with variable batch sizes
# ---------------------------------------------------------------------------

def test_extend_variable_batch():
    """Test extend kernel with varying prefix/extend lengths per sequence."""
    from sglang.srt.layers.attention.triton_ops.turboquant_extend_attention import (
        turboquant_extend_attention_fwd,
    )

    device = "cuda"
    torch.manual_seed(42)

    H_q, H_kv, D = 8, 4, 128
    bits = 3
    seed = 42
    kv_group_num = H_q // H_kv

    hadamard = HadamardTransform(D, seed, device)
    k_cb = compute_codebook(D, bits).to(device)
    v_cb = compute_codebook(D, bits).to(device)
    sm_scale = 1.0 / math.sqrt(D)

    # Three sequences with different prefix/extend lengths
    configs = [(0, 8), (32, 4), (16, 16)]
    B = len(configs)

    # Build per-sequence data and quantize into a flat pool
    all_k_raw = []
    all_v_raw = []
    all_q = []
    qo_offsets = [0]
    kv_offsets = [0]
    prefix_lens_list = []

    for p_len, e_len in configs:
        total = p_len + e_len
        all_k_raw.append(torch.randn(total, H_kv, D, device=device, dtype=torch.float32))
        all_v_raw.append(torch.randn(total, H_kv, D, device=device, dtype=torch.float32))
        all_q.append(torch.randn(e_len, H_q, D, device=device, dtype=torch.float32))
        qo_offsets.append(qo_offsets[-1] + e_len)
        kv_offsets.append(kv_offsets[-1] + total)
        prefix_lens_list.append(p_len)

    # Flatten and quantize
    k_flat = torch.cat(all_k_raw, dim=0)
    v_flat = torch.cat(all_v_raw, dim=0)
    q_flat = torch.cat(all_q, dim=0)

    k_packed, k_norms = mse_quantize(k_flat, hadamard, k_cb, bits)
    v_packed, v_norms = mse_quantize(v_flat, hadamard, v_cb, bits)

    total_q_tokens = qo_offsets[-1]
    total_kv_tokens = kv_offsets[-1]
    max_extend_len = max(e for _, e in configs)

    qo_indptr = torch.tensor(qo_offsets, dtype=torch.int32, device=device)
    kv_indptr = torch.tensor(kv_offsets, dtype=torch.int32, device=device)
    kv_indices = torch.arange(total_kv_tokens, dtype=torch.int64, device=device)
    prefix_lens_t = torch.tensor(prefix_lens_list, dtype=torch.int32, device=device)

    # Run kernel
    q_rot = hadamard.forward(q_flat)
    o_rot = torch.empty(total_q_tokens, H_q, hadamard.padded_dim, dtype=torch.float32, device=device)

    turboquant_extend_attention_fwd(
        q_rot,
        k_packed, k_norms,
        v_packed, v_norms,
        k_cb, v_cb,
        o_rot,
        qo_indptr, kv_indptr, kv_indices, prefix_lens_t,
        max_extend_len=max_extend_len,
        sm_scale=sm_scale,
        mse_bits=bits,
        v_bits=bits,
        head_dim=hadamard.padded_dim,
        is_causal=True,
    )

    kernel_out = hadamard.inverse(o_rot)

    # === Per-sequence reference comparison ===
    k_deq = mse_dequantize(k_packed, k_norms, hadamard, k_cb, bits, D)
    v_deq = mse_dequantize(v_packed, v_norms, hadamard, v_cb, bits, D)

    sims = []
    for seq_idx, (p_len, e_len) in enumerate(configs):
        total = p_len + e_len
        q_start = qo_offsets[seq_idx]
        q_end = qo_offsets[seq_idx + 1]
        kv_start = kv_offsets[seq_idx]
        kv_end = kv_offsets[seq_idx + 1]

        q_seq = q_flat[q_start:q_end]  # [e_len, H_q, D]
        k_seq = k_deq[kv_start:kv_end]  # [total, H_kv, D]
        v_seq = v_deq[kv_start:kv_end]  # [total, H_kv, D]

        # GQA expand
        k_exp = k_seq.repeat_interleave(kv_group_num, dim=1)
        v_exp = v_seq.repeat_interleave(kv_group_num, dim=1)

        # q_seq: [e_len, H_q, D] -> [H_q, e_len, D]
        q_s = q_seq.permute(1, 0, 2)
        k_s = k_exp.permute(1, 0, 2)
        v_s = v_exp.permute(1, 0, 2)

        scores_s = torch.matmul(q_s, k_s.transpose(-2, -1)) * sm_scale

        # Causal mask
        mask = torch.ones(e_len, total, dtype=torch.bool, device=device)
        for qi in range(e_len):
            for ki in range(total):
                if ki >= p_len and (ki - p_len) > qi:
                    mask[qi, ki] = False
        scores_s = scores_s.masked_fill(~mask[None, :, :], float("-inf"))
        weights_s = torch.softmax(scores_s, dim=-1)
        ref_seq = torch.matmul(weights_s, v_s).permute(1, 0, 2)  # [e_len, H_q, D]

        kernel_seq = kernel_out[q_start:q_end]
        sim = cosine_sim(ref_seq, kernel_seq)
        sims.append(sim)

    min_sim = min(sims)
    assert min_sim > 0.88, (
        f"Variable batch extend: min cosine sim {min_sim:.4f} < 0.88, all={sims}"
    )
    return f"sims={[f'{s:.4f}' for s in sims]}, min={min_sim:.4f}"


# ---------------------------------------------------------------------------
# Test: Split-channel extend kernel vs reference
# ---------------------------------------------------------------------------

def test_split_extend_vs_reference():
    """Compare split-channel Triton extend kernel against dequant+matmul reference."""
    try:
        from sglang.srt.layers.attention.triton_ops.turboquant_extend_attention import (
            turboquant_extend_attention_fwd_split,
        )
    except ImportError:
        return "SKIP: turboquant_extend_attention_fwd_split not yet implemented"

    device = "cuda"
    torch.manual_seed(42)

    B, H_q, H_kv, D = 2, 8, 4, 128
    prefix_len, extend_len = 32, 16
    total_kv = prefix_len + extend_len
    avg_bits = 3.5
    seed = 42
    kv_group_num = H_q // H_kv

    lo_bits = math.floor(avg_bits)  # 3
    hi_bits = math.ceil(avg_bits)   # 4
    lo_indices, hi_indices = select_outlier_indices(D, avg_bits)
    restore_order = torch.argsort(torch.cat([lo_indices, hi_indices]))
    d_lo, d_hi = lo_indices.shape[0], hi_indices.shape[0]

    hadamard_lo = HadamardTransform(d_lo, seed, device)
    hadamard_hi = HadamardTransform(d_hi, seed + 97, device)

    k_cb_lo = compute_codebook(d_lo, lo_bits).to(device)
    k_cb_hi = compute_codebook(d_hi, hi_bits).to(device)
    v_cb_lo = compute_codebook(d_lo, lo_bits).to(device)
    v_cb_hi = compute_codebook(d_hi, hi_bits).to(device)

    # Random data
    q = torch.randn(B, extend_len, H_q, D, device=device, dtype=torch.float32)
    k_raw = torch.randn(B, total_kv, H_kv, D, device=device, dtype=torch.float32)
    v_raw = torch.randn(B, total_kv, H_kv, D, device=device, dtype=torch.float32)

    # Flatten and quantize
    k_flat = k_raw.view(B * total_kv, H_kv, D)
    v_flat = v_raw.view(B * total_kv, H_kv, D)

    k_lo_p, k_lo_n, k_hi_p, k_hi_n = split_channel_mse_quantize(
        k_flat, lo_indices, hi_indices, hadamard_lo, hadamard_hi,
        k_cb_lo, k_cb_hi, lo_bits, hi_bits,
    )

    v_lo_p, v_lo_n, v_hi_p, v_hi_n = split_channel_mse_quantize(
        v_flat, lo_indices, hi_indices, hadamard_lo, hadamard_hi,
        v_cb_lo, v_cb_hi, lo_bits, hi_bits,
    )

    # === REFERENCE: dequant + standard attention ===
    k_deq = split_channel_mse_dequantize(
        k_lo_p, k_lo_n, k_hi_p, k_hi_n, lo_indices, hi_indices, restore_order,
        hadamard_lo, hadamard_hi,
        k_cb_lo, k_cb_hi, lo_bits, hi_bits, D,
    )
    v_deq = split_channel_mse_dequantize(
        v_lo_p, v_lo_n, v_hi_p, v_hi_n, lo_indices, hi_indices, restore_order,
        hadamard_lo, hadamard_hi,
        v_cb_lo, v_cb_hi, lo_bits, hi_bits, D,
    )

    k_deq = k_deq.view(B, total_kv, H_kv, D)
    v_deq = v_deq.view(B, total_kv, H_kv, D)
    k_expanded = k_deq.repeat_interleave(kv_group_num, dim=2)
    v_expanded = v_deq.repeat_interleave(kv_group_num, dim=2)

    sm_scale = 1.0 / math.sqrt(D)
    q_t = q.permute(0, 2, 1, 3)
    k_t = k_expanded.permute(0, 2, 1, 3)
    v_t = v_expanded.permute(0, 2, 1, 3)

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale
    causal_mask = torch.ones(extend_len, total_kv, dtype=torch.bool, device=device)
    for qi in range(extend_len):
        for ki in range(total_kv):
            if ki >= prefix_len and (ki - prefix_len) > qi:
                causal_mask[qi, ki] = False
    scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(weights, v_t).permute(0, 2, 1, 3)

    # === TRITON SPLIT KERNEL ===
    q_flat = q.view(B * extend_len, H_q, D)

    # Split queries by channel
    q_lo = q_flat.index_select(-1, lo_indices.to(device))
    q_hi = q_flat.index_select(-1, hi_indices.to(device))

    q_rot_lo = hadamard_lo.forward(q_lo)
    q_rot_hi = hadamard_hi.forward(q_hi)

    total_q = B * extend_len
    qo_indptr = torch.tensor(
        [i * extend_len for i in range(B + 1)], dtype=torch.int32, device=device
    )
    kv_indptr = torch.tensor(
        [i * total_kv for i in range(B + 1)], dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(B * total_kv, dtype=torch.int64, device=device)
    prefix_lens_t = torch.full((B,), prefix_len, dtype=torch.int32, device=device)

    padded_d_lo = hadamard_lo.padded_dim
    padded_d_hi = hadamard_hi.padded_dim
    head_dim = padded_d_lo + padded_d_hi
    o_rot_split = torch.empty(total_q, H_q, head_dim, dtype=torch.float32, device=device)

    turboquant_extend_attention_fwd_split(
        q_rot_lo, q_rot_hi,
        k_lo_p, k_lo_n, k_hi_p, k_hi_n,
        v_lo_p, v_lo_n, v_hi_p, v_hi_n,
        k_cb_lo, k_cb_hi, v_cb_lo, v_cb_hi,
        o_rot_split,
        qo_indptr, kv_indptr, kv_indices, prefix_lens_t,
        max_extend_len=extend_len,
        sm_scale=sm_scale,
        mse_bits_lo=lo_bits,
        mse_bits_hi=hi_bits,
        v_bits_lo=lo_bits,
        v_bits_hi=hi_bits,
        d_lo=padded_d_lo,
        d_hi=padded_d_hi,
        head_dim=head_dim,
        is_causal=True,
    )

    # Inverse rotate per group, reassemble
    o_lo = hadamard_lo.inverse(o_rot_split[..., :padded_d_lo])
    o_hi = hadamard_hi.inverse(o_rot_split[..., padded_d_lo:])
    output_split = torch.cat([o_lo, o_hi], dim=-1)
    kernel_out = output_split.index_select(-1, restore_order.to(device))
    kernel_out = kernel_out.view(B, extend_len, H_q, D)

    sim = cosine_sim(ref_out, kernel_out)
    assert sim > 0.85, f"Split extend vs reference cosine sim {sim:.4f} < 0.85"
    return f"cosine_sim={sim:.4f}"


# ---------------------------------------------------------------------------
# Test: Long-sequence scaling — decode and extend at varying kv_len
# ---------------------------------------------------------------------------

def test_long_sequence_decode():
    """Decode attention at varying kv_len for Qwen2.5-3B (D=128) and Qwen3.5-9B (D=256).

    Tests whether cosine similarity degrades at long context.
    This is the key diagnostic for the 0% accuracy bug on Qwen2.5-3B.
    """
    from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
        turboquant_decode_attention_fwd,
    )

    device = "cuda"
    bits = 3
    seed = 42

    configs = [
        # (label, H_q, H_kv, D, kv_len)
        ("Qwen2.5-3B_kv48",   16, 2, 128, 48),
        ("Qwen2.5-3B_kv512",  16, 2, 128, 512),
        ("Qwen2.5-3B_kv2048", 16, 2, 128, 2048),
        ("Qwen2.5-3B_kv4096", 16, 2, 128, 4096),
        ("Qwen3.5-9B_kv48",   32, 4, 256, 48),
        ("Qwen3.5-9B_kv512",  32, 4, 256, 512),
        ("Qwen3.5-9B_kv2048", 32, 4, 256, 2048),
        ("Qwen3.5-9B_kv4096", 32, 4, 256, 4096),
    ]

    results = []
    for label, H_q, H_kv, D, kv_len in configs:
        torch.manual_seed(42)
        B = 1

        hadamard = HadamardTransform(D, seed, device)
        k_cb = compute_codebook(D, bits).to(device)
        v_cb = compute_codebook(D, bits).to(device)

        q = torch.randn(B, H_q, D, device=device, dtype=torch.float32)
        k_raw = torch.randn(kv_len, H_kv, D, device=device, dtype=torch.float32)
        v_raw = torch.randn(kv_len, H_kv, D, device=device, dtype=torch.float32)

        # Quantize (both MSE)
        k_packed, k_norms = mse_quantize(k_raw, hadamard, k_cb, bits)
        v_packed, v_norms = mse_quantize(v_raw, hadamard, v_cb, bits)

        # === REFERENCE: dequant + matmul ===
        k_deq = mse_dequantize(k_packed, k_norms, hadamard, k_cb, bits, D)
        v_deq = mse_dequantize(v_packed, v_norms, hadamard, v_cb, bits, D)

        kv_group_num = H_q // H_kv
        k_expanded = k_deq.repeat_interleave(kv_group_num, dim=1)
        v_expanded = v_deq.repeat_interleave(kv_group_num, dim=1)

        sm_scale = 1.0 / math.sqrt(D)
        scores = torch.einsum("bhd,shd->bhs", q, k_expanded) * sm_scale
        weights = torch.softmax(scores, dim=-1)
        ref_out = torch.einsum("bhs,shd->bhd", weights, v_expanded)

        # === FUSED TRITON KERNEL ===
        q_rot = hadamard.forward(q)

        kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
        kv_indices = torch.arange(kv_len, dtype=torch.int32, device=device)
        max_kv_splits = max(8, (kv_len + 255) // 256)
        num_kv_splits = torch.tensor([max_kv_splits], dtype=torch.int32, device=device)

        BLOCK_DV = triton.next_power_of_2(hadamard.padded_dim)
        attn_logits = torch.empty(B, H_q, max_kv_splits, BLOCK_DV, dtype=torch.float32, device=device)
        attn_lse = torch.empty(B, H_q, max_kv_splits, dtype=torch.float32, device=device)
        o_rot = torch.empty(B, H_q, hadamard.padded_dim, dtype=torch.float32, device=device)

        turboquant_decode_attention_fwd(
            q_rot,
            k_packed, k_norms,
            v_packed, v_norms,
            k_cb, v_cb,
            o_rot,
            kv_indptr, kv_indices, num_kv_splits,
            max_kv_splits, sm_scale,
            bits, bits, hadamard.padded_dim,
            attn_logits, attn_lse,
        )
        # Apply inverse rotation externally (QR dense matmul)
        kernel_out = hadamard.inverse(o_rot)

        sim = cosine_sim(ref_out, kernel_out)
        results.append((label, sim))
        print(f"    {label}: cosine_sim={sim:.6f}")

        # Free GPU memory
        del q, k_raw, v_raw, k_packed, k_norms
        del v_packed, v_norms, k_deq, v_deq, k_expanded, v_expanded
        del attn_logits, attn_lse, o_rot
        torch.cuda.empty_cache()

    # Report and assert
    summary_parts = []
    for label, sim in results:
        summary_parts.append(f"{label}={sim:.4f}")
        assert sim > 0.85, f"{label}: cosine_sim={sim:.6f} < 0.85 — QUALITY DEGRADATION"

    return " | ".join(summary_parts)


def test_long_sequence_extend():
    """Extend attention at varying total_kv for Qwen2.5-3B (D=128) and Qwen3.5-9B (D=256).

    Simulates prefill of long prompts (NIAH-like).
    """
    from sglang.srt.layers.attention.triton_ops.turboquant_extend_attention import (
        turboquant_extend_attention_fwd,
    )

    device = "cuda"
    bits = 3
    seed = 42

    configs = [
        # (label, H_q, H_kv, D, prefix_len, extend_len)
        ("Qwen2.5-3B_ext48",   16, 2, 128, 0,    48),
        ("Qwen2.5-3B_ext512",  16, 2, 128, 256,  256),
        ("Qwen2.5-3B_ext2048", 16, 2, 128, 1792, 256),
        ("Qwen2.5-3B_ext4096", 16, 2, 128, 3840, 256),
        ("Qwen3.5-9B_ext48",   32, 4, 256, 0,    48),
        ("Qwen3.5-9B_ext512",  32, 4, 256, 256,  256),
        ("Qwen3.5-9B_ext2048", 32, 4, 256, 1792, 256),
        ("Qwen3.5-9B_ext4096", 32, 4, 256, 3840, 256),
    ]

    results = []
    for label, H_q, H_kv, D, prefix_len, extend_len in configs:
        torch.manual_seed(42)
        B = 1
        total_kv = prefix_len + extend_len
        kv_group_num = H_q // H_kv

        hadamard = HadamardTransform(D, seed, device)
        k_cb = compute_codebook(D, bits).to(device)
        v_cb = compute_codebook(D, bits).to(device)

        q = torch.randn(B, extend_len, H_q, D, device=device, dtype=torch.float32)
        k_raw = torch.randn(B, total_kv, H_kv, D, device=device, dtype=torch.float32)
        v_raw = torch.randn(B, total_kv, H_kv, D, device=device, dtype=torch.float32)

        # Quantize
        k_flat = k_raw.view(B * total_kv, H_kv, D)
        v_flat = v_raw.view(B * total_kv, H_kv, D)
        k_packed, k_norms = mse_quantize(k_flat, hadamard, k_cb, bits)
        v_packed, v_norms = mse_quantize(v_flat, hadamard, v_cb, bits)

        # === REFERENCE ===
        k_deq = mse_dequantize(k_packed, k_norms, hadamard, k_cb, bits, D)
        v_deq = mse_dequantize(v_packed, v_norms, hadamard, v_cb, bits, D)
        k_deq = k_deq.view(B, total_kv, H_kv, D)
        v_deq = v_deq.view(B, total_kv, H_kv, D)

        k_expanded = k_deq.repeat_interleave(kv_group_num, dim=2)
        v_expanded = v_deq.repeat_interleave(kv_group_num, dim=2)

        sm_scale = 1.0 / math.sqrt(D)
        q_t = q.permute(0, 2, 1, 3)
        k_t = k_expanded.permute(0, 2, 1, 3)
        v_t = v_expanded.permute(0, 2, 1, 3)

        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale

        causal_mask = torch.ones(extend_len, total_kv, dtype=torch.bool, device=device)
        for qi in range(extend_len):
            for ki in range(total_kv):
                if ki >= prefix_len and (ki - prefix_len) > qi:
                    causal_mask[qi, ki] = False
        scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        ref_out = torch.matmul(weights, v_t).permute(0, 2, 1, 3)

        # === TRITON KERNEL ===
        q_flat_k = q.view(B * extend_len, H_q, D)
        q_rot = hadamard.forward(q_flat_k)

        total_q = B * extend_len
        total_kv_tokens = B * total_kv
        qo_indptr = torch.tensor(
            [i * extend_len for i in range(B + 1)], dtype=torch.int32, device=device
        )
        kv_indptr = torch.tensor(
            [i * total_kv for i in range(B + 1)], dtype=torch.int32, device=device
        )
        kv_indices = torch.arange(total_kv_tokens, dtype=torch.int64, device=device)
        prefix_lens_t = torch.full((B,), prefix_len, dtype=torch.int32, device=device)

        o_rot = torch.empty(total_q, H_q, hadamard.padded_dim, dtype=torch.float32, device=device)

        turboquant_extend_attention_fwd(
            q_rot,
            k_packed, k_norms,
            v_packed, v_norms,
            k_cb, v_cb,
            o_rot,
            qo_indptr, kv_indptr, kv_indices, prefix_lens_t,
            max_extend_len=extend_len,
            sm_scale=sm_scale,
            mse_bits=bits,
            v_bits=bits,
            head_dim=hadamard.padded_dim,
            is_causal=True,
        )

        kernel_out = hadamard.inverse(o_rot).view(B, extend_len, H_q, D)

        sim = cosine_sim(ref_out, kernel_out)
        results.append((label, sim))
        print(f"    {label}: cosine_sim={sim:.6f}")

        # Free GPU memory
        del q, k_raw, v_raw, k_flat, v_flat, k_packed, k_norms
        del v_packed, v_norms, k_deq, v_deq, k_expanded, v_expanded
        del scores, weights, o_rot, q_rot
        torch.cuda.empty_cache()

    summary_parts = []
    for label, sim in results:
        summary_parts.append(f"{label}={sim:.4f}")
        assert sim > 0.85, f"{label}: cosine_sim={sim:.6f} < 0.85 — QUALITY DEGRADATION"

    return " | ".join(summary_parts)


# ---------------------------------------------------------------------------
# Test: rotated-space accumulation equivalence (Phase H verification)
# ---------------------------------------------------------------------------

def test_rotated_space_equivalence():
    """Verify rotated-space V accumulation equals standard per-token dequant.

    Phase H optimization: sum(a_t * H_inv(cb_t)) = H_inv(sum(a_t * cb_t))
    because H_inv is linear.
    """
    device = "cuda"
    torch.manual_seed(42)

    D = 128
    T = 32  # sequence length
    bits = 3
    seed = 42

    hadamard = HadamardTransform(D, seed, device)
    v_cb = compute_codebook(D, bits).to(device)

    # Generate random value vectors, quantize
    v_raw = torch.randn(T, D, device=device, dtype=torch.float32)
    v_packed, v_norms = mse_quantize(v_raw.unsqueeze(1), hadamard, v_cb, bits)

    # Random attention weights (softmax-like)
    alpha = torch.softmax(torch.randn(T, device=device), dim=0)

    # Method 1 (standard): dequant each V, then weighted sum
    v_deq = mse_dequantize(v_packed, v_norms, hadamard, v_cb, bits, D)
    v_deq = v_deq.squeeze(1)  # [T, D]
    out_standard = (alpha.unsqueeze(-1) * v_deq).sum(dim=0)  # [D]

    # Method 2 (rotated): accumulate codebook values, then one inverse Hadamard
    # Unpack indices manually
    v_indices = unpack_bits(v_packed.squeeze(1), bits, hadamard.padded_dim)  # [T, padded_dim]
    v_norms_flat = v_norms.squeeze(1).squeeze(-1).float()  # [T]

    rot_sum = torch.zeros(hadamard.padded_dim, device=device, dtype=torch.float32)
    for t in range(T):
        cb_vals = v_cb[v_indices[t].long()]  # [padded_dim]
        rot_sum += alpha[t] * v_norms_flat[t] * cb_vals
    out_rotated = hadamard.inverse(rot_sum.unsqueeze(0)).squeeze(0)  # [D]
    out_rotated = out_rotated[:D]

    max_diff = (out_standard - out_rotated).abs().max().item()
    sim = cosine_sim(out_standard, out_rotated)
    assert max_diff < 1e-3, f"Rotated vs standard max diff {max_diff:.2e} >= 1e-3"
    assert sim > 0.999, f"Rotated vs standard cosine sim {sim:.6f} < 0.999"
    return f"max_diff={max_diff:.2e}, cosine_sim={sim:.6f}"


# ---------------------------------------------------------------------------
# Test: gather_dequant matches full dequant at indexed positions
# ---------------------------------------------------------------------------

def test_gather_dequant():
    """gather_dequant_key/value should match get_key_buffer()[indices] exactly."""
    from sglang.srt.mem_cache.turboquant_pool import TurboQuantTokenToKVPool

    device = "cuda"
    torch.manual_seed(42)

    for bits, label in [(3, "integer_3bit"), (3.5, "split_3.5bit")]:
        pool = TurboQuantTokenToKVPool(
            size=256, page_size=1, dtype=torch.bfloat16,
            head_num=2, head_dim=128, layer_num=2,
            device=device, enable_memory_saver=False,
            turboquant_bits=bits, turboquant_seed=42,
            use_workspace=False,
        )

        # Populate some positions with random data
        N = 64
        loc = torch.arange(N, device=device)
        k_data = torch.randn(N, 2, 128, device=device, dtype=torch.bfloat16)
        v_data = torch.randn(N, 2, 128, device=device, dtype=torch.bfloat16)

        # Use a mock layer object for set_kv_buffer
        class MockLayer:
            layer_id = 0
            k_scale = None
            v_scale = None
        pool.set_kv_buffer(MockLayer(), loc, k_data, v_data)

        # Gather at a subset of positions (with some duplicates)
        indices = torch.tensor([0, 5, 10, 20, 5, 63], device=device)

        # Full dequant then index
        full_k = pool.get_key_buffer(0)
        full_v = pool.get_value_buffer(0)
        expected_k = full_k[indices]
        expected_v = full_v[indices]

        # Gather-dequant
        gathered_k = pool.gather_dequant_key(0, indices)
        gathered_v = pool.gather_dequant_value(0, indices)

        k_sim = cosine_sim(expected_k, gathered_k)
        v_sim = cosine_sim(expected_v, gathered_v)
        k_max_diff = (expected_k.float() - gathered_k.float()).abs().max().item()
        v_max_diff = (expected_v.float() - gathered_v.float()).abs().max().item()

        assert k_sim > 0.9999, f"{label} key: sim={k_sim:.6f}"
        assert v_sim > 0.9999, f"{label} val: sim={v_sim:.6f}"
        assert k_max_diff < 1e-3, f"{label} key: max_diff={k_max_diff:.2e}"
        assert v_max_diff < 1e-3, f"{label} val: max_diff={v_max_diff:.2e}"

    return "integer and split paths match full dequant"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_integer_4bit_h_kv2():
    """Integer 4-bit decode kernel at H_kv=2 (Qwen2.5-3B config). Regression test."""
    from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
        turboquant_decode_attention_fwd,
    )

    device = "cuda"
    torch.manual_seed(42)

    B, H_q, H_kv, D = 1, 16, 2, 128
    seq_len = 64
    bits = 4
    seed = 42
    kv_group_num = H_q // H_kv

    hadamard_k = HadamardTransform(D, seed, device)
    hadamard_v = HadamardTransform(D, seed + 500, device)
    k_cb = compute_codebook(D, bits).to(device)
    v_cb = compute_codebook(D, bits).to(device)

    q = torch.randn(B, H_q, D, device=device, dtype=torch.float32)
    k_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)
    v_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)

    k_packed, k_norms = mse_quantize(k_raw, hadamard_k, k_cb, bits)
    v_packed, v_norms = mse_quantize(v_raw, hadamard_v, v_cb, bits)

    # Reference: dequant + standard attention
    k_deq = mse_dequantize(k_packed, k_norms, hadamard_k, k_cb, bits, D)
    v_deq = mse_dequantize(v_packed, v_norms, hadamard_v, v_cb, bits, D)
    k_expanded = k_deq.repeat_interleave(kv_group_num, dim=1)
    v_expanded = v_deq.repeat_interleave(kv_group_num, dim=1)
    sm_scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("bhd,shd->bhs", q, k_expanded) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("bhs,shd->bhd", weights, v_expanded)

    # Triton kernel
    q_rot = hadamard_k.forward(q)
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device=device)
    max_kv_splits = 8
    num_kv_splits = torch.tensor([max_kv_splits], dtype=torch.int32, device=device)
    BLOCK_DV = triton.next_power_of_2(hadamard_k.padded_dim)
    attn_logits = torch.empty(B, H_q, max_kv_splits, BLOCK_DV, dtype=torch.float32, device=device)
    attn_lse = attn_logits[:, :, :, 0].contiguous()
    o_rot = torch.empty(B, H_q, hadamard_k.padded_dim, dtype=torch.float32, device=device)

    turboquant_decode_attention_fwd(
        q_rot, k_packed, k_norms, v_packed, v_norms,
        k_cb, v_cb, o_rot,
        kv_indptr, kv_indices, num_kv_splits,
        max_kv_splits, sm_scale, bits, bits, hadamard_k.padded_dim,
        attn_logits, attn_lse,
    )

    kernel_out = hadamard_v.inverse(o_rot)
    sim = cosine_sim(ref_out, kernel_out)
    assert sim > 0.95, f"Integer 4-bit H_kv=2 cosine sim {sim:.4f} < 0.95"
    return f"cosine_sim={sim:.4f}"


def test_split_kernel_h_kv2():
    """Split 3.5-bit decode kernel at H_kv=2 (Qwen2.5-3B config). Regression test."""
    from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
        turboquant_decode_attention_fwd_split,
    )

    device = "cuda"
    torch.manual_seed(42)

    B, H_q, H_kv, D = 1, 16, 2, 128
    seq_len = 64
    avg_bits = 3.5
    seed = 42
    kv_group_num = H_q // H_kv

    lo_bits = math.floor(avg_bits)
    hi_bits = math.ceil(avg_bits)
    lo_indices, hi_indices = select_outlier_indices(D, avg_bits)
    restore_order = torch.argsort(torch.cat([lo_indices, hi_indices]))
    d_lo, d_hi = lo_indices.shape[0], hi_indices.shape[0]

    hadamard_lo = HadamardTransform(d_lo, seed, device)
    hadamard_hi = HadamardTransform(d_hi, seed + 97, device)
    v_hadamard_lo = HadamardTransform(d_lo, seed + 500, device)
    v_hadamard_hi = HadamardTransform(d_hi, seed + 597, device)

    k_cb_lo = compute_codebook(d_lo, lo_bits).to(device)
    k_cb_hi = compute_codebook(d_hi, hi_bits).to(device)
    v_cb_lo = compute_codebook(d_lo, lo_bits).to(device)
    v_cb_hi = compute_codebook(d_hi, hi_bits).to(device)

    q = torch.randn(B, H_q, D, device=device, dtype=torch.float32)
    k_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)
    v_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)

    k_lo_p, k_lo_n, k_hi_p, k_hi_n = split_channel_mse_quantize(
        k_raw, lo_indices, hi_indices, hadamard_lo, hadamard_hi,
        k_cb_lo, k_cb_hi, lo_bits, hi_bits,
    )
    v_lo_p, v_lo_n, v_hi_p, v_hi_n = split_channel_mse_quantize(
        v_raw, lo_indices, hi_indices, v_hadamard_lo, v_hadamard_hi,
        v_cb_lo, v_cb_hi, lo_bits, hi_bits,
    )

    # Reference: dequant + standard attention
    k_deq = split_channel_mse_dequantize(
        k_lo_p, k_lo_n, k_hi_p, k_hi_n,
        lo_indices, hi_indices, restore_order,
        hadamard_lo, hadamard_hi,
        k_cb_lo, k_cb_hi, lo_bits, hi_bits, D,
    )
    v_deq = split_channel_mse_dequantize(
        v_lo_p, v_lo_n, v_hi_p, v_hi_n,
        lo_indices, hi_indices, restore_order,
        v_hadamard_lo, v_hadamard_hi,
        v_cb_lo, v_cb_hi, lo_bits, hi_bits, D,
    )
    k_expanded = k_deq.repeat_interleave(kv_group_num, dim=1)
    v_expanded = v_deq.repeat_interleave(kv_group_num, dim=1)
    sm_scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("bhd,shd->bhs", q, k_expanded) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("bhs,shd->bhd", weights, v_expanded)

    # Triton split kernel
    q_lo = q.index_select(-1, lo_indices.to(device))
    q_hi = q.index_select(-1, hi_indices.to(device))
    q_rot_lo = hadamard_lo.forward(q_lo)
    q_rot_hi = hadamard_hi.forward(q_hi)

    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device=device)
    max_kv_splits = 8
    num_kv_splits = torch.tensor([max_kv_splits], dtype=torch.int32, device=device)

    padded_d_lo = hadamard_lo.padded_dim
    padded_d_hi = hadamard_hi.padded_dim
    head_dim = padded_d_lo + padded_d_hi
    BLOCK_DV = triton.next_power_of_2(head_dim)
    attn_logits = torch.empty(B, H_q, max_kv_splits, BLOCK_DV, dtype=torch.float32, device=device)
    attn_lse = attn_logits[:, :, :, 0].contiguous()
    o_rot_split = torch.empty(B, H_q, head_dim, dtype=torch.float32, device=device)

    turboquant_decode_attention_fwd_split(
        q_rot_lo, q_rot_hi,
        k_lo_p, k_lo_n, k_hi_p, k_hi_n,
        v_lo_p, v_lo_n, v_hi_p, v_hi_n,
        k_cb_lo, k_cb_hi, v_cb_lo, v_cb_hi,
        o_rot_split,
        kv_indptr, kv_indices, num_kv_splits,
        max_kv_splits, sm_scale,
        lo_bits, hi_bits, lo_bits, hi_bits,
        padded_d_lo, padded_d_hi, head_dim,
        attn_logits, attn_lse,
    )

    o_lo = v_hadamard_lo.inverse(o_rot_split[..., :padded_d_lo])
    o_hi = v_hadamard_hi.inverse(o_rot_split[..., padded_d_lo:])
    output_split = torch.cat([o_lo, o_hi], dim=-1)
    kernel_out = output_split.index_select(-1, restore_order.to(device))

    sim = cosine_sim(ref_out, kernel_out)
    assert sim > 0.95, f"Split H_kv=2 cosine sim {sim:.4f} < 0.95"
    return f"cosine_sim={sim:.4f}"


ALL_TESTS = [
    test_bit_packing_roundtrip,
    test_select_outlier_indices,
    test_mse_roundtrip,
    test_integer_kernel_vs_reference,
    test_split_kernel_vs_reference,
    # Extend kernel tests disabled — the TQ extend kernel is not used in production
    # (backend uses selective gather-dequant + parent Triton extend).
    # test_integer_extend_vs_reference,
    # test_extend_causal_mask,
    # test_extend_variable_batch,
    # test_split_extend_vs_reference,
    test_rotated_space_equivalence,
    test_long_sequence_decode,
    # test_long_sequence_extend,
    test_gather_dequant,
    test_integer_4bit_h_kv2,
    test_split_kernel_h_kv2,
]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping kernel tests")
        sys.exit(1)

    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            summary = test_fn()
            print(f"  PASS  {name}: {summary}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
