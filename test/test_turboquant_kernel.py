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
    prod_dequantize,
    prod_quantize,
    select_outlier_indices,
    split_channel_mse_dequantize,
    split_channel_mse_quantize,
    split_channel_prod_dequantize,
    split_channel_prod_quantize,
    unpack_bits,
)
from sglang.srt.layers.quantization.turboquant.rotation import (
    projection_matrix,
    rotation_matrix,
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
    pi = rotation_matrix(D, seed).to(device)
    pi_t = pi.T.contiguous()
    cb = compute_codebook(D, bits).to(device)

    packed, norms = mse_quantize(x, pi, cb, bits)
    recon = mse_dequantize(packed, norms, pi_t, cb, bits, D)

    sim = cosine_sim(x, recon)
    assert sim > 0.98, f"MSE roundtrip cosine sim {sim:.4f} < 0.98"
    return f"cosine_sim={sim:.4f}"


# ---------------------------------------------------------------------------
# Test: Prod quantize/dequantize roundtrip
# ---------------------------------------------------------------------------

def test_prod_roundtrip():
    """Prod quantize -> dequantize should have reasonable cosine similarity."""
    device = "cuda"
    N, H, D = 8, 4, 256
    bits = 3
    seed = 42

    x = torch.randn(N, H, D, device=device)
    pi = rotation_matrix(D, seed).to(device)
    pi_t = pi.T.contiguous()
    s = projection_matrix(D, seed).to(device)
    mse_bits = max(bits - 1, 0)
    cb = compute_codebook(D, mse_bits).to(device)

    mse_p, qjl_p, norms, res_norms = prod_quantize(x, pi, s, cb, bits)
    recon = prod_dequantize(mse_p, qjl_p, norms, res_norms, pi_t, s, cb, bits, D)

    sim = cosine_sim(x, recon)
    assert sim > 0.90, f"Prod roundtrip cosine sim {sim:.4f} < 0.90"
    return f"cosine_sim={sim:.4f}"


# ---------------------------------------------------------------------------
# Test: Split-channel roundtrip
# ---------------------------------------------------------------------------

def test_split_channel_roundtrip():
    """Split-channel quantize -> dequantize roundtrip for 3.5-bit."""
    device = "cuda"
    N, H, D = 8, 4, 256
    avg_bits = 3.5
    seed = 42

    lo_bits = math.floor(avg_bits)  # 3
    hi_bits = math.ceil(avg_bits)   # 4
    lo_indices, hi_indices = select_outlier_indices(D, avg_bits)
    restore_order = torch.argsort(torch.cat([lo_indices, hi_indices]))
    d_lo, d_hi = lo_indices.shape[0], hi_indices.shape[0]

    pi_lo = rotation_matrix(d_lo, seed).to(device)
    pi_hi = rotation_matrix(d_hi, seed + 97).to(device)
    s_lo = projection_matrix(d_lo, seed).to(device)
    s_hi = projection_matrix(d_hi, seed + 97).to(device)

    k_cb_lo = compute_codebook(d_lo, max(lo_bits - 1, 0)).to(device)
    k_cb_hi = compute_codebook(d_hi, max(hi_bits - 1, 0)).to(device)
    v_cb_lo = compute_codebook(d_lo, lo_bits).to(device)
    v_cb_hi = compute_codebook(d_hi, hi_bits).to(device)

    x = torch.randn(N, H, D, device=device)

    # Key (prod) roundtrip
    results = split_channel_prod_quantize(
        x, lo_indices, hi_indices, pi_lo, pi_hi, s_lo, s_hi,
        k_cb_lo, k_cb_hi, lo_bits, hi_bits,
    )
    k_recon = split_channel_prod_dequantize(
        *results, lo_indices, hi_indices, restore_order,
        pi_lo.T.contiguous(), pi_hi.T.contiguous(),
        s_lo, s_hi, k_cb_lo, k_cb_hi, lo_bits, hi_bits, D,
    )
    k_sim = cosine_sim(x, k_recon)

    # Value (MSE) roundtrip
    lo_p, lo_n, hi_p, hi_n = split_channel_mse_quantize(
        x, lo_indices, hi_indices, pi_lo, pi_hi,
        v_cb_lo, v_cb_hi, lo_bits, hi_bits,
    )
    v_recon = split_channel_mse_dequantize(
        lo_p, lo_n, hi_p, hi_n, lo_indices, hi_indices, restore_order,
        pi_lo.T.contiguous(), pi_hi.T.contiguous(),
        v_cb_lo, v_cb_hi, lo_bits, hi_bits, D,
    )
    v_sim = cosine_sim(x, v_recon)

    assert k_sim > 0.88, f"Split prod roundtrip cosine sim {k_sim:.4f} < 0.88"
    assert v_sim > 0.97, f"Split MSE roundtrip cosine sim {v_sim:.4f} < 0.97"
    return f"prod={k_sim:.4f}, mse={v_sim:.4f}"


# ---------------------------------------------------------------------------
# Test: Integer-bit Triton kernel vs reference
# ---------------------------------------------------------------------------

def test_integer_kernel_vs_reference():
    """Compare Triton decode kernel output against dequant+matmul reference."""
    from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
        turboquant_decode_attention_fwd,
    )

    device = "cuda"
    torch.manual_seed(42)

    B, H_q, H_kv, D = 1, 8, 4, 256
    seq_len = 64
    bits = 3
    mse_bits = max(bits - 1, 0)
    seed = 42
    kv_group_num = H_q // H_kv

    # Setup codebooks, matrices
    pi = rotation_matrix(D, seed).to(device)
    pi_t = pi.T.contiguous()
    s = projection_matrix(D, seed).to(device)
    k_cb = compute_codebook(D, mse_bits).to(device)
    v_cb = compute_codebook(D, bits).to(device)

    # Random Q, K, V
    q = torch.randn(B, H_q, D, device=device, dtype=torch.float32)
    k_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)
    v_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)

    # Quantize K (prod) and V (mse)
    k_mse_p, k_qjl_p, k_norms, k_res_norms = prod_quantize(k_raw, pi, s, k_cb, bits)
    v_packed, v_norms = mse_quantize(v_raw, pi, v_cb, bits)

    # === REFERENCE: dequant + standard attention ===
    k_deq = prod_dequantize(k_mse_p, k_qjl_p, k_norms, k_res_norms, pi_t, s, k_cb, bits, D)
    v_deq = mse_dequantize(v_packed, v_norms, pi_t, v_cb, bits, D)

    # GQA expand: [seq_len, H_kv, D] -> [seq_len, H_q, D]
    k_expanded = k_deq.repeat_interleave(kv_group_num, dim=1)
    v_expanded = v_deq.repeat_interleave(kv_group_num, dim=1)

    sm_scale = 1.0 / math.sqrt(D)
    # Q: [B, H_q, D], K: [seq_len, H_q, D] -> scores: [B, H_q, seq_len]
    scores = torch.einsum("bhd,shd->bhs", q, k_expanded) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    # weights: [B, H_q, seq_len], V: [seq_len, H_q, D] -> ref_out: [B, H_q, D]
    ref_out = torch.einsum("bhs,shd->bhd", weights, v_expanded)

    # === TRITON KERNEL ===
    # Pre-rotate and project queries
    q_rot = torch.matmul(q, pi_t)
    q_proj = torch.matmul(q, s.T)

    # Page table: simple contiguous
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device=device)

    max_kv_splits = 8
    num_kv_splits = torch.tensor([max_kv_splits], dtype=torch.int32, device=device)

    # Scratch buffers
    BLOCK_DV = triton.next_power_of_2(D)
    attn_logits = torch.empty(B, H_q, max_kv_splits, BLOCK_DV, dtype=torch.float32, device=device)
    attn_lse = attn_logits[:, :, :, 0].contiguous()  # shares first element per split

    o_rot = torch.empty(B, H_q, D, dtype=torch.float32, device=device)

    qjl_scale = math.sqrt(math.pi / 2.0) / D

    # Reshape packed buffers: [seq_len, H_kv, pw] -> pool-style
    turboquant_decode_attention_fwd(
        q_rot, q_proj,
        k_mse_p, k_qjl_p, k_norms, k_res_norms,
        v_packed, v_norms,
        k_cb, v_cb,
        o_rot,
        kv_indptr, kv_indices, num_kv_splits,
        max_kv_splits, sm_scale, qjl_scale,
        mse_bits, bits, D,
        attn_logits, attn_lse,
    )

    # Inverse rotate
    kernel_out = torch.matmul(o_rot, pi)

    sim = cosine_sim(ref_out, kernel_out)
    assert sim > 0.95, f"Integer kernel vs reference cosine sim {sim:.4f} < 0.95"
    return f"cosine_sim={sim:.4f}"


# ---------------------------------------------------------------------------
# Test: Split-channel Triton kernel vs reference
# ---------------------------------------------------------------------------

def test_split_kernel_vs_reference():
    """Compare split-channel Triton decode kernel against dequant+matmul reference."""
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

    pi_lo = rotation_matrix(d_lo, seed).to(device)
    pi_hi = rotation_matrix(d_hi, seed + 97).to(device)
    s_lo = projection_matrix(d_lo, seed).to(device)
    s_hi = projection_matrix(d_hi, seed + 97).to(device)

    k_cb_lo = compute_codebook(d_lo, max(lo_bits - 1, 0)).to(device)
    k_cb_hi = compute_codebook(d_hi, max(hi_bits - 1, 0)).to(device)
    v_cb_lo = compute_codebook(d_lo, lo_bits).to(device)
    v_cb_hi = compute_codebook(d_hi, hi_bits).to(device)

    # Random Q, K, V
    q = torch.randn(B, H_q, D, device=device, dtype=torch.float32)
    k_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)
    v_raw = torch.randn(seq_len, H_kv, D, device=device, dtype=torch.float32)

    # Quantize K (split prod) and V (split mse)
    k_results = split_channel_prod_quantize(
        k_raw, lo_indices, hi_indices, pi_lo, pi_hi, s_lo, s_hi,
        k_cb_lo, k_cb_hi, lo_bits, hi_bits,
    )
    lo_mse_p, lo_qjl_p, lo_n, lo_rn, hi_mse_p, hi_qjl_p, hi_n, hi_rn = k_results

    lo_vp, lo_vn, hi_vp, hi_vn = split_channel_mse_quantize(
        v_raw, lo_indices, hi_indices, pi_lo, pi_hi,
        v_cb_lo, v_cb_hi, lo_bits, hi_bits,
    )

    # === REFERENCE: dequant + standard attention ===
    k_deq = split_channel_prod_dequantize(
        *k_results, lo_indices, hi_indices, restore_order,
        pi_lo.T.contiguous(), pi_hi.T.contiguous(),
        s_lo, s_hi, k_cb_lo, k_cb_hi, lo_bits, hi_bits, D,
    )
    v_deq = split_channel_mse_dequantize(
        lo_vp, lo_vn, hi_vp, hi_vn, lo_indices, hi_indices, restore_order,
        pi_lo.T.contiguous(), pi_hi.T.contiguous(),
        v_cb_lo, v_cb_hi, lo_bits, hi_bits, D,
    )

    k_expanded = k_deq.repeat_interleave(kv_group_num, dim=1)
    v_expanded = v_deq.repeat_interleave(kv_group_num, dim=1)

    sm_scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("bhd,shd->bhs", q, k_expanded) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("bhs,shd->bhd", weights, v_expanded)

    # === TRITON SPLIT KERNEL ===
    # Split queries by channel
    q_lo = q.index_select(-1, lo_indices.to(device))
    q_hi = q.index_select(-1, hi_indices.to(device))

    q_rot_lo = torch.matmul(q_lo, pi_lo.T.contiguous())
    q_proj_lo = torch.matmul(q_lo, s_lo.T)
    q_rot_hi = torch.matmul(q_hi, pi_hi.T.contiguous())
    q_proj_hi = torch.matmul(q_hi, s_hi.T)

    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device=device)

    max_kv_splits = 8
    num_kv_splits = torch.tensor([max_kv_splits], dtype=torch.int32, device=device)

    head_dim = d_lo + d_hi
    BLOCK_DV = triton.next_power_of_2(head_dim)
    attn_logits = torch.empty(B, H_q, max_kv_splits, BLOCK_DV, dtype=torch.float32, device=device)
    attn_lse = attn_logits[:, :, :, 0].contiguous()

    o_rot_split = torch.empty(B, H_q, head_dim, dtype=torch.float32, device=device)

    qjl_scale_lo = math.sqrt(math.pi / 2.0) / d_lo
    qjl_scale_hi = math.sqrt(math.pi / 2.0) / d_hi
    k_lo_mse_bits = max(lo_bits - 1, 0)
    k_hi_mse_bits = max(hi_bits - 1, 0)

    turboquant_decode_attention_fwd_split(
        q_rot_lo, q_proj_lo, q_rot_hi, q_proj_hi,
        lo_mse_p, lo_qjl_p, lo_n, lo_rn,
        hi_mse_p, hi_qjl_p, hi_n, hi_rn,
        lo_vp, lo_vn, hi_vp, hi_vn,
        k_cb_lo, k_cb_hi, v_cb_lo, v_cb_hi,
        o_rot_split,
        kv_indptr, kv_indices, num_kv_splits,
        max_kv_splits, sm_scale, qjl_scale_lo, qjl_scale_hi,
        k_lo_mse_bits, k_hi_mse_bits,
        lo_bits, hi_bits,
        d_lo, d_hi, head_dim,
        attn_logits, attn_lse,
    )

    # Inverse rotate per group, reassemble
    o_lo = torch.matmul(o_rot_split[..., :d_lo], pi_lo)
    o_hi = torch.matmul(o_rot_split[..., d_lo:], pi_hi)
    output_split = torch.cat([o_lo, o_hi], dim=-1)
    kernel_out = output_split.index_select(-1, restore_order.to(device))

    sim = cosine_sim(ref_out, kernel_out)
    assert sim > 0.95, f"Split kernel vs reference cosine sim {sim:.4f} < 0.95"
    return f"cosine_sim={sim:.4f}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_bit_packing_roundtrip,
    test_select_outlier_indices,
    test_mse_roundtrip,
    test_prod_roundtrip,
    test_split_channel_roundtrip,
    test_integer_kernel_vs_reference,
    test_split_kernel_vs_reference,
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
