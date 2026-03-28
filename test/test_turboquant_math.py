"""
TurboQuant mathematical validation tests.

Validates codebook correctness against MLX reference, theoretical bounds
from the paper (arXiv:2504.19874), and multi-layer error accumulation.
No server needed. GPU required for tests 2-5.

Usage: python test/test_turboquant_math.py
"""

import math
import sys

import numpy as np
import torch

from sglang.srt.layers.quantization.turboquant.codebook import (
    _beta_pdf,
    compute_codebook,
)
from sglang.srt.layers.quantization.turboquant.quant_ops import (
    mse_dequantize,
    mse_quantize,
    prod_dequantize,
    prod_quantize,
    select_outlier_indices,
    split_channel_mse_dequantize,
    split_channel_mse_quantize,
    unpack_bits,
)
from sglang.srt.layers.quantization.turboquant.rotation import (
    HadamardTransform,
    projection_matrix,
)


def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def _per_vector_cosine_sim(a, b):
    """Mean per-vector cosine similarity. a, b: [..., D]."""
    a_f = a.float()
    b_f = b.float()
    dot = (a_f * b_f).sum(dim=-1)
    norm_a = a_f.norm(dim=-1).clamp(min=1e-8)
    norm_b = b_f.norm(dim=-1).clamp(min=1e-8)
    return (dot / (norm_a * norm_b)).mean().item()


# ---------------------------------------------------------------------------
# MLX codebook reference (numpy only, no mlx.core dependency)
# ---------------------------------------------------------------------------

def _mlx_codebook_reference(dim: int, bits: int) -> np.ndarray:
    """Replicate MLX's _codebook() using only numpy.

    This is a verbatim copy of mlx-vlm/mlx_vlm/turboquant.py lines 1706-1743
    but returns numpy instead of mx.array. Intentionally omits the SGLang
    1-bit analytical fast path to match MLX exactly.
    """
    if bits <= 0:
        return np.zeros(0, dtype=np.float32)
    levels = 1 << bits
    if dim <= 1:
        return np.linspace(-1.0, 1.0, levels, dtype=np.float32)

    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 32768, dtype=np.float32)
    # Use the same _beta_pdf — formula is identical in both implementations
    weights = _beta_pdf(grid, dim)

    cdf = np.cumsum(weights)
    quantiles = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    for _ in range(100):
        boundaries = np.empty(levels + 1, dtype=np.float32)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        new_centroids = centroids.copy()
        for i in range(levels):
            if i == levels - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            else:
                mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            bucket_weights = weights[mask]
            if bucket_weights.size == 0:
                continue
            total_weight = bucket_weights.sum()
            if total_weight > 0:
                new_centroids[i] = np.sum(bucket_weights * grid[mask]) / total_weight
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids.astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: Codebook cross-implementation comparison (CPU)
# ---------------------------------------------------------------------------

def test_codebook_cross_impl():
    """SGLang vs MLX codebook: match within 1e-4 for bits >= 2."""
    results = []
    for dim in [32, 64, 128, 256]:
        for bits in [2, 3, 4]:
            sglang_cb = compute_codebook(dim, bits).numpy()
            mlx_cb = _mlx_codebook_reference(dim, bits)
            max_diff = np.max(np.abs(sglang_cb - mlx_cb))
            results.append((dim, bits, max_diff))
            assert max_diff < 1e-4, (
                f"dim={dim}, bits={bits}: max diff {max_diff:.2e} >= 1e-4"
            )
    worst = max(r[2] for r in results)
    return f"12 configs, worst max_diff={worst:.2e}"


# ---------------------------------------------------------------------------
# Test 2: Codebook optimality vs paper bounds (GPU)
# ---------------------------------------------------------------------------

def test_codebook_optimality():
    """Empirical MSE within paper bounds (Theorem 1 upper, Theorem 3 lower)."""
    device = "cuda"
    N = 100_000
    dim = 128
    seed = 42

    torch.manual_seed(seed)
    # Generate random unit vectors
    raw = torch.randn(N, 1, dim, device=device)
    unit = raw / raw.norm(dim=-1, keepdim=True)

    results = []
    for bits in [2, 3, 4]:
        hadamard = HadamardTransform(dim, seed=seed, device=device)
        codebook = compute_codebook(dim, bits).to(device)

        # Quantize/dequantize roundtrip
        packed, norms = mse_quantize(unit, hadamard, codebook, bits)
        recon = mse_dequantize(packed, norms, hadamard, codebook, bits, dim)

        # MSE for unit vectors: E[||x - x_hat||^2]
        mse = ((unit - recon) ** 2).sum(dim=-1).mean().item()

        # Paper bounds (arXiv:2504.19874, Theorems 1 and 3)
        upper_bound = math.sqrt(3) * math.pi / 2 * (1.0 / 4**bits)
        lower_bound = 1.0 / 4**bits

        results.append((bits, mse, lower_bound, upper_bound))
        assert mse <= upper_bound * 1.1, (
            f"bits={bits}: MSE {mse:.6f} > upper bound {upper_bound:.6f} * 1.1"
        )
        assert mse >= lower_bound * 0.8, (
            f"bits={bits}: MSE {mse:.6f} < lower bound {lower_bound:.6f} * 0.8"
        )

    lines = []
    for bits, mse, lb, ub in results:
        lines.append(f"b={bits}: MSE={mse:.6f} (bounds [{lb:.6f}, {ub:.6f}])")
    return "; ".join(lines)


# ---------------------------------------------------------------------------
# Test 3: Cosine similarity sweep across dim and bits (GPU)
# ---------------------------------------------------------------------------

def test_cosine_sim_sweep():
    """Cosine sim improves with dim (Beta concentration) and bits."""
    device = "cuda"
    N = 1000
    seed = 42
    torch.manual_seed(seed)

    dims = [32, 64, 128, 256]
    bits_list = [2, 3, 4]
    results = {}  # (dim, bits) -> mean cosine sim

    for dim in dims:
        for bits in bits_list:
            x = torch.randn(N, 1, dim, device=device)  # Random vectors (non-unit)
            hadamard = HadamardTransform(dim, seed=seed, device=device)
            codebook = compute_codebook(dim, bits).to(device)

            packed, norms = mse_quantize(x, hadamard, codebook, bits)
            recon = mse_dequantize(packed, norms, hadamard, codebook, bits, dim)

            sim = _per_vector_cosine_sim(x, recon)
            results[(dim, bits)] = sim

    # Check monotonicity in bits (for fixed dim)
    for dim in dims:
        for i in range(len(bits_list) - 1):
            b_lo = bits_list[i]
            b_hi = bits_list[i + 1]
            assert results[(dim, b_hi)] >= results[(dim, b_lo)] - 0.002, (
                f"dim={dim}: sim@b={b_hi} ({results[(dim, b_hi)]:.4f}) < "
                f"sim@b={b_lo} ({results[(dim, b_lo)]:.4f}) - 0.002"
            )

    # Check monotonicity in dim (for fixed bits)
    for bits in bits_list:
        for i in range(len(dims) - 1):
            d_lo = dims[i]
            d_hi = dims[i + 1]
            assert results[(d_hi, bits)] >= results[(d_lo, bits)] - 0.01, (
                f"bits={bits}: sim@d={d_hi} ({results[(d_hi, bits)]:.4f}) < "
                f"sim@d={d_lo} ({results[(d_lo, bits)]:.4f}) - 0.01"
            )

    # Absolute thresholds
    assert results[(128, 3)] >= 0.97, f"dim=128/b=3: {results[(128, 3)]:.4f} < 0.97"
    assert results[(128, 4)] >= 0.99, f"dim=128/b=4: {results[(128, 4)]:.4f} < 0.99"

    # Print table
    header = "dim\t" + "\t".join(f"b={b}" for b in bits_list)
    print(f"    {header}")
    for dim in dims:
        row = f"    {dim}\t" + "\t".join(f"{results[(dim, b)]:.4f}" for b in bits_list)
        print(row)

    return f"d128/b3={results[(128, 3)]:.4f}, d128/b4={results[(128, 4)]:.4f}"


# ---------------------------------------------------------------------------
# Test 4: Multi-layer attention quality simulation (GPU) — KEY DIAGNOSTIC
# ---------------------------------------------------------------------------

def _rms_norm(x, eps=1e-6):
    """RMSNorm (like real transformers use)."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def test_multi_layer_attention_quality():
    """Simulate N-layer transformer with quantize-dequant KV cache per layer.

    Models the error accumulation path: at each layer, K and V are quantized
    and dequantized (simulating cache storage), then attention is computed.
    The attention output plus residual connection feeds the next layer.
    Uses RMSNorm between layers (like Qwen/Llama) to prevent numerical
    instability at deep layer counts.
    """
    device = "cuda"
    # Qwen2.5-3B-like config
    H_q = 16
    H_kv = 2
    D = 128
    seq_len = 64
    seed = 42

    torch.manual_seed(seed)

    layer_counts = [1, 4, 12, 36]
    max_layers = max(layer_counts)

    # Xavier-scaled projection matrices (stable initialization)
    proj_q = [torch.randn(D, H_q * D, device=device) * math.sqrt(2.0 / (D + H_q * D)) for _ in range(max_layers)]
    proj_k = [torch.randn(D, H_kv * D, device=device) * math.sqrt(2.0 / (D + H_kv * D)) for _ in range(max_layers)]
    proj_v = [torch.randn(D, H_kv * D, device=device) * math.sqrt(2.0 / (D + H_kv * D)) for _ in range(max_layers)]
    proj_out = [torch.randn(H_q * D, D, device=device) * math.sqrt(2.0 / (H_q * D + D)) for _ in range(max_layers)]

    sm_scale = 1.0 / math.sqrt(D)
    gqa_ratio = H_q // H_kv

    all_results = {}

    for bits in [3, 4]:
        hadamard = HadamardTransform(D, seed=seed, device=device)
        codebook = compute_codebook(D, bits).to(device)

        results = {}
        for n_layers in layer_counts:
            torch.manual_seed(seed)
            hidden_q = torch.randn(seq_len, D, device=device) * 0.02
            hidden_r = hidden_q.clone()

            for layer_i in range(n_layers):
                # RMSNorm before attention (pre-norm architecture)
                h_q_norm = _rms_norm(hidden_q)
                h_r_norm = _rms_norm(hidden_r)

                # Project to Q, K, V
                q_q = (h_q_norm @ proj_q[layer_i]).view(seq_len, H_q, D)
                k_q = (h_q_norm @ proj_k[layer_i]).view(seq_len, H_kv, D)
                v_q = (h_q_norm @ proj_v[layer_i]).view(seq_len, H_kv, D)

                q_r = (h_r_norm @ proj_q[layer_i]).view(seq_len, H_q, D)
                k_r = (h_r_norm @ proj_k[layer_i]).view(seq_len, H_kv, D)
                v_r = (h_r_norm @ proj_v[layer_i]).view(seq_len, H_kv, D)

                # Quantize K, V for the quantized path
                k_packed, k_norms = mse_quantize(k_q, hadamard, codebook, bits)
                v_packed, v_norms = mse_quantize(v_q, hadamard, codebook, bits)
                k_hat = mse_dequantize(k_packed, k_norms, hadamard, codebook, bits, D)
                v_hat = mse_dequantize(v_packed, v_norms, hadamard, codebook, bits, D)

                # Expand KV heads for GQA
                k_hat_exp = k_hat.repeat_interleave(gqa_ratio, dim=1)
                v_hat_exp = v_hat.repeat_interleave(gqa_ratio, dim=1)
                k_r_exp = k_r.repeat_interleave(gqa_ratio, dim=1)
                v_r_exp = v_r.repeat_interleave(gqa_ratio, dim=1)

                # Attention with causal mask
                scores_q = torch.einsum("shd,thd->hst", q_q, k_hat_exp) * sm_scale
                scores_r = torch.einsum("shd,thd->hst", q_r, k_r_exp) * sm_scale

                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1
                )
                scores_q = scores_q + causal_mask
                scores_r = scores_r + causal_mask

                attn_q = torch.softmax(scores_q, dim=-1)
                attn_r = torch.softmax(scores_r, dim=-1)

                out_q = torch.einsum("hst,thd->shd", attn_q, v_hat_exp)
                out_r = torch.einsum("hst,thd->shd", attn_r, v_r_exp)

                # Output projection + residual connection
                out_q_flat = out_q.reshape(seq_len, H_q * D)
                out_r_flat = out_r.reshape(seq_len, H_q * D)

                hidden_q = hidden_q + (out_q_flat @ proj_out[layer_i])
                hidden_r = hidden_r + (out_r_flat @ proj_out[layer_i])

            # Check for NaN/Inf in either path
            ref_ok = torch.isfinite(hidden_r).all().item()
            quant_ok = torch.isfinite(hidden_q).all().item()

            if not ref_ok:
                sim = float("nan")  # Simulation instability, not quantization issue
            elif not quant_ok:
                sim = 0.0  # Quantization caused divergence
            else:
                sim = _per_vector_cosine_sim(hidden_q, hidden_r)

            results[n_layers] = (sim, ref_ok, quant_ok)

        all_results[bits] = results

    lines = []
    for bits in [3, 4]:
        parts = []
        for n in layer_counts:
            sim, ref_ok, quant_ok = all_results[bits][n]
            status = ""
            if not ref_ok:
                status = " (ref NaN!)"
            elif not quant_ok:
                status = " (quant NaN!)"
            parts.append(f"N={n}: {sim:.4f}{status}")
        line = f"b={bits}: " + ", ".join(parts)
        lines.append(line)
        print(f"    {line}")

    # Conservative thresholds — diagnostic test, not a hard gate
    for bits in [3, 4]:
        sim_1, ref_ok_1, _ = all_results[bits][1]
        sim_4, ref_ok_4, _ = all_results[bits][4]
        if ref_ok_1:
            assert sim_1 > 0.90, f"b={bits}/N=1: {sim_1:.4f} < 0.90"
        if ref_ok_4:
            assert sim_4 > 0.80, f"b={bits}/N=4: {sim_4:.4f} < 0.80"

    return "; ".join(lines)


# ---------------------------------------------------------------------------
# Test 5: Extend path per-layer distortion (GPU)
# ---------------------------------------------------------------------------

def test_extend_roundtrip_error():
    """Extend path: quantize K,V -> dequant -> attention vs raw K,V attention.

    The 2-stage extend kernel uses raw K,V for extend tokens (stage 2) and
    dequantized pool data for prefix tokens (stage 1). This test measures the
    distortion introduced by prefix dequantization on attention output quality.
    """
    device = "cuda"
    H_q = 16
    H_kv = 2
    D = 128
    extend_len = 16
    seed = 42
    torch.manual_seed(seed)

    sm_scale = 1.0 / math.sqrt(D)
    gqa_ratio = H_q // H_kv

    results = []

    for bits_config in [3, 3.5, 4]:
        for prefix_len in [0, 32, 128]:
            total_kv = prefix_len + extend_len

            Q = torch.randn(extend_len, H_q, D, device=device)
            K_all = torch.randn(total_kv, H_kv, D, device=device)
            V_all = torch.randn(total_kv, H_kv, D, device=device)

            # --- Quantized path (prefix quantized, extend raw) ---
            if prefix_len > 0:
                K_prefix = K_all[:prefix_len]
                V_prefix = V_all[:prefix_len]

                is_split = (bits_config != int(bits_config))

                if is_split:
                    int_bits = int(bits_config)
                    lo_idx, hi_idx = select_outlier_indices(D, bits_config)
                    lo_idx = lo_idx.to(device)
                    hi_idx = hi_idx.to(device)
                    restore = torch.argsort(torch.cat([lo_idx, hi_idx])).to(device)

                    d_lo = lo_idx.shape[0]
                    d_hi = hi_idx.shape[0]
                    had_lo = HadamardTransform(d_lo, seed=seed, device=device)
                    had_hi = HadamardTransform(d_hi, seed=seed + 1, device=device)
                    cb_lo = compute_codebook(d_lo, int_bits).to(device)
                    cb_hi = compute_codebook(d_hi, int_bits + 1).to(device)

                    kp = split_channel_mse_quantize(
                        K_prefix, lo_idx, hi_idx, had_lo, had_hi,
                        cb_lo, cb_hi, int_bits, int_bits + 1,
                    )
                    vp = split_channel_mse_quantize(
                        V_prefix, lo_idx, hi_idx, had_lo, had_hi,
                        cb_lo, cb_hi, int_bits, int_bits + 1,
                    )
                    K_prefix_hat = split_channel_mse_dequantize(
                        *kp, lo_idx, hi_idx, restore,
                        had_lo, had_hi, cb_lo, cb_hi,
                        int_bits, int_bits + 1, D,
                    )
                    V_prefix_hat = split_channel_mse_dequantize(
                        *vp, lo_idx, hi_idx, restore,
                        had_lo, had_hi, cb_lo, cb_hi,
                        int_bits, int_bits + 1, D,
                    )
                else:
                    int_bits = int(bits_config)
                    hadamard = HadamardTransform(D, seed=seed, device=device)
                    codebook = compute_codebook(D, int_bits).to(device)

                    kp, kn = mse_quantize(K_prefix, hadamard, codebook, int_bits)
                    vp, vn = mse_quantize(V_prefix, hadamard, codebook, int_bits)
                    K_prefix_hat = mse_dequantize(kp, kn, hadamard, codebook, int_bits, D)
                    V_prefix_hat = mse_dequantize(vp, vn, hadamard, codebook, int_bits, D)

                # Merge: dequantized prefix + raw extend
                K_hat = torch.cat([K_prefix_hat, K_all[prefix_len:]], dim=0)
                V_hat = torch.cat([V_prefix_hat, V_all[prefix_len:]], dim=0)
            else:
                # No prefix — everything is raw extend
                K_hat = K_all
                V_hat = V_all

            # --- Reference path (all raw) ---
            K_ref = K_all
            V_ref = V_all

            # Expand KV heads for GQA
            K_hat_exp = K_hat.repeat_interleave(gqa_ratio, dim=1)
            V_hat_exp = V_hat.repeat_interleave(gqa_ratio, dim=1)
            K_ref_exp = K_ref.repeat_interleave(gqa_ratio, dim=1)
            V_ref_exp = V_ref.repeat_interleave(gqa_ratio, dim=1)

            # Attention scores: Q [ext, Hq, D] x K [tot, Hq, D]^T -> [Hq, ext, tot]
            scores_q = torch.einsum("shd,thd->hst", Q, K_hat_exp) * sm_scale
            scores_r = torch.einsum("shd,thd->hst", Q, K_ref_exp) * sm_scale

            # Causal mask: extend token i can attend to all prefix + extend[0..i]
            mask = torch.zeros(extend_len, total_kv, device=device)
            for i in range(extend_len):
                mask[i, prefix_len + i + 1:] = float("-inf")
            scores_q = scores_q + mask
            scores_r = scores_r + mask

            attn_q = torch.softmax(scores_q, dim=-1)
            attn_r = torch.softmax(scores_r, dim=-1)

            out_q = torch.einsum("hst,thd->shd", attn_q, V_hat_exp)
            out_r = torch.einsum("hst,thd->shd", attn_r, V_ref_exp)

            sim = _per_vector_cosine_sim(out_q, out_r)
            results.append((bits_config, prefix_len, sim))

    # Print results
    print("    bits\tprefix\tcosine_sim")
    for bits, plen, sim in results:
        print(f"    {bits}\t{plen}\t{sim:.4f}")

    # Assertions
    for bits, plen, sim in results:
        if plen == 0:
            # No prefix = no quantization error
            assert sim > 0.999, f"b={bits}/p={plen}: {sim:.4f} < 0.999 (no prefix!)"
        else:
            # With prefix, expect some loss but bounded
            min_sim = 0.85 if bits == 3 else 0.90
            assert sim > min_sim, f"b={bits}/p={plen}: {sim:.4f} < {min_sim}"

    # Verify error doesn't grow significantly with prefix length
    for bits in [3, 3.5, 4]:
        sims = [(plen, sim) for b, plen, sim in results if b == bits and plen > 0]
        if len(sims) >= 2:
            sim_short = sims[0][1]
            sim_long = sims[-1][1]
            assert sim_long > sim_short - 0.05, (
                f"b={bits}: sim degrades with prefix length "
                f"({sim_short:.4f} -> {sim_long:.4f})"
            )

    return f"{len(results)} configs tested"


# ---------------------------------------------------------------------------
# Test 6: MSE vs Prod inner-product bias (GPU) — KEY DIAGNOSTIC
# ---------------------------------------------------------------------------

def test_inner_product_bias():
    """Measure MSE vs prod inner-product bias directly.

    Paper Section 3.3: MSE-optimal quantizers are biased for inner products.
    Prod (MSE+QJL) is unbiased: E[<y, dequant_prod(k)>] = <y, k>.

    Measures regression slope (alpha: quant_score ~ alpha * true_score),
    signal-dependent correlation, and compares MSE vs prod at dim=128.
    """
    device = "cuda"
    D = 128
    N = 10000
    H = 1
    seed = 42
    torch.manual_seed(seed)

    q = torch.randn(N, H, D, device=device)
    k = torch.randn(N, H, D, device=device)

    true_score = (q * k).sum(dim=-1).squeeze(-1)  # [N]

    results = {}

    for bits in [2, 3, 4]:
        hadamard = HadamardTransform(D, seed=seed, device=device)
        codebook = compute_codebook(D, bits).to(device)

        # MSE path
        packed, norms = mse_quantize(k, hadamard, codebook, bits)
        k_mse = mse_dequantize(packed, norms, hadamard, codebook, bits, D)
        mse_score = (q * k_mse).sum(dim=-1).squeeze(-1)

        # Prod path
        s_matrix = projection_matrix(D, seed).to(device)
        mse_bits_for_prod = max(bits - 1, 0)
        prod_cb = compute_codebook(D, mse_bits_for_prod).to(device) if mse_bits_for_prod > 0 else torch.zeros(1, device=device)

        mse_p, qjl_p, norms_prod, res_norms = prod_quantize(
            k, hadamard, s_matrix, prod_cb, bits
        )
        k_prod = prod_dequantize(
            mse_p, qjl_p, norms_prod, res_norms,
            hadamard, s_matrix, prod_cb, bits, D
        )
        prod_score = (q * k_prod).sum(dim=-1).squeeze(-1)

        ts = true_score.float()

        for name, qs, k_hat in [("mse", mse_score, k_mse), ("prod", prod_score, k_prod)]:
            qs_f = qs.float()
            # Regression: qs ~ alpha * ts + beta
            ts_centered = ts - ts.mean()
            alpha = (ts_centered * (qs_f - qs_f.mean())).mean() / (ts_centered ** 2).mean()
            beta = qs_f.mean() - alpha * ts.mean()

            error = qs_f - ts
            corr = torch.corrcoef(torch.stack([ts, error]))[0, 1].item()

            pred = alpha * ts + beta
            ss_res = ((qs_f - pred) ** 2).sum()
            ss_tot = ((qs_f - qs_f.mean()) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot).item()

            results[(bits, name)] = {
                'alpha': alpha.item(),
                'beta': beta.item(),
                'mean_error': error.mean().item(),
                'std_error': error.std().item(),
                'signal_corr': corr,
                'r2': r2,
                'cosine_sim': _per_vector_cosine_sim(k, k_hat),
            }

    print("    bits  method  alpha    beta       mean_err   std_err   sig_corr  R²      cos_sim")
    for bits in [2, 3, 4]:
        for method in ["mse", "prod"]:
            r = results[(bits, method)]
            print(f"    {bits}     {method:4s}    {r['alpha']:.4f}   {r['beta']:+.6f}  {r['mean_error']:+.6f}  {r['std_error']:.4f}    {r['signal_corr']:+.4f}    {r['r2']:.4f}   {r['cosine_sim']:.4f}")

    mse_alphas = [results[(b, 'mse')]['alpha'] for b in [2, 3, 4]]
    prod_alphas = [results[(b, 'prod')]['alpha'] for b in [2, 3, 4]]

    return (
        f"MSE alpha: b2={mse_alphas[0]:.4f}, b3={mse_alphas[1]:.4f}, b4={mse_alphas[2]:.4f}; "
        f"Prod alpha: b2={prod_alphas[0]:.4f}, b3={prod_alphas[1]:.4f}, b4={prod_alphas[2]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 7: Attention distribution distortion (GPU)
# ---------------------------------------------------------------------------

def test_attention_distribution_distortion():
    """Measure how MSE bias distorts softmax attention vs prod.

    Simulates realistic GQA attention (Qwen2.5-3B config). Compares
    softmax distributions from true vs MSE-quantized vs prod-quantized keys.
    Measures KL divergence, top-1 accuracy, entropy shift.
    """
    device = "cuda"
    D = 128
    H_q = 16
    H_kv = 2
    seq_len = 128
    num_queries = 100
    seed = 42
    torch.manual_seed(seed)

    sm_scale = 1.0 / math.sqrt(D)
    gqa_ratio = H_q // H_kv

    results = {}

    for bits in [3, 4]:
        hadamard = HadamardTransform(D, seed=seed, device=device)
        codebook = compute_codebook(D, bits).to(device)
        s_matrix = projection_matrix(D, seed).to(device)
        prod_cb = compute_codebook(D, max(bits - 1, 0)).to(device)

        Q = torch.randn(num_queries, H_q, D, device=device)
        K = torch.randn(seq_len, H_kv, D, device=device)

        # MSE quantize
        packed_m, norms_m = mse_quantize(K, hadamard, codebook, bits)
        K_mse = mse_dequantize(packed_m, norms_m, hadamard, codebook, bits, D)

        # Prod quantize
        mp, qp, np_, rn = prod_quantize(K, hadamard, s_matrix, prod_cb, bits)
        K_prod = prod_dequantize(mp, qp, np_, rn, hadamard, s_matrix, prod_cb, bits, D)

        # Expand KV heads for GQA
        K_exp = K.repeat_interleave(gqa_ratio, dim=1)
        K_mse_exp = K_mse.repeat_interleave(gqa_ratio, dim=1)
        K_prod_exp = K_prod.repeat_interleave(gqa_ratio, dim=1)

        # Attention scores: [num_queries, H_q, seq_len]
        scores_true = torch.einsum("qhd,shd->qhs", Q, K_exp) * sm_scale
        scores_mse = torch.einsum("qhd,shd->qhs", Q, K_mse_exp) * sm_scale
        scores_prod = torch.einsum("qhd,shd->qhs", Q, K_prod_exp) * sm_scale

        attn_true = torch.softmax(scores_true, dim=-1)
        attn_mse = torch.softmax(scores_mse, dim=-1)
        attn_prod = torch.softmax(scores_prod, dim=-1)

        eps = 1e-10
        kl_mse = (attn_true * (attn_true.clamp(min=eps).log() - attn_mse.clamp(min=eps).log())).sum(dim=-1).mean().item()
        kl_prod = (attn_true * (attn_true.clamp(min=eps).log() - attn_prod.clamp(min=eps).log())).sum(dim=-1).mean().item()

        top1_true = attn_true.argmax(dim=-1)
        top1_acc_mse = (top1_true == attn_mse.argmax(dim=-1)).float().mean().item()
        top1_acc_prod = (top1_true == attn_prod.argmax(dim=-1)).float().mean().item()

        ent_true = -(attn_true * attn_true.clamp(min=eps).log()).sum(dim=-1).mean().item()
        ent_mse = -(attn_mse * attn_mse.clamp(min=eps).log()).sum(dim=-1).mean().item()
        ent_prod = -(attn_prod * attn_prod.clamp(min=eps).log()).sum(dim=-1).mean().item()

        results[bits] = {
            'kl_mse': kl_mse, 'kl_prod': kl_prod,
            'top1_mse': top1_acc_mse, 'top1_prod': top1_acc_prod,
            'ent_true': ent_true, 'ent_mse': ent_mse, 'ent_prod': ent_prod,
        }

    print("    bits  metric       MSE        Prod       True")
    for bits in [3, 4]:
        r = results[bits]
        print(f"    {bits}     KL div       {r['kl_mse']:.6f}   {r['kl_prod']:.6f}")
        print(f"    {bits}     Top-1 acc    {r['top1_mse']:.4f}     {r['top1_prod']:.4f}")
        print(f"    {bits}     Entropy      {r['ent_mse']:.4f}     {r['ent_prod']:.4f}     {r['ent_true']:.4f}")

    return (
        f"b3: KL mse={results[3]['kl_mse']:.6f} prod={results[3]['kl_prod']:.6f}; "
        f"b4: KL mse={results[4]['kl_mse']:.6f} prod={results[4]['kl_prod']:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 8: Fused decode vs dequant extend path consistency (GPU)
# ---------------------------------------------------------------------------

def test_fused_vs_dequant_consistency():
    """Compare fused decode kernel math vs dequant-then-score extend path.

    The fused decode kernel scores in rotated space (float32 throughout).
    The extend path dequantizes to model dtype (bf16) then scores.
    If these produce different results, the prefill→decode boundary
    creates an inconsistency that compounds over 36 layers.

    Tests at both bf16 and float32 intermediate dtype to isolate
    whether the bf16 truncation is the source of divergence.
    """
    device = "cuda"
    D = 128
    H_q = 16
    H_kv = 2
    seq_len = 64
    seed = 42
    torch.manual_seed(seed)

    sm_scale = 1.0 / math.sqrt(D)
    gqa_ratio = H_q // H_kv

    results = []

    for bits in [3, 4]:
        hadamard = HadamardTransform(D, seed=seed, device=device)
        codebook = compute_codebook(D, bits).to(device)

        q = torch.randn(1, H_q, D, device=device)
        K = torch.randn(seq_len, H_kv, D, device=device)
        V = torch.randn(seq_len, H_kv, D, device=device)

        k_packed, k_norms = mse_quantize(K, hadamard, codebook, bits)
        v_packed, v_norms = mse_quantize(V, hadamard, codebook, bits)

        # ===== Path A: Fused (rotated-space, float32 throughout) =====
        q_rot = hadamard.forward(q.float())  # [1, H_q, D]

        k_idx = unpack_bits(k_packed, bits, hadamard.padded_dim)  # [seq, H_kv, D]
        k_cb = codebook[k_idx.long()]  # [seq, H_kv, D] float32
        v_idx = unpack_bits(v_packed, bits, hadamard.padded_dim)
        v_cb = codebook[v_idx.long()]  # [seq, H_kv, D] float32

        k_cb_exp = k_cb.repeat_interleave(gqa_ratio, dim=1)
        k_n_exp = k_norms.float().repeat_interleave(gqa_ratio, dim=1)

        # Score: [H_q, seq]
        scores_fused = torch.einsum("bhd,shd->hs", q_rot, k_cb_exp)
        scores_fused = scores_fused * k_n_exp.T * sm_scale

        attn_fused = torch.softmax(scores_fused, dim=-1)

        v_cb_exp = v_cb.repeat_interleave(gqa_ratio, dim=1)
        v_n_exp = v_norms.float().repeat_interleave(gqa_ratio, dim=1)

        # V weighted sum in rotated space: [H_q, D]
        weighted_v_rot = torch.einsum(
            "hs,shd,sh->hd", attn_fused, v_cb_exp, v_n_exp
        )
        output_fused = hadamard.inverse(weighted_v_rot.unsqueeze(0)).squeeze(0)

        # ===== Path B: Dequant to bf16, then score (extend path) =====
        K_bf16 = mse_dequantize(
            k_packed, k_norms, hadamard, codebook, bits, D
        ).to(torch.bfloat16)
        V_bf16 = mse_dequantize(
            v_packed, v_norms, hadamard, codebook, bits, D
        ).to(torch.bfloat16)

        K_bf16_exp = K_bf16.repeat_interleave(gqa_ratio, dim=1).float()
        V_bf16_exp = V_bf16.repeat_interleave(gqa_ratio, dim=1).float()

        scores_bf16 = torch.einsum("bhd,shd->hs", q.float(), K_bf16_exp) * sm_scale
        attn_bf16 = torch.softmax(scores_bf16, dim=-1)
        output_bf16 = torch.einsum("hs,shd->hd", attn_bf16, V_bf16_exp)

        # ===== Path C: Dequant to float32, then score (no bf16 truncation) =====
        K_f32 = mse_dequantize(
            k_packed, k_norms, hadamard, codebook, bits, D
        )
        V_f32 = mse_dequantize(
            v_packed, v_norms, hadamard, codebook, bits, D
        )

        K_f32_exp = K_f32.repeat_interleave(gqa_ratio, dim=1)
        V_f32_exp = V_f32.repeat_interleave(gqa_ratio, dim=1)

        scores_f32 = torch.einsum("bhd,shd->hs", q.float(), K_f32_exp) * sm_scale
        attn_f32 = torch.softmax(scores_f32, dim=-1)
        output_f32 = torch.einsum("hs,shd->hd", attn_f32, V_f32_exp)

        # ===== Metrics =====
        cos_fused_bf16 = _per_vector_cosine_sim(
            output_fused.unsqueeze(0), output_bf16.unsqueeze(0)
        )
        cos_fused_f32 = _per_vector_cosine_sim(
            output_fused.unsqueeze(0), output_f32.unsqueeze(0)
        )
        cos_bf16_f32 = _per_vector_cosine_sim(
            output_bf16.unsqueeze(0), output_f32.unsqueeze(0)
        )

        eps = 1e-10
        kl_fused_bf16 = (
            attn_fused
            * (attn_fused.clamp(min=eps).log() - attn_bf16.clamp(min=eps).log())
        ).sum(dim=-1).mean().item()
        kl_fused_f32 = (
            attn_fused
            * (attn_fused.clamp(min=eps).log() - attn_f32.clamp(min=eps).log())
        ).sum(dim=-1).mean().item()

        score_cos_fused_bf16 = _per_vector_cosine_sim(
            scores_fused.unsqueeze(0), scores_bf16.unsqueeze(0)
        )
        score_cos_fused_f32 = _per_vector_cosine_sim(
            scores_fused.unsqueeze(0), scores_f32.unsqueeze(0)
        )

        results.append({
            'bits': bits,
            'out_fused_bf16': cos_fused_bf16,
            'out_fused_f32': cos_fused_f32,
            'out_bf16_f32': cos_bf16_f32,
            'kl_fused_bf16': kl_fused_bf16,
            'kl_fused_f32': kl_fused_f32,
            'score_fused_bf16': score_cos_fused_bf16,
            'score_fused_f32': score_cos_fused_f32,
        })

    print("    bits  comparison       output_cos  score_cos   attn_KL")
    for r in results:
        print(f"    {r['bits']}     fused vs bf16    {r['out_fused_bf16']:.6f}    {r['score_fused_bf16']:.6f}    {r['kl_fused_bf16']:.6f}")
        print(f"    {r['bits']}     fused vs f32     {r['out_fused_f32']:.6f}    {r['score_fused_f32']:.6f}    {r['kl_fused_f32']:.6f}")
        print(f"    {r['bits']}     bf16 vs f32      {r['out_bf16_f32']:.6f}")

    return (
        f"b3: fused-bf16={results[0]['out_fused_bf16']:.6f}, fused-f32={results[0]['out_fused_f32']:.6f}; "
        f"b4: fused-bf16={results[1]['out_fused_bf16']:.6f}, fused-f32={results[1]['out_fused_f32']:.6f}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CPU_TESTS = [
    test_codebook_cross_impl,
]

GPU_TESTS = [
    test_codebook_optimality,
    test_cosine_sim_sweep,
    test_multi_layer_attention_quality,
    test_extend_roundtrip_error,
    test_inner_product_bias,
    test_attention_distribution_distortion,
    test_fused_vs_dequant_consistency,
]

ALL_TESTS = CPU_TESTS + GPU_TESTS


def main():
    has_cuda = torch.cuda.is_available()

    passed = 0
    failed = 0
    skipped = 0

    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        if test_fn in GPU_TESTS and not has_cuda:
            print(f"  SKIP  {name}: no CUDA")
            skipped += 1
            continue
        try:
            summary = test_fn()
            print(f"  PASS  {name}: {summary}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    total = passed + failed + skipped
    print(f"\n{passed}/{total} tests passed ({skipped} skipped, {failed} failed)")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
