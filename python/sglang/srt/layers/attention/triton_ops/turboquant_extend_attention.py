# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
TurboQuant fused extend/prefill attention kernel (arXiv:2504.19874).

Unified single-loop kernel that reads packed b-bit quantized KV buffers directly.
All KV (prefix + fresh extend tokens) is already quantized in the pool before
the kernel runs. The kernel reads ONLY from quantized buffers via kv_indices.

This is the extend counterpart to turboquant_decode_attention.py:
  - BLOCK_M > 1 queries (multiple query rows per block)
  - Causal masking for extend tokens
  - No Stage 2 reduction (single-stage, like the standard extend kernel)

Phase A: integer bits (e.g. 3-bit). Single set of buffers.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# =============================================================================
# Bit extraction helpers (copied from turboquant_decode_attention.py)
# =============================================================================


@triton.jit
def _extract_bits(
    packed_ptr,
    kv_loc,          # [BLOCK_N]
    stride_bs,       # stride along pool/token dimension (in elements, uint8)
    head_offset,     # cur_kv_head * stride_h (precomputed scalar)
    d_offs,          # [BLOCK_D] dimension offsets
    n_mask,          # [BLOCK_N] validity mask for tokens
    d_mask,          # [BLOCK_D] validity mask for dimensions
    BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Extract BITS-wide values from a packed uint8 buffer.

    Returns [BLOCK_N, BLOCK_D] int32 indices.

    Bit layout: dimension d occupies bits [d*BITS, (d+1)*BITS) in a flat
    bitstream stored as uint8 bytes. When a value spans a byte boundary
    (spillover), we load the next byte and merge the high bits.
    """
    bit_offsets = d_offs * BITS            # [BLOCK_D]: bit position in stream
    byte_indices = bit_offsets // 8        # [BLOCK_D]: which byte
    intra_bit = bit_offsets % 8            # [BLOCK_D]: bit offset within byte
    MASK: tl.constexpr = (1 << BITS) - 1

    # Load primary bytes: [BLOCK_N, BLOCK_D]
    ptrs = (
        packed_ptr
        + kv_loc[:, None] * stride_bs
        + head_offset
        + byte_indices[None, :]
    )
    combined_mask = n_mask[:, None] & d_mask[None, :]
    packed = tl.load(ptrs, mask=combined_mask, other=0)
    idx = (packed >> intra_bit[None, :]) & MASK

    # Handle spillover for odd bit widths where value can span byte boundary
    if BITS == 3 or BITS == 5 or BITS == 6 or BITS == 7:
        spill = intra_bit + BITS - 8  # [BLOCK_D]: >0 when spilling

        next_ptrs = (
            packed_ptr
            + kv_loc[:, None] * stride_bs
            + head_offset
            + byte_indices[None, :] + 1
        )
        next_packed = tl.load(next_ptrs, mask=combined_mask, other=0)

        # Clamp spill to >= 0 for safe shifting
        spill_clamped = tl.maximum(spill, 0)  # [BLOCK_D]

        # Extract lowest `spill` bits from next byte, shift into high position
        spill_bits = next_packed & ((1 << spill_clamped[None, :]) - 1)
        shift_up = (BITS - spill_clamped)[None, :]  # how far left to shift
        spill_contribution = spill_bits << shift_up

        # Only apply where spill > 0
        spill_mask = (spill > 0)[None, :]  # broadcast [1, BLOCK_D] -> [BLOCK_N, BLOCK_D]
        idx = tl.where(spill_mask, (idx | spill_contribution) & MASK, idx)

    return idx


@triton.jit
def _extract_sign_bits(
    sign_ptr,
    kv_loc,          # [BLOCK_N]
    stride_bs,
    head_offset,
    d_offs,          # [BLOCK_D]
    n_mask,
    d_mask,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Extract 1-bit sign values from packed uint8 QJL buffer.

    Returns [BLOCK_N, BLOCK_D] float32: +1.0 or -1.0.
    """
    byte_idx = d_offs // 8     # [BLOCK_D]
    bit_idx = d_offs % 8       # [BLOCK_D]

    ptrs = (
        sign_ptr
        + kv_loc[:, None] * stride_bs
        + head_offset
        + byte_idx[None, :]
    )
    combined_mask = n_mask[:, None] & d_mask[None, :]
    sign_byte = tl.load(ptrs, mask=combined_mask, other=0)
    bit_val = (sign_byte >> bit_idx[None, :]) & 1
    # Map {0, 1} -> {-1.0, +1.0}
    sign = bit_val.to(tl.float32) * 2.0 - 1.0
    return sign


# =============================================================================
# TurboQuant fused extend attention kernel (single-stage)
# =============================================================================


@triton.jit
def _turboquant_extend_kernel(
    # Pre-rotated/projected queries: [total_q_tokens, H_q, D]
    Q_rot,
    Q_proj,
    # Packed K buffers
    K_MSE_Buffer,      # [pool_size, H_kv, packed_mse_width] uint8
    K_QJL_Buffer,      # [pool_size, H_kv, packed_qjl_width] uint8
    K_Norms,           # [pool_size, H_kv] float16
    K_ResNorms,        # [pool_size, H_kv] float16
    # Packed V buffers
    V_Packed,          # [pool_size, H_kv, packed_v_width] uint8
    V_Norms,           # [pool_size, H_kv] float16
    # Codebooks
    K_Codebook,        # [2^mse_bits] float32
    V_Codebook,        # [2^v_bits] float32
    # Output
    O_rot,             # [total_q_tokens, H_q, D] float32 output
    # Sequence metadata
    qo_indptr,         # [batch+1] — query offsets per sequence
    kv_indptr,         # [batch+1] — unified KV offsets (prefix + extend)
    kv_indices,        # [total_kv] — pool slot indices
    prefix_lens,       # [batch] — prefix length per sequence
    # Scaling factors
    sm_scale,          # 1/sqrt(head_dim)
    qjl_scale,         # sqrt(pi/2) / D
    # Strides — Q: [total_q_tokens, H_q, D]
    stride_q_bs,
    stride_q_h,
    # Strides — K_MSE: [pool_size, H_kv, packed_mse_width]
    stride_k_mse_bs,
    stride_k_mse_h,
    # Strides — K_QJL: [pool_size, H_kv, packed_qjl_width]
    stride_k_qjl_bs,
    stride_k_qjl_h,
    # Strides — K_Norms / K_ResNorms: [pool_size, H_kv]
    stride_kn_bs,
    # Strides — V_Packed: [pool_size, H_kv, packed_v_width]
    stride_v_bs,
    stride_v_h,
    # Strides — V_Norms: [pool_size, H_kv]
    stride_vn_bs,
    # Strides — O: [total_q_tokens, H_q, D]
    stride_o_bs,
    stride_o_h,
    # Constexprs
    kv_group_num: tl.constexpr,
    MSE_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Lv: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """TurboQuant fused extend attention (single-stage).

    Grid: (batch_size, num_q_heads, cdiv(max_extend_len, BLOCK_M))

    Reads all KV from quantized buffers via kv_indices page table.
    Applies causal masking for extend tokens (prefix always visible).
    Output is in rotated space — caller must inverse-rotate.
    """
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    # Load sequence metadata
    q_start = tl.load(qo_indptr + cur_seq)
    q_end = tl.load(qo_indptr + cur_seq + 1)
    q_len = q_end - q_start
    kv_start = tl.load(kv_indptr + cur_seq)
    kv_end = tl.load(kv_indptr + cur_seq + 1)
    kv_len = kv_end - kv_start
    prefix_len = tl.load(prefix_lens + cur_seq)

    # Early exit if this block is beyond the sequence's extend length
    if cur_block_m * BLOCK_M >= q_len:
        return

    # Offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < q_len
    mask_dv = offs_dv < Lv

    # Precomputed head offsets for packed buffer access
    k_mse_head_off = cur_kv_head * stride_k_mse_h
    k_qjl_head_off = cur_kv_head * stride_k_qjl_h
    v_head_off = cur_kv_head * stride_v_h

    # Q base offsets for this block of queries
    q_row_offsets = q_start + cur_block_m * BLOCK_M + offs_m  # [BLOCK_M]

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Unified loop: process all KV tokens (prefix + extend)
    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < kv_len

        # Load kv_loc from page table
        kv_loc = tl.load(
            kv_indices + kv_start + start_n + offs_n,
            mask=mask_n,
            other=0,
        )

        # Build mask
        final_mask = mask_m[:, None] & mask_n[None, :]

        # Causal mask: prefix always visible, extend region causal
        if IS_CAUSAL:
            q_idx = cur_block_m * BLOCK_M + offs_m[:, None]
            k_idx_in_total = start_n + offs_n[None, :]
            k_is_extend = k_idx_in_total >= prefix_len
            k_idx_in_extend = k_idx_in_total - prefix_len
            causal_mask = tl.where(
                k_is_extend,
                q_idx >= k_idx_in_extend,
                True,  # No causal mask for prefix
            )
            final_mask &= causal_mask

        # ==============================================================
        # K scoring: TurboQuant_prod (paper Algorithm 1)
        # score = ||k|| * (q_rot^T @ C[idx] + scale * ||k_res|| * q_proj^T @ sign)
        # ==============================================================

        # MSE component: [BLOCK_M, BLOCK_N]
        mse_scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d_start in range(0, Lv, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < Lv

            # Load Q_rot block: [BLOCK_M, BLOCK_D]
            q_rot_ptrs = (
                q_row_offsets[:, None] * stride_q_bs
                + cur_head * stride_q_h
                + d_offs[None, :]
            )
            q_rot_block = tl.load(
                Q_rot + q_rot_ptrs,
                mask=mask_m[:, None] & d_mask[None, :],
                other=0.0,
            )

            # Extract MSE indices: [BLOCK_N, BLOCK_D]
            mse_idx = _extract_bits(
                K_MSE_Buffer, kv_loc, stride_k_mse_bs, k_mse_head_off,
                d_offs, mask_n, d_mask, MSE_BITS, BLOCK_N, BLOCK_D,
            )

            # Codebook gather: [BLOCK_N, BLOCK_D]
            k_val = tl.load(K_Codebook + mse_idx)
            k_val = tl.where(mask_n[:, None] & d_mask[None, :], k_val, 0.0)

            # Score: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            mse_scores += tl.dot(q_rot_block, tl.trans(k_val))

        # QJL component: [BLOCK_M, BLOCK_N]
        qjl_scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d_start in range(0, Lv, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < Lv

            # Load Q_proj block: [BLOCK_M, BLOCK_D]
            q_proj_ptrs = (
                q_row_offsets[:, None] * stride_q_bs
                + cur_head * stride_q_h
                + d_offs[None, :]
            )
            q_proj_block = tl.load(
                Q_proj + q_proj_ptrs,
                mask=mask_m[:, None] & d_mask[None, :],
                other=0.0,
            )

            # Extract 1-bit signs: [BLOCK_N, BLOCK_D] -> {-1, +1}
            sign = _extract_sign_bits(
                K_QJL_Buffer, kv_loc, stride_k_qjl_bs, k_qjl_head_off,
                d_offs, mask_n, d_mask, BLOCK_N, BLOCK_D,
            )
            sign = tl.where(mask_n[:, None] & d_mask[None, :], sign, 0.0)

            # Score: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            qjl_scores += tl.dot(q_proj_block, tl.trans(sign))

        # Combined score (paper Eq. 7):
        # qk = ||k|| * (mse_score + sqrt(pi/2)/D * ||k_res|| * qjl_score) / sqrt(d)
        k_norms_vec = tl.load(
            K_Norms + kv_loc * stride_kn_bs + cur_kv_head,
            mask=mask_n, other=0.0,
        ).to(tl.float32)
        k_res_norms_vec = tl.load(
            K_ResNorms + kv_loc * stride_kn_bs + cur_kv_head,
            mask=mask_n, other=0.0,
        ).to(tl.float32)

        qk = k_norms_vec[None, :] * (
            mse_scores + qjl_scale * k_res_norms_vec[None, :] * qjl_scores
        )
        qk *= sm_scale
        qk = tl.where(final_mask, qk, float("-inf"))

        # ==============================================================
        # Online softmax
        # ==============================================================
        row_max = tl.max(qk, 1)
        # Guard against -inf row_max to avoid NaN in exp
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        n_e_max = tl.maximum(row_max_fixed, e_max)

        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        # ==============================================================
        # V weighted sum: TurboQuant_mse (paper Algorithm 2)
        # ==============================================================
        v_norms_vec = tl.load(
            V_Norms + kv_loc * stride_vn_bs + cur_kv_head,
            mask=mask_n, other=0.0,
        ).to(tl.float32)

        # Extract V indices: [BLOCK_N, BLOCK_DV]
        v_idx = _extract_bits(
            V_Packed, kv_loc, stride_v_bs, v_head_off,
            offs_dv, mask_n, mask_dv, V_BITS, BLOCK_N, BLOCK_DV,
        )

        # Codebook gather: [BLOCK_N, BLOCK_DV]
        v_val = tl.load(V_Codebook + v_idx)
        v_val = tl.where(mask_n[:, None] & mask_dv[None, :], v_val, 0.0)

        # Weight = softmax_prob * v_norm: [BLOCK_M, BLOCK_N]
        pv = p * v_norms_vec[None, :]

        # Accumulate: [BLOCK_M, BLOCK_DV]
        acc = acc * re_scale[:, None] + tl.dot(pv.to(tl.float32), v_val)

        e_max = n_e_max

    # ==================================================================
    # Store output
    # ==================================================================
    # Guard deno against zero (all-masked rows)
    deno = tl.where(deno == 0.0, 1.0, deno)
    output = acc / deno[:, None]

    offs_o = (
        (q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_o_bs
        + cur_head * stride_o_h
        + offs_dv[None, :]
    )
    tl.store(
        O_rot + offs_o,
        output,
        mask=mask_m[:, None] & mask_dv[None, :],
    )


# =============================================================================
# Python wrapper
# =============================================================================


def turboquant_extend_attention_fwd(
    q_rot: torch.Tensor,            # [T, H_q, D] pre-rotated queries (float32)
    q_proj: torch.Tensor,           # [T, H_q, D] pre-projected queries (float32)
    k_mse_buffer: torch.Tensor,     # [pool_size, H_kv, packed_mse_width] uint8
    k_qjl_buffer: torch.Tensor,     # [pool_size, H_kv, packed_qjl_width] uint8
    k_norms: torch.Tensor,          # [pool_size, H_kv] float16
    k_res_norms: torch.Tensor,      # [pool_size, H_kv] float16
    v_packed_buffer: torch.Tensor,   # [pool_size, H_kv, packed_v_width] uint8
    v_norms: torch.Tensor,          # [pool_size, H_kv] float16
    k_codebook: torch.Tensor,       # [2^mse_bits] float32
    v_codebook: torch.Tensor,       # [2^v_bits] float32
    o_rot: torch.Tensor,            # [T, H_q, D] float32 output
    qo_indptr: torch.Tensor,        # [batch+1]
    kv_indptr: torch.Tensor,        # [batch+1]
    kv_indices: torch.Tensor,       # [total_kv]
    prefix_lens: torch.Tensor,      # [batch]
    max_extend_len: int,
    sm_scale: float,                # 1/sqrt(head_dim)
    qjl_scale: float,               # sqrt(pi/2) / D
    mse_bits: int,                  # MSE quantization bits
    v_bits: int,                    # V quantization bits
    head_dim: int,                  # actual head dimension
    is_causal: bool = True,
):
    """TurboQuant fused extend attention (single-stage).

    All K/V data is read from packed uint8 buffers with codebook gathers.
    Output is in rotated space — the caller must inverse-rotate.

    Unlike the decode kernel, this is single-stage (no KV splits / Stage 2
    reduction) since extend processes multiple query tokens per sequence.
    """
    batch_size = qo_indptr.shape[0] - 1
    head_num = q_rot.shape[1]
    kv_head_num = k_mse_buffer.shape[1]
    kv_group_num = head_num // kv_head_num

    BLOCK_DV = triton.next_power_of_2(head_dim)

    # Adapt block sizes to GPU shared memory constraints.
    # TurboQuant extend uses more shared memory than standard extend
    # due to codebook gathers, so be conservative on sm86/89 (~100KB).
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 9 and capability[1] == 0:
        # Hopper (H100): 228KB shared memory
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = min(128, BLOCK_DV)
        num_stages = 2
    else:
        # Ampere sm86/89 (A6000, RTX 4090): ~100KB shared memory
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_D = min(64, BLOCK_DV)
        num_stages = 1

    num_warps = 4 if head_dim <= 128 else 8

    grid = (batch_size, head_num, triton.cdiv(max_extend_len, BLOCK_M))

    _turboquant_extend_kernel[grid](
        q_rot,
        q_proj,
        k_mse_buffer,
        k_qjl_buffer,
        k_norms,
        k_res_norms,
        v_packed_buffer,
        v_norms,
        k_codebook,
        v_codebook,
        o_rot,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        sm_scale,
        qjl_scale,
        # Q strides
        q_rot.stride(0),
        q_rot.stride(1),
        # K_MSE strides
        k_mse_buffer.stride(0),
        k_mse_buffer.stride(1),
        # K_QJL strides
        k_qjl_buffer.stride(0),
        k_qjl_buffer.stride(1),
        # K_Norms stride
        k_norms.stride(0),
        # V_Packed strides
        v_packed_buffer.stride(0),
        v_packed_buffer.stride(1),
        # V_Norms stride
        v_norms.stride(0),
        # O strides
        o_rot.stride(0),
        o_rot.stride(1),
        # Constexprs
        kv_group_num=kv_group_num,
        MSE_BITS=mse_bits,
        V_BITS=v_bits,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        Lv=head_dim,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# =============================================================================
# Split-channel TurboQuant extend attention (Phase B — fractional bits)
# =============================================================================


@triton.jit
def _turboquant_extend_kernel_split(
    # Pre-rotated/projected queries per group: [total_q_tokens, H_q, D_lo/D_hi]
    Q_rot_lo,
    Q_proj_lo,
    Q_rot_hi,
    Q_proj_hi,
    # Packed K buffers — lo group
    K_MSE_Buffer_lo,    # [pool_size, H_kv, packed_mse_width_lo] uint8
    K_QJL_Buffer_lo,    # [pool_size, H_kv, packed_qjl_width_lo] uint8
    K_Norms_lo,         # [pool_size, H_kv] float16
    K_ResNorms_lo,      # [pool_size, H_kv] float16
    # Packed K buffers — hi group
    K_MSE_Buffer_hi,    # [pool_size, H_kv, packed_mse_width_hi] uint8
    K_QJL_Buffer_hi,    # [pool_size, H_kv, packed_qjl_width_hi] uint8
    K_Norms_hi,         # [pool_size, H_kv] float16
    K_ResNorms_hi,      # [pool_size, H_kv] float16
    # Packed V buffers — lo group
    V_Packed_lo,        # [pool_size, H_kv, packed_v_width_lo] uint8
    V_Norms_lo,         # [pool_size, H_kv] float16
    # Packed V buffers — hi group
    V_Packed_hi,        # [pool_size, H_kv, packed_v_width_hi] uint8
    V_Norms_hi,         # [pool_size, H_kv] float16
    # Codebooks per group
    K_Codebook_lo,      # [2^MSE_BITS_LO] float32
    K_Codebook_hi,      # [2^MSE_BITS_HI] float32
    V_Codebook_lo,      # [2^V_BITS_LO] float32
    V_Codebook_hi,      # [2^V_BITS_HI] float32
    # Output in split order: [total_q_tokens, H_q, D_lo+D_hi]
    O_rot,
    # Sequence metadata
    qo_indptr,          # [batch+1]
    kv_indptr,          # [batch+1]
    kv_indices,         # [total_kv]
    prefix_lens,        # [batch]
    # Scaling factors
    sm_scale,           # 1/sqrt(head_dim)
    qjl_scale_lo,       # sqrt(pi/2) / D_LO
    qjl_scale_hi,       # sqrt(pi/2) / D_HI
    # Strides — Q_lo: [total_q_tokens, H_q, D_lo]
    stride_qlo_bs,
    stride_qlo_h,
    # Strides — Q_hi: [total_q_tokens, H_q, D_hi]
    stride_qhi_bs,
    stride_qhi_h,
    # Strides — K_MSE_lo
    stride_k_mse_lo_bs,
    stride_k_mse_lo_h,
    # Strides — K_QJL_lo
    stride_k_qjl_lo_bs,
    stride_k_qjl_lo_h,
    # Strides — K_Norms_lo
    stride_kn_lo_bs,
    # Strides — K_MSE_hi
    stride_k_mse_hi_bs,
    stride_k_mse_hi_h,
    # Strides — K_QJL_hi
    stride_k_qjl_hi_bs,
    stride_k_qjl_hi_h,
    # Strides — K_Norms_hi
    stride_kn_hi_bs,
    # Strides — V_Packed_lo
    stride_v_lo_bs,
    stride_v_lo_h,
    # Strides — V_Norms_lo
    stride_vn_lo_bs,
    # Strides — V_Packed_hi
    stride_v_hi_bs,
    stride_v_hi_h,
    # Strides — V_Norms_hi
    stride_vn_hi_bs,
    # Strides — O: [total_q_tokens, H_q, D_lo+D_hi]
    stride_o_bs,
    stride_o_h,
    # Constexprs
    kv_group_num: tl.constexpr,
    MSE_BITS_LO: tl.constexpr,
    MSE_BITS_HI: tl.constexpr,
    V_BITS_LO: tl.constexpr,
    V_BITS_HI: tl.constexpr,
    D_LO: tl.constexpr,
    D_HI: tl.constexpr,
    BLOCK_DV_LO: tl.constexpr,
    BLOCK_DV_HI: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Lv: tl.constexpr,           # D_LO + D_HI = original head_dim
    IS_CAUSAL: tl.constexpr,
):
    """TurboQuant split-channel extend attention (single-stage).

    Two independent channel groups (lo + hi) with different bit-widths.
    K scores combine additively before softmax; V accumulates per group
    with shared softmax weights. Output stored in split order [lo | hi].
    """
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    # Load sequence metadata
    q_start = tl.load(qo_indptr + cur_seq)
    q_end = tl.load(qo_indptr + cur_seq + 1)
    q_len = q_end - q_start
    kv_start = tl.load(kv_indptr + cur_seq)
    kv_end = tl.load(kv_indptr + cur_seq + 1)
    kv_len = kv_end - kv_start
    prefix_len = tl.load(prefix_lens + cur_seq)

    # Early exit
    if cur_block_m * BLOCK_M >= q_len:
        return

    # Offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dv_lo = tl.arange(0, BLOCK_DV_LO)
    offs_dv_hi = tl.arange(0, BLOCK_DV_HI)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < q_len
    mask_dv_lo = offs_dv_lo < D_LO
    mask_dv_hi = offs_dv_hi < D_HI

    # Precomputed head offsets
    k_mse_lo_head_off = cur_kv_head * stride_k_mse_lo_h
    k_qjl_lo_head_off = cur_kv_head * stride_k_qjl_lo_h
    k_mse_hi_head_off = cur_kv_head * stride_k_mse_hi_h
    k_qjl_hi_head_off = cur_kv_head * stride_k_qjl_hi_h
    v_lo_head_off = cur_kv_head * stride_v_lo_h
    v_hi_head_off = cur_kv_head * stride_v_hi_h

    # Q row offsets for this block
    q_row_offsets = q_start + cur_block_m * BLOCK_M + offs_m  # [BLOCK_M]

    # Initialize accumulators — separate for lo and hi groups
    acc_lo = tl.zeros([BLOCK_M, BLOCK_DV_LO], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, BLOCK_DV_HI], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Unified loop: process all KV tokens (prefix + extend)
    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < kv_len

        # Load kv_loc from page table
        kv_loc = tl.load(
            kv_indices + kv_start + start_n + offs_n,
            mask=mask_n,
            other=0,
        )

        # Build mask
        final_mask = mask_m[:, None] & mask_n[None, :]

        # Causal mask
        if IS_CAUSAL:
            q_idx = cur_block_m * BLOCK_M + offs_m[:, None]
            k_idx_in_total = start_n + offs_n[None, :]
            k_is_extend = k_idx_in_total >= prefix_len
            k_idx_in_extend = k_idx_in_total - prefix_len
            causal_mask = tl.where(
                k_is_extend,
                q_idx >= k_idx_in_extend,
                True,
            )
            final_mask &= causal_mask

        # ==============================================================
        # K scoring — lo group: MSE + QJL -> [BLOCK_M, BLOCK_N]
        # ==============================================================
        mse_scores_lo = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d_start in range(0, D_LO, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_LO

            q_rot_ptrs = (
                q_row_offsets[:, None] * stride_qlo_bs
                + cur_head * stride_qlo_h
                + d_offs[None, :]
            )
            q_rot_block = tl.load(
                Q_rot_lo + q_rot_ptrs,
                mask=mask_m[:, None] & d_mask[None, :],
                other=0.0,
            )

            mse_idx = _extract_bits(
                K_MSE_Buffer_lo, kv_loc, stride_k_mse_lo_bs, k_mse_lo_head_off,
                d_offs, mask_n, d_mask, MSE_BITS_LO, BLOCK_N, BLOCK_D,
            )
            k_val = tl.load(K_Codebook_lo + mse_idx)
            k_val = tl.where(mask_n[:, None] & d_mask[None, :], k_val, 0.0)

            mse_scores_lo += tl.dot(q_rot_block, tl.trans(k_val))

        qjl_scores_lo = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d_start in range(0, D_LO, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_LO

            q_proj_ptrs = (
                q_row_offsets[:, None] * stride_qlo_bs
                + cur_head * stride_qlo_h
                + d_offs[None, :]
            )
            q_proj_block = tl.load(
                Q_proj_lo + q_proj_ptrs,
                mask=mask_m[:, None] & d_mask[None, :],
                other=0.0,
            )

            sign = _extract_sign_bits(
                K_QJL_Buffer_lo, kv_loc, stride_k_qjl_lo_bs, k_qjl_lo_head_off,
                d_offs, mask_n, d_mask, BLOCK_N, BLOCK_D,
            )
            sign = tl.where(mask_n[:, None] & d_mask[None, :], sign, 0.0)

            qjl_scores_lo += tl.dot(q_proj_block, tl.trans(sign))

        # ==============================================================
        # K scoring — hi group: MSE + QJL -> [BLOCK_M, BLOCK_N]
        # ==============================================================
        mse_scores_hi = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d_start in range(0, D_HI, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_HI

            q_rot_ptrs = (
                q_row_offsets[:, None] * stride_qhi_bs
                + cur_head * stride_qhi_h
                + d_offs[None, :]
            )
            q_rot_block = tl.load(
                Q_rot_hi + q_rot_ptrs,
                mask=mask_m[:, None] & d_mask[None, :],
                other=0.0,
            )

            mse_idx = _extract_bits(
                K_MSE_Buffer_hi, kv_loc, stride_k_mse_hi_bs, k_mse_hi_head_off,
                d_offs, mask_n, d_mask, MSE_BITS_HI, BLOCK_N, BLOCK_D,
            )
            k_val = tl.load(K_Codebook_hi + mse_idx)
            k_val = tl.where(mask_n[:, None] & d_mask[None, :], k_val, 0.0)

            mse_scores_hi += tl.dot(q_rot_block, tl.trans(k_val))

        qjl_scores_hi = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d_start in range(0, D_HI, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D_HI

            q_proj_ptrs = (
                q_row_offsets[:, None] * stride_qhi_bs
                + cur_head * stride_qhi_h
                + d_offs[None, :]
            )
            q_proj_block = tl.load(
                Q_proj_hi + q_proj_ptrs,
                mask=mask_m[:, None] & d_mask[None, :],
                other=0.0,
            )

            sign = _extract_sign_bits(
                K_QJL_Buffer_hi, kv_loc, stride_k_qjl_hi_bs, k_qjl_hi_head_off,
                d_offs, mask_n, d_mask, BLOCK_N, BLOCK_D,
            )
            sign = tl.where(mask_n[:, None] & d_mask[None, :], sign, 0.0)

            qjl_scores_hi += tl.dot(q_proj_block, tl.trans(sign))

        # ==============================================================
        # Combined K score (additive across groups, paper Section 3.3)
        # ==============================================================
        k_norms_lo = tl.load(
            K_Norms_lo + kv_loc * stride_kn_lo_bs + cur_kv_head,
            mask=mask_n, other=0.0,
        ).to(tl.float32)
        k_res_norms_lo = tl.load(
            K_ResNorms_lo + kv_loc * stride_kn_lo_bs + cur_kv_head,
            mask=mask_n, other=0.0,
        ).to(tl.float32)
        k_norms_hi = tl.load(
            K_Norms_hi + kv_loc * stride_kn_hi_bs + cur_kv_head,
            mask=mask_n, other=0.0,
        ).to(tl.float32)
        k_res_norms_hi = tl.load(
            K_ResNorms_hi + kv_loc * stride_kn_hi_bs + cur_kv_head,
            mask=mask_n, other=0.0,
        ).to(tl.float32)

        qk = (
            k_norms_lo[None, :] * (
                mse_scores_lo + qjl_scale_lo * k_res_norms_lo[None, :] * qjl_scores_lo
            )
            + k_norms_hi[None, :] * (
                mse_scores_hi + qjl_scale_hi * k_res_norms_hi[None, :] * qjl_scores_hi
            )
        )
        qk *= sm_scale
        qk = tl.where(final_mask, qk, float("-inf"))

        # ==============================================================
        # Online softmax — rescale BOTH accumulators
        # ==============================================================
        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        n_e_max = tl.maximum(row_max_fixed, e_max)

        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)
        acc_lo = acc_lo * re_scale[:, None]
        acc_hi = acc_hi * re_scale[:, None]

        # ==============================================================
        # V weighted sum — lo group
        # ==============================================================
        v_norms_lo_vec = tl.load(
            V_Norms_lo + kv_loc * stride_vn_lo_bs + cur_kv_head,
            mask=mask_n, other=0.0,
        ).to(tl.float32)

        v_idx_lo = _extract_bits(
            V_Packed_lo, kv_loc, stride_v_lo_bs, v_lo_head_off,
            offs_dv_lo, mask_n, mask_dv_lo, V_BITS_LO, BLOCK_N, BLOCK_DV_LO,
        )
        v_val_lo = tl.load(V_Codebook_lo + v_idx_lo)
        v_val_lo = tl.where(mask_n[:, None] & mask_dv_lo[None, :], v_val_lo, 0.0)

        pv_lo = p * v_norms_lo_vec[None, :]
        acc_lo += tl.dot(pv_lo.to(tl.float32), v_val_lo)

        # ==============================================================
        # V weighted sum — hi group
        # ==============================================================
        v_norms_hi_vec = tl.load(
            V_Norms_hi + kv_loc * stride_vn_hi_bs + cur_kv_head,
            mask=mask_n, other=0.0,
        ).to(tl.float32)

        v_idx_hi = _extract_bits(
            V_Packed_hi, kv_loc, stride_v_hi_bs, v_hi_head_off,
            offs_dv_hi, mask_n, mask_dv_hi, V_BITS_HI, BLOCK_N, BLOCK_DV_HI,
        )
        v_val_hi = tl.load(V_Codebook_hi + v_idx_hi)
        v_val_hi = tl.where(mask_n[:, None] & mask_dv_hi[None, :], v_val_hi, 0.0)

        pv_hi = p * v_norms_hi_vec[None, :]
        acc_hi += tl.dot(pv_hi.to(tl.float32), v_val_hi)

        e_max = n_e_max

    # ==================================================================
    # Store output in split order: [lo | hi]
    # ==================================================================
    deno = tl.where(deno == 0.0, 1.0, deno)
    out_lo = acc_lo / deno[:, None]
    out_hi = acc_hi / deno[:, None]

    o_base = (
        (q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_o_bs
        + cur_head * stride_o_h
    )

    # Store lo channels at offsets [0..D_LO-1]
    tl.store(
        O_rot + o_base + offs_dv_lo[None, :],
        out_lo,
        mask=mask_m[:, None] & mask_dv_lo[None, :],
    )

    # Store hi channels at offsets [D_LO..D_LO+D_HI-1]
    tl.store(
        O_rot + o_base + D_LO + offs_dv_hi[None, :],
        out_hi,
        mask=mask_m[:, None] & mask_dv_hi[None, :],
    )


# =============================================================================
# Python wrapper: split-channel extend attention
# =============================================================================


def turboquant_extend_attention_fwd_split(
    q_rot_lo: torch.Tensor,         # [T, H_q, D_lo] float32
    q_proj_lo: torch.Tensor,        # [T, H_q, D_lo] float32
    q_rot_hi: torch.Tensor,         # [T, H_q, D_hi] float32
    q_proj_hi: torch.Tensor,        # [T, H_q, D_hi] float32
    k_mse_lo: torch.Tensor,         # [pool_size, H_kv, packed_mse_width_lo] uint8
    k_qjl_lo: torch.Tensor,         # [pool_size, H_kv, packed_qjl_width_lo] uint8
    k_norms_lo: torch.Tensor,       # [pool_size, H_kv] float16
    k_res_norms_lo: torch.Tensor,   # [pool_size, H_kv] float16
    k_mse_hi: torch.Tensor,         # [pool_size, H_kv, packed_mse_width_hi] uint8
    k_qjl_hi: torch.Tensor,         # [pool_size, H_kv, packed_qjl_width_hi] uint8
    k_norms_hi: torch.Tensor,       # [pool_size, H_kv] float16
    k_res_norms_hi: torch.Tensor,   # [pool_size, H_kv] float16
    v_packed_lo: torch.Tensor,      # [pool_size, H_kv, packed_v_width_lo] uint8
    v_norms_lo: torch.Tensor,       # [pool_size, H_kv] float16
    v_packed_hi: torch.Tensor,      # [pool_size, H_kv, packed_v_width_hi] uint8
    v_norms_hi: torch.Tensor,       # [pool_size, H_kv] float16
    k_cb_lo: torch.Tensor,          # [2^mse_bits_lo] float32
    k_cb_hi: torch.Tensor,          # [2^mse_bits_hi] float32
    v_cb_lo: torch.Tensor,          # [2^v_bits_lo] float32
    v_cb_hi: torch.Tensor,          # [2^v_bits_hi] float32
    o_rot_split: torch.Tensor,      # [T, H_q, D_lo+D_hi] output in split order
    qo_indptr: torch.Tensor,        # [batch+1]
    kv_indptr: torch.Tensor,        # [batch+1]
    kv_indices: torch.Tensor,       # [total_kv]
    prefix_lens: torch.Tensor,      # [batch]
    max_extend_len: int,
    sm_scale: float,                # 1/sqrt(head_dim)
    qjl_scale_lo: float,            # sqrt(pi/2) / D_lo
    qjl_scale_hi: float,            # sqrt(pi/2) / D_hi
    mse_bits_lo: int,
    mse_bits_hi: int,
    v_bits_lo: int,
    v_bits_hi: int,
    d_lo: int,
    d_hi: int,
    head_dim: int,                  # d_lo + d_hi
    is_causal: bool = True,
):
    """TurboQuant split-channel fused extend attention (single-stage).

    For fractional bit-widths (e.g. 3.5-bit), channels split into lo + hi groups
    with independent bit-widths. K scores combine additively before softmax.
    V accumulates per group with shared softmax weights.
    Output is in split order [lo | hi] — the caller must inverse-rotate per group
    and reassemble in original channel order.
    """
    batch_size = qo_indptr.shape[0] - 1
    head_num = q_rot_lo.shape[1]
    kv_head_num = k_mse_lo.shape[1]
    kv_group_num = head_num // kv_head_num

    BLOCK_DV_LO = triton.next_power_of_2(d_lo)
    BLOCK_DV_HI = triton.next_power_of_2(d_hi)

    # Adapt block sizes to GPU shared memory constraints.
    # Split-channel has even higher memory pressure (two groups of buffers).
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 9 and capability[1] == 0:
        # Hopper (H100): 228KB shared memory
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = min(64, min(BLOCK_DV_LO, BLOCK_DV_HI))
        num_stages = 2
    else:
        # sm86/89 (RTX 4090, A6000): ~100KB shared memory
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_D = min(32, min(BLOCK_DV_LO, BLOCK_DV_HI))
        num_stages = 1

    num_warps = 4 if head_dim <= 128 else 8

    grid = (batch_size, head_num, triton.cdiv(max_extend_len, BLOCK_M))

    _turboquant_extend_kernel_split[grid](
        q_rot_lo,
        q_proj_lo,
        q_rot_hi,
        q_proj_hi,
        k_mse_lo,
        k_qjl_lo,
        k_norms_lo,
        k_res_norms_lo,
        k_mse_hi,
        k_qjl_hi,
        k_norms_hi,
        k_res_norms_hi,
        v_packed_lo,
        v_norms_lo,
        v_packed_hi,
        v_norms_hi,
        k_cb_lo,
        k_cb_hi,
        v_cb_lo,
        v_cb_hi,
        o_rot_split,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        sm_scale,
        qjl_scale_lo,
        qjl_scale_hi,
        # Q_lo strides
        q_rot_lo.stride(0),
        q_rot_lo.stride(1),
        # Q_hi strides
        q_rot_hi.stride(0),
        q_rot_hi.stride(1),
        # K_MSE_lo strides
        k_mse_lo.stride(0),
        k_mse_lo.stride(1),
        # K_QJL_lo strides
        k_qjl_lo.stride(0),
        k_qjl_lo.stride(1),
        # K_Norms_lo stride
        k_norms_lo.stride(0),
        # K_MSE_hi strides
        k_mse_hi.stride(0),
        k_mse_hi.stride(1),
        # K_QJL_hi strides
        k_qjl_hi.stride(0),
        k_qjl_hi.stride(1),
        # K_Norms_hi stride
        k_norms_hi.stride(0),
        # V_Packed_lo strides
        v_packed_lo.stride(0),
        v_packed_lo.stride(1),
        # V_Norms_lo stride
        v_norms_lo.stride(0),
        # V_Packed_hi strides
        v_packed_hi.stride(0),
        v_packed_hi.stride(1),
        # V_Norms_hi stride
        v_norms_hi.stride(0),
        # O strides
        o_rot_split.stride(0),
        o_rot_split.stride(1),
        # Constexprs
        kv_group_num=kv_group_num,
        MSE_BITS_LO=mse_bits_lo,
        MSE_BITS_HI=mse_bits_hi,
        V_BITS_LO=v_bits_lo,
        V_BITS_HI=v_bits_hi,
        D_LO=d_lo,
        D_HI=d_hi,
        BLOCK_DV_LO=BLOCK_DV_LO,
        BLOCK_DV_HI=BLOCK_DV_HI,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        Lv=head_dim,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# =============================================================================
# FUSED Python wrappers: extend + inline inverse FWHT (Phase H2.9)
# =============================================================================


def turboquant_extend_attention_fused_fwd(
    q_rot: torch.Tensor,            # [T, H_q, D] pre-rotated queries (float32)
    q_proj: torch.Tensor,           # [T, H_q, D] pre-projected queries (float32)
    signs: torch.Tensor,            # [padded_dim] Hadamard sign vector
    hadamard_scale: float,          # 1/sqrt(padded_dim)
    original_dim: int,              # original head_dim (for truncation)
    k_mse_buffer: torch.Tensor,
    k_qjl_buffer: torch.Tensor,
    k_norms: torch.Tensor,
    k_res_norms: torch.Tensor,
    v_packed_buffer: torch.Tensor,
    v_norms: torch.Tensor,
    k_codebook: torch.Tensor,
    v_codebook: torch.Tensor,
    o: torch.Tensor,                # [T, H_q, padded_dim] output buffer
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_extend_len: int,
    sm_scale: float,
    qjl_scale: float,
    mse_bits: int,
    v_bits: int,
    head_dim: int,
    is_causal: bool = True,
):
    """TurboQuant extend with post-kernel inverse FWHT (Phase H2.9).

    Calls the original extend kernel, then applies inverse Hadamard transform.
    Output o is de-rotated — caller just truncates to original_dim.

    Values accumulated in rotated space — inverse applied ONCE (Phase H).
    """
    from sglang.srt.layers.quantization.turboquant.triton_fwht import (
        triton_fwht_inverse,
    )

    # Original extend kernel → o gets rotated-space output
    turboquant_extend_attention_fwd(
        q_rot, q_proj,
        k_mse_buffer, k_qjl_buffer, k_norms, k_res_norms,
        v_packed_buffer, v_norms,
        k_codebook, v_codebook,
        o,
        qo_indptr, kv_indptr, kv_indices, prefix_lens,
        max_extend_len, sm_scale, qjl_scale,
        mse_bits, v_bits, head_dim, is_causal,
    )

    # Inverse FWHT on output (standalone kernel — efficient for many rows)
    o_inv = triton_fwht_inverse(o, signs, head_dim, original_dim, hadamard_scale)
    o.copy_(o_inv)


def turboquant_extend_attention_fused_fwd_split(
    q_rot_lo: torch.Tensor,        # [T, H_q, D_lo] pre-rotated (float32)
    q_proj_lo: torch.Tensor,
    q_rot_hi: torch.Tensor,        # [T, H_q, D_hi] pre-rotated (float32)
    q_proj_hi: torch.Tensor,
    signs_lo: torch.Tensor,        # [padded_D_lo]
    signs_hi: torch.Tensor,        # [padded_D_hi]
    hadamard_scale_lo: float,
    hadamard_scale_hi: float,
    original_dim_lo: int,
    original_dim_hi: int,
    k_mse_lo: torch.Tensor,
    k_qjl_lo: torch.Tensor,
    k_norms_lo: torch.Tensor,
    k_res_norms_lo: torch.Tensor,
    k_mse_hi: torch.Tensor,
    k_qjl_hi: torch.Tensor,
    k_norms_hi: torch.Tensor,
    k_res_norms_hi: torch.Tensor,
    v_packed_lo: torch.Tensor,
    v_norms_lo: torch.Tensor,
    v_packed_hi: torch.Tensor,
    v_norms_hi: torch.Tensor,
    k_cb_lo: torch.Tensor,
    k_cb_hi: torch.Tensor,
    v_cb_lo: torch.Tensor,
    v_cb_hi: torch.Tensor,
    o_split: torch.Tensor,         # [T, H_q, padded_D_lo+padded_D_hi] output buffer
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_extend_len: int,
    sm_scale: float,
    qjl_scale_lo: float,
    qjl_scale_hi: float,
    mse_bits_lo: int,
    mse_bits_hi: int,
    v_bits_lo: int,
    v_bits_hi: int,
    d_lo: int,
    d_hi: int,
    head_dim: int,
    is_causal: bool = True,
):
    """Split-channel extend with post-kernel per-group inverse FWHT (Phase H2.9).

    Output o_split is de-rotated in split order [lo | hi].
    """
    from sglang.srt.layers.quantization.turboquant.triton_fwht import (
        triton_fwht_inverse,
    )

    padded_d_lo = q_rot_lo.shape[-1]
    padded_d_hi = q_rot_hi.shape[-1]

    # Original split extend kernel → o_split gets rotated-space output
    turboquant_extend_attention_fwd_split(
        q_rot_lo, q_proj_lo, q_rot_hi, q_proj_hi,
        k_mse_lo, k_qjl_lo, k_norms_lo, k_res_norms_lo,
        k_mse_hi, k_qjl_hi, k_norms_hi, k_res_norms_hi,
        v_packed_lo, v_norms_lo, v_packed_hi, v_norms_hi,
        k_cb_lo, k_cb_hi, v_cb_lo, v_cb_hi,
        o_split,
        qo_indptr, kv_indptr, kv_indices, prefix_lens,
        max_extend_len, sm_scale, qjl_scale_lo, qjl_scale_hi,
        mse_bits_lo, mse_bits_hi, v_bits_lo, v_bits_hi,
        d_lo, d_hi, head_dim, is_causal,
    )

    # Inverse FWHT per group
    o_lo_rot = o_split[..., :padded_d_lo].contiguous()
    o_hi_rot = o_split[..., padded_d_lo:padded_d_lo + padded_d_hi].contiguous()

    o_lo_inv = triton_fwht_inverse(o_lo_rot, signs_lo, padded_d_lo, original_dim_lo, hadamard_scale_lo)
    o_hi_inv = triton_fwht_inverse(o_hi_rot, signs_hi, padded_d_hi, original_dim_hi, hadamard_scale_hi)

    o_split[..., :padded_d_lo].copy_(o_lo_inv)
    o_split[..., padded_d_lo:padded_d_lo + padded_d_hi].copy_(o_hi_inv)
