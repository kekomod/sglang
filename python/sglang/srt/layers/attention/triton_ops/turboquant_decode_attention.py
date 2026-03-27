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
TurboQuant fused decode attention kernel (arXiv:2504.19874).

Two-stage flash-decoding that reads packed b-bit quantized KV buffers directly:
  - Stage 1: TurboQuant_prod scoring (MSE codebook + QJL sign correction) for K,
             online softmax, TurboQuant_mse weighted sum for V.
  - Stage 2: Merge across KV splits (reuses SGLang's existing _fwd_kernel_stage2).

Phase A: integer bits (e.g. 3-bit). Single set of buffers.
Phase B: split-channel for fractional bits (e.g. 3.5-bit). Two groups (lo + hi)
         with independent bit-widths; scores combine additively before softmax.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _fwd_kernel_stage2,
    _MIN_BLOCK_KV,
)
from sglang.srt.layers.quantization.turboquant.triton_fwht import _log2

logger = logging.getLogger(__name__)


# =============================================================================
# Inline FWHT helper for kernel fusion (Phase G)
# =============================================================================


@triton.jit
def _fwht_butterfly_inplace(
    scratch_ptr,
    offs,
    LOG2_D: tl.constexpr,
    D: tl.constexpr,
):
    """In-place Walsh-Hadamard butterfly on a scratch buffer.

    Performs LOG2_D butterfly stages using the XOR-partner trick.
    Caller must store data to scratch_ptr before calling.
    After return, scratch_ptr contains the FWHT result (unscaled).

    Uses tl.debug_barrier() between stages to prevent reordering.
    """
    for stage in tl.static_range(LOG2_D):
        stride = 1 << stage
        top_mask = (offs & stride) == 0
        partner = offs ^ stride
        x_self = tl.load(scratch_ptr + offs)
        x_partner = tl.load(scratch_ptr + partner)
        x = tl.where(top_mask, x_self + x_partner, x_partner - x_self)
        tl.store(scratch_ptr + offs, x)
        tl.debug_barrier()


# =============================================================================
# Bit extraction helpers
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

    Spillover analysis by bit width:
      1-bit: offset cycles {0,1,...,7} — max 0+1=1 <= 8, never spills
      2-bit: offset cycles {0,2,4,6} — max 6+2=8, never spills
      3-bit: offset cycles {0,3,6,1,4,7,2,5} — spills at 6 (6+3=9) and 7 (7+3=10)
      4-bit: offset cycles {0,4} — max 4+4=8, never spills
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
# Stage 1: TurboQuant fused decode attention
# =============================================================================


@triton.jit
def _turboquant_fwd_kernel_stage1(
    # Pre-rotated/projected queries: [B, H_q, D]
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
    # Codebooks (small, fits in L1/registers)
    K_Codebook,        # [2^mse_bits] float32
    V_Codebook,        # [2^v_bits] float32
    # Scaling factors
    sm_scale,          # 1/sqrt(head_dim)
    qjl_scale,         # sqrt(pi/2) / D
    # Page table
    kv_indptr,         # [B+1]
    kv_indices,        # [total_kv_len]
    num_kv_splits,     # [B]
    # Output intermediates (shared with Stage 2)
    Att_Out,           # [B, H_q, max_kv_splits, Lv]
    Att_Lse,           # [B, H_q, max_kv_splits] (interleaved in same buffer)
    # Strides — Q: [B, H_q, D]
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
    # Strides — Att_Out: [B, H_q, max_kv_splits, Lv]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    # Constexprs
    kv_group_num: tl.constexpr,
    MSE_BITS: tl.constexpr,       # key MSE quantization bits (e.g. 2 for 3-bit total)
    V_BITS: tl.constexpr,         # value quantization bits
    BLOCK_DV: tl.constexpr,       # >= HEAD_DIM, power of 2
    BLOCK_N: tl.constexpr,        # tokens per tile
    BLOCK_D: tl.constexpr,        # dimension tile for K scoring loops
    MIN_BLOCK_KV: tl.constexpr,
    Lv: tl.constexpr,             # actual head_dim (for V output and masks)
):
    """TurboQuant decode attention Stage 1.

    For each KV split assigned to this program:
    1. Score keys via TurboQuant_prod (MSE codebook + QJL sign correction)
    2. Online softmax
    3. Weighted-sum values via TurboQuant_mse (codebook gather in rotated space)
    4. Store intermediate (acc/sum, lse) for Stage 2 merging
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    # KV range for this batch element
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # This split's range
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # V accumulator in rotated space: [BLOCK_DV]
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < Lv

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # Query base offset
        q_base = cur_batch * stride_q_bs + cur_head * stride_q_h

        # Precomputed head offsets for packed buffer access
        k_mse_head_off = cur_kv_head * stride_k_mse_h
        k_qjl_head_off = cur_kv_head * stride_k_qjl_h
        v_head_off = cur_kv_head * stride_v_h

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end

            # Load kv_loc from page table (same as standard decode kernel)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=n_mask,
                other=0,
            )

            # ==============================================================
            # K scoring: TurboQuant_prod (paper Algorithm 1)
            # score = ||k|| * (q_rot^T @ C[idx] + scale * ||k_res|| * q_proj^T @ sign)
            # ==============================================================

            # MSE component: sum_d(q_rot[d] * K_Codebook[K_mse_idx[n, d]])
            mse_score = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, Lv, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < Lv

                # Extract b-bit MSE indices: [BLOCK_N, BLOCK_D]
                mse_idx = _extract_bits(
                    K_MSE_Buffer, kv_loc, stride_k_mse_bs, k_mse_head_off,
                    d_offs, n_mask, d_mask, MSE_BITS, BLOCK_N, BLOCK_D,
                )

                # Codebook gather: [BLOCK_N, BLOCK_D]
                k_val = tl.load(K_Codebook + mse_idx)
                k_val = tl.where(n_mask[:, None] & d_mask[None, :], k_val, 0.0)

                # Query slice: [BLOCK_D]
                q_rot_slice = tl.load(
                    Q_rot + q_base + d_offs, mask=d_mask, other=0.0,
                )

                # Partial dot product: [BLOCK_N]
                mse_score += tl.sum(q_rot_slice[None, :] * k_val, axis=1)

            # QJL component: sum_d(q_proj[d] * sign[n, d])
            qjl_score = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, Lv, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < Lv

                # Extract 1-bit signs: [BLOCK_N, BLOCK_D] -> {-1, +1}
                sign = _extract_sign_bits(
                    K_QJL_Buffer, kv_loc, stride_k_qjl_bs, k_qjl_head_off,
                    d_offs, n_mask, d_mask, BLOCK_N, BLOCK_D,
                )
                sign = tl.where(n_mask[:, None] & d_mask[None, :], sign, 0.0)

                # Query slice: [BLOCK_D]
                q_proj_slice = tl.load(
                    Q_proj + q_base + d_offs, mask=d_mask, other=0.0,
                )

                qjl_score += tl.sum(q_proj_slice[None, :] * sign, axis=1)

            # Combined score (paper Eq. 7):
            # qk = ||k|| * (mse_score + sqrt(pi/2)/D * ||k_res|| * qjl_score) / sqrt(d)
            k_norms = tl.load(
                K_Norms + kv_loc * stride_kn_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            k_res_norms = tl.load(
                K_ResNorms + kv_loc * stride_kn_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)

            qk = k_norms * (mse_score + qjl_scale * k_res_norms * qjl_score)
            qk *= sm_scale
            qk = tl.where(n_mask, qk, float("-inf"))

            # ==============================================================
            # Online softmax (identical to standard decode kernel)
            # ==============================================================
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

            # ==============================================================
            # V weighted sum: TurboQuant_mse (paper Algorithm 2)
            # acc[d] += sum_n(p[n] * v_norm[n] * V_Codebook[V_idx[n, d]])
            # ==============================================================
            v_norms_vec = tl.load(
                V_Norms + kv_loc * stride_vn_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)

            # Weight = softmax_prob * v_norm: [BLOCK_N]
            w = p * v_norms_vec

            # Extract V indices for full dimension: [BLOCK_N, BLOCK_DV]
            v_idx = _extract_bits(
                V_Packed, kv_loc, stride_v_bs, v_head_off,
                offs_dv, n_mask, mask_dv, V_BITS, BLOCK_N, BLOCK_DV,
            )

            # Codebook gather: [BLOCK_N, BLOCK_DV]
            v_val = tl.load(V_Codebook + v_idx)
            v_val = tl.where(n_mask[:, None] & mask_dv[None, :], v_val, 0.0)

            # Weighted sum: acc[d] += sum_n(w[n] * v_val[n, d])
            acc += tl.sum(w[:, None] * v_val, axis=0)

        # ==================================================================
        # Store intermediate results for Stage 2 merging
        # ==================================================================
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum, mask=mask_dv)

        offs_mid_lse = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_lse, e_max + tl.log(e_sum))


# =============================================================================
# Stage 1 FUSED: inline forward FWHT (Phase G)
# =============================================================================


@triton.jit
def _turboquant_fwd_kernel_stage1_fused(
    # Raw queries (NOT pre-rotated): [B, H_q, D]
    Q_raw,
    Q_proj,
    # Hadamard transform parameters
    Signs,              # [padded_dim] float32 Rademacher sign vector
    hadamard_scale,     # 1/sqrt(padded_dim)
    # Packed K buffers
    K_MSE_Buffer,       # [pool_size, H_kv, packed_mse_width] uint8
    K_QJL_Buffer,       # [pool_size, H_kv, packed_qjl_width] uint8
    K_Norms,            # [pool_size, H_kv] float16
    K_ResNorms,         # [pool_size, H_kv] float16
    # Packed V buffers
    V_Packed,           # [pool_size, H_kv, packed_v_width] uint8
    V_Norms,            # [pool_size, H_kv] float16
    # Codebooks
    K_Codebook,         # [2^mse_bits] float32
    V_Codebook,         # [2^v_bits] float32
    # Scaling factors
    sm_scale,           # 1/sqrt(head_dim)
    qjl_scale,          # sqrt(pi/2) / D
    # Page table
    kv_indptr,          # [B+1]
    kv_indices,         # [total_kv_len]
    num_kv_splits,      # [B]
    # Output intermediates (also used as FWHT scratch at kernel start)
    Att_Out,            # [B, H_q, max_kv_splits, Lv]
    Att_Lse,            # [B, H_q, max_kv_splits]
    # Strides — Q: [B, H_q, D]
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
    # Strides — Att_Out: [B, H_q, max_kv_splits, Lv]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    # Constexprs
    kv_group_num: tl.constexpr,
    MSE_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lv: tl.constexpr,
    LOG2_D: tl.constexpr,
):
    """TurboQuant fused decode Stage 1 with inline forward FWHT (Phase G).

    Same as _turboquant_fwd_kernel_stage1 but accepts raw (unrotated) queries
    and performs the forward Hadamard transform inline using the Att_Out slot
    as scratch buffer. Eliminates the separate FWHT kernel launch.

    Values accumulated in rotated space — no per-token inverse Hadamard (Phase H).
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    # KV range for this batch element
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # This split's range
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # V accumulator in rotated space: [BLOCK_DV]
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < Lv

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    # Scratch base in Att_Out for this program's unique (batch, head, split) slot
    scratch_base = (
        cur_batch * stride_mid_ob
        + cur_head * stride_mid_oh
        + split_kv_id * stride_mid_os
    )

    if split_kv_end > split_kv_start:
        # ==============================================================
        # Inline forward FWHT on query (Phase G)
        # Uses Att_Out[batch, head, split, :] as scratch buffer.
        # This scratch is overwritten with real output at kernel end.
        # ==============================================================
        q_base = cur_batch * stride_q_bs + cur_head * stride_q_h

        # Load raw query and multiply by Hadamard signs
        q_raw = tl.load(Q_raw + q_base + offs_dv, mask=mask_dv, other=0.0)
        signs = tl.load(Signs + offs_dv, mask=mask_dv, other=1.0)
        q_signed = q_raw * signs

        # Store to scratch and run butterfly
        tl.store(Att_Out + scratch_base + offs_dv, q_signed, mask=mask_dv)
        tl.debug_barrier()
        _fwht_butterfly_inplace(Att_Out + scratch_base, offs_dv, LOG2_D, BLOCK_DV)

        # Load rotated query and apply scale
        q_rot_full = tl.load(Att_Out + scratch_base + offs_dv, mask=mask_dv, other=0.0)
        q_rot_full = q_rot_full * hadamard_scale

        # Store scaled q_rot back to scratch for tiled reads during scoring
        tl.store(Att_Out + scratch_base + offs_dv, q_rot_full, mask=mask_dv)
        tl.debug_barrier()

        # Precomputed head offsets for packed buffer access
        k_mse_head_off = cur_kv_head * stride_k_mse_h
        k_qjl_head_off = cur_kv_head * stride_k_qjl_h
        v_head_off = cur_kv_head * stride_v_h

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end

            # Load kv_loc from page table
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=n_mask,
                other=0,
            )

            # ==============================================================
            # K scoring: TurboQuant_prod (paper Algorithm 1)
            # ==============================================================

            # MSE component: read q_rot from scratch in tiles
            mse_score = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, Lv, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < Lv

                mse_idx = _extract_bits(
                    K_MSE_Buffer, kv_loc, stride_k_mse_bs, k_mse_head_off,
                    d_offs, n_mask, d_mask, MSE_BITS, BLOCK_N, BLOCK_D,
                )

                k_val = tl.load(K_Codebook + mse_idx)
                k_val = tl.where(n_mask[:, None] & d_mask[None, :], k_val, 0.0)

                # Read q_rot tile from scratch (instead of Q_rot buffer)
                q_rot_slice = tl.load(
                    Att_Out + scratch_base + d_offs, mask=d_mask, other=0.0,
                )

                mse_score += tl.sum(q_rot_slice[None, :] * k_val, axis=1)

            # QJL component: q_proj from pre-computed cuBLAS matmul
            qjl_score = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, Lv, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < Lv

                sign = _extract_sign_bits(
                    K_QJL_Buffer, kv_loc, stride_k_qjl_bs, k_qjl_head_off,
                    d_offs, n_mask, d_mask, BLOCK_N, BLOCK_D,
                )
                sign = tl.where(n_mask[:, None] & d_mask[None, :], sign, 0.0)

                q_proj_slice = tl.load(
                    Q_proj + q_base + d_offs, mask=d_mask, other=0.0,
                )

                qjl_score += tl.sum(q_proj_slice[None, :] * sign, axis=1)

            # Combined score (paper Eq. 7)
            k_norms = tl.load(
                K_Norms + kv_loc * stride_kn_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            k_res_norms = tl.load(
                K_ResNorms + kv_loc * stride_kn_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)

            qk = k_norms * (mse_score + qjl_scale * k_res_norms * qjl_score)
            qk *= sm_scale
            qk = tl.where(n_mask, qk, float("-inf"))

            # ==============================================================
            # Online softmax
            # ==============================================================
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

            # ==============================================================
            # V weighted sum in rotated space (Phase H — no per-token H_inv)
            # ==============================================================
            v_norms_vec = tl.load(
                V_Norms + kv_loc * stride_vn_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)

            w = p * v_norms_vec

            v_idx = _extract_bits(
                V_Packed, kv_loc, stride_v_bs, v_head_off,
                offs_dv, n_mask, mask_dv, V_BITS, BLOCK_N, BLOCK_DV,
            )

            v_val = tl.load(V_Codebook + v_idx)
            v_val = tl.where(n_mask[:, None] & mask_dv[None, :], v_val, 0.0)

            acc += tl.sum(w[:, None] * v_val, axis=0)

        # ==================================================================
        # Store intermediate results (overwrites FWHT scratch)
        # ==================================================================
        tl.store(Att_Out + scratch_base + offs_dv, acc / e_sum, mask=mask_dv)

        offs_mid_lse = scratch_base // Lv
        tl.store(Att_Lse + offs_mid_lse, e_max + tl.log(e_sum))


# =============================================================================
# Stage 2 FUSED: merge + inline inverse FWHT (Phase G)
# =============================================================================


@triton.jit
def _turboquant_stage2_inv_fwht_kernel(
    Mid_O,              # [B, H_q, max_kv_splits, Lv] intermediate from Stage 1
    Mid_O_1,            # [B, H_q, max_kv_splits] LSE (aliased from Mid_O)
    O,                  # [B, H_q, Lv] output buffer (used as FWHT scratch, then final)
    Signs,              # [padded_dim] Hadamard sign vector
    hadamard_scale,     # 1/sqrt(padded_dim)
    kv_indptr,          # [B+1]
    num_kv_splits,      # [B]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    LOG2_D: tl.constexpr,
):
    """Fused Stage 2 merge + inverse FWHT (Phase G).

    Combines SGLang's _fwd_kernel_stage2 merge logic with inline inverse
    Hadamard transform. Eliminates the separate FWHT inverse kernel launch.

    Inverse Hadamard applied ONCE per layer, not per token.
    Correct because H_inv is linear: sum(a_t * H_inv(x_t)) = H_inv(sum(a_t * x_t)) (Phase H).
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    # Standard Stage 2 merge across KV splits
    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # Merged result (still in rotated space)
    merged = acc / e_sum

    # ==================================================================
    # Inline inverse FWHT (Phase G)
    # Uses O buffer as scratch for butterfly stages.
    # Inverse: butterfly → scale → signs
    # ==================================================================
    o_base = cur_batch * stride_obs + cur_head * stride_oh

    # Store merged to O (scratch for butterfly)
    tl.store(O + o_base + offs_d, merged, mask=mask_d)
    tl.debug_barrier()

    # Butterfly stages
    _fwht_butterfly_inplace(O + o_base, offs_d, LOG2_D, BLOCK_DV)

    # Apply scale and signs (inverse: butterfly → scale → signs)
    result = tl.load(O + o_base + offs_d, mask=mask_d, other=0.0)
    signs = tl.load(Signs + offs_d, mask=mask_d, other=1.0)
    result = result * hadamard_scale * signs

    tl.store(O + o_base + offs_d, result, mask=mask_d)


def _turboquant_stage2_inv_fwht_fwd(
    att_out,           # [B, H_q, max_kv_splits, Lv]
    att_lse,           # [B, H_q, max_kv_splits]
    o,                 # [B, H_q, Lv] output (de-rotated)
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
    head_dim,
    signs,             # [padded_dim] float32
    hadamard_scale,    # float
):
    """Merge stage-1 intermediates + inline inverse FWHT. Output is de-rotated."""
    batch, head_num = o.shape[0], o.shape[1]
    Lv = head_dim
    BLOCK_DV = triton.next_power_of_2(Lv)
    MAX_KV_SPLITS = max_kv_splits
    LOG2_D = _log2(BLOCK_DV)

    grid = (batch, head_num)
    _turboquant_stage2_inv_fwht_kernel[grid](
        att_out,
        att_lse,
        o,
        signs,
        hadamard_scale,
        kv_indptr,
        num_kv_splits,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        LOG2_D=LOG2_D,
        num_warps=4,
        num_stages=2,
    )


# =============================================================================
# Stage 2 wrapper (calls existing SGLang kernel)
# =============================================================================


def _turboquant_stage2_fwd(
    att_out,          # [B, H_q, max_kv_splits, Lv] intermediate from stage 1
    att_lse,          # [B, H_q, max_kv_splits] log-sum-exp from stage 1
    o,                # [B, H_q, Lv] output buffer
    kv_indptr,        # [B+1]
    num_kv_splits,    # [B]
    max_kv_splits,
    head_dim,         # actual Lv
):
    """Merge stage-1 intermediates across KV splits. Reuses SGLang's stage-2 kernel."""
    batch, head_num = o.shape[0], o.shape[1]
    Lv = head_dim
    BLOCK_DV = triton.next_power_of_2(Lv)
    MAX_KV_SPLITS = max_kv_splits

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        att_out,
        att_lse,
        o,
        1.0,            # v_scale = 1.0 (norms already applied in stage 1)
        kv_indptr,
        num_kv_splits,
        None,           # sink_ptr — no sinks for TurboQuant
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        HAS_SINK=False,
        num_warps=4,
        num_stages=2,
    )


# =============================================================================
# Python wrapper: full turboquant decode attention
# =============================================================================


def turboquant_decode_attention_fwd(
    q_rot: torch.Tensor,            # [B, H_q, D] pre-rotated queries (float32)
    q_proj: torch.Tensor,           # [B, H_q, D] pre-projected queries (float32)
    k_mse_buffer: torch.Tensor,     # [pool_size, H_kv, packed_mse_width] uint8
    k_qjl_buffer: torch.Tensor,     # [pool_size, H_kv, packed_qjl_width] uint8
    k_norms: torch.Tensor,          # [pool_size, H_kv] float16
    k_res_norms: torch.Tensor,      # [pool_size, H_kv] float16
    v_packed_buffer: torch.Tensor,   # [pool_size, H_kv, packed_v_width] uint8
    v_norms: torch.Tensor,          # [pool_size, H_kv] float16
    k_codebook: torch.Tensor,       # [2^mse_bits] float32
    v_codebook: torch.Tensor,       # [2^v_bits] float32
    o_rot: torch.Tensor,            # [B, H_q, D] output (in rotated space)
    kv_indptr: torch.Tensor,        # [B+1]
    kv_indices: torch.Tensor,       # [total_kv_len]
    num_kv_splits: torch.Tensor,    # [B]
    max_kv_splits: int,
    sm_scale: float,                # 1/sqrt(head_dim)
    qjl_scale: float,               # sqrt(pi/2) / D
    mse_bits: int,                  # MSE quantization bits (e.g. 2 for 3-bit K)
    v_bits: int,                    # V quantization bits
    head_dim: int,                  # actual head dimension
    attn_logits: torch.Tensor,      # [B, H_q, max_kv_splits, head_dim] scratch
    attn_lse: torch.Tensor,         # shares storage with attn_logits
):
    """TurboQuant fused decode attention (two-stage flash-decoding).

    Stage 1: For each KV split, compute TurboQuant_prod scores (MSE + QJL),
             apply online softmax, and accumulate TurboQuant_mse V weighted sum.
    Stage 2: Merge across KV splits using SGLang's existing reduction kernel.

    All K/V data is read from packed uint8 buffers with codebook gathers.
    Output is in rotated space — the caller must inverse-rotate.
    """
    batch, head_num = q_rot.shape[0], q_rot.shape[1]
    kv_head_num = k_mse_buffer.shape[1]
    kv_group_num = head_num // kv_head_num

    BLOCK_N = 64
    BLOCK_DV = triton.next_power_of_2(head_dim)
    # K scoring dimension tile: smaller tiles reduce register pressure
    # while still being efficient for the reduction.
    BLOCK_D = min(128, BLOCK_DV)

    MAX_KV_SPLITS = max_kv_splits
    grid = (batch, head_num, MAX_KV_SPLITS)

    num_warps = 4 if kv_group_num == 1 else 2

    # Stage 1: fused scoring + softmax + V accumulation
    _turboquant_fwd_kernel_stage1[grid](
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
        sm_scale,
        qjl_scale,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        attn_logits,
        attn_lse,
        # Q strides
        q_rot.stride(0),
        q_rot.stride(1),
        # K_MSE strides
        k_mse_buffer.stride(0),
        k_mse_buffer.stride(1),
        # K_QJL strides
        k_qjl_buffer.stride(0),
        k_qjl_buffer.stride(1),
        # K_Norms stride (same layout for norms and res_norms)
        k_norms.stride(0),
        # V_Packed strides
        v_packed_buffer.stride(0),
        v_packed_buffer.stride(1),
        # V_Norms stride
        v_norms.stride(0),
        # Att_Out strides
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        # Constexprs
        kv_group_num=kv_group_num,
        MSE_BITS=mse_bits,
        V_BITS=v_bits,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        Lv=head_dim,
        num_warps=num_warps,
        num_stages=2,
    )

    # Stage 2: merge across KV splits
    _turboquant_stage2_fwd(
        attn_logits,
        attn_lse,
        o_rot,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        head_dim,
    )


# =============================================================================
# Python wrapper: FUSED turboquant decode attention (Phase G)
# =============================================================================


def turboquant_decode_attention_fused_fwd(
    q_rot: torch.Tensor,            # [B, H_q, D] pre-rotated queries (float32)
    q_proj: torch.Tensor,           # [B, H_q, D] pre-projected queries (float32)
    signs: torch.Tensor,            # [padded_dim] Hadamard sign vector (for Stage 2 inv)
    hadamard_scale: float,          # 1/sqrt(padded_dim) (for Stage 2 inv)
    k_mse_buffer: torch.Tensor,     # [pool_size, H_kv, packed_mse_width] uint8
    k_qjl_buffer: torch.Tensor,     # [pool_size, H_kv, packed_qjl_width] uint8
    k_norms: torch.Tensor,          # [pool_size, H_kv] float16
    k_res_norms: torch.Tensor,      # [pool_size, H_kv] float16
    v_packed_buffer: torch.Tensor,   # [pool_size, H_kv, packed_v_width] uint8
    v_norms: torch.Tensor,          # [pool_size, H_kv] float16
    k_codebook: torch.Tensor,       # [2^mse_bits] float32
    v_codebook: torch.Tensor,       # [2^v_bits] float32
    o: torch.Tensor,                # [B, H_q, D] output (de-rotated, final)
    kv_indptr: torch.Tensor,        # [B+1]
    kv_indices: torch.Tensor,       # [total_kv_len]
    num_kv_splits: torch.Tensor,    # [B]
    max_kv_splits: int,
    sm_scale: float,                # 1/sqrt(head_dim)
    qjl_scale: float,               # sqrt(pi/2) / D
    mse_bits: int,
    v_bits: int,
    head_dim: int,
    attn_logits: torch.Tensor,      # [B, H_q, max_kv_splits, head_dim] scratch
    attn_lse: torch.Tensor,
):
    """TurboQuant decode with fused Stage 2 inverse FWHT (Phase G/H2).

    Stage 1: Original Triton kernel (pre-rotated queries, no redundant FWHT).
    Stage 2: Fused merge + inline inverse Hadamard (saves 1 kernel launch).

    Forward FWHT on queries is done externally by caller (standalone kernel).
    Output o is de-rotated — no further inverse Hadamard needed.
    """
    batch, head_num = q_rot.shape[0], q_rot.shape[1]
    kv_head_num = k_mse_buffer.shape[1]
    kv_group_num = head_num // kv_head_num

    BLOCK_N = 64
    BLOCK_DV = triton.next_power_of_2(head_dim)
    BLOCK_D = min(128, BLOCK_DV)

    MAX_KV_SPLITS = max_kv_splits
    grid = (batch, head_num, MAX_KV_SPLITS)

    num_warps = 4 if kv_group_num == 1 else 2

    # Stage 1: original kernel with pre-rotated queries (no redundant FWHT)
    _turboquant_fwd_kernel_stage1[grid](
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
        sm_scale,
        qjl_scale,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        attn_logits,
        attn_lse,
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
        # Att_Out strides
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        # Constexprs
        kv_group_num=kv_group_num,
        MSE_BITS=mse_bits,
        V_BITS=v_bits,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        Lv=head_dim,
        num_warps=num_warps,
        num_stages=2,
    )

    # Stage 2: fused merge + inline inverse FWHT (saves 1 kernel launch)
    _turboquant_stage2_inv_fwht_fwd(
        attn_logits,
        attn_lse,
        o,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        head_dim,
        signs,
        hadamard_scale,
    )


# =============================================================================
# Stage 1: Split-channel TurboQuant decode attention (Phase B)
# =============================================================================


@triton.jit
def _turboquant_fwd_kernel_stage1_split(
    # Pre-rotated/projected queries per group
    Q_rot_lo,           # [B, H_q, D_lo]
    Q_proj_lo,          # [B, H_q, D_lo]
    # Packed K buffers — lo group
    K_MSE_Buffer_lo,    # [pool_size, H_kv, packed_mse_width_lo] uint8
    K_QJL_Buffer_lo,    # [pool_size, H_kv, packed_qjl_width_lo] uint8
    K_Norms_lo,         # [pool_size, H_kv] float16
    K_ResNorms_lo,      # [pool_size, H_kv] float16
    # Pre-rotated/projected queries — hi group
    Q_rot_hi,           # [B, H_q, D_hi]
    Q_proj_hi,          # [B, H_q, D_hi]
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
    # Scaling factors
    sm_scale,           # 1/sqrt(head_dim)
    qjl_scale_lo,       # sqrt(pi/2) / D_LO  (per-group, per paper + MLX)
    qjl_scale_hi,       # sqrt(pi/2) / D_HI
    # Page table
    kv_indptr,          # [B+1]
    kv_indices,         # [total_kv_len]
    num_kv_splits,      # [B]
    # Output intermediates (shared with Stage 2)
    Att_Out,            # [B, H_q, max_kv_splits, Lv]  where Lv=D_LO+D_HI
    Att_Lse,            # [B, H_q, max_kv_splits]
    # Strides — Q_lo: [B, H_q, D_lo]
    stride_qlo_bs,
    stride_qlo_h,
    # Strides — Q_hi: [B, H_q, D_hi]
    stride_qhi_bs,
    stride_qhi_h,
    # Strides — K_MSE_lo: [pool_size, H_kv, packed_mse_width_lo]
    stride_k_mse_lo_bs,
    stride_k_mse_lo_h,
    # Strides — K_QJL_lo: [pool_size, H_kv, packed_qjl_width_lo]
    stride_k_qjl_lo_bs,
    stride_k_qjl_lo_h,
    # Strides — K_Norms_lo / K_ResNorms_lo: [pool_size, H_kv]
    stride_kn_lo_bs,
    # Strides — K_MSE_hi: [pool_size, H_kv, packed_mse_width_hi]
    stride_k_mse_hi_bs,
    stride_k_mse_hi_h,
    # Strides — K_QJL_hi: [pool_size, H_kv, packed_qjl_width_hi]
    stride_k_qjl_hi_bs,
    stride_k_qjl_hi_h,
    # Strides — K_Norms_hi / K_ResNorms_hi: [pool_size, H_kv]
    stride_kn_hi_bs,
    # Strides — V_Packed_lo: [pool_size, H_kv, packed_v_width_lo]
    stride_v_lo_bs,
    stride_v_lo_h,
    # Strides — V_Norms_lo: [pool_size, H_kv]
    stride_vn_lo_bs,
    # Strides — V_Packed_hi: [pool_size, H_kv, packed_v_width_hi]
    stride_v_hi_bs,
    stride_v_hi_h,
    # Strides — V_Norms_hi: [pool_size, H_kv]
    stride_vn_hi_bs,
    # Strides — Att_Out: [B, H_q, max_kv_splits, Lv]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    # Constexprs
    kv_group_num: tl.constexpr,
    MSE_BITS_LO: tl.constexpr,
    MSE_BITS_HI: tl.constexpr,
    V_BITS_LO: tl.constexpr,
    V_BITS_HI: tl.constexpr,
    D_LO: tl.constexpr,
    D_HI: tl.constexpr,
    BLOCK_DV_LO: tl.constexpr,   # next_power_of_2(D_LO)
    BLOCK_DV_HI: tl.constexpr,   # next_power_of_2(D_HI)
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lv: tl.constexpr,            # D_LO + D_HI = original head_dim
):
    """TurboQuant split-channel decode attention Stage 1.

    Two independent channel groups (lo + hi) with different bit-widths.
    K scores combine additively before softmax; V accumulates per group
    with shared softmax weights. Output stored in split order [lo | hi].
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    # KV range for this batch element
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # This split's range
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # V accumulators per group
    offs_dv_lo = tl.arange(0, BLOCK_DV_LO)
    offs_dv_hi = tl.arange(0, BLOCK_DV_HI)
    mask_dv_lo = offs_dv_lo < D_LO
    mask_dv_hi = offs_dv_hi < D_HI

    e_max = -float("inf")
    e_sum = 0.0
    acc_lo = tl.zeros([BLOCK_DV_LO], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_DV_HI], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # Query base offsets per group
        qlo_base = cur_batch * stride_qlo_bs + cur_head * stride_qlo_h
        qhi_base = cur_batch * stride_qhi_bs + cur_head * stride_qhi_h

        # Precomputed head offsets for packed buffer access
        k_mse_lo_head_off = cur_kv_head * stride_k_mse_lo_h
        k_qjl_lo_head_off = cur_kv_head * stride_k_qjl_lo_h
        k_mse_hi_head_off = cur_kv_head * stride_k_mse_hi_h
        k_qjl_hi_head_off = cur_kv_head * stride_k_qjl_hi_h
        v_lo_head_off = cur_kv_head * stride_v_lo_h
        v_hi_head_off = cur_kv_head * stride_v_hi_h

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end

            # Load kv_loc from page table
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=n_mask,
                other=0,
            )

            # ==============================================================
            # K scoring — lo group: MSE + QJL
            # ==============================================================
            mse_score_lo = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, D_LO, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D_LO

                mse_idx = _extract_bits(
                    K_MSE_Buffer_lo, kv_loc, stride_k_mse_lo_bs, k_mse_lo_head_off,
                    d_offs, n_mask, d_mask, MSE_BITS_LO, BLOCK_N, BLOCK_D,
                )
                k_val = tl.load(K_Codebook_lo + mse_idx)
                k_val = tl.where(n_mask[:, None] & d_mask[None, :], k_val, 0.0)

                q_rot_slice = tl.load(
                    Q_rot_lo + qlo_base + d_offs, mask=d_mask, other=0.0,
                )
                mse_score_lo += tl.sum(q_rot_slice[None, :] * k_val, axis=1)

            qjl_score_lo = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, D_LO, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D_LO

                sign = _extract_sign_bits(
                    K_QJL_Buffer_lo, kv_loc, stride_k_qjl_lo_bs, k_qjl_lo_head_off,
                    d_offs, n_mask, d_mask, BLOCK_N, BLOCK_D,
                )
                sign = tl.where(n_mask[:, None] & d_mask[None, :], sign, 0.0)

                q_proj_slice = tl.load(
                    Q_proj_lo + qlo_base + d_offs, mask=d_mask, other=0.0,
                )
                qjl_score_lo += tl.sum(q_proj_slice[None, :] * sign, axis=1)

            # ==============================================================
            # K scoring — hi group: MSE + QJL
            # ==============================================================
            mse_score_hi = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, D_HI, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D_HI

                mse_idx = _extract_bits(
                    K_MSE_Buffer_hi, kv_loc, stride_k_mse_hi_bs, k_mse_hi_head_off,
                    d_offs, n_mask, d_mask, MSE_BITS_HI, BLOCK_N, BLOCK_D,
                )
                k_val = tl.load(K_Codebook_hi + mse_idx)
                k_val = tl.where(n_mask[:, None] & d_mask[None, :], k_val, 0.0)

                q_rot_slice = tl.load(
                    Q_rot_hi + qhi_base + d_offs, mask=d_mask, other=0.0,
                )
                mse_score_hi += tl.sum(q_rot_slice[None, :] * k_val, axis=1)

            qjl_score_hi = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, D_HI, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D_HI

                sign = _extract_sign_bits(
                    K_QJL_Buffer_hi, kv_loc, stride_k_qjl_hi_bs, k_qjl_hi_head_off,
                    d_offs, n_mask, d_mask, BLOCK_N, BLOCK_D,
                )
                sign = tl.where(n_mask[:, None] & d_mask[None, :], sign, 0.0)

                q_proj_slice = tl.load(
                    Q_proj_hi + qhi_base + d_offs, mask=d_mask, other=0.0,
                )
                qjl_score_hi += tl.sum(q_proj_slice[None, :] * sign, axis=1)

            # ==============================================================
            # Combined K score (additive across groups, paper Section 3.3)
            # ==============================================================
            k_norms_lo = tl.load(
                K_Norms_lo + kv_loc * stride_kn_lo_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            k_res_norms_lo = tl.load(
                K_ResNorms_lo + kv_loc * stride_kn_lo_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            k_norms_hi = tl.load(
                K_Norms_hi + kv_loc * stride_kn_hi_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            k_res_norms_hi = tl.load(
                K_ResNorms_hi + kv_loc * stride_kn_hi_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)

            qk = (
                k_norms_lo * (mse_score_lo + qjl_scale_lo * k_res_norms_lo * qjl_score_lo)
                + k_norms_hi * (mse_score_hi + qjl_scale_hi * k_res_norms_hi * qjl_score_hi)
            )
            qk *= sm_scale
            qk = tl.where(n_mask, qk, float("-inf"))

            # ==============================================================
            # Online softmax — rescale BOTH accumulators
            # ==============================================================
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc_lo *= re_scale
            acc_hi *= re_scale
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

            # ==============================================================
            # V weighted sum — lo group
            # ==============================================================
            v_norms_lo_vec = tl.load(
                V_Norms_lo + kv_loc * stride_vn_lo_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            w_lo = p * v_norms_lo_vec

            v_idx_lo = _extract_bits(
                V_Packed_lo, kv_loc, stride_v_lo_bs, v_lo_head_off,
                offs_dv_lo, n_mask, mask_dv_lo, V_BITS_LO, BLOCK_N, BLOCK_DV_LO,
            )
            v_val_lo = tl.load(V_Codebook_lo + v_idx_lo)
            v_val_lo = tl.where(n_mask[:, None] & mask_dv_lo[None, :], v_val_lo, 0.0)
            acc_lo += tl.sum(w_lo[:, None] * v_val_lo, axis=0)

            # ==============================================================
            # V weighted sum — hi group
            # ==============================================================
            v_norms_hi_vec = tl.load(
                V_Norms_hi + kv_loc * stride_vn_hi_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            w_hi = p * v_norms_hi_vec

            v_idx_hi = _extract_bits(
                V_Packed_hi, kv_loc, stride_v_hi_bs, v_hi_head_off,
                offs_dv_hi, n_mask, mask_dv_hi, V_BITS_HI, BLOCK_N, BLOCK_DV_HI,
            )
            v_val_hi = tl.load(V_Codebook_hi + v_idx_hi)
            v_val_hi = tl.where(n_mask[:, None] & mask_dv_hi[None, :], v_val_hi, 0.0)
            acc_hi += tl.sum(w_hi[:, None] * v_val_hi, axis=0)

        # ==================================================================
        # Store intermediate results in split order: [lo | hi]
        # ==================================================================
        mid_base = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        )

        # Store lo channels at offsets [0..D_LO-1]
        tl.store(
            Att_Out + mid_base + offs_dv_lo,
            acc_lo / e_sum,
            mask=mask_dv_lo,
        )

        # Store hi channels at offsets [D_LO..D_LO+D_HI-1]
        tl.store(
            Att_Out + mid_base + D_LO + offs_dv_hi,
            acc_hi / e_sum,
            mask=mask_dv_hi,
        )

        offs_mid_lse = mid_base // Lv
        tl.store(Att_Lse + offs_mid_lse, e_max + tl.log(e_sum))


# =============================================================================
# Python wrapper: split-channel turboquant decode attention
# =============================================================================


def turboquant_decode_attention_fwd_split(
    q_rot_lo: torch.Tensor,         # [B, H_q, D_lo] float32
    q_proj_lo: torch.Tensor,        # [B, H_q, D_lo] float32
    q_rot_hi: torch.Tensor,         # [B, H_q, D_hi] float32
    q_proj_hi: torch.Tensor,        # [B, H_q, D_hi] float32
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
    o_rot_split: torch.Tensor,      # [B, H_q, D_lo+D_hi] output in split order
    kv_indptr: torch.Tensor,        # [B+1]
    kv_indices: torch.Tensor,       # [total_kv_len]
    num_kv_splits: torch.Tensor,    # [B]
    max_kv_splits: int,
    sm_scale: float,                # 1/sqrt(head_dim)
    qjl_scale_lo: float,            # sqrt(pi/2) / D_lo  (per-group, per paper + MLX)
    qjl_scale_hi: float,            # sqrt(pi/2) / D_hi
    mse_bits_lo: int,
    mse_bits_hi: int,
    v_bits_lo: int,
    v_bits_hi: int,
    d_lo: int,
    d_hi: int,
    head_dim: int,                  # d_lo + d_hi
    attn_logits: torch.Tensor,      # [B, H_q, max_kv_splits, head_dim] scratch
    attn_lse: torch.Tensor,         # shares storage with attn_logits
):
    """TurboQuant split-channel fused decode attention (two-stage flash-decoding).

    For fractional bit-widths (e.g. 3.5-bit), channels split into lo + hi groups
    with independent bit-widths. K scores combine additively before softmax.
    V accumulates per group with shared softmax weights.
    Output is in split order [lo | hi] — the caller must inverse-rotate per group
    and reassemble in original channel order.
    """
    batch, head_num = q_rot_lo.shape[0], q_rot_lo.shape[1]
    kv_head_num = k_mse_lo.shape[1]
    kv_group_num = head_num // kv_head_num

    BLOCK_N = 64
    BLOCK_DV_LO = triton.next_power_of_2(d_lo)
    BLOCK_DV_HI = triton.next_power_of_2(d_hi)
    BLOCK_D = min(64, min(BLOCK_DV_LO, BLOCK_DV_HI))

    MAX_KV_SPLITS = max_kv_splits
    grid = (batch, head_num, MAX_KV_SPLITS)

    num_warps = 4 if kv_group_num == 1 else 2

    # Stage 1: split-channel fused scoring + softmax + V accumulation
    _turboquant_fwd_kernel_stage1_split[grid](
        q_rot_lo,
        q_proj_lo,
        k_mse_lo,
        k_qjl_lo,
        k_norms_lo,
        k_res_norms_lo,
        q_rot_hi,
        q_proj_hi,
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
        sm_scale,
        qjl_scale_lo,
        qjl_scale_hi,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        attn_logits,
        attn_lse,
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
        # Att_Out strides
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
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
        BLOCK_D=BLOCK_D,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        Lv=head_dim,
        num_warps=num_warps,
        num_stages=2,
    )

    # Stage 2: merge across KV splits (reuse existing stage 2)
    _turboquant_stage2_fwd(
        attn_logits,
        attn_lse,
        o_rot_split,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        head_dim,
    )


# =============================================================================
# Split-channel FUSED Stage 1: inline forward FWHT (Phase G)
# =============================================================================


@triton.jit
def _turboquant_fwd_kernel_stage1_split_fused(
    # Raw queries per group (NOT pre-rotated)
    Q_lo,               # [B, H_q, D_lo]
    Q_proj_lo,          # [B, H_q, D_lo]
    Signs_lo,           # [padded_D_lo] float32
    hadamard_scale_lo,
    # Packed K buffers — lo group
    K_MSE_Buffer_lo,
    K_QJL_Buffer_lo,
    K_Norms_lo,
    K_ResNorms_lo,
    # Raw queries — hi group
    Q_hi,               # [B, H_q, D_hi]
    Q_proj_hi,          # [B, H_q, D_hi]
    Signs_hi,           # [padded_D_hi] float32
    hadamard_scale_hi,
    # Packed K buffers — hi group
    K_MSE_Buffer_hi,
    K_QJL_Buffer_hi,
    K_Norms_hi,
    K_ResNorms_hi,
    # Packed V buffers — lo group
    V_Packed_lo,
    V_Norms_lo,
    # Packed V buffers — hi group
    V_Packed_hi,
    V_Norms_hi,
    # Codebooks per group
    K_Codebook_lo,
    K_Codebook_hi,
    V_Codebook_lo,
    V_Codebook_hi,
    # Scaling factors
    sm_scale,
    qjl_scale_lo,
    qjl_scale_hi,
    # Page table
    kv_indptr,
    kv_indices,
    num_kv_splits,
    # Output intermediates (also FWHT scratch)
    Att_Out,            # [B, H_q, max_kv_splits, Lv] where Lv=padded_D_lo+padded_D_hi
    Att_Lse,
    # Strides — Q_lo: [B, H_q, D_lo]
    stride_qlo_bs,
    stride_qlo_h,
    # Strides — Q_hi: [B, H_q, D_hi]
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
    # Strides — Att_Out
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
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
    BLOCK_D: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lv: tl.constexpr,
    LOG2_D_LO: tl.constexpr,
    LOG2_D_HI: tl.constexpr,
):
    """Fused split-channel decode Stage 1 with inline forward FWHT (Phase G).

    Same as _turboquant_fwd_kernel_stage1_split but performs forward Hadamard
    transform inline for both lo and hi groups using Att_Out as scratch.
    Values accumulated in rotated space (Phase H).
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    offs_dv_lo = tl.arange(0, BLOCK_DV_LO)
    offs_dv_hi = tl.arange(0, BLOCK_DV_HI)
    mask_dv_lo = offs_dv_lo < D_LO
    mask_dv_hi = offs_dv_hi < D_HI

    e_max = -float("inf")
    e_sum = 0.0
    acc_lo = tl.zeros([BLOCK_DV_LO], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_DV_HI], dtype=tl.float32)

    # Scratch base in Att_Out
    mid_base = (
        cur_batch * stride_mid_ob
        + cur_head * stride_mid_oh
        + split_kv_id * stride_mid_os
    )

    if split_kv_end > split_kv_start:
        # ==============================================================
        # Inline forward FWHT — lo group (Phase G)
        # Uses Att_Out[..., :BLOCK_DV_LO] as scratch
        # ==============================================================
        qlo_base = cur_batch * stride_qlo_bs + cur_head * stride_qlo_h

        q_lo_raw = tl.load(Q_lo + qlo_base + offs_dv_lo, mask=mask_dv_lo, other=0.0)
        signs_lo = tl.load(Signs_lo + offs_dv_lo, mask=mask_dv_lo, other=1.0)
        q_lo_signed = q_lo_raw * signs_lo

        tl.store(Att_Out + mid_base + offs_dv_lo, q_lo_signed, mask=mask_dv_lo)
        tl.debug_barrier()
        _fwht_butterfly_inplace(Att_Out + mid_base, offs_dv_lo, LOG2_D_LO, BLOCK_DV_LO)

        q_rot_lo_full = tl.load(Att_Out + mid_base + offs_dv_lo, mask=mask_dv_lo, other=0.0)
        q_rot_lo_full = q_rot_lo_full * hadamard_scale_lo
        tl.store(Att_Out + mid_base + offs_dv_lo, q_rot_lo_full, mask=mask_dv_lo)
        tl.debug_barrier()

        # ==============================================================
        # Inline forward FWHT — hi group (Phase G)
        # Uses Att_Out[..., D_LO:D_LO+BLOCK_DV_HI] as scratch
        # ==============================================================
        qhi_base = cur_batch * stride_qhi_bs + cur_head * stride_qhi_h

        q_hi_raw = tl.load(Q_hi + qhi_base + offs_dv_hi, mask=mask_dv_hi, other=0.0)
        signs_hi = tl.load(Signs_hi + offs_dv_hi, mask=mask_dv_hi, other=1.0)
        q_hi_signed = q_hi_raw * signs_hi

        tl.store(Att_Out + mid_base + D_LO + offs_dv_hi, q_hi_signed, mask=mask_dv_hi)
        tl.debug_barrier()
        _fwht_butterfly_inplace(Att_Out + mid_base + D_LO, offs_dv_hi, LOG2_D_HI, BLOCK_DV_HI)

        q_rot_hi_full = tl.load(Att_Out + mid_base + D_LO + offs_dv_hi, mask=mask_dv_hi, other=0.0)
        q_rot_hi_full = q_rot_hi_full * hadamard_scale_hi
        tl.store(Att_Out + mid_base + D_LO + offs_dv_hi, q_rot_hi_full, mask=mask_dv_hi)
        tl.debug_barrier()

        # Precomputed head offsets
        k_mse_lo_head_off = cur_kv_head * stride_k_mse_lo_h
        k_qjl_lo_head_off = cur_kv_head * stride_k_qjl_lo_h
        k_mse_hi_head_off = cur_kv_head * stride_k_mse_hi_h
        k_qjl_hi_head_off = cur_kv_head * stride_k_qjl_hi_h
        v_lo_head_off = cur_kv_head * stride_v_lo_h
        v_hi_head_off = cur_kv_head * stride_v_hi_h

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end

            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=n_mask, other=0,
            )

            # K scoring — lo group (read q_rot from scratch)
            mse_score_lo = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, D_LO, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D_LO

                mse_idx = _extract_bits(
                    K_MSE_Buffer_lo, kv_loc, stride_k_mse_lo_bs, k_mse_lo_head_off,
                    d_offs, n_mask, d_mask, MSE_BITS_LO, BLOCK_N, BLOCK_D,
                )
                k_val = tl.load(K_Codebook_lo + mse_idx)
                k_val = tl.where(n_mask[:, None] & d_mask[None, :], k_val, 0.0)

                q_rot_slice = tl.load(
                    Att_Out + mid_base + d_offs, mask=d_mask, other=0.0,
                )
                mse_score_lo += tl.sum(q_rot_slice[None, :] * k_val, axis=1)

            qjl_score_lo = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, D_LO, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D_LO

                sign = _extract_sign_bits(
                    K_QJL_Buffer_lo, kv_loc, stride_k_qjl_lo_bs, k_qjl_lo_head_off,
                    d_offs, n_mask, d_mask, BLOCK_N, BLOCK_D,
                )
                sign = tl.where(n_mask[:, None] & d_mask[None, :], sign, 0.0)

                q_proj_slice = tl.load(
                    Q_proj_lo + qlo_base + d_offs, mask=d_mask, other=0.0,
                )
                qjl_score_lo += tl.sum(q_proj_slice[None, :] * sign, axis=1)

            # K scoring — hi group (read q_rot from scratch)
            mse_score_hi = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, D_HI, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D_HI

                mse_idx = _extract_bits(
                    K_MSE_Buffer_hi, kv_loc, stride_k_mse_hi_bs, k_mse_hi_head_off,
                    d_offs, n_mask, d_mask, MSE_BITS_HI, BLOCK_N, BLOCK_D,
                )
                k_val = tl.load(K_Codebook_hi + mse_idx)
                k_val = tl.where(n_mask[:, None] & d_mask[None, :], k_val, 0.0)

                q_rot_slice = tl.load(
                    Att_Out + mid_base + D_LO + d_offs, mask=d_mask, other=0.0,
                )
                mse_score_hi += tl.sum(q_rot_slice[None, :] * k_val, axis=1)

            qjl_score_hi = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_start in range(0, D_HI, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D_HI

                sign = _extract_sign_bits(
                    K_QJL_Buffer_hi, kv_loc, stride_k_qjl_hi_bs, k_qjl_hi_head_off,
                    d_offs, n_mask, d_mask, BLOCK_N, BLOCK_D,
                )
                sign = tl.where(n_mask[:, None] & d_mask[None, :], sign, 0.0)

                q_proj_slice = tl.load(
                    Q_proj_hi + qhi_base + d_offs, mask=d_mask, other=0.0,
                )
                qjl_score_hi += tl.sum(q_proj_slice[None, :] * sign, axis=1)

            # Combined K score (additive across groups)
            k_norms_lo = tl.load(
                K_Norms_lo + kv_loc * stride_kn_lo_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            k_res_norms_lo = tl.load(
                K_ResNorms_lo + kv_loc * stride_kn_lo_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            k_norms_hi = tl.load(
                K_Norms_hi + kv_loc * stride_kn_hi_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            k_res_norms_hi = tl.load(
                K_ResNorms_hi + kv_loc * stride_kn_hi_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)

            qk = (
                k_norms_lo * (mse_score_lo + qjl_scale_lo * k_res_norms_lo * qjl_score_lo)
                + k_norms_hi * (mse_score_hi + qjl_scale_hi * k_res_norms_hi * qjl_score_hi)
            )
            qk *= sm_scale
            qk = tl.where(n_mask, qk, float("-inf"))

            # Online softmax — rescale BOTH accumulators
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc_lo *= re_scale
            acc_hi *= re_scale
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

            # V weighted sum — lo group (rotated space, Phase H)
            v_norms_lo_vec = tl.load(
                V_Norms_lo + kv_loc * stride_vn_lo_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            w_lo = p * v_norms_lo_vec

            v_idx_lo = _extract_bits(
                V_Packed_lo, kv_loc, stride_v_lo_bs, v_lo_head_off,
                offs_dv_lo, n_mask, mask_dv_lo, V_BITS_LO, BLOCK_N, BLOCK_DV_LO,
            )
            v_val_lo = tl.load(V_Codebook_lo + v_idx_lo)
            v_val_lo = tl.where(n_mask[:, None] & mask_dv_lo[None, :], v_val_lo, 0.0)
            acc_lo += tl.sum(w_lo[:, None] * v_val_lo, axis=0)

            # V weighted sum — hi group (rotated space, Phase H)
            v_norms_hi_vec = tl.load(
                V_Norms_hi + kv_loc * stride_vn_hi_bs + cur_kv_head,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            w_hi = p * v_norms_hi_vec

            v_idx_hi = _extract_bits(
                V_Packed_hi, kv_loc, stride_v_hi_bs, v_hi_head_off,
                offs_dv_hi, n_mask, mask_dv_hi, V_BITS_HI, BLOCK_N, BLOCK_DV_HI,
            )
            v_val_hi = tl.load(V_Codebook_hi + v_idx_hi)
            v_val_hi = tl.where(n_mask[:, None] & mask_dv_hi[None, :], v_val_hi, 0.0)
            acc_hi += tl.sum(w_hi[:, None] * v_val_hi, axis=0)

        # Store intermediate results (overwrites FWHT scratch)
        tl.store(Att_Out + mid_base + offs_dv_lo, acc_lo / e_sum, mask=mask_dv_lo)
        tl.store(Att_Out + mid_base + D_LO + offs_dv_hi, acc_hi / e_sum, mask=mask_dv_hi)

        offs_mid_lse = mid_base // Lv
        tl.store(Att_Lse + offs_mid_lse, e_max + tl.log(e_sum))


# =============================================================================
# Split-channel FUSED Stage 2: merge + inverse FWHT per group (Phase G)
# =============================================================================


@triton.jit
def _turboquant_stage2_inv_fwht_split_kernel(
    Mid_O,
    Mid_O_1,
    O,                  # [B, H_q, Lv] output (de-rotated, split order)
    Signs_lo,           # [padded_D_lo]
    Signs_hi,           # [padded_D_hi]
    hadamard_scale_lo,
    hadamard_scale_hi,
    kv_indptr,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV_LO: tl.constexpr,
    BLOCK_DV_HI: tl.constexpr,
    D_LO: tl.constexpr,
    D_HI: tl.constexpr,
    Lv: tl.constexpr,
    LOG2_D_LO: tl.constexpr,
    LOG2_D_HI: tl.constexpr,
):
    """Fused split-channel Stage 2 merge + per-group inverse FWHT (Phase G).

    Merges across KV splits, then applies inverse Hadamard to lo and hi
    groups independently. Output is de-rotated in split order [lo | hi].
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # Full dimension offsets for loading intermediates
    offs_d_full = tl.arange(0, BLOCK_DV_LO + BLOCK_DV_HI)
    mask_d_lo = offs_d_full < D_LO
    mask_d_hi = (offs_d_full >= D_LO) & (offs_d_full < D_LO + D_HI)
    mask_d_full = offs_d_full < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV_LO + BLOCK_DV_HI], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d_full
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    # Standard Stage 2 merge
    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d_full, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    merged = acc / e_sum

    # ==================================================================
    # Store full merged result to O, then apply inverse FWHT per group
    # ==================================================================
    o_base = cur_batch * stride_obs + cur_head * stride_oh

    # Store full merged result to O (scratch for per-group butterfly)
    tl.store(O + o_base + offs_d_full, merged, mask=mask_d_full)
    tl.debug_barrier()

    # ==================================================================
    # Inverse FWHT — lo group (Phase G)
    # ==================================================================
    offs_lo = tl.arange(0, BLOCK_DV_LO)
    mask_lo = offs_lo < D_LO

    _fwht_butterfly_inplace(O + o_base, offs_lo, LOG2_D_LO, BLOCK_DV_LO)

    result_lo = tl.load(O + o_base + offs_lo, mask=mask_lo, other=0.0)
    signs_lo = tl.load(Signs_lo + offs_lo, mask=mask_lo, other=1.0)
    result_lo = result_lo * hadamard_scale_lo * signs_lo
    tl.store(O + o_base + offs_lo, result_lo, mask=mask_lo)

    # ==================================================================
    # Inverse FWHT — hi group (Phase G)
    # ==================================================================
    offs_hi = tl.arange(0, BLOCK_DV_HI)
    mask_hi = offs_hi < D_HI

    _fwht_butterfly_inplace(O + o_base + D_LO, offs_hi, LOG2_D_HI, BLOCK_DV_HI)

    result_hi = tl.load(O + o_base + D_LO + offs_hi, mask=mask_hi, other=0.0)
    signs_hi = tl.load(Signs_hi + offs_hi, mask=mask_hi, other=1.0)
    result_hi = result_hi * hadamard_scale_hi * signs_hi
    tl.store(O + o_base + D_LO + offs_hi, result_hi, mask=mask_hi)


def _turboquant_stage2_inv_fwht_split_fwd(
    att_out,
    att_lse,
    o,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
    head_dim,
    d_lo,
    d_hi,
    signs_lo,
    signs_hi,
    hadamard_scale_lo,
    hadamard_scale_hi,
):
    """Merge stage-1 split intermediates + per-group inverse FWHT."""
    batch, head_num = o.shape[0], o.shape[1]
    Lv = head_dim
    BLOCK_DV_LO = triton.next_power_of_2(d_lo)
    BLOCK_DV_HI = triton.next_power_of_2(d_hi)
    LOG2_D_LO = _log2(BLOCK_DV_LO)
    LOG2_D_HI = _log2(BLOCK_DV_HI)

    grid = (batch, head_num)
    _turboquant_stage2_inv_fwht_split_kernel[grid](
        att_out,
        att_lse,
        o,
        signs_lo,
        signs_hi,
        hadamard_scale_lo,
        hadamard_scale_hi,
        kv_indptr,
        num_kv_splits,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=max_kv_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV_LO=BLOCK_DV_LO,
        BLOCK_DV_HI=BLOCK_DV_HI,
        D_LO=d_lo,
        D_HI=d_hi,
        Lv=Lv,
        LOG2_D_LO=LOG2_D_LO,
        LOG2_D_HI=LOG2_D_HI,
        num_warps=4,
        num_stages=2,
    )


# =============================================================================
# Python wrapper: FUSED split-channel decode attention (Phase G)
# =============================================================================


def turboquant_decode_attention_fused_fwd_split(
    q_lo: torch.Tensor,             # [B, H_q, D_lo] raw (NOT rotated, float32)
    q_proj_lo: torch.Tensor,        # [B, H_q, D_lo] float32
    q_hi: torch.Tensor,             # [B, H_q, D_hi] raw (NOT rotated, float32)
    q_proj_hi: torch.Tensor,        # [B, H_q, D_hi] float32
    signs_lo: torch.Tensor,         # [padded_D_lo] float32
    signs_hi: torch.Tensor,         # [padded_D_hi] float32
    hadamard_scale_lo: float,
    hadamard_scale_hi: float,
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
    o_split: torch.Tensor,          # [B, H_q, D_lo+D_hi] output (de-rotated, split order)
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
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
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
):
    """Split-channel decode with fused Stage 2 inverse FWHT (Phase G/H2).

    Stage 1: Original split-channel kernel (pre-rotated queries).
    Stage 2: Fused merge + per-group inline inverse Hadamard.

    Forward FWHT on queries is done externally by caller (standalone kernel).
    Output o_split is de-rotated in split order [lo | hi].
    """
    batch, head_num = q_lo.shape[0], q_lo.shape[1]
    kv_head_num = k_mse_lo.shape[1]
    kv_group_num = head_num // kv_head_num

    BLOCK_N = 64
    BLOCK_DV_LO = triton.next_power_of_2(d_lo)
    BLOCK_DV_HI = triton.next_power_of_2(d_hi)
    BLOCK_D = min(64, min(BLOCK_DV_LO, BLOCK_DV_HI))

    MAX_KV_SPLITS = max_kv_splits
    grid = (batch, head_num, MAX_KV_SPLITS)

    num_warps = 4 if kv_group_num == 1 else 2

    # Stage 1: original split-channel kernel with pre-rotated queries
    _turboquant_fwd_kernel_stage1_split[grid](
        q_lo, q_proj_lo,
        k_mse_lo, k_qjl_lo, k_norms_lo, k_res_norms_lo,
        q_hi, q_proj_hi,
        k_mse_hi, k_qjl_hi, k_norms_hi, k_res_norms_hi,
        v_packed_lo, v_norms_lo,
        v_packed_hi, v_norms_hi,
        k_cb_lo, k_cb_hi, v_cb_lo, v_cb_hi,
        sm_scale, qjl_scale_lo, qjl_scale_hi,
        kv_indptr, kv_indices, num_kv_splits,
        attn_logits, attn_lse,
        # Q_lo strides
        q_lo.stride(0), q_lo.stride(1),
        # Q_hi strides
        q_hi.stride(0), q_hi.stride(1),
        # K_MSE_lo strides
        k_mse_lo.stride(0), k_mse_lo.stride(1),
        # K_QJL_lo strides
        k_qjl_lo.stride(0), k_qjl_lo.stride(1),
        # K_Norms_lo stride
        k_norms_lo.stride(0),
        # K_MSE_hi strides
        k_mse_hi.stride(0), k_mse_hi.stride(1),
        # K_QJL_hi strides
        k_qjl_hi.stride(0), k_qjl_hi.stride(1),
        # K_Norms_hi stride
        k_norms_hi.stride(0),
        # V_Packed_lo strides
        v_packed_lo.stride(0), v_packed_lo.stride(1),
        # V_Norms_lo stride
        v_norms_lo.stride(0),
        # V_Packed_hi strides
        v_packed_hi.stride(0), v_packed_hi.stride(1),
        # V_Norms_hi stride
        v_norms_hi.stride(0),
        # Att_Out strides
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        # Constexprs
        kv_group_num=kv_group_num,
        MSE_BITS_LO=mse_bits_lo, MSE_BITS_HI=mse_bits_hi,
        V_BITS_LO=v_bits_lo, V_BITS_HI=v_bits_hi,
        D_LO=d_lo, D_HI=d_hi,
        BLOCK_DV_LO=BLOCK_DV_LO, BLOCK_DV_HI=BLOCK_DV_HI,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        Lv=head_dim,
        num_warps=num_warps,
        num_stages=2,
    )

    # Stage 2: fused merge + per-group inverse FWHT
    _turboquant_stage2_inv_fwht_split_fwd(
        attn_logits, attn_lse, o_split,
        kv_indptr, num_kv_splits, max_kv_splits,
        head_dim, d_lo, d_hi,
        signs_lo, signs_hi,
        hadamard_scale_lo, hadamard_scale_hi,
    )
