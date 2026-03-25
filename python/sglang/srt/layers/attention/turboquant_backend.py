"""TurboQuant attention backend (arXiv:2504.19874).

Inherits from TritonAttnBackend and overrides only `forward_decode` to call
the fused TurboQuant decode kernel that reads packed quantized KV buffers
directly (no dequantization).

Prefill/extend reuses the inherited TritonAttnBackend path, which calls
pool.get_key_buffer() / get_value_buffer() for on-demand dequant +
standard Triton extend kernel.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class TurboQuantAttnBackend(TritonAttnBackend):
    """Attention backend that uses TurboQuant fused decode kernel.

    Extends TritonAttnBackend — inherits init_forward_metadata, forward_extend,
    and all buffer setup. Only forward_decode is overridden to use the
    TurboQuant Triton kernel that reads packed uint8 quantized KV directly.
    """

    def __init__(self, model_runner: ModelRunner, **kwargs):
        super().__init__(model_runner, **kwargs)

        # Lazy import of TurboQuant decode kernels
        from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
            turboquant_decode_attention_fwd,
            turboquant_decode_attention_fwd_split,
        )

        self.turboquant_decode_attention_fwd = torch.compiler.disable(
            turboquant_decode_attention_fwd
        )
        self.turboquant_decode_attention_fwd_split = torch.compiler.disable(
            turboquant_decode_attention_fwd_split
        )

    def _get_tq_pool(self, forward_batch: ForwardBatch):
        """Resolve the TurboQuantTokenToKVPool from the forward batch.

        Handles both direct TurboQuantTokenToKVPool and HybridLinearKVPool
        wrapping.
        """
        pool = forward_batch.token_to_kv_pool
        if hasattr(pool, "full_kv_pool"):
            return pool, pool.full_kv_pool
        return pool, pool

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # Reshape q to standard 2D layout
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # Get pool references
        pool, tq_pool = self._get_tq_pool(forward_batch)

        # Save KV cache (quantize into compact buffers)
        if save_kv_cache:
            pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        # Map layer_id to internal pool index
        if hasattr(pool, "_transfer_full_attention_id"):
            mapped_id = pool._transfer_full_attention_id(layer.layer_id)
        else:
            mapped_id = layer.layer_id
        li = mapped_id - tq_pool.start_layer

        # Get page table from forward metadata
        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices

        if not tq_pool.is_split:
            # ============================================================
            # Integer-bit path (e.g. 3-bit)
            # ============================================================
            head_dim = tq_pool.head_dim

            # Get quantized buffers
            q_bufs = tq_pool.get_quantized_k_buffers(mapped_id)
            v_bufs = tq_pool.get_quantized_v_buffers(mapped_id)

            # Pre-rotate and project queries
            # q: [B, H_q * D] -> [B, H_q, D] -> float32
            q_float = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).float()

            # pi_t_matrices[li] is [D, D], q_float is [B, H_q, D]
            # matmul: [B, H_q, D] x [D, D] -> [B, H_q, D]
            q_rot = torch.matmul(q_float, tq_pool.pi_t_matrices[li])
            q_proj = torch.matmul(q_float, tq_pool.s_matrices[li].T)

            # Output buffer in rotated space
            o_rot = torch.empty(
                (q_float.shape[0], layer.tp_q_head_num, head_dim),
                dtype=torch.float32,
                device=q.device,
            )

            # QJL scale: sqrt(pi/2) / D (paper Eq. 7)
            qjl_scale = math.sqrt(math.pi / 2.0) / head_dim

            # Call TurboQuant fused decode kernel
            self.turboquant_decode_attention_fwd(
                q_rot,
                q_proj,
                q_bufs["mse_packed"],
                q_bufs["qjl_packed"],
                q_bufs["norms"],
                q_bufs["res_norms"],
                v_bufs["packed"],
                v_bufs["norms"],
                tq_pool.k_codebook,
                tq_pool.v_codebook,
                o_rot,
                kv_indptr,
                kv_indices,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                qjl_scale,
                tq_pool.k_mse_bits,
                tq_pool.int_bits,  # v_bits = full integer bits
                head_dim,
                self.forward_metadata.attn_logits,
                self.forward_metadata.attn_lse,
            )

            # Inverse rotate output: [B, H_q, D] x [D, D] -> [B, H_q, D]
            output = torch.matmul(o_rot, tq_pool.pi_matrices[li])
            return output.to(q.dtype).view(-1, layer.tp_q_head_num * layer.v_head_dim)

        else:
            # ============================================================
            # Split-channel path (fractional bits, e.g. 3.5-bit)
            # ============================================================
            d_lo, d_hi = tq_pool.d_lo, tq_pool.d_hi
            head_dim = tq_pool.head_dim  # = d_lo + d_hi

            # Get split quantized buffers
            k_bufs = tq_pool.get_quantized_k_buffers(mapped_id)  # 8-item dict
            v_bufs = tq_pool.get_quantized_v_buffers(mapped_id)  # 4-item dict

            # Split queries by channel indices, rotate/project per group
            q_float = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).float()
            lo_idx = tq_pool.lo_indices.to(q.device)
            hi_idx = tq_pool.hi_indices.to(q.device)

            q_lo = q_float.index_select(-1, lo_idx)  # [B, H_q, d_lo]
            q_hi = q_float.index_select(-1, hi_idx)  # [B, H_q, d_hi]

            q_rot_lo = torch.matmul(q_lo, tq_pool.pi_t_lo[li])
            q_proj_lo = torch.matmul(q_lo, tq_pool.s_lo[li].T)
            q_rot_hi = torch.matmul(q_hi, tq_pool.pi_t_hi[li])
            q_proj_hi = torch.matmul(q_hi, tq_pool.s_hi[li].T)

            # Output buffer in split order [d_lo + d_hi]
            o_rot_split = torch.empty(
                (q_float.shape[0], layer.tp_q_head_num, head_dim),
                dtype=torch.float32,
                device=q.device,
            )

            # QJL scale: per-group sqrt(pi/2) / D_group (paper + MLX _SplitCodec)
            qjl_scale_lo = math.sqrt(math.pi / 2.0) / d_lo
            qjl_scale_hi = math.sqrt(math.pi / 2.0) / d_hi

            # Call split-channel kernel
            self.turboquant_decode_attention_fwd_split(
                q_rot_lo, q_proj_lo,
                q_rot_hi, q_proj_hi,
                k_bufs["lo_mse_packed"], k_bufs["lo_qjl_packed"],
                k_bufs["lo_norms"], k_bufs["lo_res_norms"],
                k_bufs["hi_mse_packed"], k_bufs["hi_qjl_packed"],
                k_bufs["hi_norms"], k_bufs["hi_res_norms"],
                v_bufs["lo_packed"], v_bufs["lo_norms"],
                v_bufs["hi_packed"], v_bufs["hi_norms"],
                tq_pool.k_cb_lo, tq_pool.k_cb_hi,
                tq_pool.v_cb_lo, tq_pool.v_cb_hi,
                o_rot_split,
                kv_indptr, kv_indices,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits,
                layer.scaling, qjl_scale_lo, qjl_scale_hi,
                tq_pool.k_lo_mse_bits, tq_pool.k_hi_mse_bits,
                tq_pool.v_lo_bits, tq_pool.v_hi_bits,
                d_lo, d_hi, head_dim,
                self.forward_metadata.attn_logits,
                self.forward_metadata.attn_lse,
            )

            # Inverse rotate per group (output is in split order: [lo | hi])
            o_lo = torch.matmul(o_rot_split[..., :d_lo], tq_pool.pi_lo[li])
            o_hi = torch.matmul(o_rot_split[..., d_lo:], tq_pool.pi_hi[li])

            # Reassemble in original channel order
            output_split = torch.cat([o_lo, o_hi], dim=-1)
            output = output_split.index_select(-1, tq_pool.restore_order.to(q.device))

            return output.to(q.dtype).view(-1, layer.tp_q_head_num * layer.v_head_dim)
