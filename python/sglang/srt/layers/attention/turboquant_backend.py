"""TurboQuant attention backend (arXiv:2504.19874).

Inherits from TritonAttnBackend and overrides forward_decode and
forward_extend to call fused TurboQuant Triton kernels that read packed
quantized KV buffers directly (no dequantization).
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

        # Lazy import of TurboQuant kernels
        from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
            turboquant_decode_attention_fwd,
            turboquant_decode_attention_fwd_split,
            turboquant_decode_attention_fused_fwd,
            turboquant_decode_attention_fused_fwd_split,
        )
        from sglang.srt.layers.attention.triton_ops.turboquant_extend_attention import (
            turboquant_extend_attention_fwd,
            turboquant_extend_attention_fwd_split,
            turboquant_extend_attention_fused_fwd,
            turboquant_extend_attention_fused_fwd_split,
        )

        self.turboquant_decode_attention_fwd = torch.compiler.disable(
            turboquant_decode_attention_fwd
        )
        self.turboquant_decode_attention_fwd_split = torch.compiler.disable(
            turboquant_decode_attention_fwd_split
        )
        self.turboquant_decode_attention_fused_fwd = torch.compiler.disable(
            turboquant_decode_attention_fused_fwd
        )
        self.turboquant_decode_attention_fused_fwd_split = torch.compiler.disable(
            turboquant_decode_attention_fused_fwd_split
        )
        self.turboquant_extend_attention_fwd = torch.compiler.disable(
            turboquant_extend_attention_fwd
        )
        self.turboquant_extend_attention_fwd_split = torch.compiler.disable(
            turboquant_extend_attention_fwd_split
        )
        self.turboquant_extend_attention_fused_fwd = torch.compiler.disable(
            turboquant_extend_attention_fused_fwd
        )
        self.turboquant_extend_attention_fused_fwd_split = torch.compiler.disable(
            turboquant_extend_attention_fused_fwd_split
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

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        """TurboQuant fused extend: quantize KV, then attend over packed buffers.

        Flow: save KV → build unified page table → pre-rotate/project queries
        → call TurboQuant extend kernel → inverse-rotate output.
        """
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

        # Page table from forward metadata
        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices
        qo_indptr = self.forward_metadata.qo_indptr
        prefix_lens = forward_batch.extend_prefix_lens
        max_extend_len = self.forward_metadata.max_extend_len

        # Total query tokens
        total_q = q.shape[0]

        if not tq_pool.is_split:
            # ============================================================
            # Integer-bit path — fused extend with inverse FWHT
            # ============================================================
            head_dim = tq_pool.head_dim

            q_bufs = tq_pool.get_quantized_k_buffers(mapped_id)
            v_bufs = tq_pool.get_quantized_v_buffers(mapped_id)

            q_float = q.view(total_q, layer.tp_q_head_num, layer.qk_head_dim).float()
            q_rot = tq_pool.k_hadamard[li].forward(q_float)
            q_proj = torch.matmul(q_float, tq_pool.s_matrices[li].T)

            padded_dim = tq_pool.k_hadamard[li].padded_dim
            output = torch.empty(
                (total_q, layer.tp_q_head_num, padded_dim),
                dtype=torch.float32,
                device=q.device,
            )

            qjl_scale = math.sqrt(math.pi / 2.0) / head_dim

            # Fused extend: kernel + inverse FWHT handled internally
            self.turboquant_extend_attention_fused_fwd(
                q_rot, q_proj,
                tq_pool.k_hadamard[li].signs,
                tq_pool.k_hadamard[li].scale,
                tq_pool.k_hadamard[li].dim,
                q_bufs["mse_packed"], q_bufs["qjl_packed"],
                q_bufs["norms"], q_bufs["res_norms"],
                v_bufs["packed"], v_bufs["norms"],
                tq_pool.k_codebook, tq_pool.v_codebook,
                output,
                qo_indptr, kv_indptr, kv_indices, prefix_lens,
                max_extend_len, layer.scaling, qjl_scale,
                tq_pool.k_mse_bits, tq_pool.int_bits,
                padded_dim,
            )

            # Output already de-rotated — truncate padding
            output = output[..., :tq_pool.k_hadamard[li].dim]
            return output.to(q.dtype).view(-1, layer.tp_q_head_num * layer.v_head_dim)

        else:
            # ============================================================
            # Split-channel path — fused extend with per-group inverse FWHT
            # ============================================================
            d_lo, d_hi = tq_pool.d_lo, tq_pool.d_hi
            head_dim = tq_pool.head_dim

            k_bufs = tq_pool.get_quantized_k_buffers(mapped_id)
            v_bufs = tq_pool.get_quantized_v_buffers(mapped_id)

            q_float = q.view(total_q, layer.tp_q_head_num, layer.qk_head_dim).float()
            lo_idx = tq_pool.lo_indices.to(q.device)
            hi_idx = tq_pool.hi_indices.to(q.device)

            q_lo = q_float.index_select(-1, lo_idx)
            q_hi = q_float.index_select(-1, hi_idx)

            q_rot_lo = tq_pool.hadamard_lo[li].forward(q_lo)
            q_proj_lo = torch.matmul(q_lo, tq_pool.s_lo[li].T)
            q_rot_hi = tq_pool.hadamard_hi[li].forward(q_hi)
            q_proj_hi = torch.matmul(q_hi, tq_pool.s_hi[li].T)

            padded_d_lo = tq_pool.hadamard_lo[li].padded_dim
            padded_d_hi = tq_pool.hadamard_hi[li].padded_dim
            output_split = torch.empty(
                (total_q, layer.tp_q_head_num, padded_d_lo + padded_d_hi),
                dtype=torch.float32,
                device=q.device,
            )

            qjl_scale_lo = math.sqrt(math.pi / 2.0) / d_lo
            qjl_scale_hi = math.sqrt(math.pi / 2.0) / d_hi

            # Fused extend: kernel + per-group inverse FWHT handled internally
            self.turboquant_extend_attention_fused_fwd_split(
                q_rot_lo, q_proj_lo,
                q_rot_hi, q_proj_hi,
                tq_pool.hadamard_lo[li].signs,
                tq_pool.hadamard_hi[li].signs,
                tq_pool.hadamard_lo[li].scale,
                tq_pool.hadamard_hi[li].scale,
                tq_pool.hadamard_lo[li].dim,
                tq_pool.hadamard_hi[li].dim,
                k_bufs["lo_mse_packed"], k_bufs["lo_qjl_packed"],
                k_bufs["lo_norms"], k_bufs["lo_res_norms"],
                k_bufs["hi_mse_packed"], k_bufs["hi_qjl_packed"],
                k_bufs["hi_norms"], k_bufs["hi_res_norms"],
                v_bufs["lo_packed"], v_bufs["lo_norms"],
                v_bufs["hi_packed"], v_bufs["hi_norms"],
                tq_pool.k_cb_lo, tq_pool.k_cb_hi,
                tq_pool.v_cb_lo, tq_pool.v_cb_hi,
                output_split,
                qo_indptr, kv_indptr, kv_indices, prefix_lens,
                max_extend_len, layer.scaling,
                qjl_scale_lo, qjl_scale_hi,
                tq_pool.k_lo_mse_bits, tq_pool.k_hi_mse_bits,
                tq_pool.v_lo_bits, tq_pool.v_hi_bits,
                padded_d_lo, padded_d_hi, padded_d_lo + padded_d_hi,
            )

            # Output already de-rotated — truncate and reassemble
            o_lo = output_split[..., :tq_pool.hadamard_lo[li].dim]
            o_hi = output_split[..., padded_d_lo:padded_d_lo + tq_pool.hadamard_hi[li].dim]

            output_cat = torch.cat([o_lo, o_hi], dim=-1)
            output = output_cat.index_select(-1, tq_pool.restore_order.to(q.device))
            return output.to(q.dtype).view(-1, layer.tp_q_head_num * layer.v_head_dim)

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
            # Integer-bit path (e.g. 3-bit) — fused Stage 2 inverse FWHT
            # ============================================================
            head_dim = tq_pool.head_dim

            # Get quantized buffers
            q_bufs = tq_pool.get_quantized_k_buffers(mapped_id)
            v_bufs = tq_pool.get_quantized_v_buffers(mapped_id)

            # q: [B, H_q * D] -> [B, H_q, D] -> float32
            q_float = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).float()

            # Pre-rotate query via standalone FWHT kernel (tiny, fast)
            q_rot = tq_pool.k_hadamard[li].forward(q_float)
            # QJL projection (cuBLAS)
            q_proj = torch.matmul(q_float, tq_pool.s_matrices[li].T)

            # Output buffer (de-rotated by fused Stage 2)
            padded_dim = tq_pool.k_hadamard[li].padded_dim
            output = torch.empty(
                (q_float.shape[0], layer.tp_q_head_num, padded_dim),
                dtype=torch.float32,
                device=q.device,
            )

            # QJL scale: sqrt(pi/2) / D (paper Eq. 7)
            qjl_scale = math.sqrt(math.pi / 2.0) / head_dim

            # Fused decode: Stage 1 (pre-rotated q) + Stage 2 (inline inverse FWHT)
            self.turboquant_decode_attention_fused_fwd(
                q_rot,
                q_proj,
                tq_pool.k_hadamard[li].signs,
                tq_pool.k_hadamard[li].scale,
                q_bufs["mse_packed"],
                q_bufs["qjl_packed"],
                q_bufs["norms"],
                q_bufs["res_norms"],
                v_bufs["packed"],
                v_bufs["norms"],
                tq_pool.k_codebook,
                tq_pool.v_codebook,
                output,
                kv_indptr,
                kv_indices,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                qjl_scale,
                tq_pool.k_mse_bits,
                tq_pool.int_bits,  # v_bits = full integer bits
                padded_dim,
                self.forward_metadata.attn_logits,
                self.forward_metadata.attn_lse,
            )

            # Output already de-rotated by fused Stage 2 — truncate padding
            output = output[..., :tq_pool.k_hadamard[li].dim]
            return output.to(q.dtype).view(-1, layer.tp_q_head_num * layer.v_head_dim)

        else:
            # ============================================================
            # Split-channel path (fractional bits, e.g. 3.5-bit) — fused FWHT (Phase G)
            # ============================================================
            d_lo, d_hi = tq_pool.d_lo, tq_pool.d_hi
            head_dim = tq_pool.head_dim  # = d_lo + d_hi

            # Get split quantized buffers
            k_bufs = tq_pool.get_quantized_k_buffers(mapped_id)  # 8-item dict
            v_bufs = tq_pool.get_quantized_v_buffers(mapped_id)  # 4-item dict

            # Split queries by channel indices, rotate, project
            q_float = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).float()
            lo_idx = tq_pool.lo_indices.to(q.device)
            hi_idx = tq_pool.hi_indices.to(q.device)

            q_lo = q_float.index_select(-1, lo_idx)  # [B, H_q, d_lo]
            q_hi = q_float.index_select(-1, hi_idx)  # [B, H_q, d_hi]

            # Pre-rotate queries via standalone FWHT kernel (tiny, fast)
            q_rot_lo = tq_pool.hadamard_lo[li].forward(q_lo)
            q_rot_hi = tq_pool.hadamard_hi[li].forward(q_hi)
            # QJL projections (cuBLAS)
            q_proj_lo = torch.matmul(q_lo, tq_pool.s_lo[li].T)
            q_proj_hi = torch.matmul(q_hi, tq_pool.s_hi[li].T)

            # Output buffer (de-rotated by fused Stage 2, split order)
            padded_d_lo = tq_pool.hadamard_lo[li].padded_dim
            padded_d_hi = tq_pool.hadamard_hi[li].padded_dim
            output_split = torch.empty(
                (q_float.shape[0], layer.tp_q_head_num, padded_d_lo + padded_d_hi),
                dtype=torch.float32,
                device=q.device,
            )

            # QJL scale: per-group sqrt(pi/2) / D_group
            qjl_scale_lo = math.sqrt(math.pi / 2.0) / d_lo
            qjl_scale_hi = math.sqrt(math.pi / 2.0) / d_hi

            # Fused decode: Stage 1 (pre-rotated q) + Stage 2 (inline inverse FWHT)
            self.turboquant_decode_attention_fused_fwd_split(
                q_rot_lo, q_proj_lo,
                q_rot_hi, q_proj_hi,
                tq_pool.hadamard_lo[li].signs,
                tq_pool.hadamard_hi[li].signs,
                tq_pool.hadamard_lo[li].scale,
                tq_pool.hadamard_hi[li].scale,
                k_bufs["lo_mse_packed"], k_bufs["lo_qjl_packed"],
                k_bufs["lo_norms"], k_bufs["lo_res_norms"],
                k_bufs["hi_mse_packed"], k_bufs["hi_qjl_packed"],
                k_bufs["hi_norms"], k_bufs["hi_res_norms"],
                v_bufs["lo_packed"], v_bufs["lo_norms"],
                v_bufs["hi_packed"], v_bufs["hi_norms"],
                tq_pool.k_cb_lo, tq_pool.k_cb_hi,
                tq_pool.v_cb_lo, tq_pool.v_cb_hi,
                output_split,
                kv_indptr, kv_indices,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits,
                layer.scaling, qjl_scale_lo, qjl_scale_hi,
                tq_pool.k_lo_mse_bits, tq_pool.k_hi_mse_bits,
                tq_pool.v_lo_bits, tq_pool.v_hi_bits,
                padded_d_lo, padded_d_hi, padded_d_lo + padded_d_hi,
                self.forward_metadata.attn_logits,
                self.forward_metadata.attn_lse,
            )

            # Output already de-rotated — truncate padding and reassemble
            o_lo = output_split[..., :tq_pool.hadamard_lo[li].dim]
            o_hi = output_split[..., padded_d_lo:padded_d_lo + tq_pool.hadamard_hi[li].dim]

            # Reassemble in original channel order
            output_cat = torch.cat([o_lo, o_hi], dim=-1)
            output = output_cat.index_select(-1, tq_pool.restore_order.to(q.device))

            return output.to(q.dtype).view(-1, layer.tp_q_head_num * layer.v_head_dim)
