"""TurboQuant attention backend (arXiv:2504.19874).

Inherits from TritonAttnBackend and overrides:
- forward_decode: fused Triton kernel reads packed uint8 quantized KV directly
- forward_extend: selective gather-dequant (O(prefix) instead of O(pool_size))

The 2-stage extend kernel uses raw K,V for extend tokens (stage 2, zero
quantization error) and dequantized pool data for prefix tokens (stage 1).
The override replaces the parent's full-pool dequant with selective
gather-dequant at only the positions referenced by kv_indices.

Both keys and values use MSE-only quantization (no QJL). The fused
decode kernel scores directly from codebook indices * norms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class TurboQuantAttnBackend(TritonAttnBackend):
    """Attention backend with TurboQuant fused decode and optimized extend.

    Extends TritonAttnBackend. forward_decode reads packed uint8 quantized
    KV directly via the TQ Triton kernel. forward_extend uses selective
    gather-dequant (O(prefix) not O(pool_size)) for the prefix buffer.
    """

    def __init__(self, model_runner: ModelRunner, **kwargs):
        super().__init__(model_runner, **kwargs)

        # Lazy import of TurboQuant decode kernels (extend uses parent's Triton kernel)
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
        """Extend with selective gather-dequant (O(prefix) instead of O(pool)).

        The 2-stage Triton extend kernel already uses raw K,V for extend tokens
        (stage 2) and only reads dequantized pool data for prefix tokens (stage 1).
        This override replaces the parent's O(pool_size) full-pool dequant with
        O(prefix_tokens) selective gather-dequant, saving memory and compute.
        """
        from sglang.srt.layers.attention.triton_backend import logit_capping_mod
        from sglang.srt.layers.radix_attention import AttentionType

        # Allocate output
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

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

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        causal = True
        if (
            layer.is_cross_attention
            or layer.attn_type == AttentionType.ENCODER_ONLY
            or (
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
                and self.allow_bidirectional_attention_in_extend
            )
        ):
            causal = False

        # Deterministic mode: delegate to parent (uses different kernel)
        if self.enable_deterministic:
            return self._forward_extend_unified(
                q, o, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        # Resolve kv metadata
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = layer.sliding_window_size
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None

        # Map layer_id for hybrid pools
        if hasattr(pool, "_transfer_full_attention_id"):
            mapped_id = pool._transfer_full_attention_id(layer.layer_id)
        else:
            mapped_id = layer.layer_id

        # Gather-dequant: only dequantize the prefix positions referenced by kv_indices
        if kv_indices.numel() > 0:
            k_buffer = tq_pool.gather_dequant_key(mapped_id, kv_indices)
            v_buffer = tq_pool.gather_dequant_value(mapped_id, kv_indices)
            remapped_kv_indices = torch.arange(
                kv_indices.numel(), device=kv_indices.device, dtype=kv_indices.dtype
            )
        else:
            # No prefix tokens — use parent's full-pool dequant as fallback
            # (returns [pool_size, H, D] but stage 1 loop won't execute with 0 prefix)
            k_buffer = pool.get_key_buffer(layer.layer_id)
            v_buffer = pool.get_value_buffer(layer.layer_id)
            remapped_kv_indices = kv_indices

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k_buffer,
            v_buffer,
            self.forward_metadata.qo_indptr,
            kv_indptr,
            remapped_kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            1.0,  # k_descale (TQ handles scaling via norms, not FP8 scales)
            1.0,  # v_descale
            layer.scaling,
            logit_cap=logits_soft_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
        )
        return o

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
            k_bufs = tq_pool.get_quantized_k_buffers(mapped_id)
            v_bufs = tq_pool.get_quantized_v_buffers(mapped_id)

            q_float = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).float()
            q_rot = tq_pool.k_hadamard[li].forward(q_float)

            padded_dim = tq_pool.k_hadamard[li].padded_dim
            o_rot = torch.empty(
                (q_float.shape[0], layer.tp_q_head_num, padded_dim),
                dtype=torch.float32, device=q.device,
            )

            self.turboquant_decode_attention_fwd(
                q_rot,
                k_bufs["mse_packed"], k_bufs["norms"],
                v_bufs["packed"], v_bufs["norms"],
                tq_pool.k_codebook, tq_pool.v_codebook,
                o_rot,
                kv_indptr, kv_indices,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits, layer.scaling,
                tq_pool.k_mse_bits, tq_pool.int_bits,
                padded_dim,
                self.forward_metadata.attn_logits,
                self.forward_metadata.attn_lse,
            )

            # Inverse rotation with V's rotation (V accumulated in V's rotated space)
            output = tq_pool.v_hadamard[li].inverse(o_rot)
            return output.to(q.dtype).view(-1, layer.tp_q_head_num * layer.v_head_dim)

        else:
            # ============================================================
            # Split-channel path (fractional bits, e.g. 3.5-bit)
            # ============================================================
            k_bufs = tq_pool.get_quantized_k_buffers(mapped_id)
            v_bufs = tq_pool.get_quantized_v_buffers(mapped_id)

            q_float = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).float()
            lo_idx = tq_pool.lo_indices.to(q.device)
            hi_idx = tq_pool.hi_indices.to(q.device)

            q_lo = q_float.index_select(-1, lo_idx)
            q_hi = q_float.index_select(-1, hi_idx)

            q_rot_lo = tq_pool.hadamard_lo[li].forward(q_lo)
            q_rot_hi = tq_pool.hadamard_hi[li].forward(q_hi)

            padded_d_lo = tq_pool.hadamard_lo[li].padded_dim
            padded_d_hi = tq_pool.hadamard_hi[li].padded_dim
            combined_dim = padded_d_lo + padded_d_hi

            o_rot_split = torch.empty(
                (q_float.shape[0], layer.tp_q_head_num, combined_dim),
                dtype=torch.float32, device=q.device,
            )

            self.turboquant_decode_attention_fwd_split(
                q_rot_lo, q_rot_hi,
                k_bufs["lo_packed"], k_bufs["lo_norms"],
                k_bufs["hi_packed"], k_bufs["hi_norms"],
                v_bufs["lo_packed"], v_bufs["lo_norms"],
                v_bufs["hi_packed"], v_bufs["hi_norms"],
                tq_pool.k_cb_lo, tq_pool.k_cb_hi,
                tq_pool.v_cb_lo, tq_pool.v_cb_hi,
                o_rot_split,
                kv_indptr, kv_indices,
                self.forward_metadata.num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                tq_pool.k_lo_bits, tq_pool.k_hi_bits,
                tq_pool.v_lo_bits, tq_pool.v_hi_bits,
                padded_d_lo, padded_d_hi, combined_dim,
                self.forward_metadata.attn_logits,
                self.forward_metadata.attn_lse,
            )

            # Inverse rotation per group with V's rotations, then reassemble
            o_lo = tq_pool.v_hadamard_lo[li].inverse(o_rot_split[..., :padded_d_lo])
            o_hi = tq_pool.v_hadamard_hi[li].inverse(
                o_rot_split[..., padded_d_lo:padded_d_lo + padded_d_hi]
            )

            output_cat = torch.cat([o_lo, o_hi], dim=-1)
            output = output_cat.index_select(-1, tq_pool.restore_order.to(q.device))
            return output.to(q.dtype).view(-1, layer.tp_q_head_num * layer.v_head_dim)
