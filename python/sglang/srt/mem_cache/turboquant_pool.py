"""TurboQuant KV cache memory pool — Compact Quantized Storage (Step 2).

Stores KV cache in compact bit-packed buffers (uint8 + fp16 norms) rather
than full BF16. Provides on-demand dequantization for prefill/extend and
direct quantized-buffer access for the Triton decode kernel.

Supports both integer bits (e.g. 3) and fractional bits via split-channel
codec (e.g. 3.5 → lo-group at floor(3.5)=3 bits + hi-group at ceil(3.5)=4 bits).

Reference: arXiv:2504.19874, Algorithms 1 and 2
"""

import logging
import math
from typing import Optional

import torch

from sglang.srt.layers.quantization.turboquant.codebook import (
    compute_codebook,
    packed_width,
)
from sglang.srt.layers.quantization.turboquant.quant_ops import (
    mse_dequantize,
    mse_quantize,
    prod_dequantize,
    prod_quantize,
    select_outlier_indices,
    split_channel_mse_dequantize,
    split_channel_mse_quantize,
    split_channel_prod_dequantize,
    split_channel_prod_quantize,
)
from sglang.srt.layers.quantization.turboquant.rotation import (
    HadamardTransform,
    projection_matrix,
)
from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class TurboQuantTokenToKVPool(KVCache):
    """Compact TurboQuant pool: stores quantized KV in bit-packed buffers.

    Inherits from KVCache directly (not MHATokenToKVPool) to avoid
    allocating wasted BF16 buffers. Provides:
    - set_kv_buffer: quantize incoming K/V and store compactly
    - get_key_buffer / get_value_buffer: on-demand dequant for prefill
    - get_quantized_k_buffers / get_quantized_v_buffers: raw buffers for Triton kernel
    - move_kv_cache: copy all packed buffers between locations
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        turboquant_bits: float = 3,
        turboquant_seed: int = 42,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim
        self.tq_bits = turboquant_bits
        self.tq_seed = turboquant_seed

        total_slots = size + page_size

        # Detect split vs integer mode
        self.is_split = not math.isclose(
            turboquant_bits, round(turboquant_bits), abs_tol=1e-9
        )

        if self.is_split:
            self._init_split(total_slots)
        else:
            self._init_integer(total_slots)

        # BF16 workspace for CUDA graph mode — mirrors packed storage
        self.k_workspace = [
            torch.zeros(total_slots, head_num, head_dim, dtype=torch.bfloat16, device=device)
            for _ in range(layer_num)
        ]
        self.v_workspace = [
            torch.zeros(total_slots, head_num, head_dim, dtype=torch.bfloat16, device=device)
            for _ in range(layer_num)
        ]
        self._graph_mode = False

        self._finalize_allocation_log(size)

    # ------------------------------------------------------------------
    # Split-channel initialization (fractional bits, e.g. 3.5)
    # ------------------------------------------------------------------

    def _init_split(self, total_slots: int):
        bits = self.tq_bits
        lo_bits = math.floor(bits)
        hi_bits = math.ceil(bits)
        self.lo_bits = lo_bits
        self.hi_bits = hi_bits

        # Channel split — move to device at init to avoid runtime host→device copies
        lo_indices, hi_indices = select_outlier_indices(self.head_dim, bits)
        self.lo_indices = lo_indices.to(self.device)
        self.hi_indices = hi_indices.to(self.device)
        self.restore_order = torch.argsort(
            torch.cat([lo_indices, hi_indices])
        ).to(self.device)
        d_lo = lo_indices.shape[0]
        d_hi = hi_indices.shape[0]
        self.d_lo = d_lo
        self.d_hi = d_hi

        # Key bits: prod uses (group_bits - 1) for MSE + 1 for QJL
        self.k_lo_mse_bits = max(lo_bits - 1, 0)
        self.k_hi_mse_bits = max(hi_bits - 1, 0)
        # Value bits: mse uses full group_bits
        self.v_lo_bits = lo_bits
        self.v_hi_bits = hi_bits

        logger.info(
            f"TurboQuant compact pool (split-channel): {bits}-bit, "
            f"lo={lo_bits}b ({d_lo}ch), hi={hi_bits}b ({d_hi}ch), "
            f"heads={self.head_num}, layers={self.layer_num}"
        )

        # Per-group codebooks
        self.k_cb_lo = compute_codebook(d_lo, self.k_lo_mse_bits).to(self.device)
        self.k_cb_hi = compute_codebook(d_hi, self.k_hi_mse_bits).to(self.device)
        self.v_cb_lo = compute_codebook(d_lo, self.v_lo_bits).to(self.device)
        self.v_cb_hi = compute_codebook(d_hi, self.v_hi_bits).to(self.device)

        # Per-layer Hadamard transforms and QJL projection matrices (separate per group)
        self.hadamard_lo = []
        self.hadamard_hi = []
        self.s_lo = []
        self.s_hi = []
        for i in range(self.layer_num):
            seed_base = self.tq_seed + i * 1000
            self.hadamard_lo.append(HadamardTransform(d_lo, seed_base, torch.device(self.device)))
            self.hadamard_hi.append(HadamardTransform(d_hi, seed_base + 97, torch.device(self.device)))
            self.s_lo.append(projection_matrix(d_lo, seed_base).to(self.device))
            self.s_hi.append(projection_matrix(d_hi, seed_base + 97).to(self.device))

        # Allocate packed buffers — use padded_dim from Hadamard
        H = self.head_num
        padded_d_lo = self.hadamard_lo[0].padded_dim
        padded_d_hi = self.hadamard_hi[0].padded_dim
        self.padded_d_lo = padded_d_lo
        self.padded_d_hi = padded_d_hi
        k_lo_mse_pw = packed_width(padded_d_lo, self.k_lo_mse_bits)
        k_lo_qjl_pw = packed_width(d_lo, 1)  # QJL on original dim
        k_hi_mse_pw = packed_width(padded_d_hi, self.k_hi_mse_bits)
        k_hi_qjl_pw = packed_width(d_hi, 1)  # QJL on original dim
        v_lo_pw = packed_width(padded_d_lo, self.v_lo_bits)
        v_hi_pw = packed_width(padded_d_hi, self.v_hi_bits)

        def _alloc_uint8(last_dim):
            return [
                torch.zeros(total_slots, H, last_dim, dtype=torch.uint8, device=self.device)
                for _ in range(self.layer_num)
            ]

        def _alloc_fp16():
            return [
                torch.zeros(total_slots, H, dtype=torch.float16, device=self.device)
                for _ in range(self.layer_num)
            ]

        # Key buffers (prod: MSE + QJL per group)
        self.k_lo_mse_packed = _alloc_uint8(k_lo_mse_pw)
        self.k_lo_qjl_packed = _alloc_uint8(k_lo_qjl_pw)
        self.k_hi_mse_packed = _alloc_uint8(k_hi_mse_pw)
        self.k_hi_qjl_packed = _alloc_uint8(k_hi_qjl_pw)
        self.k_lo_norms = _alloc_fp16()
        self.k_lo_res_norms = _alloc_fp16()
        self.k_hi_norms = _alloc_fp16()
        self.k_hi_res_norms = _alloc_fp16()

        # Value buffers (mse per group)
        self.v_lo_packed = _alloc_uint8(v_lo_pw)
        self.v_hi_packed = _alloc_uint8(v_hi_pw)
        self.v_lo_norms = _alloc_fp16()
        self.v_hi_norms = _alloc_fp16()

    # ------------------------------------------------------------------
    # Integer-bit initialization (e.g. 3-bit)
    # ------------------------------------------------------------------

    def _init_integer(self, total_slots: int):
        bits = int(round(self.tq_bits))
        self.int_bits = bits
        mse_bits = max(bits - 1, 0)
        self.k_mse_bits = mse_bits

        logger.info(
            f"TurboQuant compact pool (integer): {bits}-bit, "
            f"keys={mse_bits}+1 (prod), values={bits} (mse), "
            f"head_dim={self.head_dim}, heads={self.head_num}, layers={self.layer_num}"
        )

        D = self.head_dim

        # Codebooks
        self.k_codebook = compute_codebook(D, mse_bits).to(self.device)
        self.v_codebook = compute_codebook(D, bits).to(self.device)

        # Per-layer Hadamard transforms and QJL projection matrices
        self.k_hadamard = []
        self.s_matrices = []
        for i in range(self.layer_num):
            seed = self.tq_seed + i * 1000
            self.k_hadamard.append(HadamardTransform(D, seed, torch.device(self.device)))
            self.s_matrices.append(projection_matrix(D, seed).to(self.device))

        # Allocate packed buffers — use padded_dim from Hadamard
        H = self.head_num
        padded_D = self.k_hadamard[0].padded_dim
        self.padded_dim = padded_D
        k_mse_pw = packed_width(padded_D, mse_bits)
        k_qjl_pw = packed_width(D, 1)  # QJL is on original dim (residual in original space)
        v_pw = packed_width(padded_D, bits)

        def _alloc_uint8(last_dim):
            return [
                torch.zeros(total_slots, H, last_dim, dtype=torch.uint8, device=self.device)
                for _ in range(self.layer_num)
            ]

        def _alloc_fp16():
            return [
                torch.zeros(total_slots, H, dtype=torch.float16, device=self.device)
                for _ in range(self.layer_num)
            ]

        self.k_mse_packed = _alloc_uint8(k_mse_pw)
        self.k_qjl_packed = _alloc_uint8(k_qjl_pw)
        self.k_norms = _alloc_fp16()
        self.k_res_norms = _alloc_fp16()
        self.v_packed = _alloc_uint8(v_pw)
        self.v_norms = _alloc_fp16()

    # ------------------------------------------------------------------
    # CUDA graph mode control
    # ------------------------------------------------------------------

    def set_graph_mode(self, enabled: bool):
        """Enable/disable CUDA graph mode.

        In graph mode, set_kv_buffer only writes to BF16 workspace (graph-safe
        indexed scatter) and skips quantization. After graph replay, call
        quant_new_tokens() to quantize the workspace into packed storage.
        """
        self._graph_mode = enabled

    def quant_new_tokens(self, loc: torch.Tensor):
        """Quantize newly generated tokens from BF16 workspace into packed storage.

        Called after CUDA graph replay to persist the BF16 workspace data
        into the compact quantized buffers.
        """
        for li in range(self.layer_num):
            k = self.k_workspace[li][loc]  # [batch, H, D] bf16
            v = self.v_workspace[li][loc]  # [batch, H, D] bf16
            if self.is_split:
                self._set_kv_split(li, loc, k, v)
            else:
                self._set_kv_integer(li, loc, k, v)

    # ------------------------------------------------------------------
    # set_kv_buffer — quantize and store compactly
    # ------------------------------------------------------------------

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ) -> None:
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        li = layer_id - self.start_layer

        k = cache_k.view(-1, self.head_num, self.head_dim)
        v = cache_v.view(-1, self.head_num, self.head_dim)

        # Always write BF16 workspace (graph-safe indexed scatter)
        self.k_workspace[li][loc] = k.to(torch.bfloat16)
        self.v_workspace[li][loc] = v.to(torch.bfloat16)

        # Quantize to packed storage only outside graph capture
        if not self._graph_mode:
            if self.is_split:
                self._set_kv_split(li, loc, k, v)
            else:
                self._set_kv_integer(li, loc, k, v)

    def _set_kv_split(self, li: int, loc: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Keys (prod quantize per group)
        (lo_mse_p, lo_qjl_p, lo_n, lo_rn,
         hi_mse_p, hi_qjl_p, hi_n, hi_rn) = split_channel_prod_quantize(
            k, self.lo_indices, self.hi_indices,
            self.hadamard_lo[li], self.hadamard_hi[li],
            self.s_lo[li], self.s_hi[li],
            self.k_cb_lo, self.k_cb_hi,
            self.lo_bits, self.hi_bits,
        )
        self.k_lo_mse_packed[li][loc] = lo_mse_p
        self.k_lo_qjl_packed[li][loc] = lo_qjl_p
        self.k_lo_norms[li][loc] = lo_n
        self.k_lo_res_norms[li][loc] = lo_rn
        self.k_hi_mse_packed[li][loc] = hi_mse_p
        self.k_hi_qjl_packed[li][loc] = hi_qjl_p
        self.k_hi_norms[li][loc] = hi_n
        self.k_hi_res_norms[li][loc] = hi_rn

        # Values (mse quantize per group)
        lo_vp, lo_vn, hi_vp, hi_vn = split_channel_mse_quantize(
            v, self.lo_indices, self.hi_indices,
            self.hadamard_lo[li], self.hadamard_hi[li],
            self.v_cb_lo, self.v_cb_hi,
            self.v_lo_bits, self.v_hi_bits,
        )
        self.v_lo_packed[li][loc] = lo_vp
        self.v_lo_norms[li][loc] = lo_vn
        self.v_hi_packed[li][loc] = hi_vp
        self.v_hi_norms[li][loc] = hi_vn

    def _set_kv_integer(self, li: int, loc: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        bits = self.int_bits

        # Keys (prod quantize)
        mse_p, qjl_p, k_n, k_rn = prod_quantize(
            k, self.k_hadamard[li], self.s_matrices[li],
            self.k_codebook, bits,
        )
        self.k_mse_packed[li][loc] = mse_p
        self.k_qjl_packed[li][loc] = qjl_p
        self.k_norms[li][loc] = k_n
        self.k_res_norms[li][loc] = k_rn

        # Values (mse quantize)
        v_p, v_n = mse_quantize(
            v, self.k_hadamard[li], self.v_codebook, bits,
        )
        self.v_packed[li][loc] = v_p
        self.v_norms[li][loc] = v_n

    # ------------------------------------------------------------------
    # get_quantized_*_buffers — raw buffers for Triton decode kernel
    # ------------------------------------------------------------------

    def get_quantized_k_buffers(self, layer_id: int) -> dict:
        """Return all K packed buffers for the Triton decode kernel."""
        li = layer_id - self.start_layer
        if self.is_split:
            return {
                "lo_mse_packed": self.k_lo_mse_packed[li],
                "lo_qjl_packed": self.k_lo_qjl_packed[li],
                "lo_norms": self.k_lo_norms[li],
                "lo_res_norms": self.k_lo_res_norms[li],
                "hi_mse_packed": self.k_hi_mse_packed[li],
                "hi_qjl_packed": self.k_hi_qjl_packed[li],
                "hi_norms": self.k_hi_norms[li],
                "hi_res_norms": self.k_hi_res_norms[li],
            }
        else:
            return {
                "mse_packed": self.k_mse_packed[li],
                "qjl_packed": self.k_qjl_packed[li],
                "norms": self.k_norms[li],
                "res_norms": self.k_res_norms[li],
            }

    def get_quantized_v_buffers(self, layer_id: int) -> dict:
        """Return all V packed buffers for the Triton decode kernel."""
        li = layer_id - self.start_layer
        if self.is_split:
            return {
                "lo_packed": self.v_lo_packed[li],
                "lo_norms": self.v_lo_norms[li],
                "hi_packed": self.v_hi_packed[li],
                "hi_norms": self.v_hi_norms[li],
            }
        else:
            return {
                "packed": self.v_packed[li],
                "norms": self.v_norms[li],
            }

    # ------------------------------------------------------------------
    # get_key_buffer / get_value_buffer — on-demand dequant for prefill
    # ------------------------------------------------------------------

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Dequantize the entire key buffer for a layer. Returns [pool_size+page, H, D]."""
        if self._graph_mode:
            li = layer_id - self.start_layer
            return self.k_workspace[li]

        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        li = layer_id - self.start_layer

        if self.is_split:
            return split_channel_prod_dequantize(
                self.k_lo_mse_packed[li], self.k_lo_qjl_packed[li],
                self.k_lo_norms[li], self.k_lo_res_norms[li],
                self.k_hi_mse_packed[li], self.k_hi_qjl_packed[li],
                self.k_hi_norms[li], self.k_hi_res_norms[li],
                self.lo_indices, self.hi_indices, self.restore_order,
                self.hadamard_lo[li], self.hadamard_hi[li],
                self.s_lo[li], self.s_hi[li],
                self.k_cb_lo, self.k_cb_hi,
                self.lo_bits, self.hi_bits, self.head_dim,
            ).to(self.dtype)
        else:
            return prod_dequantize(
                self.k_mse_packed[li], self.k_qjl_packed[li],
                self.k_norms[li], self.k_res_norms[li],
                self.k_hadamard[li], self.s_matrices[li],
                self.k_codebook, self.int_bits, self.head_dim,
            ).to(self.dtype)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """Dequantize the entire value buffer for a layer. Returns [pool_size+page, H, D]."""
        if self._graph_mode:
            li = layer_id - self.start_layer
            return self.v_workspace[li]

        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        li = layer_id - self.start_layer

        if self.is_split:
            return split_channel_mse_dequantize(
                self.v_lo_packed[li], self.v_lo_norms[li],
                self.v_hi_packed[li], self.v_hi_norms[li],
                self.lo_indices, self.hi_indices, self.restore_order,
                self.hadamard_lo[li], self.hadamard_hi[li],
                self.v_cb_lo, self.v_cb_hi,
                self.v_lo_bits, self.v_hi_bits, self.head_dim,
            ).to(self.dtype)
        else:
            return mse_dequantize(
                self.v_packed[li], self.v_norms[li],
                self.k_hadamard[li], self.v_codebook,
                self.int_bits, self.head_dim,
            ).to(self.dtype)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    # ------------------------------------------------------------------
    # move_kv_cache — copy packed buffers between token locations
    # ------------------------------------------------------------------

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return

        if self.is_split:
            for li in range(self.layer_num):
                # Key buffers
                self.k_lo_mse_packed[li][tgt_loc] = self.k_lo_mse_packed[li][src_loc]
                self.k_lo_qjl_packed[li][tgt_loc] = self.k_lo_qjl_packed[li][src_loc]
                self.k_hi_mse_packed[li][tgt_loc] = self.k_hi_mse_packed[li][src_loc]
                self.k_hi_qjl_packed[li][tgt_loc] = self.k_hi_qjl_packed[li][src_loc]
                self.k_lo_norms[li][tgt_loc] = self.k_lo_norms[li][src_loc]
                self.k_lo_res_norms[li][tgt_loc] = self.k_lo_res_norms[li][src_loc]
                self.k_hi_norms[li][tgt_loc] = self.k_hi_norms[li][src_loc]
                self.k_hi_res_norms[li][tgt_loc] = self.k_hi_res_norms[li][src_loc]
                # Value buffers
                self.v_lo_packed[li][tgt_loc] = self.v_lo_packed[li][src_loc]
                self.v_hi_packed[li][tgt_loc] = self.v_hi_packed[li][src_loc]
                self.v_lo_norms[li][tgt_loc] = self.v_lo_norms[li][src_loc]
                self.v_hi_norms[li][tgt_loc] = self.v_hi_norms[li][src_loc]
                # Workspace buffers
                self.k_workspace[li][tgt_loc] = self.k_workspace[li][src_loc]
                self.v_workspace[li][tgt_loc] = self.v_workspace[li][src_loc]
        else:
            for li in range(self.layer_num):
                self.k_mse_packed[li][tgt_loc] = self.k_mse_packed[li][src_loc]
                self.k_qjl_packed[li][tgt_loc] = self.k_qjl_packed[li][src_loc]
                self.k_norms[li][tgt_loc] = self.k_norms[li][src_loc]
                self.k_res_norms[li][tgt_loc] = self.k_res_norms[li][src_loc]
                self.v_packed[li][tgt_loc] = self.v_packed[li][src_loc]
                self.v_norms[li][tgt_loc] = self.v_norms[li][src_loc]
                # Workspace buffers
                self.k_workspace[li][tgt_loc] = self.k_workspace[li][src_loc]
                self.v_workspace[li][tgt_loc] = self.v_workspace[li][src_loc]

    # ------------------------------------------------------------------
    # get_kv_size_bytes — total memory usage
    # ------------------------------------------------------------------

    def get_kv_size_bytes(self):
        k_bytes = 0
        v_bytes = 0

        if self.is_split:
            for li in range(self.layer_num):
                k_bytes += (
                    self.k_lo_mse_packed[li].nbytes
                    + self.k_lo_qjl_packed[li].nbytes
                    + self.k_hi_mse_packed[li].nbytes
                    + self.k_hi_qjl_packed[li].nbytes
                    + self.k_lo_norms[li].nbytes
                    + self.k_lo_res_norms[li].nbytes
                    + self.k_hi_norms[li].nbytes
                    + self.k_hi_res_norms[li].nbytes
                )
                v_bytes += (
                    self.v_lo_packed[li].nbytes
                    + self.v_hi_packed[li].nbytes
                    + self.v_lo_norms[li].nbytes
                    + self.v_hi_norms[li].nbytes
                )
        else:
            for li in range(self.layer_num):
                k_bytes += (
                    self.k_mse_packed[li].nbytes
                    + self.k_qjl_packed[li].nbytes
                    + self.k_norms[li].nbytes
                    + self.k_res_norms[li].nbytes
                )
                v_bytes += (
                    self.v_packed[li].nbytes
                    + self.v_norms[li].nbytes
                )

        # Add workspace buffer sizes
        for li in range(self.layer_num):
            k_bytes += self.k_workspace[li].nbytes
            v_bytes += self.v_workspace[li].nbytes

        return k_bytes, v_bytes
