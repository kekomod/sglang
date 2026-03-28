"""TurboQuant KV cache memory pool — Compact Quantized Storage (Step 2).

Stores KV cache in compact bit-packed buffers (uint8 + fp16 norms) rather
than full BF16. Provides on-demand dequantization for prefill/extend and
direct quantized-buffer access for the Triton decode kernel.

Both keys and values use MSE-only quantization (no QJL). At 3.5-bit with
dim=64 split groups, MSE-only gives 0.989 per-token cosine sim vs prod's
0.948 — QJL's variance dominates the bias correction at these dimensions.

Supports both integer bits (e.g. 3) and fractional bits via split-channel
codec (e.g. 3.5 → lo-group at floor(3.5)=3 bits + hi-group at ceil(3.5)=4 bits).

Reference: arXiv:2504.19874, Algorithm 1
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
    select_outlier_indices,
    split_channel_mse_dequantize,
    split_channel_mse_quantize,
)
from sglang.srt.layers.quantization.turboquant.rotation import (
    HadamardTransform,
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
        use_workspace: bool = True,
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

        # BF16 workspace — only needed for FlashInfer backend (expects pool-sized BF16 tensors)
        self._use_workspace = use_workspace
        if self._use_workspace:
            self.k_workspace = [
                torch.zeros(total_slots, head_num, head_dim, dtype=torch.bfloat16, device=device)
                for _ in range(layer_num)
            ]
            self.v_workspace = [
                torch.zeros(total_slots, head_num, head_dim, dtype=torch.bfloat16, device=device)
                for _ in range(layer_num)
            ]
        else:
            self.k_workspace = None
            self.v_workspace = None
        self._graph_mode = False

        if self._use_workspace:
            ws_bytes = sum(t.nbytes for t in self.k_workspace + self.v_workspace)
            logger.info(f"TurboQuant BF16 workspace allocated: {ws_bytes / 1e9:.2f} GB")
        else:
            logger.info("TurboQuant: no BF16 workspace (fused backend)")

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

        # Both keys and values use full MSE bits per group
        self.k_lo_bits = lo_bits
        self.k_hi_bits = hi_bits
        self.v_lo_bits = lo_bits
        self.v_hi_bits = hi_bits

        logger.info(
            f"TurboQuant compact pool (split-channel): {bits}-bit, "
            f"lo={lo_bits}b ({d_lo}ch), hi={hi_bits}b ({d_hi}ch), "
            f"heads={self.head_num}, layers={self.layer_num}"
        )

        # Per-group codebooks (keys and values use same codebooks at same bits)
        self.k_cb_lo = compute_codebook(d_lo, lo_bits).to(self.device)
        self.k_cb_hi = compute_codebook(d_hi, hi_bits).to(self.device)
        self.v_cb_lo = compute_codebook(d_lo, lo_bits).to(self.device)
        self.v_cb_hi = compute_codebook(d_hi, hi_bits).to(self.device)

        # Per-layer Hadamard transforms (separate per group, separate K vs V)
        self.hadamard_lo = []
        self.hadamard_hi = []
        self.v_hadamard_lo = []
        self.v_hadamard_hi = []
        for i in range(self.layer_num):
            seed_base = self.tq_seed + i * 1000
            self.hadamard_lo.append(HadamardTransform(d_lo, seed_base, torch.device(self.device)))
            self.hadamard_hi.append(HadamardTransform(d_hi, seed_base + 97, torch.device(self.device)))
            self.v_hadamard_lo.append(HadamardTransform(d_lo, seed_base + 500, torch.device(self.device)))
            self.v_hadamard_hi.append(HadamardTransform(d_hi, seed_base + 597, torch.device(self.device)))

        # Allocate packed buffers — use padded_dim from Hadamard
        H = self.head_num
        padded_d_lo = self.hadamard_lo[0].padded_dim
        padded_d_hi = self.hadamard_hi[0].padded_dim
        self.padded_d_lo = padded_d_lo
        self.padded_d_hi = padded_d_hi
        k_lo_pw = packed_width(padded_d_lo, lo_bits)
        k_hi_pw = packed_width(padded_d_hi, hi_bits)
        v_lo_pw = packed_width(padded_d_lo, lo_bits)
        v_hi_pw = packed_width(padded_d_hi, hi_bits)

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

        # Key buffers (MSE per group)
        self.k_lo_packed = _alloc_uint8(k_lo_pw)
        self.k_hi_packed = _alloc_uint8(k_hi_pw)
        self.k_lo_norms = _alloc_fp16()
        self.k_hi_norms = _alloc_fp16()

        # Value buffers (MSE per group)
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
        self.k_mse_bits = bits

        logger.info(
            f"TurboQuant compact pool (integer): {bits}-bit, "
            f"keys={bits} (mse), values={bits} (mse), "
            f"head_dim={self.head_dim}, heads={self.head_num}, layers={self.layer_num}"
        )

        D = self.head_dim

        # Codebooks (keys and values use same codebook at same bits)
        self.k_codebook = compute_codebook(D, bits).to(self.device)
        self.v_codebook = compute_codebook(D, bits).to(self.device)

        # Per-layer Hadamard transforms (separate for K and V — MLX uses seed vs seed+1)
        self.k_hadamard = []
        self.v_hadamard = []
        for i in range(self.layer_num):
            seed = self.tq_seed + i * 1000
            self.k_hadamard.append(HadamardTransform(D, seed, torch.device(self.device)))
            self.v_hadamard.append(HadamardTransform(D, seed + 500, torch.device(self.device)))

        # Allocate packed buffers — use padded_dim from Hadamard
        H = self.head_num
        padded_D = self.k_hadamard[0].padded_dim
        self.padded_dim = padded_D
        k_pw = packed_width(padded_D, bits)
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

        self.k_mse_packed = _alloc_uint8(k_pw)
        self.k_norms = _alloc_fp16()
        self.v_packed = _alloc_uint8(v_pw)
        self.v_norms = _alloc_fp16()

    # ------------------------------------------------------------------
    # CUDA graph mode control
    # ------------------------------------------------------------------

    def set_graph_mode(self, enabled: bool):
        """Enable/disable CUDA graph mode.

        Controls get_key_buffer/get_value_buffer routing: in graph mode with
        workspace, returns BF16 workspace directly (FlashInfer path).
        """
        self._graph_mode = enabled

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

        # Write BF16 workspace if allocated (FlashInfer backend)
        if self._use_workspace:
            self.k_workspace[li][loc] = k.to(torch.bfloat16)
            self.v_workspace[li][loc] = v.to(torch.bfloat16)

        # Always quantize to packed storage (graph-safe)
        if self.is_split:
            self._set_kv_split(li, loc, k, v)
        else:
            self._set_kv_integer(li, loc, k, v)

    def _set_kv_split(self, li: int, loc: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Keys (MSE quantize per group, K rotation)
        k_lo_p, k_lo_n, k_hi_p, k_hi_n = split_channel_mse_quantize(
            k, self.lo_indices, self.hi_indices,
            self.hadamard_lo[li], self.hadamard_hi[li],
            self.k_cb_lo, self.k_cb_hi,
            self.k_lo_bits, self.k_hi_bits,
        )
        self.k_lo_packed[li][loc] = k_lo_p
        self.k_lo_norms[li][loc] = k_lo_n
        self.k_hi_packed[li][loc] = k_hi_p
        self.k_hi_norms[li][loc] = k_hi_n

        # Values (MSE quantize per group, V rotation — independent from K)
        v_lo_p, v_lo_n, v_hi_p, v_hi_n = split_channel_mse_quantize(
            v, self.lo_indices, self.hi_indices,
            self.v_hadamard_lo[li], self.v_hadamard_hi[li],
            self.v_cb_lo, self.v_cb_hi,
            self.v_lo_bits, self.v_hi_bits,
        )
        self.v_lo_packed[li][loc] = v_lo_p
        self.v_lo_norms[li][loc] = v_lo_n
        self.v_hi_packed[li][loc] = v_hi_p
        self.v_hi_norms[li][loc] = v_hi_n

    def _set_kv_integer(self, li: int, loc: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        bits = self.int_bits

        # Keys (MSE quantize with K rotation)
        k_p, k_n = mse_quantize(
            k, self.k_hadamard[li], self.k_codebook, bits,
        )
        self.k_mse_packed[li][loc] = k_p
        self.k_norms[li][loc] = k_n

        # Values (MSE quantize with V rotation — independent from K)
        v_p, v_n = mse_quantize(
            v, self.v_hadamard[li], self.v_codebook, bits,
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
                "lo_packed": self.k_lo_packed[li],
                "lo_norms": self.k_lo_norms[li],
                "hi_packed": self.k_hi_packed[li],
                "hi_norms": self.k_hi_norms[li],
            }
        else:
            return {
                "mse_packed": self.k_mse_packed[li],
                "norms": self.k_norms[li],
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
    # gather_dequant — selective dequant at specific positions only
    # ------------------------------------------------------------------

    def gather_dequant_key(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize keys at specific pool positions only. Returns [len(indices), H, D]."""
        li = layer_id - self.start_layer

        if self.is_split:
            return split_channel_mse_dequantize(
                self.k_lo_packed[li][indices], self.k_lo_norms[li][indices],
                self.k_hi_packed[li][indices], self.k_hi_norms[li][indices],
                self.lo_indices, self.hi_indices, self.restore_order,
                self.hadamard_lo[li], self.hadamard_hi[li],
                self.k_cb_lo, self.k_cb_hi,
                self.k_lo_bits, self.k_hi_bits, self.head_dim,
            ).to(self.dtype)
        else:
            return mse_dequantize(
                self.k_mse_packed[li][indices], self.k_norms[li][indices],
                self.k_hadamard[li], self.k_codebook,
                self.int_bits, self.head_dim,
            ).to(self.dtype)

    def gather_dequant_value(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize values at specific pool positions only. Returns [len(indices), H, D]."""
        li = layer_id - self.start_layer

        if self.is_split:
            return split_channel_mse_dequantize(
                self.v_lo_packed[li][indices], self.v_lo_norms[li][indices],
                self.v_hi_packed[li][indices], self.v_hi_norms[li][indices],
                self.lo_indices, self.hi_indices, self.restore_order,
                self.v_hadamard_lo[li], self.v_hadamard_hi[li],
                self.v_cb_lo, self.v_cb_hi,
                self.v_lo_bits, self.v_hi_bits, self.head_dim,
            ).to(self.dtype)
        else:
            return mse_dequantize(
                self.v_packed[li][indices], self.v_norms[li][indices],
                self.v_hadamard[li], self.v_codebook,
                self.int_bits, self.head_dim,
            ).to(self.dtype)

    # ------------------------------------------------------------------
    # get_key_buffer / get_value_buffer — on-demand dequant for prefill
    # ------------------------------------------------------------------

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Dequantize the entire key buffer for a layer. Returns [pool_size+page, H, D]."""
        if self._use_workspace and self._graph_mode:
            li = layer_id - self.start_layer
            return self.k_workspace[li]

        # No workspace: dequantize from packed buffers (used by parent extend kernel)
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        li = layer_id - self.start_layer

        if self.is_split:
            return split_channel_mse_dequantize(
                self.k_lo_packed[li], self.k_lo_norms[li],
                self.k_hi_packed[li], self.k_hi_norms[li],
                self.lo_indices, self.hi_indices, self.restore_order,
                self.hadamard_lo[li], self.hadamard_hi[li],
                self.k_cb_lo, self.k_cb_hi,
                self.k_lo_bits, self.k_hi_bits, self.head_dim,
            ).to(self.dtype)
        else:
            return mse_dequantize(
                self.k_mse_packed[li], self.k_norms[li],
                self.k_hadamard[li], self.k_codebook,
                self.int_bits, self.head_dim,
            ).to(self.dtype)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """Dequantize the entire value buffer for a layer. Returns [pool_size+page, H, D]."""
        if self._use_workspace and self._graph_mode:
            li = layer_id - self.start_layer
            return self.v_workspace[li]

        # No workspace: dequantize from packed buffers (used by parent extend kernel)

        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        li = layer_id - self.start_layer

        if self.is_split:
            return split_channel_mse_dequantize(
                self.v_lo_packed[li], self.v_lo_norms[li],
                self.v_hi_packed[li], self.v_hi_norms[li],
                self.lo_indices, self.hi_indices, self.restore_order,
                self.v_hadamard_lo[li], self.v_hadamard_hi[li],
                self.v_cb_lo, self.v_cb_hi,
                self.v_lo_bits, self.v_hi_bits, self.head_dim,
            ).to(self.dtype)
        else:
            return mse_dequantize(
                self.v_packed[li], self.v_norms[li],
                self.v_hadamard[li], self.v_codebook,
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
                self.k_lo_packed[li][tgt_loc] = self.k_lo_packed[li][src_loc]
                self.k_hi_packed[li][tgt_loc] = self.k_hi_packed[li][src_loc]
                self.k_lo_norms[li][tgt_loc] = self.k_lo_norms[li][src_loc]
                self.k_hi_norms[li][tgt_loc] = self.k_hi_norms[li][src_loc]
                # Value buffers
                self.v_lo_packed[li][tgt_loc] = self.v_lo_packed[li][src_loc]
                self.v_hi_packed[li][tgt_loc] = self.v_hi_packed[li][src_loc]
                self.v_lo_norms[li][tgt_loc] = self.v_lo_norms[li][src_loc]
                self.v_hi_norms[li][tgt_loc] = self.v_hi_norms[li][src_loc]
                # Workspace buffers (only if allocated)
                if self._use_workspace:
                    self.k_workspace[li][tgt_loc] = self.k_workspace[li][src_loc]
                    self.v_workspace[li][tgt_loc] = self.v_workspace[li][src_loc]
        else:
            for li in range(self.layer_num):
                self.k_mse_packed[li][tgt_loc] = self.k_mse_packed[li][src_loc]
                self.k_norms[li][tgt_loc] = self.k_norms[li][src_loc]
                self.v_packed[li][tgt_loc] = self.v_packed[li][src_loc]
                self.v_norms[li][tgt_loc] = self.v_norms[li][src_loc]
                # Workspace buffers (only if allocated)
                if self._use_workspace:
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
                    self.k_lo_packed[li].nbytes
                    + self.k_hi_packed[li].nbytes
                    + self.k_lo_norms[li].nbytes
                    + self.k_hi_norms[li].nbytes
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
                    + self.k_norms[li].nbytes
                )
                v_bytes += (
                    self.v_packed[li].nbytes
                    + self.v_norms[li].nbytes
                )

        # Add workspace buffer sizes (only if allocated)
        if self._use_workspace:
            for li in range(self.layer_num):
                k_bytes += self.k_workspace[li].nbytes
                v_bytes += self.v_workspace[li].nbytes

        return k_bytes, v_bytes
