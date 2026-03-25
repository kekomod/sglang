"""TurboQuant KV cache memory pool for SGLang.

Stores KV cache in TurboQuant compressed format. Dequantizes into
shared temporary buffers on read so existing attention backends
(FlashInfer, etc.) work unchanged.

Memory architecture:
- Compact quantized buffers per layer (keys: prod, values: mse)
- 1 shared dequant buffer pair reused across layers (since layers
  are processed sequentially)

Reference: arXiv:2504.19874, Algorithms 1 and 2
"""

import logging
from typing import Optional, Tuple

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
)
from sglang.srt.layers.quantization.turboquant.rotation import (
    projection_matrix,
    rotation_matrix,
)
from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class TurboQuantTokenToKVPool(KVCache):
    """KV cache pool with TurboQuant quantized storage.

    Keys use TurboQuant_prod (b-1 bits MSE + 1-bit QJL) for unbiased
    inner product estimation. Values use TurboQuant_mse (b bits MSE)
    for optimal reconstruction.
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
        turboquant_bits: int = 3,
        turboquant_seed: int = 42,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size, page_size, dtype, layer_num, device,
            enable_memory_saver, start_layer, end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim
        self.bits = turboquant_bits
        self.seed = turboquant_seed
        self.pool_size = size + page_size

        mse_bits = max(turboquant_bits - 1, 0)  # For keys (prod codec)
        val_bits = turboquant_bits  # For values (mse codec)

        logger.info(
            f"TurboQuant pool: {turboquant_bits}-bit, "
            f"keys={mse_bits}+1 (prod), values={val_bits} (mse), "
            f"head_dim={head_dim}, heads={head_num}, layers={layer_num}"
        )

        # Pre-compute codebooks (shared across layers, depends on dim+bits)
        self.k_codebook = compute_codebook(head_dim, mse_bits).to(device)
        self.v_codebook = compute_codebook(head_dim, val_bits).to(device)

        # Per-layer rotation and projection matrices
        self.pi_matrices = []
        self.pi_t_matrices = []
        self.s_matrices = []
        for i in range(layer_num):
            layer_seed = turboquant_seed + i * 1000
            pi = rotation_matrix(head_dim, layer_seed).to(device)
            self.pi_matrices.append(pi)
            self.pi_t_matrices.append(pi.T.contiguous())
            s = projection_matrix(head_dim, layer_seed).to(device)
            self.s_matrices.append(s)

        # Quantized storage buffers (compact)
        k_mse_pw = packed_width(head_dim, mse_bits)
        k_qjl_pw = packed_width(head_dim, 1)
        v_pw = packed_width(head_dim, val_bits)

        self.k_mse_buffer = [
            torch.zeros(self.pool_size, head_num, k_mse_pw, dtype=torch.uint8, device=device)
            for _ in range(layer_num)
        ]
        self.k_qjl_buffer = [
            torch.zeros(self.pool_size, head_num, k_qjl_pw, dtype=torch.uint8, device=device)
            for _ in range(layer_num)
        ]
        self.k_norm_buffer = [
            torch.zeros(self.pool_size, head_num, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ]
        self.k_residual_norm_buffer = [
            torch.zeros(self.pool_size, head_num, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ]
        self.v_packed_buffer = [
            torch.zeros(self.pool_size, head_num, v_pw, dtype=torch.uint8, device=device)
            for _ in range(layer_num)
        ]
        self.v_norm_buffer = [
            torch.zeros(self.pool_size, head_num, dtype=torch.float16, device=device)
            for _ in range(layer_num)
        ]

        # Shared dequant buffers (1 pair, reused across layers)
        self.k_dequant = torch.zeros(
            self.pool_size, head_num, head_dim, dtype=self.store_dtype, device=device
        )
        self.v_dequant = torch.zeros(
            self.pool_size, head_num, head_dim, dtype=self.store_dtype, device=device
        )
        self._dequant_layer_id = -1

        # Memory stats
        quant_bytes = sum(
            b.nelement() * b.element_size()
            for bufs in [
                self.k_mse_buffer, self.k_qjl_buffer, self.k_norm_buffer,
                self.k_residual_norm_buffer, self.v_packed_buffer, self.v_norm_buffer,
            ]
            for b in bufs
        )
        dequant_bytes = (self.k_dequant.nelement() + self.v_dequant.nelement()) * self.k_dequant.element_size()
        full_bytes = self.pool_size * head_num * head_dim * 2 * 2 * layer_num  # FP16 K+V
        logger.info(
            f"TurboQuant memory: quantized={quant_bytes / 1e9:.2f}GB, "
            f"dequant_buffer={dequant_bytes / 1e9:.2f}GB, "
            f"vs full FP16={full_bytes / 1e9:.2f}GB, "
            f"savings={full_bytes / (quant_bytes + dequant_bytes):.1f}x"
        )

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

        # Reshape to [N, H, D] if needed
        k = cache_k.view(-1, self.head_num, self.head_dim)
        v = cache_v.view(-1, self.head_num, self.head_dim)

        # Quantize keys (TurboQuant_prod)
        mse_packed, qjl_packed, k_norms, k_res_norms = prod_quantize(
            k, self.pi_matrices[li], self.s_matrices[li],
            self.k_codebook, self.bits,
        )
        self.k_mse_buffer[li][loc] = mse_packed
        self.k_qjl_buffer[li][loc] = qjl_packed
        self.k_norm_buffer[li][loc] = k_norms
        self.k_residual_norm_buffer[li][loc] = k_res_norms

        # Quantize values (TurboQuant_mse)
        v_packed, v_norms = mse_quantize(
            v, self.pi_matrices[li], self.v_codebook, self.bits,
        )
        self.v_packed_buffer[li][loc] = v_packed
        self.v_norm_buffer[li][loc] = v_norms

        # Write dequantized to shared buffer for immediate attention use
        k_recon = prod_dequantize(
            mse_packed, qjl_packed, k_norms, k_res_norms,
            self.pi_t_matrices[li], self.s_matrices[li],
            self.k_codebook, self.bits, self.head_dim,
        )
        self.k_dequant[loc] = k_recon.to(self.store_dtype)

        v_recon = mse_dequantize(
            v_packed, v_norms,
            self.pi_t_matrices[li], self.v_codebook,
            self.bits, self.head_dim,
        )
        self.v_dequant[loc] = v_recon.to(self.store_dtype)

        self._dequant_layer_id = layer_id

    def _ensure_dequant(self, layer_id: int) -> None:
        """Dequantize the full pool for a given layer into shared buffers."""
        if self._dequant_layer_id == layer_id:
            return

        li = layer_id - self.start_layer

        k_recon = prod_dequantize(
            self.k_mse_buffer[li], self.k_qjl_buffer[li],
            self.k_norm_buffer[li], self.k_residual_norm_buffer[li],
            self.pi_t_matrices[li], self.s_matrices[li],
            self.k_codebook, self.bits, self.head_dim,
        )
        self.k_dequant.copy_(k_recon.to(self.store_dtype))

        v_recon = mse_dequantize(
            self.v_packed_buffer[li], self.v_norm_buffer[li],
            self.pi_t_matrices[li], self.v_codebook,
            self.bits, self.head_dim,
        )
        self.v_dequant.copy_(v_recon.to(self.store_dtype))

        self._dequant_layer_id = layer_id

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._ensure_dequant(layer_id)
        if self.store_dtype != self.dtype:
            return self.k_dequant.view(self.dtype)
        return self.k_dequant

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        self._ensure_dequant(layer_id)
        if self.store_dtype != self.dtype:
            return self.v_dequant.view(self.dtype)
        return self.v_dequant

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_kv_size_bytes(self):
        """Return (k_size_bytes, v_size_bytes) for memory accounting."""
        k_bytes = sum(
            b.nelement() * b.element_size()
            for bufs in [self.k_mse_buffer, self.k_qjl_buffer,
                         self.k_norm_buffer, self.k_residual_norm_buffer]
            for b in bufs
        )
        # Include shared dequant buffer for keys
        k_bytes += self.k_dequant.nelement() * self.k_dequant.element_size()

        v_bytes = sum(
            b.nelement() * b.element_size()
            for bufs in [self.v_packed_buffer, self.v_norm_buffer]
            for b in bufs
        )
        v_bytes += self.v_dequant.nelement() * self.v_dequant.element_size()

        return k_bytes, v_bytes

    def get_contiguous_buf_infos(self):
        """Return buffer info for disaggregated serving (uses dequant buffers)."""
        kv_data_ptrs = [self.k_dequant.data_ptr(), self.v_dequant.data_ptr()]
        kv_data_lens = [
            self.k_dequant.nelement() * self.k_dequant.element_size(),
            self.v_dequant.nelement() * self.v_dequant.element_size(),
        ]
        kv_item_lens = [
            self.head_num * self.head_dim * self.k_dequant.element_size(),
            self.head_num * self.head_dim * self.v_dequant.element_size(),
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens
