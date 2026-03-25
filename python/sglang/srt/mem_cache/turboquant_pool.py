"""TurboQuant KV cache memory pool for SGLang — Passthrough (Step 1).

Quantizes then immediately dequantizes KV vectors, storing the result
as BF16 in a standard MHATokenToKVPool. This validates the TurboQuant
quantization math without memory savings. Memory savings come in Step 2
with a custom Triton attention kernel that reads quantized data directly.

Reference: arXiv:2504.19874, Algorithms 1 and 2
"""

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.turboquant.codebook import compute_codebook
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
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)


class TurboQuantTokenToKVPool(MHATokenToKVPool):
    """Passthrough TurboQuant pool: quantize -> dequantize -> store as BF16.

    Inherits all buffer management from MHATokenToKVPool. Only overrides
    set_kv_buffer() to apply quant/dequant distortion. FlashInfer and all
    other attention backends work unchanged.
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
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.tq_bits = turboquant_bits
        self.tq_head_dim = head_dim
        self.tq_head_num = head_num

        mse_bits = max(turboquant_bits - 1, 0)

        logger.info(
            f"TurboQuant passthrough pool: {turboquant_bits}-bit, "
            f"keys={mse_bits}+1 (prod), values={turboquant_bits} (mse), "
            f"head_dim={head_dim}, heads={head_num}, layers={layer_num}"
        )

        # Pre-compute codebooks
        self.k_codebook = compute_codebook(head_dim, mse_bits).to(device)
        self.v_codebook = compute_codebook(head_dim, turboquant_bits).to(device)

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

        k = cache_k.view(-1, self.tq_head_num, self.tq_head_dim)
        v = cache_v.view(-1, self.tq_head_num, self.tq_head_dim)

        # Quantize keys (TurboQuant_prod) then immediately dequantize
        mse_p, qjl_p, k_n, k_rn = prod_quantize(
            k, self.pi_matrices[li], self.s_matrices[li],
            self.k_codebook, self.tq_bits,
        )
        k_recon = prod_dequantize(
            mse_p, qjl_p, k_n, k_rn,
            self.pi_t_matrices[li], self.s_matrices[li],
            self.k_codebook, self.tq_bits, self.tq_head_dim,
        )

        # Quantize values (TurboQuant_mse) then immediately dequantize
        v_p, v_n = mse_quantize(
            v, self.pi_matrices[li], self.v_codebook, self.tq_bits,
        )
        v_recon = mse_dequantize(
            v_p, v_n,
            self.pi_t_matrices[li], self.v_codebook,
            self.tq_bits, self.tq_head_dim,
        )

        # Store the dequantized (lossy) result in parent's standard buffers
        super().set_kv_buffer(
            layer, loc,
            k_recon.to(cache_k.dtype),
            v_recon.to(cache_v.dtype),
            k_scale=k_scale,
            v_scale=v_scale,
            layer_id_override=layer_id_override,
        )
