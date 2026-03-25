"""TurboQuant KV cache quantization method.

Attaches TurboQuant configuration to RadixAttention layers.
The actual quantization/dequantization happens in the memory pool
(TurboQuantTokenToKVPool), not here.
"""

from typing import Optional

import torch

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase


class TurboQuantKVCacheMethod(QuantizeMethodBase):
    """Quantization method that gets attached to RadixAttention layers."""

    def __init__(self, config):
        self.config = config

    def create_weights(self, layer: torch.nn.Module):
        """Set scale factors to 1.0 — TurboQuant manages its own scaling."""
        layer.k_scale = 1.0
        layer.v_scale = 1.0
        layer.k_scale_float = 1.0
        layer.v_scale_float = 1.0
        layer.turboquant_config = self.config

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise RuntimeError(
            "TurboQuantKVCacheMethod.apply should not be called directly. "
            "Quantization happens in TurboQuantTokenToKVPool.set_kv_buffer()."
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass  # No checkpoint weights to process
