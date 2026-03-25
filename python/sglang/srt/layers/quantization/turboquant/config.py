"""TurboQuant quantization configuration for SGLang.

TurboQuant is a KV cache quantization method (not weight quantization).
It quantizes Key and Value vectors during inference using near-optimal
vector quantization based on random rotation + scalar codebook.

Reference: arXiv:2504.19874 (ICLR 2026)
"""

from typing import Any, Dict, List, Optional, Type

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.turboquant.kv_cache_method import (
    TurboQuantKVCacheMethod,
)


class TurboQuantConfig(QuantizationConfig):
    """Configuration for TurboQuant KV cache quantization.

    TurboQuant compresses the KV cache using:
    - Keys: TurboQuant_prod (b-1 bits MSE + 1-bit QJL) for unbiased inner products
    - Values: TurboQuant_mse (b bits MSE) for optimal reconstruction

    Enabled via --kv-cache-quantization turboquant.
    """

    def __init__(self, bits: int = 3, seed: int = 42):
        super().__init__()
        self.bits = int(bits)
        self.seed = seed

    def get_name(self) -> str:
        return "turboquant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # Ampere+

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []  # TurboQuant is runtime, not checkpoint-based

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TurboQuantConfig":
        return cls(
            bits=config.get("bits", 3),
            seed=config.get("seed", 42),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.radix_attention import RadixAttention

        if isinstance(layer, RadixAttention):
            return TurboQuantKVCacheMethod(self)
        return None  # No weight quantization

    def get_scaled_act_names(self) -> List[str]:
        return []
